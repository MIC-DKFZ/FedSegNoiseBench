import copy
import datetime
from functools import reduce
from operator import add
from typing import Any, Tuple, Optional, cast
import matplotlib.pyplot as plt

import torch
from torch import autocast

import torch.nn.functional as F
from torch import Tensor


from nnunetv2.utilities.helpers import dummy_context

from methods.fedavg.fedavg import FedAvg


class FedDM(FedAvg):
    """
    FedDM: Federated Weakly Supervised Segmentation via Annotation Calibration and Gradient De-Conflicting
    Meilu Zhu et al., 2023, IEEE Transactions on Medical Imaging
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10013742&tag=1
    """

    def __init__(
        self,
        clients: list = None,
        feddm_gamma_hgd_smoothing: float = None,
        feddm_ratio_cac_pixelselection: float = None,
        feddm_cac_label_correction: str = None,
        feddm_loss: str = None,
    ):
        super().__init__(clients=clients)

        self.name = "feddm"
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # originally they initialize grad_history w/ key=layer_name and value=None
        # we are working based on model weight#s addresses
        model_weights0 = self.clients[0].model.current_model_weights
        keys = list(model_weights0.keys())
        address_key_dict = {}
        for k in keys:
            address = model_weights0[k].data_ptr()
            if address not in address_key_dict.keys():
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)
        self.grad_history = {x: None for x in address_key_dict.keys()}
        self.grad_len = self.initialize_grad_len(
            model_weights0, address_key_dict, self.grad_history
        )
        self.grad_history["grad_len"] = self.grad_len

        self.mc_n_samples = 100  # for noisy predictions
        self.batch_size = self.clients[
            0
        ].model.nnunet_trainer.configuration_manager.batch_size
        self.patch_size = self.clients[
            0
        ].model.nnunet_trainer.configuration_manager.patch_size
        self.num_channels = len(
            self.clients[0].model.nnunet_trainer.dataset_json["channel_names"]
        )
        self.n_classes = len(
            self.clients[0].model.nnunet_trainer.dataset_json["labels"]
        )
        # Monte-Carlo Sampling
        self.gauss_noise = torch.rand(
            self.mc_n_samples, self.num_channels, *self.patch_size
        )
        self.embeddings = [
            torch.zeros(self.mc_n_samples, self.n_classes, *self.patch_size)
            for _ in self.clients
        ]

        # some other hparams
        self.sm_temp = 1.0  # softmax temperature
        self.ratio = feddm_ratio_cac_pixelselection  # ratio for pixel selection
        # feddm stop epoch ~= 1/4 of total training epochs
        self.stop_epoch = self.clients[0].fl_args["num_rounds"] // 4
        assert self.stop_epoch > 0, "FedDM stop_epoch must be > 0!"
        # self.stop_epoch = 50
        self.eps = 1e-6
        self.cac_label_correction = feddm_cac_label_correction
        self.feddm_loss = feddm_loss

        # HDG parameters
        self.gamma_hgd_smoothing = feddm_gamma_hgd_smoothing

        self.clients_peers = {
            id: {x: {"id": None, "model": None} for x in ["nearest", "farthest"]}
            for id in range(len(clients))
        }

    def initialize_grad_len(self, model, address_key_dict, grad_history):
        """
        Initialize gradient length for each key in the grad_history via model weight's address.
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L608C1-L616C20
        """
        grad_len = {add: 0 for add in address_key_dict.keys()}
        for add in grad_len.keys():
            dims = model[address_key_dict[add][0]].shape
            grad_len[add] += dims.numel()
        return grad_len

    def do_trainstep_peer(
        self,
        net: Any,
        batch: Any,
        epoch: int,
        loss_fn: Any = None,
        optimizer: Any = None,
        peer_models=None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L207
        Adapted to work with nnUNetv2 and as nnUNetv2's def train_step().
        """
        # determine whether processed images are 2d or 3d
        is_3d = len(batch["data"].shape) == 5  # B, C, H, W, D

        # get nearest peer model and set weights
        peer_model_nearst_statedict = self.clients_peers[next(iter(peer_models))][
            "nearest"
        ]["model"]
        nearest_idx = self.clients_peers[next(iter(peer_models))]["nearest"]["id"]
        nnunet_trainer_w_new_weights_nearst, actual_statedict_nearst = (
            self.assign_model_weights_to_trainer(
                client_idx=nearest_idx,
                new_statedict=peer_model_nearst_statedict,
            )
        )
        peer_model_nearst = nnunet_trainer_w_new_weights_nearst.network.eval()
        peer_model_nearst.to("cpu")
        # get farthest peer model and set weights
        peer_model_farthest_statedict = self.clients_peers[next(iter(peer_models))][
            "farthest"
        ]["model"]
        farthest_idx = self.clients_peers[next(iter(peer_models))]["farthest"]["id"]
        nnunet_trainer_w_new_weights_farthest, actual_statedict_farthest = (
            self.assign_model_weights_to_trainer(
                client_idx=farthest_idx,
                new_statedict=peer_model_farthest_statedict,
            )
        )
        peer_model_farthest = nnunet_trainer_w_new_weights_farthest.network.eval()
        peer_model_farthest.to("cpu")

        # define pixel selection ratio
        p = 1 - (self.ratio * epoch / self.stop_epoch)
        if epoch > self.stop_epoch:
            p = 1 - self.ratio

        # get img and label
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)

        # one-hot encode target (on all resolution levels)
        onehot_highres_targets = []
        for t in target:
            onehot_highres_target_ = F.one_hot(
                t.squeeze(1).long(), num_classes=self.n_classes
            )
            onehot_highres_target = (
                onehot_highres_target_.permute(0, 3, 1, 2)
                if not is_3d
                else onehot_highres_target_.permute(0, 4, 1, 2, 3)
            )
            onehot_highres_targets.append(onehot_highres_target.to(self.device))
        print(f"One-hot encoded target's computed: {datetime.datetime.now()}")

        # prepare for training step
        optimizer.zero_grad(set_to_none=True)
        net.train()
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # output are logits, NOT softmax-ed outputs
            output = net(data)
        print(f"Model prediction computed: {datetime.datetime.now()}")

        # just logits, not softmax-ed outputs
        # prediction of both peer models on GPU
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            peer_model_nearst.to(self.device)
            peer_model_nearst = peer_model_nearst.float()
            pred_logits1 = peer_model_nearst(data.float())
            pred_logits1 = [
                logit.detach().to("cpu") for logit in pred_logits1
            ]  # Move each tensor to CPU after prediction
            peer_model_nearst.to("cpu")

            peer_model_farthest.to(self.device)
            peer_model_farthest = peer_model_farthest.float()
            pred_logits2 = peer_model_farthest(data.float())
            pred_logits2 = [
                logit.detach().to("cpu") for logit in pred_logits2
            ]  # Move each tensor to CPU after prediction
            peer_model_farthest.to("cpu")
        print(f"Peer models predictions computed: {datetime.datetime.now()}")

        # Local-CAC to obtain corrected mask (on all resolution levels)
        clean_masks = []
        for i in range(len(onehot_highres_targets)):
            clean_mask = self.pixel_selection_by_Peers(
                output[i].detach().to("cpu"),
                pred_logits1[i].detach(),
                pred_logits2[i].detach(),
                onehot_highres_targets[i].detach().to("cpu"),
                p=p,
                is_3d=is_3d,
                device=torch.device("cpu"),
            )
            clean_masks.append(clean_mask)
        print(f"Ambigous (clean) masks computed: {datetime.datetime.now()}")

        # softmax each output (all resolution levels)
        pred_probs = [F.softmax(self.sm_temp * out, dim=1) for out in output]

        # # check if all tensors used for loss calculation require grad
        # print(f"pred_probs: {pred_probs.requires_grad=}, grad_fn: {pred_probs.grad_fn=}")
        # print(f"onehot_highres_targets: {onehot_highres_targets.requires_grad=}, grad_fn: {onehot_highres_targets.grad_fn=}")
        # print(f"clean_mask: {clean_mask.requires_grad=}, grad_fn: {clean_mask.grad_fn=}")

        # label correction (on all resolution levels)
        corrected_targets = []
        for i in range(len(onehot_highres_targets)):
            corrected_target = self.label_correction(
                target=onehot_highres_targets[i], clean_mask=clean_masks[i], is_3d=is_3d
            )
            corrected_targets.append(corrected_target)
        print(f"Ground-truth labels corrected: {datetime.datetime.now()}")

        # loss computation
        if self.feddm_loss == "feddm_focal_loss":
            # their loss implementation
            loss1 = self.Focal_Cross_Entropy(
                probs=pred_probs,
                target=corrected_targets,
                clean_mask=clean_mask,
                is_3d=is_3d,
            )
            losses = [loss1]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape
        if self.feddm_loss == "feddm_nnunets_loss":
            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                # take loss configured in nnunet_trainer
                loss = loss_fn(output, corrected_targets)
        print(f"Loss computed: {datetime.datetime.now()}")

        # Not sure whether necessary, but reassign correct weigths to peer models
        _, _ = self.assign_model_weights_to_trainer(
            client_idx=nearest_idx, new_statedict=actual_statedict_nearst
        )
        _, _ = self.assign_model_weights_to_trainer(
            client_idx=farthest_idx, new_statedict=actual_statedict_farthest
        )
        peer_model_nearst.train()
        peer_model_farthest.train()
        # del nnunet_trainer_w_new_weights_nearst
        # del actual_statedict_nearst
        # del nnunet_trainer_w_new_weights_farthest
        # del actual_statedict_farthest

        # Backward
        assert loss.requires_grad, "Loss does not require grad!"
        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
            optimizer.step()
            torch.cuda.empty_cache()
        print(datetime.datetime.now())

        return {"loss": loss.detach().cpu().numpy()}

    def assign_model_weights_to_trainer(
        self, client_idx: int = None, new_statedict: dict = None
    ):
        """
        Assign new model weights to nnunet_trainer.
        """
        # set client_checkpoint to model
        # get nnunet_trainer of current client
        client_nnunet_trainer = self.clients[client_idx].model.nnunet_trainer
        # save current model weights of current client
        current_client_model_weights = (
            client_nnunet_trainer.get_model_weights_from_checkpoint()
        )
        # set client_checkpoint to current client model
        client_nnunet_trainer.set_model_weights_to_checkpoint(new_statedict)

        return client_nnunet_trainer, current_client_model_weights

    def label_correction(self, target, clean_mask, is_3d=False):
        """
        Multi-class label correction.
        Ambiguous pixels in clean_mask are corrected to biggest/smallest fg class depending on self.cac_label_correction.

        Input:
        - target: ground-truth mask including bg
        - clean_mask: ambiguity mask with px_val=1 indicating certain fg, and px_val=2 indicating ambiguous fg; determined via peer models
        - is_3d: whether the input is 3D
        """
        # clone target to avoid in-place modification issues
        target = target.clone()
        # handle 2D and 3D cases
        if not is_3d and target.ndim == 4:
            B, C, H, W = target.shape  # C = num_classes
            _, F, _, _ = clean_mask.shape  # F = num foreground classes (C - 1)
        elif is_3d and target.ndim == 5:
            B, C, H, W, D = target.shape
            _, F, _, _, _ = clean_mask.shape
        else:
            raise ValueError("Target must be 4D or 5D tensor.")

        # Apply correction to target where label is uncertain (==2)
        # set uncertain pixels to smallest/largest fg class (dependent on self.cac_label_correction)
        for b in range(B):
            ambiguity_mask = (clean_mask[b] == 2).any(dim=0)  # shape: HxW or DxHxW

            if ambiguity_mask.sum().item() == 0:
                continue

            # Get per-foreground-class counts of "certain" (==1) pixels
            counts = [(clean_mask[b, f] == 1).sum().item() for f in range(F)]
            if sum(counts) == 0:
                continue

            # Determine label correction class index
            # +1 to account for background in `target`
            if self.cac_label_correction == "smallest":
                labelcorrection_class_idx = counts.index(min(counts)) + 1
            elif self.cac_label_correction == "largest":
                labelcorrection_class_idx = counts.index(max(counts)) + 1

            # Clear all classes (incl. bg) at ambiguous pixels
            target[b, :, ...][..., ambiguity_mask] = 0

            # Set smallest/largest foreground class at those pixels
            target[b, labelcorrection_class_idx, ...][..., ambiguity_mask] = 1

        return target

    def Focal_Cross_Entropy(
        self, probs, target, clean_mask, alpha=0.25, gamma=2.0, is_3d=False
    ):
        """
        Focal Cross-Entropy Loss with pixel selection via clean_mask.

        Input:
        - probs: softmax-ed model predictions
        - target: ground-truth mask including bg
        - clean_mask: ambiguity mask with px_val=1 indicating certain fg, and px_val=2 indicating ambiguous fg; determined via peer models
        - alpha: focal loss alpha
        - gamma: focal loss gamma
        - is_3d: whether the input is 3D

        Notes:
        - Uncertain fg pixels are always set to bg
        """
        # clone target to avoid in-place modification issues
        target = target.clone()
        # handle 2D and 3D cases
        if not is_3d and target.ndim == 4:
            B, C, H, W = target.shape  # C = num_classes
            _, F, _, _ = clean_mask.shape  # F = num foreground classes (C - 1)
        elif is_3d and target.ndim == 5:
            B, C, H, W, D = target.shape
            _, F, _, _, _ = clean_mask.shape
        else:
            raise ValueError("Target must be 4D or 5D tensor.")

        mask: Tensor = cast(Tensor, target.type(torch.float32))
        log_p: Tensor = (probs + self.eps).log()

        # compute class-wise focal losses
        total_loss = None  # torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        total_pixels = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        idx_mask = torch.zeros_like(mask, device=probs.device, dtype=torch.bool)
        loss_c = torch.zeros_like(probs, device=probs.device, dtype=probs.dtype)

        for class_idx in range(C):
            probs_c = probs[:, class_idx, ...]
            log_p_c = log_p[:, class_idx, ...]
            mask_c = mask[:, class_idx, ...]

            # Weighting for focal loss
            # background
            if class_idx == 0:
                weight = (1 - alpha) * torch.pow(1 - probs_c, gamma)
            # foreground
            else:
                weight = alpha * torch.pow(1 - probs_c, gamma)
                idx_mask[:, class_idx, ...] = (
                    clean_mask[:, class_idx - 1, ...] == 1
                ) | (clean_mask[:, class_idx - 1, ...] == 2)
            # compute class-wise loss
            loss_c[:, class_idx, ...] = -weight * mask_c * log_p_c

        # determine idx_mask for bg class, i.e. set pixels in bg's idx_mask to True is False in all fg classes
        fg_idx_mask_sum = idx_mask[:, 1:, ...].sum(dim=1)
        idx_mask[:, 0, ...] = fg_idx_mask_sum == 0

        # mask total_loss with idx_mask
        final_loss = loss_c[idx_mask].sum() / (idx_mask.sum() + self.eps)
        return final_loss

    # def pixel_selection_by_Peers(self, logits, logits1, logits2, labels, p=0):
    #     """
    #     Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L356

    #     Inputs:
    #     - labels and all logits are in the shape of B, C, H, W w/ C = bg + #fg_classes
    #     """
    #     # extract the background and (multi-channel) foreground masks
    #     bg_mask = labels[:, 0, ...]  # B, H, W
    #     fg_mask = labels[:, 1:, ...]  # B, #fg_classes, H, W

    #     # softmax to get probabilities for each class
    #     pred = torch.softmax(logits, dim=1)  # B, C, H, W
    #     pred1 = torch.softmax(logits1, dim=1)  # B, C, H, W
    #     pred2 = torch.softmax(logits2, dim=1)  # B, C, H, W

    #     # log the preds
    #     log_p: Tensor = (pred + self.eps).log()  # B, C, H, W
    #     log_p1: Tensor = (pred1 + self.eps).log()  # B, C, H, W
    #     log_p2: Tensor = (pred2 + self.eps).log()  # B, C, H, W

    #     clean_mask = torch.zeros_like(fg_mask)  # B, #fg_classes, H, W

    #     # iterate over each batch element
    #     for b in range(fg_mask.size(0)):
    #         # iterate over each fg class
    #         for class_idx in range(fg_mask.size(1)):
    #             # Calculate the class-wise loss
    #             loss = (
    #                 -fg_mask[b, class_idx, ...] * log_p[b, (class_idx + 1), ...]
    #             )  # H, W
    #             loss1 = (
    #                 -fg_mask[b, class_idx, ...] * log_p1[b, (class_idx + 1), ...]
    #             )  # H, W
    #             loss2 = (
    #                 -fg_mask[b, class_idx, ...] * log_p2[b, (class_idx + 1), ...]
    #             )  # H, W
    #             # Flatten the losses
    #             loss_flat = loss.flatten()  # H*W
    #             loss1_flat = loss1.flatten()  # H*W
    #             loss2_flat = loss2.flatten()  # H*W
    #             # Select the number of pixels to keep based on p
    #             fg_num_selected = (
    #                 (fg_mask[b, class_idx, ...].sum() * p).type(torch.int).item()
    #             )
    #             threshold = fg_num_selected  # + bg_mask[b].sum()

    #             # Apply selection for each peer model (logits, logits1, logits2)
    #             if fg_num_selected > 5:
    #                 value_fg, _ = torch.topk(
    #                     loss_flat, threshold, largest=False, sorted=True
    #                 )
    #                 thresh_fg = value_fg[-1]
    #                 value_fg1, _ = torch.topk(
    #                     loss1_flat, threshold, largest=False, sorted=True
    #                 )
    #                 thresh_fg1 = value_fg1[-1]
    #                 value_fg2, _ = torch.topk(
    #                     loss2_flat, threshold, largest=False, sorted=True
    #                 )
    #                 thresh_fg2 = value_fg2[-1]

    #                 clean_mask_ = loss <= thresh_fg
    #                 clean_mask1_ = loss1 <= thresh_fg1
    #                 clean_mask2_ = loss2 <= thresh_fg2

    #                 # pixels w/ low loss for current model and distal model are clean
    #                 clean_mask[b, class_idx, ...][(clean_mask_ & clean_mask2_)] = 1.0
    #                 # pixels w/ disagreement between current model and proximal model are uncertain
    #                 clean_mask[b, class_idx, ...][
    #                     (clean_mask_ | clean_mask1_) ^ (clean_mask_ & clean_mask1_)
    #                 ] = 2
    #             else:
    #                 # if label is in gt smaller than 5 pixels, pixels are clean
    #                 clean_mask[b, class_idx, ...] = 1.0

    #     # Ensure background pixels are preserved
    #     clean_mask_final = clean_mask * fg_mask + bg_mask.unsqueeze(
    #         1
    #     )  # B, num_classes, H, W

    #     # self.plot_it(clean_mask_final, 5, 2)

    #     return clean_mask_final

    def pixel_selection_by_Peers(
        self,
        logits: Tensor,  # [B, C, H, W], C = 1 (bg) + F (fg classes)
        logits1: Tensor,  # peer 1
        logits2: Tensor,  # peer 2
        labels: Tensor,  # one-hot [B, C, H, W]
        p: float = 0.0,  # fraction of class pixels to keep as "selected"
        min_fg_keep: int = 5,  # safeguard: if floor(p*|class|) <= this, mark all class pixels as clean
        is_3d: bool = False,
        device: torch.device = None,
    ) -> Tensor:
        """
        Returns:
            clean_mask_fg: [B, F, H, W] float32 in {0., 1., 2.}
            For each foreground class channel k (0..F-1 corresponds to class id k+1):
                - 1.0 on pixels considered "clean" for that class (main ∧ peer2).
                - 2.0 on pixels considered "uncertain" for that class (main ⊕ peer1).
                - 0.0 otherwise.
            Background pixels are 0.0 in all F channels.
        """
        assert (
            logits.shape == labels.shape
        ), "logits and labels must have identical shapes [B, C, H, W] or [B, C, H, W, D]"

        if is_3d:
            B, C, H, W, D = logits.shape
        else:
            B, C, H, W = logits.shape
        assert C >= 2, "Need at least background + one foreground class"
        F = C - 1

        device = logits.device
        dtype = torch.float32

        # Extract masks
        bg_mask = labels[:, 0, ...].bool()  # [B, H, W]
        fg_masks = labels[:, 1:, ...].bool()  # [B, F, H, W]; one-hot per class

        # Softmax -> log probs
        log_p = (torch.softmax(logits, dim=1) + self.eps).log()  # [B, C, H, W]
        log_p1 = (torch.softmax(logits1, dim=1) + self.eps).log()
        log_p2 = (torch.softmax(logits2, dim=1) + self.eps).log()

        # Output: per-foreground-class triage map
        # clean_mask_fg = torch.zeros((B, F, H, W))
        clean_mask_fg = (
            torch.zeros((B, F, H, W, D), dtype=torch.float32, device=logits.device)
            if is_3d
            else torch.zeros((B, F, H, W), dtype=torch.float32, device=logits.device)
        )

        for b in range(B):
            # iterate foreground classes (channel k+1 in logits/labels)
            for k in range(F):
                fg_idx = fg_masks[b, k]  # [H, W] bool for class (k+1)
                fg_count = int(fg_idx.sum().item())
                if fg_count == 0:
                    # No pixels of this class in this image -> remain zeros
                    continue

                # Number to keep
                fg_num_selected = int(fg_count * p)

                if fg_num_selected <= min_fg_keep:
                    # Safeguard: if too small, mark all class-k pixels as clean (=1)
                    if is_3d:
                        cm_tmp = torch.zeros((H, W, D), device=device)
                    else:
                        cm_tmp = torch.zeros((H, W), device=device)
                    cm_tmp[fg_idx] = 1.0
                    clean_mask_fg[b, k] = cm_tmp
                    continue

                # Collect class-only NLL vectors (no contamination from other classes/bg)
                # Current class is channel (k+1) because channel 0 is background.
                losses = (-log_p[b, k + 1])[fg_idx]  # [fg_count]
                losses1 = (-log_p1[b, k + 1])[fg_idx]
                losses2 = (-log_p2[b, k + 1])[fg_idx]

                # Guard k_keep ∈ [1, fg_count]
                k_keep = max(1, min(fg_num_selected, fg_count))

                # Thresholds from topk on class-only vectors
                val, _ = torch.topk(losses, k_keep, largest=False, sorted=True)
                thr = val[-1]
                val1, _ = torch.topk(losses1, k_keep, largest=False, sorted=True)
                thr1 = val1[-1]
                val2, _ = torch.topk(losses2, k_keep, largest=False, sorted=True)
                thr2 = val2[-1]

                sel = losses <= thr  # main selected
                sel1 = losses1 <= thr1  # peer1 selected (proximal)
                sel2 = losses2 <= thr2  # peer2 selected (distal)

                # Build per-image, per-class triage grid
                if is_3d:
                    cm_tmp = torch.zeros((H, W, D), device=device)
                else:
                    cm_tmp = torch.zeros((H, W), device=device)

                # Map 1D selections back into 2D at fg_idx
                status = torch.zeros_like(losses, device=device)
                # Set uncertain (main ⊕ peer1) -- no overwrite of clean since xor is false if sel&sel1
                status[sel ^ sel1] = 2.0
                # Set clean (main ∧ peer2)
                status[sel & sel2] = 1.0

                cm_tmp[fg_idx] = status.to(cm_tmp.dtype)
                clean_mask_fg[b, k] = cm_tmp

        # Note: We intentionally DO NOT write background into every foreground channel.
        # Background pixels remain 0.0 across all F channels.
        return clean_mask_fg

    def plot_it(self, tensor, batch_to_plot, channels, save_path="clean_mask_plot.png"):
        fig, axs = plt.subplots(
            batch_to_plot, channels, figsize=(6 * channels, 5 * batch_to_plot)
        )

        for b in range(batch_to_plot):
            for c in range(channels):
                ax = axs[b, c] if batch_to_plot > 1 else axs[c]
                ax.imshow(tensor[b, c].cpu(), cmap="gray")
                ax.set_title(f"Batch {b}, Channel {c}")
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory

    # def communication_FedDM(
    def feddm_central_steps(
        self,
        client_checkpoints: dict = None,
        server_model=None,
    ):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L578
        """

        # let's rather use the client_checkpoints

        # Central-CAC: Sets peers per client in self.clients_peers
        _, embeddings, nearest_clients_bulk, farthest_clients_bulk = (
            self.find_customized_peers(client_checkpoints)
        )

        grads = []
        for client_idx, client_checkpoint in client_checkpoints.items():
            grads.append(self.get_grads_(client_checkpoint, server_model))

        # Central-HGD: Compute de-conflicted gradients
        new_grads, self.grad_history = self.pcgrad_hierarchy(grads)

        # clients models get intermediately updated by adding new gradients to server model
        intermediate_updated_client_checkpoints = {}
        for k, client_checkpoint in client_checkpoints.items():
            intermediate_updated_client_checkpoints[k] = self.set_grads_(
                client_checkpoint, server_model, new_grads
            )

        # perform FedAvg of intermediately updated client models
        server_model_weights = self.fed_avg(intermediate_updated_client_checkpoints)
        return server_model_weights

    def find_customized_peers(self, client_checkpoints):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L419
        """
        customized_peers = []
        self.gauss_noise = self.gauss_noise.to(self.device)
        for client_idx, client_checkpoint in client_checkpoints.items():
            # set client_checkpoint to model
            client_nnunet_trainer, current_client_model_weights = (
                self.assign_model_weights_to_trainer(
                    client_idx=client_idx, new_statedict=client_checkpoint
                )
            )
            # get nnunet_trainer of current client
            model = client_nnunet_trainer.network

            model.eval()
            with torch.no_grad():
                # increase the sampling size by batch processing
                for i in range((self.mc_n_samples // self.batch_size) - 1):
                    gauss_noise_ = self.gauss_noise[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ]
                    out_ = model(gauss_noise_)
                    out = torch.softmax(out_[0], dim=1)
                    self.embeddings[client_idx][
                        i * self.batch_size : (i + 1) * self.batch_size
                    ] = out

            # set current_client_model_weights back to nnunet_trainer of current client
            self.clients[
                client_idx
            ].model.nnunet_trainer.set_model_weights_to_checkpoint(
                current_client_model_weights
            )
        self.gauss_noise = self.gauss_noise.to("cpu")

        nearest_clients_bulk = torch.zeros(len(self.embeddings))
        farthest_clients_bulk = torch.zeros(len(self.embeddings))
        for client_i in range(len(self.embeddings)):
            embedding = self.embeddings[client_i].reshape(
                self.embeddings[client_i].size(0), -1
            )
            nearest_samples_bulk = torch.zeros(len(self.embeddings))
            farthest_samples_bulk = torch.zeros(len(self.embeddings))
            for b in range(embedding.size(0)):
                distances = torch.zeros(len(self.embeddings))
                for client_j in range(len(self.embeddings)):
                    if client_i == client_j:
                        distances[client_j] = 1.0
                    else:
                        embedding_o = self.embeddings[client_j][b].view(-1)
                        distances[client_j] = torch.norm(
                            embedding[b] - embedding_o, p=2
                        )

                distances[client_i] = 1e10
                nearest_idx = distances.argmin()
                nearest_samples_bulk[nearest_idx] += 1
                distances[client_i] = -1e10
                farthest_idx = distances.argmax()
                farthest_samples_bulk[farthest_idx] += 1

            nearest_samples_bulk[client_i] = 0.0
            farthest_samples_bulk[client_i] = 0.0
            assert nearest_samples_bulk.sum() == embedding.size(0)
            assert farthest_samples_bulk.sum() == embedding.size(0)
            nearest_samples_bulk[client_i] = -1e10
            nearest_idx = nearest_samples_bulk.argmax()
            nearest_clients_bulk[nearest_idx] += 1
            farthest_samples_bulk[client_i] = -1e10
            farthest_idx = farthest_samples_bulk.argmax()
            farthest_clients_bulk[farthest_idx] += 1
            customized_peers.append(
                [
                    client_checkpoints[int(nearest_idx)],
                    client_checkpoints[int(farthest_idx)],
                ]
            )
            self.clients_peers[client_i]["nearest"]["id"] = int(nearest_idx)
            self.clients_peers[client_i]["nearest"]["model"] = copy.deepcopy(
                client_checkpoints[int(nearest_idx)]
            )
            self.clients_peers[client_i]["farthest"]["id"] = int(farthest_idx)
            self.clients_peers[client_i]["farthest"]["model"] = copy.deepcopy(
                client_checkpoints[int(farthest_idx)]
            )
            print(
                f"Client {client_i} nearest client: {nearest_idx}, farthest client: {farthest_idx}"
            )

        return (
            customized_peers,
            self.embeddings,
            nearest_clients_bulk,
            farthest_clients_bulk,
        )

    def get_grads_(self, client_checkpoint, server_model):
        """
        Highly adapted to work based on the model weight's addresses (data_ptr()) instead of the keys
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L487
        """
        grads = []

        # get addresses of keys instead of using keys directly
        keys = list(server_model.keys())
        address_key_dict = {}
        for k in keys:
            address = server_model[k].data_ptr()
            if address not in address_key_dict.keys():
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)

        # per address, compute gradient
        for a in address_key_dict.keys():
            grads.append(
                client_checkpoint[address_key_dict[a][0]]
                .data.clone()
                .detach()
                .flatten()
                - server_model[address_key_dict[a][0]].data.clone().detach().flatten()
            )
        return torch.cat(grads)

    def set_grads_(self, client_checkpoint, server_model, new_grads):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L494
        """
        # Build address-to-key mapping for server_model
        keys = list(server_model.keys())
        address_key_dict = {}
        for k in keys:
            address = server_model[k].data_ptr()
            if address not in address_key_dict.keys():
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)

        # Apply the gradients to model
        start = 0
        for address, key_list in address_key_dict.items():
            key = key_list[0]  # take the first key pointing to this address
            if "num_batches_tracked" not in key:
                dims = client_checkpoint[key].shape
                end = start + dims.numel()
                client_checkpoint[key].data.copy_(
                    server_model[key].data.clone().detach()
                    + new_grads[start:end].reshape(dims).clone()
                )
                start = end
        return client_checkpoint

    def pcgrad_hierarchy(self, client_grads):
        """
        Projecting conflicting gradients
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L505
        """
        # initialize
        client_grads_ = torch.stack(client_grads)
        grads = []
        grad_len = self.grad_history["grad_len"]
        start = 0

        # iterate over gradient keys
        for key in grad_len.keys():
            g_len = grad_len[key]
            end = start + g_len
            layer_grad_history = self.grad_history[key]

            # grad_history exists from 2nd fl_round
            if layer_grad_history is not None:
                pc_v = layer_grad_history.unsqueeze(0)
                client_grads_layer = client_grads_[:, start:end]
                while True:
                    num = client_grads_layer.size(0)

                    # more than 2 clients left
                    if num > 2:
                        # compute similarity across clients and sort w.r.t. similarity
                        inner_prod = torch.mul(client_grads_layer, pc_v).sum(1)
                        project = inner_prod / (pc_v**2).sum().sqrt()
                        _, ind = project.sort(descending=True)

                        # compose pairs
                        pair_list = []
                        if num % 2 == 0:
                            for i in range(num // 2):
                                pair_list.append([ind[i], ind[num - i - 1]])
                        else:
                            for i in range(num // 2):
                                pair_list.append([ind[i], ind[num - i - 1]])
                            pair_list.append([ind[num // 2]])

                        # De-conflict and fuse gradients
                        client_grads_new = []
                        for pair in pair_list:
                            # if pair is really a pair, de-conflict and sum gradients
                            if len(pair) > 1:
                                grad_0 = client_grads_layer[pair[0]]
                                grad_1 = client_grads_layer[pair[1]]
                                inner_prod = torch.dot(grad_0, grad_1)
                                if inner_prod < 0:
                                    # Sustract the conflicting component
                                    grad_pc_0 = (
                                        grad_0 - inner_prod / (grad_1**2).sum() * grad_1
                                    )
                                    grad_pc_1 = (
                                        grad_1 - inner_prod / (grad_0**2).sum() * grad_0
                                    )
                                else:
                                    grad_pc_0 = grad_0
                                    grad_pc_1 = grad_1
                                grad_pc_0_1 = grad_pc_0 + grad_pc_1
                                client_grads_new.append(grad_pc_0_1)

                            # if pair is a single client, just append the gradient
                            else:
                                grad_single = client_grads_layer[pair[0]]
                                client_grads_new.append(grad_single)
                        client_grads_layer = torch.stack(client_grads_new)

                    # just 2 clients left -> de-conflict and sum gradients
                    elif num == 2:
                        grad_pc_0 = client_grads_layer[0]
                        grad_pc_1 = client_grads_layer[1]
                        inner_prod = torch.dot(grad_pc_0, grad_pc_1)
                        if inner_prod < 0:
                            # Sustract the conflicting component
                            grad_pc_0 = (
                                grad_pc_0
                                - inner_prod / (grad_pc_1**2).sum() * grad_pc_1
                            )
                            grad_pc_1 = (
                                grad_pc_1
                                - inner_prod / (grad_pc_0**2).sum() * grad_pc_0
                            )

                        grad_pc_0_1 = grad_pc_0 + grad_pc_1
                        grad_new = grad_pc_0_1 / len(self.clients)
                        break
                    else:
                        raise NotImplementedError("No implementation for ONE client!")

                # update grad_history
                gamma = self.gamma_hgd_smoothing
                self.grad_history[key] = (
                    gamma * self.grad_history[key] + (1 - gamma) * grad_new
                )
                grads.append(grad_new)

            # grad_history does not exist from 1st fl_round
            else:
                grad_new = client_grads_[:, start:end].mean(0)
                self.grad_history[key] = grad_new
                grads.append(grad_new)

            # update start iterator for next model weights in stacked client_grads_
            start = end

        # concatenate all gradients
        grad_new = torch.cat(grads)

        return grad_new, self.grad_history
