import copy
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

    def __init__(self, clients: list = None):
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
        self.ratio = 0.6  # ratio for pixel selection
        # feddm stop epoch ~= 1/4 of total training epochs
        # self.stop_epoch = self.clients[0].model.nnunet_trainer.num_peochs // 4
        self.stop_epoch = 50
        self.eps = 1e-6

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
        optimizer: Any = None,
        peer_models=None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L207
        Adapted to work with nnUNetv2 and as nnUNetv2's def train_step().
        """
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
        for p in peer_model_nearst.parameters():
            p.requires_grad = False
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
        for p in peer_model_farthest.parameters():
            p.requires_grad = False

        # define pixel selection ratio
        p = 1 - (self.ratio * epoch / self.stop_epoch)
        if epoch > self.stop_epoch:
            p = 1 - self.ratio

        # get img and label
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        # if isinstance(target, list):
        #     target = [i.to(self.device, non_blocking=True) for i in target]
        # else:
        #     target = target.to(self.device, non_blocking=True)
        onehot_highres_targets_ = F.one_hot(
            target[0].squeeze(1).long(), num_classes=self.n_classes
        )
        onehot_highres_targets = (
            onehot_highres_targets_.permute(0, 3, 1, 2)
            if onehot_highres_targets_.ndim == 4
            else onehot_highres_targets_.permute(0, 4, 1, 2, 3)
        )
        onehot_highres_targets = onehot_highres_targets.to(self.device)

        optimizer.zero_grad(set_to_none=True)
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

            # just logits, not softmax-ed outputs
            with torch.no_grad():
                pred_logits1 = peer_model_nearst(data)
                pred_logits2 = peer_model_farthest(data)

            # Local-CAC to obtain corrected mask
            clean_mask = self.pixel_selection_by_Peers(
                output[0].detach(),
                pred_logits1[0].detach(),
                pred_logits2[0].detach(),
                onehot_highres_targets,
                p=p,
            )

            # softmax model predictions
            pred_probs: Tensor = F.softmax(self.sm_temp * output[0], dim=1)

            # check if all tensors used for loss calculation require grad
            print(f"pred_probs: {pred_probs.requires_grad=}, grad_fn: {pred_probs.grad_fn=}")
            print(f"onehot_highres_targets: {onehot_highres_targets.requires_grad=}, grad_fn: {onehot_highres_targets.grad_fn=}")
            print(f"clean_mask: {clean_mask.requires_grad=}, grad_fn: {clean_mask.grad_fn=}")

            # label correction and loss computation
            loss1 = self.Focal_Cross_Entropy(
                pred_probs, onehot_highres_targets, clean_mask
            )
            losses = [loss1]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape

        # Not sure whether necessary, but reassign correct weigths to peer models
        _, _ = self.assign_model_weights_to_trainer(
            client_idx=nearest_idx, new_statedict=actual_statedict_nearst
        )
        _, _ = self.assign_model_weights_to_trainer(
            client_idx=farthest_idx, new_statedict=actual_statedict_farthest
        )
        del nnunet_trainer_w_new_weights_nearst
        del actual_statedict_nearst
        del nnunet_trainer_w_new_weights_farthest
        del actual_statedict_farthest

        # Backward
        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
            optimizer.step()
            torch.cuda.empty_cache()

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

    def Focal_Cross_Entropy(self, probs, target, clean_mask, alpha=0.25, gamma=2.0):
        """
        Correct ground_trough seg mask and compute multi-class focal loss.

        Notes:
        - Uncertain fg pixels are always set to bg
        """
        # Clone target to avoid in-place modification issues
        target = target.clone()
        if target.ndim == 4:
            B, C, H, W = target.shape  # C = num_classes
            _, F, _, _ = clean_mask.shape  # F = num foreground classes (C - 1)
        elif target.ndim == 5:
            B, C, D, H, W = target.shape
            _, F, _, _, _ = clean_mask.shape
        else:
            raise ValueError("Target must be 4D or 5D tensor.")

        # Step 1: Apply correction to target where label is uncertain (==2)
        # set uncertain pixels to dominant fg class
        # for b in range(B):
        #     # find dominant fg class
        #     counts = [(clean_mask[b, f] == 1).sum().item() for f in range(F)]
        #     if sum(counts) == 0:
        #         # no certain fg -> skip adjustment entirely
        #         continue
        #     dominant_class_idx = counts.index(max(counts)) + 1  # +1 because class indices in target
        #     # set uncertain pixels to dominant class
        #     for class_idx in range(1, C):
        #         # set dominant class
        #         target[b, dominant_class_idx, ...][
        #             clean_mask[b, class_idx - 1, ...] == 2
        #         ] = 1
        #         # clear other classes
        #         target[b, class_idx, ...][
        #             clean_mask[b, class_idx - 1, ...] == 2
        #         ] = 0
        #         # ensure that background is also cleared
        #         target[b, 0, ...][
        #             clean_mask[b, class_idx - 1, ...] == 2
        #         ] = 0
        # set uncertain pixels to background
        for class_idx in range(1, C):  # Skip background at index 0
            target[:, 0, ...][
                clean_mask[:, class_idx - 1, ...] == 2
            ] = 1  # set background
            target[:, class_idx, ...][
                clean_mask[:, class_idx - 1, ...] == 2
            ] = 0  # clear class

        mask: Tensor = cast(Tensor, target.type(torch.float32))
        log_p: Tensor = (probs + self.eps).log()

        # Step 2: Compute class-wise focal losses
        total_loss = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        total_pixels = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)


        for class_idx in range(C):
            probs_c = probs[:, class_idx, ...]
            log_p_c = log_p[:, class_idx, ...]
            mask_c = mask[:, class_idx, ...]

            # Weighting for focal loss
            if class_idx == 0:  # background
                weight = (1 - alpha) * torch.pow(1 - probs_c, gamma)
                idx_mask = torch.ones_like(
                    probs_c, dtype=torch.bool
                )  # apply to all pixels
            else:  # foreground
                weight = alpha * torch.pow(1 - probs_c, gamma)
                idx_mask = (clean_mask[:, class_idx - 1, ...] == 1) | (
                    clean_mask[:, class_idx - 1, ...] == 2
                )

            class_loss = -weight * mask_c * log_p_c
            total_loss += class_loss[idx_mask].sum()
            total_pixels += idx_mask.sum()

        final_loss = total_loss / (total_pixels + self.eps)
        return final_loss

    def pixel_selection_by_Peers(self, logits, logits1, logits2, labels, p=0):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L356

        Inputs:
        - labels and all logits are in the shape of B, C, H, W w/ C = bg + #fg_classes
        """
        # extract the background and (multi-channel) foreground masks
        bg_mask = labels[:, 0, ...]  # B, H, W
        fg_mask = labels[:, 1:, ...]  # B, #fg_classes, H, W

        # softmax to get probabilities for each class
        pred = torch.softmax(logits, dim=1)  # B, C, H, W
        pred1 = torch.softmax(logits1, dim=1)  # B, C, H, W
        pred2 = torch.softmax(logits2, dim=1)  # B, C, H, W

        # log the preds
        log_p: Tensor = (pred + self.eps).log()  # B, C, H, W
        log_p1: Tensor = (pred1 + self.eps).log()  # B, C, H, W
        log_p2: Tensor = (pred2 + self.eps).log()  # B, C, H, W

        clean_mask = torch.zeros_like(fg_mask)  # B, #fg_classes, H, W

        # iterate over each batch
        for b in range(fg_mask.size(0)):
            # iterate over each fg class
            for class_idx in range(fg_mask.size(1)):
                # Calculate the class-wise loss
                loss = (
                    -fg_mask[b, class_idx, ...] * log_p[b, (class_idx + 1), ...]
                )  # H, W
                loss1 = (
                    -fg_mask[b, class_idx, ...] * log_p1[b, (class_idx + 1), ...]
                )  # H, W
                loss2 = (
                    -fg_mask[b, class_idx, ...] * log_p2[b, (class_idx + 1), ...]
                )  # H, W
                # Flatten the losses
                loss_flat = loss.flatten()  # H*W
                loss1_flat = loss1.flatten()  # H*W
                loss2_flat = loss2.flatten()  # H*W
                # Select the number of pixels to keep based on p
                fg_num_selected = (
                    (fg_mask[b, class_idx, ...].sum() * p).type(torch.int).item()
                )
                threshold = fg_num_selected  # + bg_mask[b].sum()

                # Apply selection for each peer model (logits, logits1, logits2)
                if fg_num_selected > 5:
                    value_fg, _ = torch.topk(
                        loss_flat, threshold, largest=False, sorted=True
                    )
                    thresh_fg = value_fg[-1]
                    value_fg1, _ = torch.topk(
                        loss1_flat, threshold, largest=False, sorted=True
                    )
                    thresh_fg1 = value_fg1[-1]
                    value_fg2, _ = torch.topk(
                        loss2_flat, threshold, largest=False, sorted=True
                    )
                    thresh_fg2 = value_fg2[-1]

                    clean_mask_ = loss <= thresh_fg
                    clean_mask1_ = loss1 <= thresh_fg1
                    clean_mask2_ = loss2 <= thresh_fg2

                    # pixels w/ low loss for current model and distal model are clean
                    clean_mask[b, class_idx, ...][(clean_mask_ & clean_mask2_)] = 1.0
                    # pixels w/ disagreement between current model and proximal model are uncertain
                    clean_mask[b, class_idx, ...][
                        (clean_mask_ | clean_mask1_) ^ (clean_mask_ & clean_mask1_)
                    ] = 2
                else:
                    # if label is in gt smaller than 5 pixels, pixels are clean
                    clean_mask[b, class_idx, ...] = 1.0

        # Ensure background pixels are preserved
        clean_mask_final = clean_mask * fg_mask + bg_mask.unsqueeze(
            1
        )  # B, num_classes, H, W

        # self.plot_it(clean_mask_final, 5, 2)

        return clean_mask_final

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
                gamma = 0.99
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
