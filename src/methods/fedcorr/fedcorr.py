import copy
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import logging
import torch
from torch import autocast
import numpy as np
from scipy.spatial.distance import cdist

from methods.fedavg.fedavg import FedAvg

from nnunetv2.utilities.helpers import dummy_context


class FedCorr(FedAvg):
    """
    FedCorr: Multi-Stage Federated Learning for Label Noise Correction
    J Xu et al., 2022, CVPR
    https://arxiv.org/abs/2204.04677

    Args:
        clients (list): List of FL clients.
        feda3i_warmup_rounds (int): Number of warmup rounds before starting quality-based aggregation.
        feda3i_interw (float): Interpolation weight between expand and shrink clients for quality-based aggregation.
    """

    def __init__(
        self,
        clients: list = None,
        num_rounds: int = None,
        fedcorr_preproc_rounds_frac: float = None,
        fedcorr_relabel_ratio: float = None,
        fedcorr_relabel_confidence_thres: float = None,
        fedcorr_proxterm_beta: float = None,
    ):
        super().__init__(clients=clients)
        self.name = "fedcorr"
        self.fedcorr_preproc_rounds = int(fedcorr_preproc_rounds_frac * num_rounds)
        self.fedcorr_relabel_ratio = fedcorr_relabel_ratio
        self.fedcorr_relabel_confidence_thres = fedcorr_relabel_confidence_thres
        self.fedcorr_proxterm_beta = fedcorr_proxterm_beta

        # variables for client and sample noise estimation
        self.noisy_clients = []
        self.clean_clients = []
        self.LID_whole = {c_id: None for c_id in range(len(self.clients))}
        self.LID_client = {c_id: None for c_id in range(len(self.clients))}
        self.LID_accumulative_client = {c_id: 0.0 for c_id in range(len(self.clients))}
        self.loss_whole = {c_id: None for c_id in range(len(self.clients))}
        self.loss_accumulative_whole = {c_id: {} for c_id in range(len(self.clients))}

        # compute number of fl_rounds for fine-tuning and full-training stages
        self.fedcorr_finetune_rounds, self.fedcorr_fulltrain_rounds = (
            self._compute_stage_rounds(num_rounds)
        )

        # initialize estimated noisy levels for clients
        self.clients_estimated_noisy_level = {
            client.client_id: 0.0 for client in self.clients
        }

        # global model weights for proximal term computation are set from orchestrator
        self.global_fl_model_weights = None

        logging.info(
            f"FedCorr initialized with preproc rounds frac {self.fedcorr_preproc_rounds}, relabel ratio {self.fedcorr_relabel_ratio}, and proxterm beta {self.fedcorr_proxterm_beta}!"
        )

    def _compute_stage_rounds(self, num_rounds: int):
        """Compute number of FL rounds for fine-tuning and full-training stages."""
        # compute number of fl_rounds for fine-tuning and full-training stages
        finetune_rounds = (num_rounds - self.fedcorr_preproc_rounds) // 2
        fulltrain_rounds = finetune_rounds

        # ensure self.fedcorr_preproc_rounds + finetune_rounds + fulltrain_rounds == num_rounds
        while (
            self.fedcorr_preproc_rounds + finetune_rounds + fulltrain_rounds
        ) != num_rounds:
            if (
                self.fedcorr_preproc_rounds + finetune_rounds + fulltrain_rounds
            ) < num_rounds:
                fulltrain_rounds += 1
            else:
                finetune_rounds -= 1

        return finetune_rounds, fulltrain_rounds

    def compute_proximal_loss_term(self, local_model_weights, c_id: int):
        """
        Compute FedCorr's proximal loss term.
        Prox term = beta * mu * ||w_diff||^2 w/ w_diff = local_model_weights - global_model_weights

        Args:
            glob_model_weights: Current global model weights.
        """
        # compute weight difference norm
        local_model = copy.deepcopy(local_model_weights.state_dict())
        w_diff = torch.tensor(0.0).to(next(iter(local_model.values())).device)
        # get addresses of keys
        keys = list(local_model.keys())
        address_key_dict = {}
        for k in keys:
            address = local_model[k].data_ptr()
            if address not in address_key_dict.keys():
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)
        # compute w_diff
        for a in address_key_dict.keys():
            diff = (
                local_model[address_key_dict[a][0]]
                - self.global_fl_model_weights[
                    address_key_dict[a][0].replace("_orig_mod.", "")
                ]
            )
            w_diff += torch.pow(torch.norm(diff), 2)
        w_diff = torch.sqrt(w_diff)

        # compute proximal term
        prox_term = (
            self.fedcorr_proxterm_beta
            * self.clients_estimated_noisy_level[c_id]
            * w_diff
        )
        return prox_term

    def get_output_seg(self, nnunet_trainer):
        """
        Generate per-sample outputs and losses in the training dataloader.

        Args:
            nnunet_trainer: Trainer object containing network, dataloader, device, and loss criterion

        Retruns:
            Tuple of (output_whole, loss_whole) where output_whole is a list of network outputs
                 and loss_whole is a list of corresponding loss values
        """

        # get net, loader, device, criterion from nnunet_trainer
        net = nnunet_trainer.network
        net.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader = nnunet_trainer.dataloader_train
        device = nnunet_trainer.device
        criterion = nnunet_trainer.loss

        # prepare to collect outputs and losses
        output_whole = []
        loss_whole = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(loader, total=nnunet_trainer.num_iterations_per_epoch)
            ):
                if batch_idx >= nnunet_trainer.num_iterations_per_epoch:
                    break
                for batch_element_idx in range(len(batch["data"])):
                    # get data, target, and keys
                    data = batch["data"][batch_element_idx].unsqueeze(0)
                    _target = batch["target"]
                    key = batch["keys"][batch_element_idx]
                    target = (
                        [tg[batch_element_idx].unsqueeze(0) for tg in _target]
                        if isinstance(_target, list)
                        else _target[batch_element_idx].unsqueeze(0)
                    )
                    data = data.to(device, non_blocking=True)
                    if isinstance(target, list):
                        target = [i.to(device, non_blocking=True) for i in target]
                    else:
                        target = target.to(device, non_blocking=True)

                    with (
                        autocast(device.type, enabled=True)
                        if device.type == "cuda"
                        else dummy_context()
                    ):
                        output = net(data)
                        # del data
                        l = criterion(output, target)

                    # collect outputs and losses
                    output_whole.append([out.cpu() for out in output])
                    if key not in loss_whole:
                        loss_whole[key] = []
                    loss_whole[key].append(l.cpu())

        # get highres outputs
        output_whole_highres = [out[0] for out in output_whole]

        # get average loss per sample
        for key in loss_whole.keys():
            loss_whole[key] = torch.mean(torch.stack(loss_whole[key]))

        return output_whole, output_whole_highres, loss_whole

    def lid_term_batched(self, X, k=20, batch_size=128, eps=1e-6):
        """
        Compute Local Intrinsic Dimensionality (LID) for a batch of samples.

        Calculates LID scores by computing k-nearest neighbor distances in batches
        to manage memory efficiently, then applies the LID formula to estimate
        the intrinsic dimensionality of the data.

        Args:
            X: Input data as list of tensors or stacked tensor of shape (N, C, ..., H, W)
            k: Number of nearest neighbors to consider (default: 20)
            batch_size: Batch size for distance computation (default: 128)
            eps: Small epsilon value to avoid division by zero (default: 1e-6)

        Returns:
            Array of LID scores for each sample
        """
        # X is list of tensors, each (1, C, (D), H, W)
        if isinstance(X, list):
            X = torch.stack(X, dim=0)  # (N, C, (D), H, W)

        N = X.size(0)
        X_flat = X.view(N, -1).cpu().numpy()

        # allocate full distance matrix on CPU
        distances = np.empty((N, N))

        # fill in pairwise distances row-wise in batches
        for i in range(0, N, batch_size):
            print(
                f"Computing distances for samples {i} to {min(i + batch_size, N)} / {N}"
            )
            X_i = X_flat[i : i + batch_size]
            d_block = cdist(X_i, X_flat, metric="euclidean")
            distances[i : i + batch_size] = d_block

        # compute LID scores
        f = lambda v: -k / (np.sum(np.log(v / (v[-1] + eps))) + eps)
        sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[
            :, 1 : k + 1
        ]
        m, n = sort_indices.shape
        row_idx = np.arange(m)[:, None]
        col_idx = sort_indices
        distances_ = distances[row_idx, col_idx]
        lids = np.apply_along_axis(f, axis=1, arr=distances_)
        return lids

    def set_lid(self, lid_values, c_id: int):
        """
        Set LID values for a specific client.

        Args:
            lid_values: List or array of LID values for the client's samples
            c_id: Client ID
        """
        self.LID_whole[c_id] = lid_values
        self.LID_client[c_id] = np.mean(lid_values)
        self.LID_accumulative_client[c_id] += np.mean(lid_values)

    def set_loss(self, loss_values, c_id: int):
        """
        Set loss values for a specific client.

        Args:
            loss_values: Dict with sample IDs as keys and loss values as values
            c_id: Client ID
        """
        self.loss_whole[c_id] = loss_values
        for key, value in loss_values.items():
            if key not in self.loss_accumulative_whole[c_id]:
                self.loss_accumulative_whole[c_id][key] = 0.0
            self.loss_accumulative_whole[c_id][key] += value.item()

    def preproc_central_steps(self, fl_round: int, seed: int = 42):
        """
        Centralized steps for FedCorr's pre-processing stage.
        Identification of noisy clients via GMM on client's LID scores.
        For each noisy client, identification of noisy samples via per-sample loss value including label correction.

        Args:
            fl_round (int): Current federated learning round.
        """
        logging.info(
            f"FedCorr pre-processing stage centralized steps at FL round {fl_round}!"
        )
        # identify noisy clients via GMM on client's LID scores
        lid_accumulative_values = np.array(
            [self.LID_accumulative_client[c_id] for c_id in range(len(self.clients))]
        ).reshape(-1, 1)
        # fit two-component GMM
        gmm_LID = GaussianMixture(n_components=2, random_state=seed).fit(
            lid_accumulative_values
        )
        # predict client labels grouping them into two groups
        labels_LID = gmm_LID.predict(lid_accumulative_values)
        # client group with smaller mean LID is considered clean
        clean_label = np.argsort(gmm_LID.means_[:, 0])[0]
        # select noisy and clean clients
        self.noisy_clients = np.where(labels_LID != clean_label)[0]
        self.clean_clients = np.where(labels_LID == clean_label)[0]
        logging.info(f"Identified noisy clients: {self.noisy_clients.tolist()}")
        logging.info(f"Identified clean clients: {self.clean_clients.tolist()}")

        # for each noisy client:
        # - estimate noise level via GMM on per-sample loss values
        for c_id in self.noisy_clients:
            logging.info(
                f"Processing noisy client {c_id} for noisy sample identification..."
            )
            # get per-sample losses
            per_sample_losses = np.array(
                list(self.loss_accumulative_whole[c_id].values())
            ).reshape(-1, 1)
            # fit two-component GMM
            gmm_loss = GaussianMixture(n_components=2, random_state=seed).fit(
                per_sample_losses
            )
            # predict sample labels grouping them into two groups
            labels_loss = gmm_loss.predict(per_sample_losses)
            # sample group with smaller mean loss is considered clean
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]
            # select noisy samples (or get their indices)
            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            # compute estimated noisy level for this client
            self.clients_estimated_noisy_level[c_id] = len(pred_n) / len(
                per_sample_losses
            )
            logging.info(
                f"Client {c_id} estimated noisy level: {self.clients_estimated_noisy_level[c_id]:.4f}"
            )

    def label_correction(
        self,
        sample_idx: list,
        output: torch.Tensor,
        target: torch.Tensor,
        client_id: int,
        topk: int = 1000,
    ):
        """
        FedCorr's label correction based on per-sample loss values and model output confidence.
        Based on per-sample loss values, identify samples to be relabelled according to estimated noisy level.
        Further filter selected samples based on model output confidence.
        For selected samples, interchange target with model output (argmax over channels).

        Args:
            sample_idx (list): List of sample indices in the current batch.
            output (torch.Tensor): Model outputs for the current batch.
            target (torch.Tensor): Ground truth targets for the current batch.
            client_id (int): ID of the client performing label correction.
            topk (int): Number of top probabilities to consider for confidence estimation.

        Returns:
            torch.Tensor: Corrected targets after label correction.
        """
        # get deep copies of target to avoid in-place modifications
        target_ = [t.clone() for t in target]

        # get current accumulated losses for this client and sample indices
        acc_losses_all = self.loss_accumulative_whole[client_id]
        loss = []
        for idx, key in enumerate(sample_idx):
            if key in acc_losses_all.keys():
                loss.append(acc_losses_all[key])
            else:
                loss.append(None)
        loss = np.array(loss)

        # determine indices that need to be relabelled
        # 1. based on estimated noisy level and relabel ratio
        relabel_idx = (-loss).argsort()[
            : int(
                len(sample_idx)
                * self.clients_estimated_noisy_level[client_id]
                * self.fedcorr_relabel_ratio
            )
        ]
        logging.debug(
            f"Client {client_id} relabelling {len(relabel_idx)} samples based on loss!"
        )
        # 2. based on confidence threshold on model outputs
        highres_output = output[0]
        probs_flat = highres_output.reshape(highres_output.shape[0], -1)
        topk_vals = np.partition(probs_flat.cpu().detach().numpy(), -topk, axis=1)[
            :, -topk:
        ]
        conf_per_image = topk_vals.mean(axis=1)
        high_conf_idx = np.where(
            conf_per_image > self.fedcorr_relabel_confidence_thres
        )[0]

        # intersect both criteria
        loss_n_conf_relabel_idx = list(set(high_conf_idx) & set(relabel_idx))
        logging.debug(
            f"Client {client_id} relabelling {len(loss_n_conf_relabel_idx)} samples after confidence check!"
        )

        # interchange target with output for selected indices
        logging.info(
            f"Client {client_id}: Relabelling {len(loss_n_conf_relabel_idx)} samples out of {len(sample_idx)} samples!"
        )
        if len(loss_n_conf_relabel_idx) > 0:
            for idx in loss_n_conf_relabel_idx:
                for res_lvl in range(len(target_)):
                    # argmax over output channels
                    argmaxed_nononehot_output = torch.argmax(
                        output[res_lvl][idx], dim=0, keepdim=True
                    )
                    # replace target with argmaxed output
                    target_[res_lvl][idx] = argmaxed_nononehot_output
        return target_
