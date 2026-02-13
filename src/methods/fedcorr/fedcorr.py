import os
import copy
import ast
import datetime
from glob import glob
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
        fl_strategy_state: dict = None,
    ):
        super().__init__(clients=clients)
        self.experiment_id = None  # to be set when saving state

        self.name = "fedcorr"
        self.fedcorr_preproc_rounds = int(fedcorr_preproc_rounds_frac * num_rounds)
        self.fedcorr_relabel_ratio = fedcorr_relabel_ratio
        self.fedcorr_relabel_confidence_thres = fedcorr_relabel_confidence_thres
        self.fedcorr_proxterm_beta = fedcorr_proxterm_beta

        # variables for client and sample noise estimation
        self.noisy_clients = (
            []
            if fl_strategy_state is None
            else fl_strategy_state.get("noisy_clients", [])
        )
        self.clean_clients = (
            []
            if fl_strategy_state is None
            else fl_strategy_state.get("clean_clients", [])
        )
        self.LID_whole = (
            {c_id: None for c_id in range(len(self.clients))}
            if fl_strategy_state is None
            else fl_strategy_state.get("LID_whole", {})
        )
        self.LID_client = (
            {c_id: None for c_id in range(len(self.clients))}
            if fl_strategy_state is None
            else fl_strategy_state.get("LID_client", {})
        )
        self.LID_accumulative_client = (
            {c_id: 0.0 for c_id in range(len(self.clients))}
            if fl_strategy_state is None
            else fl_strategy_state.get("LID_accumulative_client", {})
        )
        self.loss_whole = (
            {c_id: None for c_id in range(len(self.clients))}
            if fl_strategy_state is None
            else fl_strategy_state.get("loss_whole", {})
        )
        self.loss_accumulative_whole = (
            {c_id: {} for c_id in range(len(self.clients))}
            if fl_strategy_state is None
            else fl_strategy_state.get("loss_accumulative_whole", {})
        )
        self.clients_estimated_noisy_level = (
            {client.client_id: 0.0 for client in self.clients}
            if fl_strategy_state is None
            else fl_strategy_state.get("clients_estimated_noisy_level", {})
        )
        self.path_to_global_fl_model_weights = (
            None
            if fl_strategy_state is None
            else fl_strategy_state.get("path_to_global_fl_model_weights", None)
        )
        # check if noisy and clean clients are lists
        if not isinstance(self.noisy_clients, list):
            self.noisy_clients = ast.literal_eval(self.noisy_clients.replace(" ", ", "))
        if not isinstance(self.clean_clients, list):
            self.clean_clients = ast.literal_eval(self.clean_clients.replace(" ", ", "))
        # check if keys are strings, convert to int if needed
        if fl_strategy_state is not None:
            if isinstance(self.LID_whole.keys().__iter__().__next__(), str):
                self.LID_whole = {int(k): v for k, v in self.LID_whole.items()}
            if isinstance(self.LID_client.keys().__iter__().__next__(), str):
                self.LID_client = {int(k): v for k, v in self.LID_client.items()}
            if isinstance(
                self.LID_accumulative_client.keys().__iter__().__next__(), str
            ):
                self.LID_accumulative_client = {
                    int(k): v for k, v in self.LID_accumulative_client.items()
                }
            if isinstance(self.loss_whole.keys().__iter__().__next__(), str):
                self.loss_whole = {int(k): v for k, v in self.loss_whole.items()}
            if isinstance(
                self.loss_accumulative_whole.keys().__iter__().__next__(), str
            ):
                self.loss_accumulative_whole = {
                    int(k): v for k, v in self.loss_accumulative_whole.items()
                }
            if isinstance(
                self.clients_estimated_noisy_level.keys().__iter__().__next__(), str
            ):
                self.clients_estimated_noisy_level = {
                    int(k): v for k, v in self.clients_estimated_noisy_level.items()
                }

        # compute number of fl_rounds for fine-tuning and full-training stages
        self.fedcorr_finetune_rounds, self.fedcorr_fulltrain_rounds = (
            self._compute_stage_rounds(num_rounds)
        )

        # initialize estimated noisy levels for clients

        # global model weights for proximal term computation are set from orchestrator
        self.global_fl_model_weights = (
            None
            if self.path_to_global_fl_model_weights is None
            else torch.load(self.path_to_global_fl_model_weights)
        )

        logging.info(
            f"FedCorr initialized with preproc rounds frac {self.fedcorr_preproc_rounds}, \
                finetune rounds {self.fedcorr_finetune_rounds}, \
                fulltrain rounds {self.fedcorr_fulltrain_rounds}, \
                relabel ratio {self.fedcorr_relabel_ratio}, \
                relabel confidence thres {self.fedcorr_relabel_confidence_thres}, \
                and proxterm beta {self.fedcorr_proxterm_beta}!"
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

    def compute_proximal_loss_term(
        self, local_model_weights, c_id: int, device: torch.device
    ):
        """
        Compute FedCorr's proximal loss term.
        Prox term = beta * mu * ||w_diff||^2 w/ w_diff = local_model_weights - global_model_weights

        Args:
            glob_model_weights: Current global model weights.
        """
        # compute weight difference norm
        local_state = (
            local_model_weights.state_dict()
            if hasattr(local_model_weights, "state_dict")
            else local_model_weights
        )
        global_state = (
            self.global_fl_model_weights.state_dict()
            if hasattr(self.global_fl_model_weights, "state_dict")
            else self.global_fl_model_weights
        )
        w_diff = torch.tensor(0.0, device=device)
        # get addresses of keys
        keys = list(local_state.keys())
        address_key_dict = {}
        for k in keys:
            address = local_state[k].data_ptr()
            if address not in address_key_dict.keys():
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)
        # compute w_diff
        with torch.no_grad():
            for a in address_key_dict.keys():
                key = address_key_dict[a][0]
                global_key = key.replace("_orig_mod.", "")
                local_tensor = local_state[key].to(device, non_blocking=True)
                global_tensor = global_state[global_key].to(device, non_blocking=True)
                diff = local_tensor - global_tensor
                w_diff += torch.pow(torch.norm(diff), 2)
                del local_tensor, global_tensor, diff
        w_diff = torch.sqrt(w_diff)

        # compute proximal term
        prox_term = (
            self.fedcorr_proxterm_beta
            * self.clients_estimated_noisy_level[c_id]
            * w_diff
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        net = net.to(nnunet_trainer.device)
        net.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader = nnunet_trainer.dataloader_train
        device = nnunet_trainer.device
        criterion = nnunet_trainer.loss

        # prepare to collect outputs and losses
        output_whole = []
        loss_whole = {}

        logging.info("Getting per-sample outputs and losses...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
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
                    data.float()
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

    def lid_term_batched(self, X, k=20, eps=1e-6):
        """
        Compute Local Intrinsic Dimensionality (LID) for a batch of samples.

        Memory-efficient version that computes only k-nearest neighbors instead
        of the full distance matrix, reducing memory from O(N²) to O(N×k).

        Args:
            X: Input data as list of tensors or stacked tensor of shape (N, C, ..., H, W)
            k: Number of nearest neighbors to consider (default: 20)
            eps: Small epsilon value to avoid division by zero (default: 1e-6)

        Returns:
            Array of LID scores for each sample
        """
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info("Computing LID scores in batched manner...")
        logging.info(f"Time: {start_time}")
        # ensure batch size does not exceed dataloader batch size
        default_batch_size = 16
        batch_size = min(
            default_batch_size, self.clients[0].model.nnunet_trainer.batch_size
        )

        # X is list of tensors, each (1, C, (D), H, W)
        if isinstance(X, list):
            X = torch.stack(X, dim=0)  # (N, C, (D), H, W)

        N = X.size(0)
        if N <= 1:
            return np.zeros((N,))

        X = X.to(self.clients[0].model.nnunet_trainer.device, non_blocking=True)
        X_flat = X.view(N, -1).float()

        # Clamp k to avoid requesting more neighbors than exist (exclude self)
        k_eff = min(k, N - 1)

        # Allocate only k-nearest neighbor distances instead of full N×N matrix
        # This reduces memory from O(N²) to O(N×k)
        k_distances = torch.zeros((N, k_eff), device=X_flat.device)

        # Compute k-nearest neighbors in batches to save memory
        for i in range(0, N, batch_size):
            logging.info(
                f"Computing distances for samples {i} to {min(i + batch_size, N)} / {N}"
            )
            X_i = X_flat[i : i + batch_size]
            # Compute distances only for current batch (GPU)
            d_block = torch.cdist(X_i, X_flat, p=2)

            # For each sample in batch, extract only k+1 nearest neighbors
            k_plus_one = k_eff + 1
            d_small = torch.topk(d_block, k_plus_one, dim=1, largest=False).values
            # Skip first (self with distance 0), keep next k
            k_distances[i : i + d_small.shape[0]] = d_small[:, 1:k_plus_one]

        # Compute LID scores from k-nearest neighbor distances
        # v contains k nearest neighbor distances (sorted, excluding self)
        # LID = -k / sum(log(v_i / v_k)) where v_k is the k-th neighbor
        v_k = k_distances[:, -1].unsqueeze(1)
        ratio = k_distances / (v_k + eps)
        log_sum = torch.sum(torch.log(ratio + eps), dim=1)
        lids = -k_eff / (log_sum + eps)

        endtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Finished computing LID scores! Time: {endtime}")
        return lids.detach().cpu().numpy()

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

    def central_noisy_client_identification(self, fl_round: int, seed: int = 42):
        """
        Centralized steps for FedCorr's pre-processing stage.
        Identification of noisy clients via GMM on client's LID scores.
        For each noisy client, identification of noisy samples via per-sample loss value including label correction.

        Args:
            fl_round (int): Current federated learning round.
        """
        logging.info(
            f"FedCorr central steps at FL round {fl_round}: Identifying noisy clients!"
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

    def central_noise_level_estimation(self, fl_round: int, seed: int = 42):
        """
        Centralized steps for FedCorr's pre-processing stage.
        For each noisy client, identification of noisy samples via per-sample loss value including label correction
        Args:
            fl_round (int): Current federated learning round.
        """
        logging.info(
            f"FedCorr central steps at FL round {fl_round}: Identifying noisy samples!"
        )

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
        data: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        client_id: int,
        is_fedcorr_fulltrain_stage: bool = False,
        fedcorr_global_network=None,
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
            is_fedcorr_fulltrain_stage (bool): Whether currently in FedCorr full-training stage.
            fedcorr_global_network: Global model network for full-training stage (if needed).
            topk (int): Number of top probabilities to consider for confidence estimation.

        Returns:
            torch.Tensor: Corrected targets after label correction.
        """
        # if is_fedcorr_fulltrain_stage is True, output has to be recomputed form global model state after finetuning
        if is_fedcorr_fulltrain_stage and fedcorr_global_network is not None:
            # predict via global model state
            with torch.no_grad():
                output = fedcorr_global_network(data)

        # get deep copies of target to avoid in-place modifications
        target_ = [t.clone() for t in target]

        # get current accumulated losses for this client and sample indices
        acc_losses_all = self.loss_accumulative_whole[client_id]
        loss = []
        sentinel = -1e9  # very small so it ranks last after (-loss)
        for idx, key in enumerate(sample_idx):
            if key in acc_losses_all.keys():
                loss.append(acc_losses_all[key])
            else:
                loss.append(sentinel)
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

    def save_global_model_weights(self):
        """
        Save the global model weights to disk for potential restart, and set self.path_to_global_fl_model_weights.
        """
        if self.path_to_global_fl_model_weights is None:
            exp_id_folder = self.clients[0].results_dir
            self.path_to_global_fl_model_weights = os.path.join(
                exp_id_folder, "fedcorr_global_fl_model_weights.pth"
            )
            torch.save(
                self.global_fl_model_weights, self.path_to_global_fl_model_weights
            )

            # update state
            logging.info(
                f"Saved FedCorr global model weights to {self.path_to_global_fl_model_weights}!"
            )
            self.save_state(exp_id=self.experiment_id, client_id=None)

    def save_state(self, exp_id: str, client_id: int = None):
        """
        Save the current state of FedCorr method to experiment's cli args file.
        """
        # compose fl_strategy_state dict
        fl_strategy_state = {
            "LID_whole": self.LID_whole,
            "LID_client": self.LID_client,
            "LID_accumulative_client": self.LID_accumulative_client,
            "loss_whole": self.loss_whole,
            "loss_accumulative_whole": self.loss_accumulative_whole,
            "noisy_clients": self.noisy_clients,
            "clean_clients": self.clean_clients,
            "clients_estimated_noisy_level": self.clients_estimated_noisy_level,
            "path_to_global_fl_model_weights": self.path_to_global_fl_model_weights,
        }
        self.experiment_id = exp_id

        args_file = self.save_fl_strategy_state_to_file(
            fl_strategy_state=fl_strategy_state, exp_id=exp_id
        )

        logging.info(f"Saved FedCorr state to {args_file}")
