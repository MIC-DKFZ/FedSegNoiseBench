"""
FedSelect: Proxy-Validated Importance-Aware Federated Sample Selection with Meta Learning
Reference: Zhang et al. 2025 (KDD 2025)
Paper: https://dl.acm.org/doi/epdf/10.1145/3711896.3737093
"""

import copy
import os
import torch
import torch.nn.functional as F
import logging
import random
import heapq
from glob import glob
from torch.utils.data import WeightedRandomSampler, Dataset
from torch.optim.sgd import SGD

from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)

from methods.fedavg.fedavg import FedAvg


class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if "." in name:
            name_split = name.split(".")
            module_name = name_split[0]
            rest_name = ".".join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]
        lr = group["lr"]

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            if grad is None:
                continue
            parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad

            if momentum != 0 and "momentum_buffer" in self.state[parameter]:
                buffer = self.state[parameter]["momentum_buffer"]
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1 - dampening)
            else:
                grad_b = grad_wd

            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b

            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))


class ProxyValidationDataset(Dataset):
    """
    Custom dataset wrapper that selects a subset of samples by index.
    Used for creating proxy validation datasets with top-k important samples.
    """

    def __init__(self, base_dataset, selected_identifiers: list):
        """
        Args:
            base_dataset: The original dataset to wrap
            selected_identifiers: List of identifiers to include in this dataset
        """
        self.base_dataset = base_dataset
        self.selected_identifiers = selected_identifiers

    def __len__(self):
        return len(self.selected_identifiers)

    def __getitem__(self, idx):
        identifier = self.selected_identifiers[idx]
        return self.base_dataset[identifier]


class VNet(torch.nn.Module):
    """
    Validation Network (VNet) used for sample importance weighting.
    Maps sample loss to importance weight between 0 and 1.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 100, output_dim: int = 1):
        super(VNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class FedSelect(FedAvg):
    """
    FedSelect: Proxy-Validated Importance-Aware Federated Sample Selection with Meta Learning

    This class implements the FedSelect FL strategy which:
    1. Uses meta-learning with a proxy validation dataset to select important samples
    2. Implements momentum-based meta-margin function for discovering influential samples
    3. Selects high-quality clients and samples for training

    Inherits from FedAvg for baseline aggregation functionality.
    """

    def __init__(
        self,
        clients: list = None,
        num_rounds: int = None,
        warmup_rounds_frac: float = None,
        client_select_ratio: float = None,
        sample_select_ratio: float = None,
        meta_momentum: float = None,
        reward_data_size_frac: float = None,
        fl_strategy_state: dict = None,
    ):
        """
        Initialize FedSelect strategy.

        Args:
            clients: List of client objects
            num_rounds: Total number of FL rounds
            warmup_rounds_frac: Fraction of rounds for warmup phase (default: 0.1)
            client_select_ratio: Ratio of clients to select in each round (default: 0.4)
            sample_select_ratio: Ratio of samples to select per client (default: 0.6)
            meta_momentum: Momentum for meta-margin computation (default: 0.9)
            reward_data_size_frac: Fraction of reward/proxy validation dataset size (default: 0.1)
            fl_strategy_state: Dictionary containing saved state for restart (default: None)
        """
        # Call parent FedAvg constructor
        super().__init__(clients=clients)

        self.name = "fedselect"

        # Load from saved state if available, otherwise use provided parameters
        if fl_strategy_state is not None:
            self.warmup_rounds = fl_strategy_state["warmup_rounds"]
            self.client_select_ratio = fl_strategy_state["client_select_ratio"]
            self.sample_select_ratio = fl_strategy_state["sample_select_ratio"]
            self.meta_momentum = fl_strategy_state["meta_momentum"]
            self.meta_lr = fl_strategy_state["meta_lr"]
            self.reward_data_size_frac = fl_strategy_state["reward_data_size_frac"]
            # Convert string keys back to integers (from JSON serialization)
            self.selected_clients = [
                int(cid) if isinstance(cid, str) else cid
                for cid in fl_strategy_state.get("selected_clients", [])
            ]
            client_weights_raw = fl_strategy_state.get("client_weights", {})
            self.client_weights = {
                int(k) if isinstance(k, str) else k: v
                for k, v in client_weights_raw.items()
            }
            logging.info(
                f"FedSelect: Loading from saved state with warmup_rounds={self.warmup_rounds}"
            )
        else:
            self.warmup_rounds = int(warmup_rounds_frac * num_rounds)
            self.client_select_ratio = client_select_ratio
            self.sample_select_ratio = sample_select_ratio
            self.meta_momentum = meta_momentum
            self.meta_lr = 1e-3  # default learning rate for VNet meta model
            self.reward_data_size_frac = reward_data_size_frac
            self.selected_clients = []  # Currently selected clients
            self.client_weights = {i: 0.0 for i in range(len(clients))}

        # Initialize dictionaries for all clients using consistent indexing
        client_ids = range(len(self.clients))
        # Load saved per-client state if available
        if fl_strategy_state is not None:
            meta_margin_raw = fl_strategy_state.get("meta_margin_pre", {})
            sample_total_loss_raw = fl_strategy_state.get("sample_total_loss_pre", {})
            sample_weights_raw = fl_strategy_state.get("clients_sample_weights", {})

            def _coerce_client_keyed_dict(raw_dict):
                coerced = {}
                for k, v in raw_dict.items():
                    try:
                        k_int = int(k)
                    except Exception:
                        k_int = k
                    coerced[k_int] = v if isinstance(v, dict) else {}
                return coerced

            meta_margin_loaded = _coerce_client_keyed_dict(meta_margin_raw)
            sample_total_loss_loaded = _coerce_client_keyed_dict(sample_total_loss_raw)
            sample_weights_loaded = _coerce_client_keyed_dict(sample_weights_raw)
        else:
            meta_margin_loaded = {}
            sample_total_loss_loaded = {}
            sample_weights_loaded = {}

        self.meta_margin_pre = {
            c_id: meta_margin_loaded.get(c_id, {}) for c_id in client_ids
        }
        self.sample_total_loss_pre = {
            c_id: sample_total_loss_loaded.get(c_id, {}) for c_id in client_ids
        }
        self.client_meta_models = {c_id: VNet() for c_id in client_ids}
        self.clients_sample_weights = {
            c_id: sample_weights_loaded.get(c_id, None) for c_id in client_ids
        }
        self.clients_proxy_validation_dataloaders = {c_id: None for c_id in client_ids}
        self.client_meta_optimizers = {c_id: None for c_id in client_ids}

        # Load meta models and optimizers from checkpoints if restarting
        if fl_strategy_state is not None:
            meta_model_paths = fl_strategy_state.get("meta_model_paths", {})
            meta_optimizer_paths = fl_strategy_state.get("meta_optimizer_paths", {})

            for c_id in client_ids:
                # Load meta model if checkpoint exists
                if (
                    str(c_id) in meta_model_paths
                    and meta_model_paths[str(c_id)] is not None
                ) or (c_id in meta_model_paths and meta_model_paths[c_id] is not None):
                    # Try both string and int keys
                    model_path = meta_model_paths.get(
                        str(c_id)
                    ) or meta_model_paths.get(c_id)
                    if model_path and os.path.exists(model_path):
                        logging.info(
                            f"FedSelect: Loading meta model for client {c_id} from {model_path}"
                        )
                        self.client_meta_models[c_id].load_state_dict(
                            torch.load(
                                model_path, map_location="cpu", weights_only=True
                            )
                        )
                    elif model_path:
                        logging.warning(
                            f"FedSelect: Meta model checkpoint not found at {model_path}"
                        )

                # Load optimizer if checkpoint exists
                if (
                    str(c_id) in meta_optimizer_paths
                    and meta_optimizer_paths[str(c_id)] is not None
                ) or (
                    c_id in meta_optimizer_paths
                    and meta_optimizer_paths[c_id] is not None
                ):
                    # Try both string and int keys
                    optimizer_path = meta_optimizer_paths.get(
                        str(c_id)
                    ) or meta_optimizer_paths.get(c_id)
                    if optimizer_path and os.path.exists(optimizer_path):
                        logging.info(
                            f"FedSelect: Loading meta optimizer for client {c_id} from {optimizer_path}"
                        )
                        # Initialize optimizer first (will be properly set up during training)
                        self.client_meta_optimizers[c_id] = torch.optim.Adam(
                            self.client_meta_models[c_id].parameters(), lr=self.meta_lr
                        )
                        self.client_meta_optimizers[c_id].load_state_dict(
                            torch.load(
                                optimizer_path, map_location="cpu", weights_only=True
                            )
                        )
                    elif optimizer_path:
                        logging.warning(
                            f"FedSelect: Meta optimizer checkpoint not found at {optimizer_path}"
                        )

        # Initialize fl_strategy_state dictionary (will be updated when save_state is called)
        self.fl_strategy_state = {
            "warmup_rounds": self.warmup_rounds,
            "client_select_ratio": self.client_select_ratio,
            "sample_select_ratio": self.sample_select_ratio,
            "meta_momentum": self.meta_momentum,
            "meta_lr": self.meta_lr,
            "reward_data_size_frac": self.reward_data_size_frac,
            "selected_clients": self.selected_clients,
            "client_weights": self.client_weights,
            "meta_margin_pre": self.meta_margin_pre,
            "sample_total_loss_pre": self.sample_total_loss_pre,
            "clients_sample_weights": self.clients_sample_weights,
            "meta_model_paths": (
                {
                    int(k) if isinstance(k, str) else k: v
                    for k, v in fl_strategy_state.get("meta_model_paths", {}).items()
                }
                if fl_strategy_state is not None
                else {c_id: None for c_id in client_ids}
            ),
            "meta_optimizer_paths": (
                {
                    int(k) if isinstance(k, str) else k: v
                    for k, v in fl_strategy_state.get(
                        "meta_optimizer_paths", {}
                    ).items()
                }
                if fl_strategy_state is not None
                else {c_id: None for c_id in client_ids}
            ),
        }

        logging.info(
            f"Initialized FedSelect with warmup_rounds={self.warmup_rounds}, "
            f"client_select_ratio={self.client_select_ratio}, "
            f"sample_select_ratio={self.sample_select_ratio}"
        )

    def fedselect_aggregate(self, client_checkpoints: dict = {}, fl_round: int = 0):
        """
        FedSelect aggregation strategy.

        During warmup phase: Use FedAvg (inherited from parent)
        During training phase: Use client-weighted aggregation based on importance
        """
        if fl_round < self.warmup_rounds:
            logging.info(
                f"FedSelect warmup phase (round {fl_round}/{self.warmup_rounds}): using FedAvg"
            )
            return self.fed_avg(client_checkpoints)
        else:
            logging.info(
                f"FedSelect training phase (round {fl_round}): using importance-weighted aggregation"
            )
            # Use weighted aggregation based on client importance
            return self._weighted_aggregate(client_checkpoints)

    def _weighted_aggregate(self, client_checkpoints: dict = {}):
        """
        Perform client-weighted aggregation using computed client importance weights.
        """
        # keep direct reference (no deepcopy to avoid GPU OOM)
        _client_checkpoints = client_checkpoints

        first_key = list(_client_checkpoints.keys())[0]
        first_client_weights = _client_checkpoints[first_key]
        _server_model_weights = {
            k: v.detach().clone() for k, v in first_client_weights.items()
        }
        keys = list(first_client_weights.keys())

        # Create address-to-key mapping
        address_key_dict = {}
        for k in keys:
            address = first_client_weights[k].data_ptr()
            if address not in address_key_dict:
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)

        # Normalize client weights
        aggregation_weights = {
            cid: self.client_weights[cid] if cid in self.selected_clients else 0.0
            for cid in _client_checkpoints.keys()
        }
        total_weight = sum(aggregation_weights.values())
        normalized_weights = {
            cid: w / total_weight for cid, w in aggregation_weights.items()
        }

        # Perform weighted aggregation
        for address in address_key_dict.keys():
            _server_model_weights[address_key_dict[address][0]] = None
            for client_id, client_weights in _client_checkpoints.items():
                weight = normalized_weights.get(client_id, 0.0)
                if _server_model_weights[address_key_dict[address][0]] is None:
                    _server_model_weights[address_key_dict[address][0]] = (
                        client_weights[address_key_dict[address][0]] * weight
                    )
                else:
                    _server_model_weights[address_key_dict[address][0]] += (
                        client_weights[address_key_dict[address][0]] * weight
                    )

        return _server_model_weights

    def get_most_influential_client(self):
        """
        Meta model is updated with most influential selected clients,
        i.e. client with highest client weight.
        """
        most_influential_client_id = max(
            self.client_weights, key=self.client_weights.get
        )
        logging.info(
            f"FedSelect: most influential client selected for meta model update: {most_influential_client_id} with weight {self.client_weights[most_influential_client_id]:.4f}"
        )
        return most_influential_client_id

    def select_clients(self, fl_round: int, num_clients_to_select: int = None):
        """
        Select clients based on their importance weights.

        Args:
            fl_round: Current FL round
            num_clients_to_select: Number of clients to select (uses default ratio if None)
        """
        num_to_select = int(len(self.clients) * self.client_select_ratio)
        if fl_round < self.warmup_rounds:
            # During warmup, select clients randomly
            selected = random.sample(
                [c.client_id for c in self.clients],
                min(num_to_select, len(self.clients)),
            )
            logging.info(
                f"FedSelect warmup: randomly selected {len(selected)} clients: {selected}"
            )
            self.selected_clients = selected
        else:
            # During training, select based on importance weights using WeightedRandomSampler
            client_ids = [c.client_id for c in self.clients]
            weights = [self.client_weights[cid] for cid in client_ids]

            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=min(num_to_select, len(self.clients)),
                replacement=False,
            )
            selected = [client_ids[i] for i in sampler]
            logging.info(
                f"FedSelect: selected {len(selected)} clients based on weighted importance: {selected}"
            )
            self.selected_clients = selected

    def compute_sample_weights(
        self, nnunet_trainer, client_id: int = None, fl_round: int = 0
    ):
        """
        Compute sample weights for a client's train data.
        During warmup, assign uniform weights.
        After warmup, compute weights using its VNet meta model.

        Args:
            nnunet_trainer: nnUNet trainer object for the client
            client_id: Client identifier
            fl_round: Current FL round
        """
        # during warmup, assign uniform weights
        if fl_round < self.warmup_rounds:
            logging.info(
                f"FedSelect warmup phase (round {fl_round}): skipping sample weight computation"
            )
            self.clients_sample_weights[client_id] = {
                key: 1.0 for key in nnunet_trainer.tr_keys
            }
            return
        # after warmup, compute sample weights using VNet
        logging.info(
            f"FedSelect training phase (round {fl_round}): computing sample weights for client {client_id}"
        )
        # get net, loader, device, criterion from nnunet_trainer
        net = nnunet_trainer.network.to(nnunet_trainer.device)
        net.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader = nnunet_trainer.dataloader_train
        device = nnunet_trainer.device
        criterion = nnunet_trainer.loss

        # get meta model for this client
        meta_model = self.client_meta_models[client_id].to(device)
        meta_model.eval()

        # forward pass through training data to compute sample losses
        processed_batch_el_keys = set()
        sample_weights = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= nnunet_trainer.num_iterations_per_epoch:
                    break
                for batch_element_idx in range(len(batch["data"])):
                    # get data, target, keys for this batch element
                    data = batch["data"][batch_element_idx].unsqueeze(0)
                    _target = batch["target"]
                    key = batch["keys"][batch_element_idx]

                    # check if batch element already processed
                    if key in processed_batch_el_keys:
                        continue

                    # some target handling and data to device
                    target = (
                        [tg[batch_element_idx].unsqueeze(0) for tg in _target]
                        if isinstance(_target, list)
                        else _target[batch_element_idx].unsqueeze(0)
                    )
                    if isinstance(target, list):
                        target = [i.to(device, non_blocking=True) for i in target]
                    else:
                        target = target.to(device, non_blocking=True)
                    data = data.to(device, non_blocking=True)

                    # forward pass
                    with (
                        torch.autocast(device.type, enabled=True)
                        if device.type == "cuda"
                        else dummy_context()
                    ):
                        output = net(data)
                        l = criterion(output, target)

                    # feed loss through meta model to get sample weight
                    weight = meta_model(l.unsqueeze(0).reshape(1, -1))
                    sample_weights[key] = float(weight.detach().mean().item())
                    processed_batch_el_keys.add(key)

        self.clients_sample_weights[client_id] = sample_weights

        logging.info(
            f"FedSelect: computed sample weights for client {client_id}, total samples: {len(sample_weights)}"
        )
        del net, loader, criterion, data, target, output, l, weight
        empty_cache(device)

    def compute_client_weights(self, nnunet_trainer, client_id: int):
        """
        Compute client importance weight based on sample weights.
        client weight re-computed each round after sample weights are updated,
        used for client selection and aggregation.

        Args:
            client_id: Client identifier
        """
        sample_weights = self.clients_sample_weights.get(client_id, {})
        total_weight = sum(sample_weights.values()) if sample_weights else 0.0
        self.client_weights[client_id] = total_weight / max(
            len(nnunet_trainer.tr_keys), 1
        )
        logging.info(
            f"FedSelect: computed client weight for client {client_id}: {self.client_weights[client_id]:.4f}"
        )

    def compute_meta_margin_scores(
        self, nnunet_trainer, client_id: int = None, fl_round: int = 0
    ):
        """
        Compute meta-margin scores for a client's samples.
        Meta-margin is the difference in sample loss before and after training, used to identify influential samples.

        Args:
            nnunet_trainer: nnUNet trainer object for the client
            client_id: Client identifier
            fl_round: Current FL round
        """
        logging.info(
            f"FedSelect training phase (round {fl_round}): computing meta margin scores for client {client_id}"
        )
        # get net, loader, device, criterion from nnunet_trainer
        net = nnunet_trainer.network.to(nnunet_trainer.device)
        net.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader = nnunet_trainer.dataloader_train
        device = nnunet_trainer.device
        criterion = nnunet_trainer.loss

        # forward pass through training data to compute per-sample losses
        processed_batch_el_keys = {}
        sample_total_loss = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= nnunet_trainer.num_iterations_per_epoch:
                    break
                for batch_element_idx in range(len(batch["data"])):
                    # get data, target, keys for this batch element
                    data = batch["data"][batch_element_idx].unsqueeze(0)
                    _target = batch["target"]
                    key = batch["keys"][batch_element_idx]

                    # check if batch element already processed
                    if key in processed_batch_el_keys:
                        continue

                    # some target handling and data to device
                    target = (
                        [tg[batch_element_idx].unsqueeze(0) for tg in _target]
                        if isinstance(_target, list)
                        else _target[batch_element_idx].unsqueeze(0)
                    )
                    if isinstance(target, list):
                        target = [i.to(device, non_blocking=True) for i in target]
                    else:
                        target = target.to(device, non_blocking=True)
                    data = data.to(device, non_blocking=True)

                    # forward pass
                    with (
                        torch.autocast(device.type, enabled=True)
                        if device.type == "cuda"
                        else dummy_context()
                    ):
                        output = net(data)
                        l = criterion(output, target).detach()

                    # store scalar loss per sample key
                    loss_value = l.view(-1).mean().item()
                    sample_total_loss[key] = loss_value
                    processed_batch_el_keys[key] = loss_value

                    # Clean up after each batch
                    del output, l, data, target
                    empty_cache(device)

        # GPU clean up
        nnunet_trainer.network.to("cpu")
        del net, loader, criterion
        empty_cache(device)

        if len(sample_total_loss) == 0:
            logging.warning(
                f"FedSelect: No sample losses computed for client {client_id}."
            )
            return

        # compute meta margin scores (per key)
        prev_losses = self.sample_total_loss_pre.get(client_id, {})
        prev_margins = self.meta_margin_pre.get(client_id, {})

        keys = list(sample_total_loss.keys())
        current_losses = torch.tensor(
            [sample_total_loss[k] for k in keys], device=device
        )
        previous_losses = torch.tensor(
            [prev_losses.get(k, 0.0) for k in keys], device=device
        )

        # equation (7) in the paper
        meta_margin = previous_losses - current_losses
        meta_margin = self._normalize_tensor(meta_margin)

        # equation (8) in the paper
        prev_margin_tensor = torch.tensor(
            [prev_margins.get(k, 0.0) for k in keys], device=device
        )
        meta_margin_hat = (
            self.meta_momentum * prev_margin_tensor
            + (1 - self.meta_momentum) * meta_margin
        )
        meta_margin_hat = self._normalize_tensor(meta_margin_hat)

        # update stored dicts with per-sample values
        self.sample_total_loss_pre[client_id] = sample_total_loss
        self.meta_margin_pre[client_id] = {
            k: meta_margin_hat[i].item() for i, k in enumerate(keys)
        }
        # TODO: in original code, they do it like the following, but doesn't align with paper
        # self.meta_margin_pre[client_id] = meta_margin

    def _normalize_tensor(self, tensor: torch.Tensor):
        """
        Normalize tensor to range [0, 1].
        """
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if max_val == min_val:
            return torch.zeros_like(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def update_proxy_validation_dataset(
        self, nnunet_trainer=None, client_id: int = None
    ):
        """
        Update the proxy validation dataset for a client using top-k samples based on meta-margin scores.

        This dataset is used as a proxy for the true validation set when updating the VNet meta model.
        Samples with higher meta-margin scores (indicating higher influence) are selected.

        Args:
            client_id: Client identifier
            nnunet_trainer: nnUNet trainer object for the client (optional, for dataset access)

        Returns:
            dataloader: DataLoader with selected proxy validation samples
            total_value: Sum of weights for selected samples
        """
        if client_id is None:
            logging.warning(
                "FedSelect: client_id is required to update proxy validation dataset"
            )
            return None, 0.0

        # Get meta-margin scores for this client
        if not self.meta_margin_pre.get(client_id):
            logging.warning(
                f"FedSelect: No meta-margin scores available for client {client_id}, "
                "cannot update proxy validation dataset"
            )
            return None, 0.0
        meta_margin = self.meta_margin_pre[client_id]

        # Determine number of samples to select for proxy validation
        num_samples = len(meta_margin)
        select_num = max(1, int(self.reward_data_size_frac * num_samples))

        # Select top-k samples by meta-margin score using heapq
        top_k_keys = heapq.nlargest(
            select_num, meta_margin.keys(), key=lambda k: meta_margin[k]
        )

        # Get the base dataset from trainer if available
        if nnunet_trainer is None:
            nnunet_trainer = getattr(self.clients[client_id], "model", None)
            nnunet_trainer = getattr(nnunet_trainer, "nnunet_trainer", None)

        if nnunet_trainer is not None and hasattr(nnunet_trainer, "dataloader_train"):
            base_dataset = copy.deepcopy(
                nnunet_trainer.dataloader_train.generator._data
            )
        else:
            logging.warning(
                f"FedSelect: No trainer provided for client {client_id}, "
                "cannot create proxy validation dataloader"
            )
            return None, 0.0

        # Ensure selected keys exist in dataset identifiers
        dataset_ids = getattr(base_dataset, "identifiers", None)
        if dataset_ids is None:
            logging.warning(
                f"FedSelect: base_dataset has no identifiers for client {client_id}, "
                "cannot align proxy validation samples"
            )
            return None, 0.0
        dataset_id_set = set(dataset_ids)
        top_k_identifiers = [k for k in top_k_keys if k in dataset_id_set]
        if len(top_k_identifiers) == 0:
            logging.warning(
                f"FedSelect: No matching dataset indices found for client {client_id} "
                "when building proxy validation dataset"
            )
            return None, 0.0

        # Create proxy validation dataset with selected identifiers
        proxy_dataset = copy.deepcopy(base_dataset)
        proxy_dataset.identifiers = top_k_identifiers
        if getattr(proxy_dataset, "identifiers_noise", None) is not None:
            proxy_dataset.identifiers_noise = {
                k: proxy_dataset.identifiers_noise[k]
                for k in top_k_identifiers
                if k in proxy_dataset.identifiers_noise
            }

        # Build nnUNet dataloader (dict batches) for proxy validation
        patch_size = nnunet_trainer.configuration_manager.patch_size
        # redoce patch size for proxy validation if needed
        # patch_size = [(ps//2) if ps > (max(patch_size)//2) else ps for ps in patch_size]
        dim = len(patch_size)
        deep_supervision_scales = nnunet_trainer._get_deep_supervision_scales()
        val_transforms = nnunet_trainer.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=nnunet_trainer.is_cascaded,
            foreground_labels=nnunet_trainer.label_manager.foreground_labels,
            regions=(
                nnunet_trainer.label_manager.foreground_regions
                if nnunet_trainer.label_manager.has_regions
                else None
            ),
            ignore_label=nnunet_trainer.label_manager.ignore_label,
        )

        proxy_batch_size = 1  # min(nnunet_trainer.batch_size, len(top_k_identifiers))
        if dim == 2:
            dl_proxy = nnUNetDataLoader2D(
                proxy_dataset,
                proxy_batch_size,
                patch_size,
                patch_size,
                nnunet_trainer.label_manager,
                oversample_foreground_percent=nnunet_trainer.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )
        else:
            dl_proxy = nnUNetDataLoader3D(
                proxy_dataset,
                proxy_batch_size,
                patch_size,
                patch_size,
                nnunet_trainer.label_manager,
                oversample_foreground_percent=nnunet_trainer.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )

        proxy_dataloader = SingleThreadedAugmenter(dl_proxy, None)

        # Compute total weight for selected samples
        total_value = sum(meta_margin[k] for k in top_k_keys if k in meta_margin)

        logging.info(
            f"FedSelect: Updated proxy validation dataset for client {client_id}: "
            f"selected {len(top_k_identifiers)}/{num_samples} samples, total weight: {total_value:.4f}"
        )

        # Store proxy validation dataloader and scores for later use
        self.clients_proxy_validation_dataloaders[client_id] = proxy_dataloader

        return proxy_dataloader, total_value

    def train_meta_model(self, client_id: int, fl_round: int = 0):
        """
        Train the VNet meta model using the proxy validation dataset.

        This performs meta-learning to train the VNet to assign importance weights to samples.
        For each proxy validation batch:
        1. Create a copy of the local model
        2. For ONE training batch: compute weighted loss, perform pseudo gradient step
        3. Evaluate on proxy validation batch
        4. Backprop validation loss to update VNet

        Args:
            client_id: ID of the most influential client for meta model training
            fl_round: Current FL round
        """
        # Get proxy validation dataloader
        proxy_dataloader = self.clients_proxy_validation_dataloaders[client_id]
        if proxy_dataloader is None:
            logging.warning(
                f"FedSelect: No proxy validation dataloader for client {client_id}, "
                "cannot train meta model"
            )
            return

        # Avoid functorch donated buffer issues with create_graph=True
        try:
            import torch._functorch.config as functorch_config  # type: ignore

            functorch_config.donated_buffer = False
        except Exception:
            pass

        # Disable torch.compile/torchdynamo for meta-training to avoid fake tensor/double backward errors
        try:
            import torch._dynamo as dynamo  # type: ignore

            dynamo.config.suppress_errors = True
            dynamo.disable = dynamo.disable  # keep lint happy
        except Exception:
            dynamo = None

        # Get trainer and device
        nnunet_trainer = self.clients[client_id].model.nnunet_trainer
        device = nnunet_trainer.device
        empty_cache(device)
        train_loader = nnunet_trainer.dataloader_train
        base_net = nnunet_trainer.network
        net = base_net
        if hasattr(net, "_orig_mod"):
            net = net._orig_mod
        # move base model to CPU during meta-training to reduce peak GPU memory
        net_cpu = net.to("cpu")
        empty_cache(device)
        criterion = nnunet_trainer.loss

        # Get or create meta model optimizer
        vnet = self.client_meta_models[client_id].to(device)
        momentum_pseudo_optimizer = 0.5
        current_meta_lr = self.meta_lr * (
            (0.1 ** int(fl_round >= 80)) * (0.1 ** int(fl_round >= 100))
        )
        meta_optimizer = self.client_meta_optimizers.get(client_id)
        if meta_optimizer is None:
            meta_optimizer = torch.optim.Adam(
                vnet.parameters(), lr=current_meta_lr, weight_decay=1e-4
            )
            self.client_meta_optimizers[client_id] = meta_optimizer
        else:
            for param_group in meta_optimizer.param_groups:
                param_group["lr"] = current_meta_lr

        # Ensure optimizer state tensors are on the same device as vnet
        for state in meta_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.device != device:
                    state[k] = v.to(device)

        vnet.train()
        meta_loss_total = 0.0

        logging.info(
            f"FedSelect: Training meta model for client {client_id} "
            f"(FL round {fl_round})"
        )

        num_it_max = (
            len(proxy_dataloader.data_loader.indices)
            // proxy_dataloader.data_loader.batch_size
        )
        # Iterate through proxy validation batches (outer loop)
        for proxy_batch_idx, proxy_batch in enumerate(proxy_dataloader):
            if proxy_batch_idx >= num_it_max:
                break
            # Create a copy of the local model for this meta-training iteration
            local_model_copy = copy.deepcopy(net_cpu)
            if dynamo is not None:
                try:
                    local_model_copy = dynamo.disable(local_model_copy)
                except Exception:
                    pass
            local_model_copy.to(device)
            local_model_copy.train()

            # Process ONE training batch to compute weighted loss and pseudo gradient step
            for train_batch_idx, train_batch in enumerate(train_loader):
                data = train_batch["data"].to(device, non_blocking=True)
                _target = train_batch["target"]

                if isinstance(_target, list):
                    target = [t.to(device, non_blocking=True) for t in _target]
                else:
                    target = _target.to(device, non_blocking=True)

                # Forward pass through copied model
                with (
                    torch.autocast(device.type, enabled=True)
                    if device.type == "cuda"
                    else dummy_context()
                ):
                    output = local_model_copy(data)
                    cost = criterion(output, target)
                    cost_v = cost.view(-1, 1)

                    # Compute importance weights using VNet
                    v_lambda = vnet(cost_v)

                    # Compute weighted loss
                    l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)

                    # Compute gradients with create_graph=True to allow backprop through this step
                    grads = torch.autograd.grad(
                        l_f_meta,
                        local_model_copy.parameters(),
                        create_graph=True,
                        # create_graph=False,  # set to False to avoid functorch donated buffer issues, but means we can't backprop through the pseudo step
                        allow_unused=True,
                    )

                # Perform meta-SGD step (inner loop optimization)
                pseudo_optimizer = MetaSGD(
                    local_model_copy,
                    local_model_copy.parameters(),
                    lr=current_meta_lr,
                    momentum=momentum_pseudo_optimizer,
                )
                pseudo_optimizer.meta_step(grads)
                del output, cost, cost_v, v_lambda, l_f_meta, grads
                del data, target
                empty_cache(device)
                break

            # Free memory
            empty_cache(device)

            # Evaluate on proxy validation batch
            # get data and target from proxy batch
            local_model_copy.eval()
            data_val = proxy_batch["data"].to(device, non_blocking=True)
            _target_val = proxy_batch["target"]
            if isinstance(_target_val, list):
                target_val = [t.to(device, non_blocking=True) for t in _target_val]
            else:
                target_val = _target_val.to(device, non_blocking=True)

            # feed through model and compute validation loss (meta-loss)
            with (
                torch.autocast(device.type, enabled=True)
                if device.type == "cuda"
                else dummy_context()
            ):
                y_g_hat = local_model_copy(data_val)
                print(
                    f"FedSelect: Meta-training iteration {proxy_batch_idx}, y_g_hat computed."
                )
                l_g_meta = criterion(y_g_hat, target_val)
                print(
                    f"FedSelect: Meta-training iteration {proxy_batch_idx}, meta-loss computed: {l_g_meta.item():.4f}"
                )

            # Backprop validation loss to update VNet
            meta_optimizer.zero_grad()
            l_g_meta.backward()
            meta_optimizer.step()
            del y_g_hat, data_val, target_val
            empty_cache(device)

            meta_loss_total += l_g_meta.item()
            del local_model_copy, l_g_meta
            empty_cache(device)

        vnet.eval()

        # # restore base model back to GPU for subsequent training steps
        # base_net.to(device)

        avg_meta_loss = meta_loss_total / max(
            len(proxy_dataloader.data_loader.indices), 1
        )
        logging.info(
            f"FedSelect: Meta model training completed for client {client_id}, "
            f"average meta loss: {avg_meta_loss:.4f}"
        )

        # finally set updated vnet to all clients' meta model
        self._set_meta_model_n_opti_to_all_clients(vnet, meta_optimizer)

    def _set_meta_model_n_opti_to_all_clients(
        self, meta_model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ):
        """
        Set the given meta model and optimizer to all clients.

        Args:
            meta_model: The VNet meta model to set
            optimizer: The optimizer associated with the meta model
        """
        for cid in self.client_meta_models.keys():
            self.client_meta_models[cid] = copy.deepcopy(meta_model)
            self.client_meta_optimizers[cid] = copy.deepcopy(optimizer)
        logging.info(f"FedSelect: Updated meta model and optimizer set for all clients")

    def save_state(self, exp_id: str, client_id: int = None):
        """
        Save the current state of FedSelect method to experiment's cli args file.
        Following the same pattern as IOP-FL: only save checkpoint for the specific client_id.
        """

        # Save meta model and optimizer checkpoints to experiment directory
        # get exp directory
        results_dir = os.getenv("nnUNet_results") or os.getcwd()
        dataset_id = exp_id.split("_")[0].strip("D")
        exp_dir = glob(
            os.path.join(results_dir, f"Dataset{dataset_id}_*", "*", "*", exp_id)
        )[0]
        assert (
            exp_dir is not None
        ), f"Could not find experiment directory for exp_id {exp_id}!"

        # Only save checkpoints for the specific client_id (like IOP-FL does)
        for cid, meta_model in self.client_meta_models.items():
            if meta_model is not None and cid == client_id:
                model_path = os.path.join(
                    exp_dir, f"fedselect_meta_model_client_{client_id}_checkpoint.pth"
                )
                torch.save(meta_model.state_dict(), model_path)
                self.fl_strategy_state["meta_model_paths"][cid] = model_path

            optimizer = self.client_meta_optimizers.get(cid)
            if optimizer is not None and cid == client_id:
                optimizer_path = os.path.join(
                    exp_dir,
                    f"fedselect_meta_optimizer_client_{client_id}_checkpoint.pth",
                )
                torch.save(optimizer.state_dict(), optimizer_path)
                self.fl_strategy_state["meta_optimizer_paths"][cid] = optimizer_path

        # set experiment id
        self.experiment_id = exp_id

        # update fl_strategy_state with latest per-client state
        self.fl_strategy_state.update(
            {
                "selected_clients": self.selected_clients,
                "client_weights": self.client_weights,
                "meta_margin_pre": self.meta_margin_pre,
                "sample_total_loss_pre": self.sample_total_loss_pre,
                "clients_sample_weights": self.clients_sample_weights,
            }
        )

        args_file = self.save_fl_strategy_state_to_file(self.fl_strategy_state, exp_id)

        logging.info(f"Saved FedSelect state to {args_file}")
