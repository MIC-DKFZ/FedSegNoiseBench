"""
FedSelect: Proxy-Validated Importance-Aware Federated Sample Selection with Meta Learning
Reference: Zhang et al. 2025 (KDD 2025)
Paper: https://dl.acm.org/doi/epdf/10.1145/3711896.3737093
"""

import copy
import torch
import torch.nn.functional as F
import logging
import random
import heapq
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset
from torch.optim.sgd import SGD

from nnunetv2.utilities.helpers import dummy_context

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

    def __init__(self, base_dataset, selected_indices: list):
        """
        Args:
            base_dataset: The original dataset to wrap
            selected_indices: List of indices to include in this dataset
        """
        self.base_dataset = base_dataset
        self.selected_indices = selected_indices
        # Map from new index to original index
        self.index_map = {i: idx for i, idx in enumerate(selected_indices)}

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        original_idx = self.index_map[idx]
        return self.base_dataset[original_idx]


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
        """
        # Call parent FedAvg constructor
        super().__init__(clients=clients)

        self.name = "fedselect"

        self.warmup_rounds = int(warmup_rounds_frac * num_rounds)
        self.client_select_ratio = client_select_ratio
        self.sample_select_ratio = sample_select_ratio
        self.meta_momentum = meta_momentum
        self.meta_lr = 1e-3  # default learning rate for VNet meta model
        self.reward_data_size_frac = reward_data_size_frac

        # Initialize per-client tracking
        self.selected_clients = []  # Currently selected clients
        
        # Initialize dictionaries for all clients using consistent indexing
        client_ids = range(len(self.clients))
        self.meta_margin_pre = {c_id: None for c_id in client_ids}
        self.sample_total_loss_pre = {c_id: None for c_id in client_ids}
        self.client_meta_models = {c_id: VNet() for c_id in client_ids}
        self.clients_sample_weights = {c_id: None for c_id in client_ids}
        self.client_weights = {c_id: 1.0 for c_id in client_ids}
        self.clients_proxy_validation_dataloaders = {c_id: None for c_id in client_ids}

        logging.info(
            f"Initialized FedSelect with warmup_rounds={self.warmup_rounds}, "
            f"client_select_ratio={self.client_select_ratio}, "
            f"sample_select_ratio={self.sample_select_ratio}"
        )

    # def get_num_datasamples_client(self, client_id: int = None):
    #     """
    #     Get number of training samples of a client with client_id or of all clients.
    #     Inherited from FedAvg.
    #     """
    #     return super().get_num_datasamples_client(client_id)

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
        _client_checkpoints = copy.deepcopy(client_checkpoints)

        first_key = list(_client_checkpoints.keys())[0]
        _server_model_weights = _client_checkpoints[first_key]
        keys = list(_server_model_weights.keys())

        # Create address-to-key mapping
        address_key_dict = {}
        for k in keys:
            address = _server_model_weights[k].data_ptr()
            if address not in address_key_dict:
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)

        # Normalize client weights
        total_weight = sum(self.client_weights.values())
        normalized_weights = {
            cid: w / total_weight for cid, w in self.client_weights.items()
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
        return most_influential_client_id

    # def compute_client_importance(
    #     self,
    #     client_id: int,
    #     sample_losses_pre: torch.Tensor,
    #     sample_losses_cur: torch.Tensor,
    # ):
    #     """
    #     Compute client importance weight using meta-margin (difference in loss).
    #     meta_margin = loss_before - loss_after

    #     Args:
    #         client_id: Client identifier
    #         sample_losses_pre: Sample losses from previous epoch
    #         sample_losses_cur: Sample losses from current epoch

    #     Returns:
    #         client_weight: Importance weight for the client
    #     """
    #     if sample_losses_pre is None or sample_losses_cur is None:
    #         return 1.0

    #     # Compute meta-margin: positive margin indicates improvement
    #     meta_margin = (sample_losses_pre - sample_losses_cur).clamp(min=0.0)

    #     # Apply momentum to meta-margin
    #     if self.meta_margin_pre.get(client_id) is not None:
    #         meta_margin_momentum = (
    #             self.meta_momentum * self.meta_margin_pre[client_id]
    #             + (1 - self.meta_momentum) * meta_margin
    #         )
    #     else:
    #         meta_margin_momentum = meta_margin

    #     # Normalize meta-margin
    #     if meta_margin_momentum.max() > 0:
    #         meta_margin_momentum = meta_margin_momentum / (
    #             meta_margin_momentum.max() + 1e-8
    #         )

    #     self.meta_margin_pre[client_id] = meta_margin_momentum

    #     # Client weight is the sum of normalized meta-margins
    #     client_weight = meta_margin_momentum.sum().item() / max(
    #         len(meta_margin_momentum), 1
    #     )
    #     self.client_weight[client_id] = max(client_weight, 0.1)  # Minimum weight 0.1

    #     logging.info(
    #         f"FedSelect: Client {client_id} importance weight = {self.client_weight[client_id]:.4f}"
    #     )
    #     return self.client_weight[client_id]

    # def compute_sample_importance(
    #     self, client_id: int, sample_losses: torch.Tensor, device: torch.device = None
    # ):
    #     """
    #     Compute sample importance weights using the VNet meta model.

    #     Args:
    #         client_id: Client identifier
    #         sample_losses: Tensor of sample losses
    #         device: Device for computation

    #     Returns:
    #         sample_weights: Importance weights for each sample
    #     """
    #     if device is None:
    #         device = torch.device("cpu")

    #     vnet = self.client_meta_models[client_id].to(device)
    #     vnet.eval()

    #     with torch.no_grad():
    #         # Reshape losses and compute importance
    #         sample_losses_reshaped = sample_losses.view(-1, 1).to(device)
    #         sample_weights = vnet(sample_losses_reshaped)
    #         sample_weights = sample_weights.view(-1)

    #     return sample_weights

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
            logging.info(f"FedSelect warmup: randomly selected {len(selected)} clients")
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
            self.clients_sample_weights[client_id] = torch.ones(
                self.get_num_datasamples_client(client_id), device=torch.device("cpu")
            )
            return
        # after warmup, compute sample weights using VNet
        logging.info(
            f"FedSelect training phase (round {fl_round}): computing sample weights for client {client_id}"
        )
        # get net, loader, device, criterion from nnunet_trainer
        net = nnunet_trainer.network
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
        processed_batch_el_keys = {}
        sample_weights = torch.ones(0, device=device)
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
                    if key in processed_batch_el_keys.keys():
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
                        l = criterion(output, target).cpu()
                        processed_batch_el_keys[key] = l

                    # feed loss through meta model to get sample weight
                    weight = meta_model(torch.reshape(l, (len(l), -1)))
                    weight = torch.reshape(weight, l.shape)
                    sample_weights = torch.cat((sample_weights, weight.to(device)), -1)

        self.clients_sample_weights[client_id] = sample_weights
        logging.info(
            f"FedSelect: computed sample weights for client {client_id}, total samples: {len(sample_weights)}"
        )

    def compute_client_weights(self, client_id: int):
        """
        Compute client importance weight based on sample weights.

        Args:
            client_id: Client identifier
        """
        self.client_weights[client_id] = self.clients_sample_weights[
            client_id
        ].sum().item() / self.get_num_datasamples_client(client_id)
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
        net = nnunet_trainer.network
        net.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader = nnunet_trainer.dataloader_train
        device = nnunet_trainer.device
        criterion = nnunet_trainer.loss

        # forward pass through training data to compute sample losses
        processed_batch_el_keys = {}
        sample_total_loss = torch.ones(0)
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
                    if key in processed_batch_el_keys.keys():
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
                        l = criterion(output, target).cpu()
                        sample_total_loss = torch.cat(
                            (sample_total_loss, l.to(device)), -1
                        )
                        processed_batch_el_keys[key] = l

        # compute meta margin scores
        # equation (7) in the paper
        meta_margin = self.sample_total_loss_pre[client_id].to(
            device
        ) - sample_total_loss.to(device)
        meta_margin = self._normalize_tensor(meta_margin)
        # equation (8) in the paper
        meta_margin_hat = (
            self.meta_momentum * self.meta_margin_pre[client_id].to(device)
            + (1 - self.meta_momentum) * meta_margin
        )
        meta_margin_hat = self._normalize_tensor(meta_margin_hat)

        self.sample_total_loss_pre[client_id] = sample_total_loss
        self.meta_margin_pre[client_id] = meta_margin_hat
        # TODO: in original code, they do it like the following, but doesn't align with paper
        # self.meta_margin_pre[client_id] = meta_margin

    def _normalize_tensor(self, tensor: torch.Tensor):
        """
        Normalize tensor to range [0, 1].
        """
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def update_proxy_validation_dataset(self, client_id: int, nnunet_trainer=None):
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
        # Get meta-margin scores for this client, and ensure np.array and device CPU
        if self.meta_margin_pre.get(client_id) is None:
            logging.warning(
                f"FedSelect: No meta-margin scores available for client {client_id}, "
                "cannot update proxy validation dataset"
            )
            return None, 0.0
        meta_margin = self.meta_margin_pre[client_id]
        if isinstance(meta_margin, torch.Tensor):
            weights = meta_margin.cpu().numpy()
        else:
            weights = meta_margin

        # Determine number of samples to select for proxy validation
        num_samples = len(weights)
        select_num = int(self.reward_data_size_frac * num_samples)

        # Select top-k samples by meta-margin score using heapq
        top_k_indices = heapq.nlargest(
            select_num, range(num_samples), key=lambda i: weights[i]
        )

        # Get the base dataset from trainer if available
        if nnunet_trainer is not None and hasattr(nnunet_trainer, "dataloader_train"):
            base_dataset = copy.deepcopy(nnunet_trainer.dataloader_train.dataset)
        else:
            logging.warning(
                f"FedSelect: No trainer provided for client {client_id}, "
                "cannot create proxy validation dataloader"
            )
            return None, 0.0

        # Create proxy validation dataset with selected indices
        proxy_dataset = ProxyValidationDataset(base_dataset, top_k_indices)

        # Create dataloader for proxy validation dataset
        proxy_dataloader = DataLoader(
            proxy_dataset,
            batch_size=select_num,
            shuffle=False,
            num_workers=0,
        )

        # Compute total weight for selected samples
        if isinstance(meta_margin, torch.Tensor):
            total_value = meta_margin[top_k_indices].sum().item()
        else:
            total_value = weights[top_k_indices].sum()

        logging.info(
            f"FedSelect: Updated proxy validation dataset for client {client_id}: "
            f"selected {len(top_k_indices)}/{num_samples} samples, total weight: {total_value:.4f}"
        )

        # Store proxy validation dataloader and scores for later use
        self.clients_proxy_validation_dataloaders[client_id] = proxy_dataloader

        return proxy_dataloader, total_value

    # def update_vnet(
    #     self,
    #     client_id: int,
    #     sample_losses_train: torch.Tensor,
    #     sample_losses_val: torch.Tensor,
    #     device: torch.device = None,
    # ):
    #     """
    #     Update the VNet model for sample importance weighting using validation data.

    #     Args:
    #         client_id: Client identifier
    #         sample_losses_train: Training sample losses
    #         sample_losses_val: Validation sample losses (proxy for true validation)
    #         device: Device for computation
    #     """
    #     if device is None:
    #         device = torch.device("cpu")

    #     vnet = self.client_meta_models[client_id].to(device)
    #     optimizer = torch.optim.Adam(
    #         vnet.parameters(), lr=self.meta_lr, weight_decay=1e-4
    #     )

    #     vnet.train()
    #     for _ in range(5):  # Few meta-update steps
    #         # Forward pass on training data
    #         sample_losses_train = sample_losses_train.view(-1, 1).to(device)
    #         sample_weights = vnet(sample_losses_train)

    #         # Compute weighted loss
    #         weighted_loss = (
    #             sample_weights.view(-1) * sample_losses_train.view(-1)
    #         ).mean()

    #         # Backward on validation data (meta-loss)
    #         sample_losses_val = sample_losses_val.view(-1, 1).to(device)
    #         sample_weights_val = vnet(sample_losses_val)
    #         meta_loss = F.mse_loss(
    #             sample_weights_val, torch.ones_like(sample_weights_val) * 0.5
    #         )

    #         # Combined loss
    #         loss = weighted_loss + meta_loss

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     vnet.eval()

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

        # Get trainer and device
        nnunet_trainer = self.clients[client_id].trainer
        device = nnunet_trainer.device
        train_loader = nnunet_trainer.dataloader_train
        net = nnunet_trainer.network
        criterion = nnunet_trainer.loss

        # Get or create meta model optimizer
        vnet = self.client_meta_models[client_id].to(device)
        current_meta_lr = self.meta_lr * (
            (0.1 ** int(fl_round >= 80)) * (0.1 ** int(fl_round >= 100))
        )
        meta_optimizer = torch.optim.Adam(
            vnet.parameters(), lr=current_meta_lr, weight_decay=1e-4
        )

        vnet.train()
        meta_loss_total = 0.0

        logging.info(
            f"FedSelect: Training meta model for client {client_id} "
            f"(FL round {fl_round})"
        )

        # Iterate through proxy validation batches (outer loop)
        for proxy_batch_idx, proxy_batch in enumerate(proxy_dataloader):
            # Create a copy of the local model for this meta-training iteration
            local_model_copy = copy.deepcopy(net)
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
                    loss = criterion(output, target)

                    # Ensure loss is a 1D tensor for VNet input
                    if loss.dim() > 1:
                        loss = loss.view(-1)

                    # Compute importance weights using VNet (detach to prevent backprop through VNet here)
                    loss_reshaped = loss.view(-1, 1).detach()
                    v_lambda = vnet(loss_reshaped)

                    # Compute weighted loss
                    l_f_meta = torch.sum(loss * v_lambda.view(-1)) / max(len(loss), 1)

                    # Compute gradients with create_graph=True to allow backprop through this step
                    grads = torch.autograd.grad(
                        l_f_meta,
                        local_model_copy.parameters(),
                        create_graph=True,
                        allow_unused=True,
                    )

                # Perform meta-SGD step (inner loop optimization)
                pseudo_optimizer = MetaSGD(
                    local_model_copy, local_model_copy.parameters(), lr=current_meta_lr
                )
                pseudo_optimizer.meta_step(grads)
                del grads
                break

            # Evaluate on proxy validation batch
            local_model_copy.eval()
            data_val = proxy_batch["data"].to(device, non_blocking=True)
            _target_val = proxy_batch["target"]

            if isinstance(_target_val, list):
                target_val = [t.to(device, non_blocking=True) for t in _target_val]
            else:
                target_val = _target_val.to(device, non_blocking=True)

            with (
                torch.autocast(device.type, enabled=True)
                if device.type == "cuda"
                else dummy_context()
            ):
                y_g_hat = local_model_copy(data_val)
                l_g_meta = criterion(y_g_hat, target_val)

            # Backprop validation loss to update VNet
            meta_optimizer.zero_grad()
            l_g_meta.backward()
            meta_optimizer.step()

            meta_loss_total += l_g_meta.item()

        vnet.eval()

        avg_meta_loss = meta_loss_total / max(len(proxy_dataloader), 1)
        logging.info(
            f"FedSelect: Meta model training completed for client {client_id}, "
            f"average meta loss: {avg_meta_loss:.4f}"
        )

        # finally set updated vnet to all clients' meta model
        self._set_meta_model_to_all_clients(vnet)

    def _set_meta_model_to_all_clients(self, meta_model: torch.nn.Module):
        """
        Set the given meta model to all clients.

        Args:
            meta_model: The VNet meta model to set
        """
        for cid in self.client_meta_models.keys():
            self.client_meta_models[cid] = copy.deepcopy(meta_model)
