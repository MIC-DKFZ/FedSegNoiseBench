"""
FedSelect: Proxy-Validated Importance-Aware Federated Sample Selection with Meta Learning
Reference: Zhang et al. 2025 (KDD 2025)
Paper: https://dl.acm.org/doi/epdf/10.1145/3711896.3737093
"""
import copy
import torch
import torch.nn.functional as F
import logging


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


class FedSelect:
    """
    FedSelect: Proxy-Validated Importance-Aware Federated Sample Selection with Meta Learning

    This class implements the FedSelect FL strategy which:
    1. Uses meta-learning with a proxy validation dataset to select important samples
    2. Implements momentum-based meta-margin function for discovering influential samples
    3. Selects high-quality clients and samples for training
    """

    def __init__(
        self,
        clients: list = None,
        num_rounds: int = 100,
        warmup_rounds_frac: float = 0.1,
        client_select_ratio: float = 0.4,
        sample_select_ratio: float = 0.6,
        meta_momentum: float = 0.9,
        meta_lr: float = 1e-3,
        reward_data_size: int = 1000,
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
            meta_lr: Learning rate for meta model (default: 1e-3)
            reward_data_size: Size of reward/proxy validation dataset (default: 1000)
        """
        self.name = "fedselect"
        self.clients = clients
        self.num_rounds = num_rounds
        self.warmup_rounds = int(warmup_rounds_frac * num_rounds)
        self.client_select_ratio = client_select_ratio
        self.sample_select_ratio = sample_select_ratio
        self.meta_momentum = meta_momentum
        self.meta_lr = meta_lr
        self.reward_data_size = reward_data_size

        # Initialize per-client tracking
        self.meta_margin_pre = {}  # Previous meta-margin per client
        self.sample_total_loss_pre = {}  # Previous sample losses per client
        self.client_weight = {}  # Client importance weights
        self.client_meta_models = {}  # Per-client VNet models

        # Initialize VNet models for each client
        for client in self.clients:
            self.client_meta_models[client.client_id] = VNet()
            self.meta_margin_pre[client.client_id] = None
            self.sample_total_loss_pre[client.client_id] = None
            self.client_weight[client.client_id] = 1.0

        logging.info(
            f"Initialized FedSelect with warmup_rounds={self.warmup_rounds}, "
            f"client_select_ratio={self.client_select_ratio}, "
            f"sample_select_ratio={self.sample_select_ratio}"
        )

    def get_num_datasamples_client(self, client_id: int = None):
        """
        Get number of training samples of a client with client_id or of all clients.
        """
        if client_id is None:
            all_samples = sum(
                [client.dataset_json["numTraining"] for client in self.clients]
            )
            return all_samples
        else:
            client = next((c for c in self.clients if c.client_id == client_id), None)
            if client:
                return client.dataset_json["numTraining"]
            return 0

    def fed_avg(self, client_checkpoints: dict = {}):
        """
        FedAvg aggregation: weighted by number of samples per client.
        """
        _client_checkpoints = copy.deepcopy(client_checkpoints)

        # Get number of samples for each client
        num_samples_list = [
            self.get_num_datasamples_client(client_id)
            for client_id in _client_checkpoints.keys()
        ]
        num_samples_all = sum(num_samples_list)

        if num_samples_all == 0:
            logging.warning("No samples found in any client!")
            return _client_checkpoints[list(_client_checkpoints.keys())[0]]

        # Get first client's model weights as reference
        first_key = list(_client_checkpoints.keys())[0]
        _server_model_weights = _client_checkpoints[first_key]
        keys = list(_server_model_weights.keys())

        # Create address-to-key mapping for efficient aggregation
        address_key_dict = {}
        for k in keys:
            address = _server_model_weights[k].data_ptr()
            if address not in address_key_dict:
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)

        # Perform weighted averaging
        for address in address_key_dict.keys():
            for client_id, client_weights in _client_checkpoints.items():
                num_samples = self.get_num_datasamples_client(client_id)
                if client_id == first_key:
                    _server_model_weights[address_key_dict[address][0]] = (
                        _server_model_weights[address_key_dict[address][0]]
                        * num_samples
                    )
                else:
                    _server_model_weights[address_key_dict[address][0]] += (
                        client_weights[address_key_dict[address][0]] * num_samples
                    )
            _server_model_weights[address_key_dict[address][0]] /= num_samples_all

        return _server_model_weights

    def fedselect_aggregate(self, client_checkpoints: dict = {}, fl_round: int = 0):
        """
        FedSelect aggregation strategy.

        During warmup phase: Use FedAvg
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
        total_weight = sum(self.client_weight.values())
        normalized_weights = {
            cid: w / total_weight for cid, w in self.client_weight.items()
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

    def compute_client_importance(
        self, client_id: int, sample_losses_pre: torch.Tensor, sample_losses_cur: torch.Tensor
    ):
        """
        Compute client importance weight using meta-margin (difference in loss).
        meta_margin = loss_before - loss_after

        Args:
            client_id: Client identifier
            sample_losses_pre: Sample losses from previous epoch
            sample_losses_cur: Sample losses from current epoch

        Returns:
            client_weight: Importance weight for the client
        """
        if sample_losses_pre is None or sample_losses_cur is None:
            return 1.0

        # Compute meta-margin: positive margin indicates improvement
        meta_margin = (sample_losses_pre - sample_losses_cur).clamp(min=0.0)

        # Apply momentum to meta-margin
        if self.meta_margin_pre.get(client_id) is not None:
            meta_margin_momentum = (
                self.meta_momentum * self.meta_margin_pre[client_id]
                + (1 - self.meta_momentum) * meta_margin
            )
        else:
            meta_margin_momentum = meta_margin

        # Normalize meta-margin
        if meta_margin_momentum.max() > 0:
            meta_margin_momentum = meta_margin_momentum / (
                meta_margin_momentum.max() + 1e-8
            )

        self.meta_margin_pre[client_id] = meta_margin_momentum

        # Client weight is the sum of normalized meta-margins
        client_weight = meta_margin_momentum.sum().item() / max(
            len(meta_margin_momentum), 1
        )
        self.client_weight[client_id] = max(client_weight, 0.1)  # Minimum weight 0.1

        logging.info(
            f"FedSelect: Client {client_id} importance weight = {self.client_weight[client_id]:.4f}"
        )
        return self.client_weight[client_id]

    def compute_sample_importance(
        self, client_id: int, sample_losses: torch.Tensor, device: torch.device = None
    ):
        """
        Compute sample importance weights using the VNet meta model.

        Args:
            client_id: Client identifier
            sample_losses: Tensor of sample losses
            device: Device for computation

        Returns:
            sample_weights: Importance weights for each sample
        """
        if device is None:
            device = torch.device("cpu")

        vnet = self.client_meta_models[client_id].to(device)
        vnet.eval()

        with torch.no_grad():
            # Reshape losses and compute importance
            sample_losses_reshaped = sample_losses.view(-1, 1).to(device)
            sample_weights = vnet(sample_losses_reshaped)
            sample_weights = sample_weights.view(-1)

        return sample_weights

    def select_clients(self, fl_round: int, num_clients_to_select: int = None):
        """
        Select clients based on their importance weights.

        Args:
            fl_round: Current FL round
            num_clients_to_select: Number of clients to select (uses default ratio if None)

        Returns:
            selected_client_ids: List of selected client IDs
        """
        if fl_round < self.warmup_rounds:
            # During warmup, select clients randomly
            import random
            num_to_select = num_clients_to_select or int(
                len(self.clients) * self.client_select_ratio
            )
            selected = random.sample(
                [c.client_id for c in self.clients],
                min(num_to_select, len(self.clients)),
            )
            logging.info(f"FedSelect warmup: randomly selected {len(selected)} clients")
            return selected
        else:
            # During training, select based on importance weights
            num_to_select = num_clients_to_select or int(
                len(self.clients) * self.client_select_ratio
            )
            # Sort clients by weight
            sorted_clients = sorted(
                self.client_weight.items(), key=lambda x: x[1], reverse=True
            )
            selected = [cid for cid, _ in sorted_clients[:num_to_select]]
            logging.info(
                f"FedSelect: selected {len(selected)} clients based on importance: {selected}"
            )
            return selected

    def update_vnet(
        self,
        client_id: int,
        sample_losses_train: torch.Tensor,
        sample_losses_val: torch.Tensor,
        device: torch.device = None,
    ):
        """
        Update the VNet model for sample importance weighting using validation data.

        Args:
            client_id: Client identifier
            sample_losses_train: Training sample losses
            sample_losses_val: Validation sample losses (proxy for true validation)
            device: Device for computation
        """
        if device is None:
            device = torch.device("cpu")

        vnet = self.client_meta_models[client_id].to(device)
        optimizer = torch.optim.Adam(vnet.parameters(), lr=self.meta_lr, weight_decay=1e-4)

        vnet.train()
        for _ in range(5):  # Few meta-update steps
            # Forward pass on training data
            sample_losses_train = sample_losses_train.view(-1, 1).to(device)
            sample_weights = vnet(sample_losses_train)

            # Compute weighted loss
            weighted_loss = (sample_weights.view(-1) * sample_losses_train.view(-1)).mean()

            # Backward on validation data (meta-loss)
            sample_losses_val = sample_losses_val.view(-1, 1).to(device)
            sample_weights_val = vnet(sample_losses_val)
            meta_loss = F.mse_loss(sample_weights_val, torch.ones_like(sample_weights_val) * 0.5)

            # Combined loss
            loss = weighted_loss + meta_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        vnet.eval()
