import copy
import os
import json


class FedAvg:
    def __init__(self, clients: list = None):
        self.name = "fedavg"
        self.clients = clients

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
            return self.clients[client_id].dataset_json["numTraining"]

    def fed_avg(self, client_checkpoints: dict = {}):
        """
        FedAvg: Communication-efficient Learning of Deep networks from Decentralized Data (https://arxiv.org/abs/1602.05629)
        Sum model_weights up.
        Divide summed model_weights by number of samples per client.
        Return a checkpoint with FedAvg-aggregated model weights.

        Note:
        * perform averaging based on pointers on the dict keys obtained with .data_ptr()
        * solution from https://github.com/MIC-DKFZ/nnUNet/issues/2553
        """
        # create deepcopy of client model weights to not directly modify them
        _client_checkpoints = copy.deepcopy(client_checkpoints)

        # for sample-weighted averaging, get number of all samples of all clients
        num_samples_all_clients_list = [
            self.get_num_datasamples_client(client_id)
            for client_id in _client_checkpoints.keys()
        ]
        num_samples_all_clients = sum(num_samples_all_clients_list)

        # initialize _server_model_weights with model weights of first client
        first_key = list(_client_checkpoints.keys())[0]
        _server_model_weights = _client_checkpoints[first_key]
        # get addresses of keys
        keys = list(_server_model_weights.keys())
        address_key_dict = {}
        for k in keys:
            address = _server_model_weights[k].data_ptr()
            if address not in address_key_dict.keys():
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)

        # perform the fedavg
        for a in address_key_dict.keys():
            for client_id, client_model_weights in _client_checkpoints.items():
                if client_id == first_key:
                    # network weights of client_id="0" are already in _server_model_weights
                    # we still need to weight client 0's model params with it's dataset size
                    _server_model_weights[address_key_dict[a][0]] = (
                        _server_model_weights[address_key_dict[a][0]]
                        * self.get_num_datasamples_client(client_id)
                    )
                else:
                    # weighted sum
                    _server_model_weights[
                        address_key_dict[a][0]
                    ] += client_model_weights[
                        address_key_dict[a][0]
                    ] * self.get_num_datasamples_client(
                        client_id
                    )
            # divided by num_all_samples
            _server_model_weights[address_key_dict[a][0]] /= num_samples_all_clients
            # modifying the 0th keys is sufficient as the other keys point to the same data

        return _server_model_weights

    def save_state(self, exp_id: str = None):
        print(
            f"Saving state of {self.name} not implemented yet for FL strategy {self.name}!"
        )

    def save_fl_strategy_state_to_file(
        self, fl_strategy_state: dict = None, exp_id: str = None
    ):
        """
        Save FL strategy state to experiment's cli args file.

        Args:
            fl_strategy_state (dict): State of the FL strategy to be saved.
            exp_id (str): Experiment ID.
        Returns:
            args_file (str): Path to the updated args file.
        """

        # save fl_strategy_state to experiment's cli args file
        results_dir = os.getenv("nnUNet_results") or os.getcwd()
        # strip from exp_id "DXXX_" prefix if exists
        if exp_id.startswith("D") and "_" in exp_id:
            exp_id = "_".join(exp_id.split("_")[1:])
        args_file = os.path.join(results_dir, f"ExperimentArgs_{exp_id}.json")

        # Read existing args if file exists, otherwise create empty dict
        if os.path.exists(args_file):
            with open(args_file, "r") as f:
                args_data = json.load(f)
        else:
            args_data = {}

        # Add fl_strategy_state to args data
        args_data["fl_strategy_state"] = fl_strategy_state

        # Write updated args data back to file
        with open(args_file, "w") as f:
            json.dump(args_data, f, indent=2, default=str)

        return args_file
