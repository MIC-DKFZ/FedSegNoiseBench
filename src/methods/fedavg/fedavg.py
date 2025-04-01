import copy

class FedAvg:
    def __init__(self, clients: list = None):
        self.clients = clients

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
        num_samples_all_clients = sum(
            [client.dataset_json["numTraining"] for client in self.clients]
        )

        # initialize _server_model_weights with model weights of first client
        _server_model_weights = _client_checkpoints[0]
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
                if client_id == "0":
                    # network weights of client_id="0" are already in _server_model_weights
                    # we still need to weight client 0's model params with it's dataset size
                    _server_model_weights[address_key_dict[a][0]] = (
                        _server_model_weights[address_key_dict[a][0]]
                        * self.clients[client_id].dataset_json["numTraining"]
                    )
                else:
                    # weighted sum
                    _server_model_weights[address_key_dict[a][0]] += (
                        client_model_weights[address_key_dict[a][0]]
                        * self.clients[client_id].dataset_json["numTraining"]
                    )
            # divided by num_all_samples
            _server_model_weights[address_key_dict[a][0]] /= num_samples_all_clients
            # modifying the 0th keys is sufficient as the other keys point to the same data

        return _server_model_weights