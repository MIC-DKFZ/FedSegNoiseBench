import logging
import time
import copy

from client import Client


class Orchestrator:
    def __init__(self, clients: list, fl_args: dict = {}):
        self.clients = clients
        self.num_rounds = fl_args["num_rounds"]
        self.server_model_weights = None

    def fl_run(self):
        orchestrator_start_time = time.time()
        # aggregate initial model weights of clients
        self.aggregate()

        # iterate over fl rounds
        for i, fl_round in enumerate(range(0, self.num_rounds)):
            logging.info(f"Start FL round {i}!")

            # distribute current orchestrator model to clients
            self.update_clients()

            orchestrator_end_time = time.time()
            logging.info(
                f"Orchestrator processing time in FL round {fl_round}: {orchestrator_end_time - orchestrator_start_time:.2f} seconds!"
            )

            # iterate over clients
            for client in self.clients:
                # local training
                client.fed_round(fl_round)

            orchestrator_start_time = time.time()

            # aggregation
            self.aggregate()

        # distiribute flinal fl models to clients
        self.update_clients(checkpoint_name="server_checkpoint_final.pth")

        # very last fl round to just predict
        for client in self.clients:
            # empty client.model.current_model_weights to None such that run_training loads model weights from checkpoint
            client.update_model(server_model_weights=None)
            client.fed_round(
                very_last_fl_predict_round=True,
                only_run_validation=True,
                fl_round=self.num_rounds,
            )

        return self.server_model_weights

    def aggregate(self):
        """
        Aggregate model weights from clients.
        Output:
            server_model_weights (dict): Aggregated model weights.
        """

        client_checkpoints = {
            client.client_id: client.model.current_model_weights
            for client in self.clients
        }

        # aggregate model weights with aggreation strategy
        self.server_model_weights = self.fed_avg(client_checkpoints)

    def update_clients(self, checkpoint_name: str = None):
        """
        Update clients with current server model weights.
        """
        if not checkpoint_name:
            for client in self.clients:
                client.update_model(self.server_model_weights)
        elif checkpoint_name == "server_checkpoint_final.pth":
            for client in self.clients:
                client.update_model(self.server_model_weights, checkpoint_name)

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
