import logging
import time

from client import Client


class Orchestrator:
    def __init__(self, clients: list, fl_args: dict = {}):
        self.clients = clients
        self.num_rounds = fl_args["num_rounds"]
        self.server_model_weights = None

    def fl_run(self):
        orchestrator_start_time = time.time()
        # aggregate initial model weights of clients
        self.aggregate(checkpoint_name="checkpoint_initial.pth")

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
                client.fed_round()

            orchestrator_start_time = time.time()

            # aggregation
            self.aggregate(checkpoint_name="checkpoint_final.pth")

        return self.server_model_weights

    def aggregate(self, checkpoint_name: str = None):
        """
        Aggregate model weights from clients.
        Loads checkpoint files of clients names according to checkpoint_name.
        Input:
            checkpoint_name (str): Name of checkpoint file to load from clients.
        Output:
            server_model_weights (dict): Aggregated model weights.
        """
        # load model weights from clients
        client_checkpoints = {
            client.client_id: client.load_checkpoint(checkpoint_name)
            for client in self.clients
        }

        # aggregate model weights with aggreation strategy
        self.server_model_weights = self.fed_avg(client_checkpoints)

    def update_clients(self):
        """
        Update clients with current server model weights.
        """
        # # compose complete checkpoint from source checkpoint with aggregated server model weights
        # if fl_round == 0:
        #     checkpoint_name = "checkpoint_initial.pth"
        # else:
        #     checkpoint_name = "checkpoint_final.pth"

        for client in self.clients:
            client.update_model(self.server_model_weights)

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
        # for sample-weighted averaging, get number of all samples of all clients
        num_samples_all_clients = sum(
            [client.dataset_json["numTraining"] for client in self.clients]
        )

        # initialize _server_model_weights with model weights of first client
        _server_model_weights = client_checkpoints[0]["network_weights"]
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
            for client_id, client_model_weights in client_checkpoints.items():
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
                        client_model_weights["network_weights"][address_key_dict[a][0]]
                        * self.clients[client_id].dataset_json["numTraining"]
                    )
            # divided by num_all_samples
            _server_model_weights[address_key_dict[a][0]] /= num_samples_all_clients
            # modifying the 0th keys is sufficient as the other keys point to the same data

        return _server_model_weights
