import logging
import time
import copy

from client import Client
from methods.fedavg.fedavg import FedAvg
from methods.feda3i.feda3i import FedA3I


class Orchestrator:
    def __init__(self, clients: list, fl_args: dict = {}):
        self.clients = clients
        self.num_rounds = fl_args["num_rounds"]
        self.server_model_weights = None

        # set FL strategy
        if fl_args["strategy"].lower() == "fedavg":
            self.fl_strategy = FedAvg(self.clients)
        elif fl_args["strategy"].lower() == "feda3i":
            self.fl_strategy = FedA3I()
        else:
            raise NotImplementedError(
                f"Federated learning strategy {fl_args['strategy']} not implemented!"
            )

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
        self.server_model_weights = self.fl_strategy.fed_avg(client_checkpoints)

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
