import logging
import time
import copy

from client import Client
from methods.fedavg.fedavg import FedAvg
from methods.feda3i.feda3i import FedA3I
from methods.feddm.feddm import FedDM


class Orchestrator:
    def __init__(self, clients: list, fl_args: dict = {}):
        self.clients = clients
        self.num_rounds = fl_args["num_rounds"]
        self.server_model_weights = None

        # set FL strategy
        if fl_args["strategy"].lower() == "fedavg":
            self.fl_strategy = FedAvg(self.clients)
        elif fl_args["strategy"].lower() == "feda3i":
            self.fl_strategy = FedA3I(
                self.clients,
                int(fl_args["feda3i_warmup_rounds_frac"] * self.num_rounds),
                fl_args["feda3i_interw"],
            )
        elif fl_args["strategy"].lower() == "feddm":
            self.fl_strategy = FedDM(
                self.clients,
                fl_args["feddm_gamma_hgd_smoothing"],
                fl_args["feddm_ratio_cac_pixelselection"],
                fl_args["feddm_cac_label_correction"],
            )
        else:
            raise NotImplementedError(
                f"Federated learning strategy {fl_args['strategy']} not implemented!"
            )

    def fl_run(self):
        orchestrator_start_time = time.time()
        # aggregate initial model weights of clients
        self.aggregate(strategy="fedavg")

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
                client.fed_round(fl_round, fl_strategy=self.fl_strategy)

            orchestrator_start_time = time.time()

            # aggregation with selected FL strategy
            # FEDAVG
            if self.fl_strategy.name == "fedavg":
                logging.info("Aggregating model weights with FedAvg strategy!")
                # compute server_model_weights via FedAvg
                self.aggregate(strategy=self.fl_strategy.name)

            # FEDA3I
            elif self.fl_strategy.name == "feda3i":
                # warm up phase
                if fl_round <= self.fl_strategy.feda3i_warmup_rounds:
                    logging.info(
                        "FedA3I warum up stage; aggregating model weights with FedAvg strategy!"
                    )
                    # compute server_model_weights via FedAvg
                    self.aggregate(strategy="fedavg")
                    # last fl_round of warmup phase
                    if fl_round == self.fl_strategy.feda3i_warmup_rounds:
                        logging.info(
                            f"FedA3I warmup phase finished; starting to compute quality-based aggregation weights!"
                        )
                        # compute quality aggregation weights
                        self.fl_strategy.feda3i_compute_quality_agg_weights()
                # training phase
                else:
                    logging.info("Aggregating model weights with FedA3I strategy!")
                    # compute server_model_weights via FedA3I
                    self.aggregate(strategy=self.fl_strategy.name)

            # FEDDM
            elif self.fl_strategy.name == "feddm":
                logging.info(
                    "Central steps of FedDM strategy: "
                    "Collaborative Annotation Calibration and Hierarchical Gradient De-Conflicting!"
                )
                # compute server_model_weights via FedDM
                self.aggregate(strategy=self.fl_strategy.name)
            else:
                raise NotImplementedError(
                    f"Federated learning strategy {self.fl_strategy.name} not implemented!"
                )

        # distiribute flinal fl models to clients
        self.update_clients(checkpoint_name="server_checkpoint_final.pth")

        # very last fl round to just predict
        for client in self.clients:
            # empty client.model.current_model_weights to None such that nnUNet's run_training loads model weights from checkpoint (via def maybe_load_checkpoint())
            client.update_model(server_model_weights=None)
            client.fed_round(
                very_last_fl_predict_round=True,
                only_run_validation=True,
                fl_round=self.num_rounds,
                fl_strategy=self.fl_strategy,
            )

        return self.server_model_weights

    def aggregate(self, strategy: str = None):
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
        if strategy == "fedavg":
            self.server_model_weights = self.fl_strategy.fed_avg(client_checkpoints)
        elif strategy == "feda3i":
            self.server_model_weights = self.fl_strategy.feda3i_aggregate(
                client_checkpoints
            )
        elif strategy == "feddm":
            self.server_model_weights = self.fl_strategy.feddm_central_steps(
                client_checkpoints, self.server_model_weights
            )
        else:
            raise NotImplementedError(
                f"Federated learning strategy {strategy} not implemented!"
            )

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
