import logging
import time
import copy

import torch

from client import Client
from methods.fedavg.fedavg import FedAvg
from methods.feda3i.feda3i import FedA3I
from methods.feddm.feddm import FedDM
from methods.iopfl.iopfl import IOPFL
from methods.fedcorr.fedcorr import FedCorr
from methods.fedselect.fedselect import FedSelect


class Orchestrator:
    def __init__(self, clients: list, fl_args: dict = {}):
        self.clients = clients
        self.num_rounds = fl_args["num_rounds"]
        self.server_model_weights = None
        # start FL round set for continuing experiments
        self.start_fl_round = fl_args.get("start_fl_round", 0)
        logging.info(
            f"Orchestrator initialized for {self.num_rounds} FL rounds, starting from FL round {self.start_fl_round}!"
        )

        # set FL strategy
        if fl_args["strategy"].lower() == "fedavg":
            self.fl_strategy = FedAvg(self.clients)
        elif fl_args["strategy"].lower() == "feda3i":
            self.fl_strategy = FedA3I(
                self.clients,
                int(fl_args["feda3i_warmup_rounds_frac"] * self.num_rounds),
                fl_args["feda3i_interw"],
                fl_strategy_state=fl_args.get("fl_strategy_state", None),
            )
        elif fl_args["strategy"].lower() == "feddm":
            self.fl_strategy = FedDM(
                self.clients,
                fl_args["feddm_gamma_hgd_smoothing"],
                fl_args["feddm_ratio_cac_pixelselection"],
                fl_args["feddm_cac_label_correction"],
                fl_args["feddm_loss"],
            )
        elif fl_args["strategy"].lower() == "iopfl":
            self.fl_strategy = IOPFL(
                self.clients,
                fl_args["iopfl_alpha"],
                fl_strategy_state=fl_args.get("fl_strategy_state", None),
            )
        elif fl_args["strategy"].lower() == "fedcorr":
            self.fl_strategy = FedCorr(
                self.clients,
                self.num_rounds,
                fl_args["fedcorr_preproc_rounds_frac"],
                fl_args["fedcorr_relabel_ratio"],
                fl_args["fedcorr_relabel_confidence_thres"],
                fl_args["fedcorr_proxterm_beta"],
                fl_strategy_state=fl_args.get("fl_strategy_state", None),
            )
        elif fl_args["strategy"].lower() == "fedselect":
            self.fl_strategy = FedSelect(
                self.clients,
                self.num_rounds,
                fl_args["fedselect_warmup_rounds_frac"],
                fl_args["fedselect_client_select_ratio"],
                fl_args["fedselect_sample_select_ratio"],
                fl_args["fedselect_meta_momentum"],
                fl_args["fedselect_reward_data_size_frac"],
                fl_strategy_state=fl_args.get("fl_strategy_state", None),
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
        for i, fl_round in enumerate(range(self.start_fl_round, self.num_rounds)):
            logging.info(f"Start FL round {i}!")

            # distribute current orchestrator model to clients
            self.update_clients(fl_round)

            orchestrator_end_time = time.time()
            logging.info(
                f"Orchestrator processing time in FL round {fl_round}: {orchestrator_end_time - orchestrator_start_time:.2f} seconds!"
            )

            # FedSelect: do client selection before local training
            if self.fl_strategy.name == "fedselect":
                self.fl_strategy.select_clients(fl_round)

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

            # IOP-FL
            elif self.fl_strategy.name == "iopfl":
                logging.info(
                    "Aggregating model weights with FedAvg strategy for IOP-FL!"
                )
                # compute server_model_weights via FedAvg
                self.aggregate(strategy="fedavg")

            # FedCorr
            elif self.fl_strategy.name == "fedcorr":
                # Aggregation
                # do normal FedAvg aggregation in FedCorr's pre-processing and full-training rounds
                if (fl_round < self.fl_strategy.fedcorr_preproc_rounds) or (
                    fl_round
                    >= self.fl_strategy.fedcorr_preproc_rounds
                    + self.fl_strategy.fedcorr_finetune_rounds
                ):
                    # compute server_model_weights via FedAvg
                    logging.info(
                        "Aggregating model weights with FedAvg strategy for FedCorr!"
                    )
                    self.aggregate(strategy="fedavg")
                # do FedCorr's adapted FedAvg aggregation in fine-tuning rounds
                else:
                    logging.info(
                        "Aggregating model weights with FedCorr's adapted FedAvg strategy for fine-tuning stage!"
                    )
                    self.aggregate(strategy="fedcorr")

                # Further central steps for FedCorr

                # FedCorr's pre-processing round: do noisy client identification
                if fl_round < self.fl_strategy.fedcorr_preproc_rounds:
                    # identify noisy clients and noisy samples
                    self.fl_strategy.central_noisy_client_identification(fl_round)

                # FedCorr's pre-processing or fine-tuning: do noise level estimation
                if (fl_round < self.fl_strategy.fedcorr_preproc_rounds) or (
                    fl_round
                    < self.fl_strategy.fedcorr_preproc_rounds
                    + self.fl_strategy.fedcorr_finetune_rounds
                ):
                    self.fl_strategy.central_noise_level_estimation(fl_round)

                # FedCorr's last fine-tuning round: set global model weights for full-training stage
                if fl_round == (
                    self.fl_strategy.fedcorr_preproc_rounds
                    + self.fl_strategy.fedcorr_finetune_rounds
                    - 1
                ):
                    logging.info(
                        "FedCorr fine-tuning stage finished: Set global model weights for full-training stage!"
                    )
                    self.fl_strategy.global_fl_model_weights = copy.deepcopy(
                        self.server_model_weights
                    )

                    # also save it to disk to have it availble for potential restart
                    self.fl_strategy.save_global_model_weights()

            # FedSelect
            elif self.fl_strategy.name == "fedselect":
                # FedSelect's aggregation
                logging.info("Aggregating model weights with FedSelect strategy!")
                self.aggregate(strategy=self.fl_strategy.name, fl_round=fl_round)

                # train meta model of most influential selected clients
                most_influential_client_id = (
                    self.fl_strategy.get_most_influential_client()
                )
                torch.cuda.empty_cache()
                self.fl_strategy.train_meta_model(most_influential_client_id, fl_round)
            else:
                raise NotImplementedError(
                    f"Federated learning strategy {self.fl_strategy.name} not implemented!"
                )

        # distiribute flinal fl models to clients
        self.update_clients(checkpoint_name="server_checkpoint_final.pth")
        torch.cuda.empty_cache()

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

    def aggregate(self, strategy: str = None, fl_round: int = 0):
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
        elif strategy == "fedcorr":
            if len(self.fl_strategy.clean_clients) > 0:
                clean_client_checkpoints = {
                    c_id: client_checkpoints[c_id]
                    for c_id in self.fl_strategy.clean_clients
                }
                logging.info(
                    f"FedCorr fine-tuning stage: aggregating model weights from {len(self.fl_strategy.clean_clients)} clean clients {self.fl_strategy.clean_clients}!"
                )
                self.server_model_weights = self.fl_strategy.fed_avg(
                    clean_client_checkpoints
                )
        elif strategy == "fedselect":
            self.server_model_weights = self.fl_strategy.fedselect_aggregate(
                client_checkpoints, fl_round
            )
        else:
            raise NotImplementedError(
                f"Federated learning strategy {strategy} not implemented!"
            )

    def update_clients(self, fl_round: int = None, checkpoint_name: str = None):
        """
        Update clients with current server model weights.
        And if self.fl_strategy is FedCorr, also global_fl_model_weights is FedCorr.
        """
        # update clients with current server model weights
        if not checkpoint_name:
            for client in self.clients:
                client.update_model(self.server_model_weights)
        elif checkpoint_name == "server_checkpoint_final.pth":
            if self.fl_strategy.name == "iopfl":
                for client in self.clients:
                    client.update_model(
                        self.fl_strategy.trajectory[client.client_id], checkpoint_name
                    )
            else:
                for client in self.clients:
                    client.update_model(self.server_model_weights, checkpoint_name)

        # if FedCorr and FedCorr's pre-processing stage, also update global_fl_model_weights
        if self.fl_strategy.name == "fedcorr":
            if (
                fl_round is not None
                and fl_round < self.fl_strategy.fedcorr_preproc_rounds
            ):
                self.fl_strategy.global_fl_model_weights = copy.deepcopy(
                    self.server_model_weights
                )
