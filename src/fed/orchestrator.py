import logging
import time
import copy

import torch

from methods.fedavg.fedavg import FedAvg
from methods.feda3i.feda3i import FedA3I
from methods.feddm.feddm import FedDM
from methods.iopfl.iopfl import IOPFL
from methods.fedcorr.fedcorr import FedCorr
from methods.fedselect.fedselect import FedSelect


class Orchestrator:
    def __init__(self, clients: list, fl_args: dict = None):
        fl_args = fl_args or {}
        self.clients = clients
        self.num_rounds = fl_args["num_rounds"]
        self.server_model_weights = None
        # start FL round set for continuing experiments
        self.start_fl_round = fl_args.get("start_fl_round", 0)
        logging.info(
            f"Orchestrator initialized for {self.num_rounds} FL rounds, starting from FL round {self.start_fl_round}!"
        )

        self.fl_strategy = self._build_fl_strategy(fl_args)

    def _build_fl_strategy(self, fl_args: dict):
        strategy_name = fl_args["strategy"].lower()
        fl_strategy_state = fl_args.get("fl_strategy_state", None)

        if strategy_name == "fedavg":
            return FedAvg(self.clients)
        if strategy_name == "feda3i":
            return FedA3I(
                self.clients,
                int(fl_args["feda3i_warmup_rounds_frac"] * self.num_rounds),
                fl_args["feda3i_interw"],
                fl_strategy_state=fl_strategy_state,
            )
        if strategy_name == "feddm":
            return FedDM(
                self.clients,
                fl_args["feddm_gamma_hgd_smoothing"],
                fl_args["feddm_ratio_cac_pixelselection"],
                fl_args["feddm_cac_label_correction"],
                fl_args["feddm_loss"],
            )
        if strategy_name == "iopfl":
            return IOPFL(
                self.clients,
                fl_args["iopfl_alpha"],
                fl_strategy_state=fl_strategy_state,
            )
        if strategy_name == "fedcorr":
            return FedCorr(
                self.clients,
                self.num_rounds,
                fl_args["fedcorr_preproc_rounds_frac"],
                fl_args["fedcorr_relabel_ratio"],
                fl_args["fedcorr_relabel_confidence_thres"],
                fl_args["fedcorr_proxterm_beta"],
                fl_strategy_state=fl_strategy_state,
            )
        if strategy_name == "fedselect":
            return FedSelect(
                self.clients,
                self.num_rounds,
                fl_args["fedselect_warmup_rounds_frac"],
                fl_args["fedselect_client_select_ratio"],
                fl_args["fedselect_sample_select_ratio"],
                fl_args["fedselect_meta_momentum"],
                fl_args["fedselect_reward_data_size_frac"],
                fl_args["fedselect_proxy_batch_size"],
                fl_strategy_state=fl_strategy_state,
            )
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
            self._run_server_step(fl_round)

        # distribute final FL models to clients
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

    def _run_server_step(self, fl_round: int):
        strategy_name = self.fl_strategy.name
        if strategy_name == "fedavg":
            self._run_fedavg_server_step()
        elif strategy_name == "feda3i":
            self._run_feda3i_server_step(fl_round)
        elif strategy_name == "feddm":
            self._run_feddm_server_step()
        elif strategy_name == "iopfl":
            self._run_iopfl_server_step()
        elif strategy_name == "fedcorr":
            self._run_fedcorr_server_step(fl_round)
        elif strategy_name == "fedselect":
            self._run_fedselect_server_step(fl_round)
        else:
            raise NotImplementedError(
                f"Federated learning strategy {strategy_name} not implemented!"
            )

    def _run_fedavg_server_step(self):
        logging.info("Aggregating model weights with FedAvg strategy!")
        self.aggregate(strategy="fedavg")

    def _run_feda3i_server_step(self, fl_round: int):
        if fl_round <= self.fl_strategy.feda3i_warmup_rounds:
            logging.info("FedA3I warmup stage; aggregating model weights with FedAvg strategy!")
            self.aggregate(strategy="fedavg")
            if fl_round == self.fl_strategy.feda3i_warmup_rounds:
                logging.info(
                    "FedA3I warmup phase finished; computing quality-based aggregation weights!"
                )
                self.fl_strategy.feda3i_compute_quality_agg_weights()
            return

        logging.info("Aggregating model weights with FedA3I strategy!")
        self.aggregate(strategy="feda3i")

    def _run_feddm_server_step(self):
        logging.info(
            "Central steps of FedDM strategy: "
            "Collaborative Annotation Calibration and Hierarchical Gradient De-Conflicting!"
        )
        self.aggregate(strategy="feddm")

    def _run_iopfl_server_step(self):
        logging.info("Aggregating model weights with FedAvg strategy for IOP-FL!")
        self.aggregate(strategy="fedavg")

    def _run_fedcorr_server_step(self, fl_round: int):
        in_preproc = fl_round < self.fl_strategy.fedcorr_preproc_rounds
        in_finetune = fl_round < (
            self.fl_strategy.fedcorr_preproc_rounds
            + self.fl_strategy.fedcorr_finetune_rounds
        )

        if in_preproc or not in_finetune:
            logging.info("Aggregating model weights with FedAvg strategy for FedCorr!")
            self.aggregate(strategy="fedavg")
        else:
            logging.info(
                "Aggregating model weights with FedCorr's adapted FedAvg strategy for fine-tuning stage!"
            )
            self.aggregate(strategy="fedcorr")

        if in_preproc:
            self.fl_strategy.central_noisy_client_identification(fl_round)
        if in_preproc or in_finetune:
            self.fl_strategy.central_noise_level_estimation(fl_round)
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
            self.fl_strategy.save_global_model_weights()

    def _run_fedselect_server_step(self, fl_round: int):
        logging.info("Aggregating model weights with FedSelect strategy!")
        self.aggregate(strategy="fedselect", fl_round=fl_round)

        most_influential_client_id = self.fl_strategy.get_most_influential_client()
        self._log_cuda_memory("FedSelect before trainer offload")
        self._move_all_clients_trainers_to_device(torch.device("cpu"))
        torch.cuda.empty_cache()
        self._log_cuda_memory("FedSelect after trainer offload+empty_cache")
        try:
            self._log_cuda_memory("FedSelect before meta model training")
            self.fl_strategy.train_meta_model(most_influential_client_id, fl_round)
        finally:
            self._move_all_clients_trainers_to_device(torch.device("cuda"))
            torch.cuda.empty_cache()
            self._log_cuda_memory("FedSelect after trainer restore+empty_cache")

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

    def _move_optimizer_state_to_device(self, optimizer, device: torch.device):
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _log_cuda_memory(self, tag: str):
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda")
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        allocated_gib = torch.cuda.memory_allocated(device) / (1024**3)
        reserved_gib = torch.cuda.memory_reserved(device) / (1024**3)
        max_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)
        logging.info(
            f"{tag}: cuda_allocated={allocated_gib:.2f} GiB, "
            f"cuda_reserved={reserved_gib:.2f} GiB, "
            f"cuda_max_allocated={max_allocated_gib:.2f} GiB, "
            f"cuda_free={free_gib:.2f}/{total_gib:.2f} GiB"
        )

    def _move_all_clients_trainers_to_device(self, device: torch.device):
        """
        Move all available client nnUNet trainer modules/state to the target device.
        This is used to free GPU memory before FedSelect meta-model training and to
        restore trainer state afterwards.
        """
        for client in self.clients:
            trainer = getattr(client.model, "nnunet_trainer", None)
            if trainer is None:
                continue
            try:
                if getattr(trainer, "network", None) is not None:
                    trainer.network = trainer.network.to(device)
                loss_module = getattr(trainer, "loss", None)
                if hasattr(loss_module, "to"):
                    trainer.loss = loss_module.to(device)
                self._move_optimizer_state_to_device(
                    getattr(trainer, "optimizer", None), device
                )
            except Exception as e:
                logging.warning(
                    f"Could not move trainer state for client {client.client_id} "
                    f"to {device}: {e}"
                )
