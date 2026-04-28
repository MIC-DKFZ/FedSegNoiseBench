import glob
import json
import logging
import os
import time

import torch
from model import nnUNetv2_fed


class Client:
    def __init__(
        self, client_id: int = None, model_args: dict = None, fl_args: dict = None
    ):
        model_args = model_args or {}
        fl_args = fl_args or {}
        logging.info(f"Initialize client {client_id}!")
        logging.info(
            f"Client {client_id} with Experiment ID: {model_args['experiment_id']}"
        )

        # input args
        self.client_id = client_id
        self.model_args = model_args
        self.fl_args = fl_args

        # initialize model, dataset, and results dir
        self.model = nnUNetv2_fed(**self.model_args)
        self.dataset_id = self.model_args["dataset_id"]
        self.dataset_name = next(
            iter(
                glob.glob(
                    os.path.join(
                        os.getenv("nnUNet_preprocessed"), f"Dataset{self.dataset_id}_*"
                    )
                )
            ),
            None,
        )
        self.dataset_json = json.loads(
            open(
                os.path.join(
                    os.getenv("nnUNet_preprocessed"), self.dataset_name, "dataset.json"
                )
            ).read()
        )
        self.results_dir = os.path.join(
            os.getenv("nnUNet_results"),
            os.path.basename(self.dataset_name),
            f"{self.model_args['trainer']}__{self.model_args['plan']}__{self.model_args['configuration']}",
            f"fold_{self.model_args['fold']}",
            self.model_args["experiment_id"],
        )

        # other
        self.current_epoch = fl_args.get("start_epoch", 0)

    def _get_round_context(self, fl_round: int):
        target_num_epochs = self.current_epoch + self.fl_args["num_local_epochs"]
        total_rounds = int(self.fl_args.get("num_rounds", 0))
        if total_rounds > 0:
            last_fl_round = fl_round == total_rounds - 1
        else:
            total_epochs = int(self.fl_args.get("num_rounds", 0)) * int(
                self.fl_args.get("num_local_epochs", 0)
            )
            last_fl_round = target_num_epochs >= total_epochs
        logging.info(
            f"fl_round={fl_round}; total_rounds={total_rounds}; "
            f"num_local_epochs={self.fl_args.get('num_local_epochs')}; "
            f"target_num_epochs={target_num_epochs} => last_fl_round={last_fl_round}"
        )
        return target_num_epochs, last_fl_round

    def _base_run_kwargs(
        self,
        target_num_epochs: int,
        last_fl_round: bool,
        very_last_fl_predict_round: bool,
        only_run_validation: bool,
        fl_strategy=None,
    ):
        return {
            "initialize_fed_training": False,
            "num_epochs": target_num_epochs,
            "current_epoch": self.current_epoch,
            "epochs_per_round": self.fl_args["num_local_epochs"],
            "last_fl_round": last_fl_round,
            "very_last_fl_predict_round": very_last_fl_predict_round,
            "only_run_validation": only_run_validation,
            "fl_strategy": fl_strategy,
        }

    def _run_default_round(self, run_kwargs: dict):
        run_kwargs["fl_strategy"] = None
        self.model.run(**run_kwargs)

    def _run_feddm_round(self, run_kwargs: dict, fl_round: int, fl_strategy):
        if fl_round <= 0:
            run_kwargs["fl_strategy"] = None
            self.model.run(**run_kwargs)
            return

        feddm_client_peers = {
            key: {
                "nearest": peers["nearest"]["id"],
                "farthest": peers["farthest"]["id"],
            }
            for key, peers in fl_strategy.clients_peers.items()
            if key == self.client_id
        }
        run_kwargs["feddm_client_peers"] = feddm_client_peers
        self.model.run(**run_kwargs)

    def _run_fedcorr_round(self, run_kwargs: dict, fl_round: int, fl_strategy):
        is_noisy_client = self.client_id in fl_strategy.noisy_clients
        run_kwargs.update(
            {
                "fl_client_id": self.client_id,
                "is_fedcorr_noisyclient": is_noisy_client,
            }
        )

        if fl_round < fl_strategy.fedcorr_preproc_rounds:
            run_kwargs["is_fedcorr_preproc_stage"] = True
            self.model.run(**run_kwargs)

            local_output, local_output_highres, loss = fl_strategy.get_output_seg(
                self.model.nnunet_trainer
            )
            LID_local = list(fl_strategy.lid_term_batched(local_output_highres))
            fl_strategy.set_lid(LID_local, self.client_id)
            fl_strategy.set_loss(loss, self.client_id)
            logging.info(
                f"Client {self.client_id} LID_local: {fl_strategy.LID_client[self.client_id]}"
            )
            return

        if fl_round < (
            fl_strategy.fedcorr_preproc_rounds + fl_strategy.fedcorr_finetune_rounds
        ):
            logging.info(
                f"FedCorr fine-tuning round at FL round {fl_round} on client {self.client_id}!"
            )
            run_kwargs["is_fedcorr_finetune_stage"] = True
            self.model.run(**run_kwargs)
            _, _, loss = fl_strategy.get_output_seg(self.model.nnunet_trainer)
            fl_strategy.set_loss(loss, self.client_id)
            return

        logging.info(
            f"FedCorr full-training round at FL round {fl_round} on client {self.client_id}!"
        )
        run_kwargs["is_fedcorr_fulltrain_stage"] = True
        self.model.run(**run_kwargs)

    def _run_fedselect_round(self, run_kwargs: dict, fl_round: int, fl_strategy):
        is_selected = self.client_id in fl_strategy.selected_clients
        if is_selected:
            fl_strategy.compute_sample_weights(
                self.model.nnunet_trainer, self.client_id, fl_round
            )
            fl_strategy.compute_client_weights(self.model.nnunet_trainer, self.client_id)

        run_kwargs.update(
            {
                "fl_client_id": self.client_id,
                "is_fedselect_selected_client": is_selected,
            }
        )
        self.model.run(**run_kwargs)

        if is_selected:
            fl_strategy.compute_meta_margin_scores(
                self.model.nnunet_trainer, self.client_id, fl_round
            )
            fl_strategy.update_proxy_validation_dataset(
                self.model.nnunet_trainer, self.client_id
            )

    def _run_strategy_round(self, fl_round: int, run_kwargs: dict, fl_strategy):
        strategy_name = getattr(fl_strategy, "name", "fedavg")
        if strategy_name == "feddm":
            self._run_feddm_round(run_kwargs, fl_round, fl_strategy)
        elif strategy_name == "fedcorr":
            self._run_fedcorr_round(run_kwargs, fl_round, fl_strategy)
        elif strategy_name == "fedselect":
            self._run_fedselect_round(run_kwargs, fl_round, fl_strategy)
        else:
            self._run_default_round(run_kwargs)

    def fed_round(
        self,
        fl_round,
        very_last_fl_predict_round: bool = False,
        only_run_validation: bool = False,
        fl_strategy=None,
    ):
        """
        Perform a federated learning round on the client.
        """
        logging.info(f"Start federated round {fl_round} on client {self.client_id}!")
        start_time = time.time()

        target_num_epochs, last_fl_round = self._get_round_context(fl_round)
        run_kwargs = self._base_run_kwargs(
            target_num_epochs,
            last_fl_round,
            very_last_fl_predict_round,
            only_run_validation,
            fl_strategy,
        )
        self._run_strategy_round(fl_round, run_kwargs, fl_strategy)

        # IOP-FL: compute personalized model after local training
        if getattr(fl_strategy, "name", None) == "iopfl" and not very_last_fl_predict_round:
            fl_strategy.compute_trajectory(
                self.model.current_model_weights, self.client_id
            )

        # update current epoch
        self.current_epoch = target_num_epochs

        # periodically save fl_strategy state
        if fl_round % self.model.save_every == 0:
            fl_strategy.save_state(exp_id=self.model_args["experiment_id"], client_id=self.client_id)

        # log time
        end_time = time.time()
        logging.info(
            f"Local federated round on client {self.client_id}: {end_time - start_time:.2f} seconds!"
        )

    def update_model(
        self, server_model_weights: dict = None, checkpoint_name: str = None
    ):
        """
        Takes the server model weights and updates the client model with them by writing it as the current checkpoint.
        """
        if not checkpoint_name:
            self.model.current_model_weights = server_model_weights
        elif checkpoint_name:
            # load "checkpoint_final.pth" from client results directory
            client_checkpoint = torch.load(
                os.path.join(self.results_dir, "checkpoint_final.pth"),
                weights_only=False,
            )
            # update this checkpoint's model weights with provided server model weights
            if "model_state_dict" in client_checkpoint:
                client_checkpoint["model_state_dict"] = server_model_weights
            elif "network_weights" in client_checkpoint:
                client_checkpoint["network_weights"] = server_model_weights
            else:
                raise ValueError(
                    "Loaded client checkpoint does not contain model weights! Therefore cannot update with server model weights."
                )
            # write torch checkpoint to clients directory
            torch.save(
                client_checkpoint, os.path.join(self.results_dir, checkpoint_name)
            )
        else:
            raise ValueError("No server model weights or checkpoint name provided!")
