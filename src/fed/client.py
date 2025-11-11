import os
import glob
import json
import logging
import time
import torch

from model import nnUNetv2_fed


class Client:
    def __init__(
        self, client_id: int = None, model_args: dict = {}, fl_args: dict = {}
    ):
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

        # set target number of epochs of current fl round
        target_num_epochs = self.current_epoch + self.fl_args["num_local_epochs"]

        # run local training
        last_fl_round = (
            abs(
                target_num_epochs
                - (self.fl_args["num_rounds"] * self.fl_args["num_local_epochs"])
            )
            == 0
        )
        logging.info(
            f"{target_num_epochs=} ; {self.fl_args['num_rounds']=} ; {self.fl_args['num_local_epochs']=} ==> Setting {last_fl_round}"
        )

        # run client's local training
        if fl_strategy.name == "feddm" and fl_round > 0:
            # # don't give feddm_client_peers incl. models to nnUNet
            feddm_client_peers = {
                key: {
                    "nearest": d["nearest"]["id"],
                    "farthest": d["farthest"]["id"],
                }
                for key, d in fl_strategy.clients_peers.items()
                if key == self.client_id
            }
            # FedDM's peer training
            self.model.run(
                initialize_fed_training=False,
                num_epochs=target_num_epochs,
                current_epoch=self.current_epoch,
                epochs_per_round=self.fl_args["num_local_epochs"],
                last_fl_round=last_fl_round,
                very_last_fl_predict_round=very_last_fl_predict_round,
                only_run_validation=only_run_validation,
                fl_strategy=fl_strategy,
                feddm_client_peers=feddm_client_peers,
            )
        else:
            self.model.run(
                initialize_fed_training=False,
                # continue_training=True,
                num_epochs=target_num_epochs,
                current_epoch=self.current_epoch,
                epochs_per_round=self.fl_args["num_local_epochs"],
                last_fl_round=last_fl_round,
                very_last_fl_predict_round=very_last_fl_predict_round,
                only_run_validation=only_run_validation,
            )
        self.current_epoch = target_num_epochs

        # log time
        end_time = time.time()
        logging.info(
            f"Local federated round on client {self.client_id}: {end_time - start_time:.2f} seconds!"
        )

    def update_model(
        self, server_model_weights: dict = {}, checkpoint_name: str = None
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
