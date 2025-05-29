import os
import glob
import json
import logging
import time
import torch

from prediction_classes.pred_model import nnUNetv2_fed


class Prediction_Client:
    def __init__(
        self, client_id: int = None, model_args: dict = {},
    ):
        logging.info(f"Initialize prediction on client {client_id}!")
        logging.info(
            f"Client {client_id} with Experiment ID: {model_args['experiment_id']}"
        )

        # input args
        self.client_id = client_id
        self.model_args = model_args

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

    def fed_round(
        self,
        very_last_fl_predict_round: bool = False,
        only_run_validation: bool = False,
    ):
        """
        Perform a federated prediction round on the client.
        """
        logging.info(f"Start federated prediction round on client {self.client_id}!")
        start_time = time.time()

        # run prediction
        self.model.run(
            very_last_fl_predict_round=very_last_fl_predict_round,
            only_run_validation=only_run_validation,
        )

        # log time
        end_time = time.time()
        logging.info(
            f"Local federated round on client {self.client_id}: {end_time - start_time:.2f} seconds!"
        )

    def update_model(
        self, server_model_weights: dict = {}, checkpoint_name: str = None
    ):
        """
        Takes the server model weights and updates the client model with them by writing it as the current hceckpoint.
        """
        if not checkpoint_name:
            self.model.current_model_weights = server_model_weights
        elif checkpoint_name:
            # load "checkpoint_final.pth" from client results directory
            client_checkpoint = torch.load(
                os.path.join(self.results_dir, "checkpoint_final.pth")
            )
            # update this checkpoint's model weights with provided server model weights
            client_checkpoint["model_state_dict"] = server_model_weights
            # write torch checkpoint to clients directory
            torch.save(
                client_checkpoint, os.path.join(self.results_dir, checkpoint_name)
            )
        else:
            raise ValueError("No server model weights or checkpoint name provided!")
