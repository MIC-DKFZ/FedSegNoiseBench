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
        )

        # other
        self.current_epoch = 0
        self.current_checkpoint = None

    def fed_round(self):
        """
        Perform a federated learning round on the client.
        """
        logging.info(f"Start local training on client {self.client_id}!")
        start_time = time.time()

        # set target number of epochs of current fl round
        target_num_epochs = self.current_epoch + self.fl_args["num_local_epochs"]

        # run local training
        self.model.run(
            initialize_fed_training=False,
            continue_training=True,
            num_epochs=target_num_epochs,
            current_epoch=self.current_epoch,
            epochs_per_round=self.fl_args["num_local_epochs"],
            last_fl_round=(
                True if target_num_epochs == self.fl_args["num_rounds"] else False
            ),
        )
        self.current_epoch = target_num_epochs

        # log time
        end_time = time.time()
        logging.info(
            f"Local federated round on client {self.client_id}: {end_time - start_time:.2f} seconds!"
        )

    def update_model(self, server_model_weights: dict = {}):
        """
        Takes the server model weights and updates the client model with them by writing it as the current hceckpoint.
        """
        # update "network_weights" of self.current_checkpoint with server model weights, and write to new checkpoint
        self.current_checkpoint["network_weights"] = server_model_weights
        # write server model weights to checkpoint
        checkpoint_path = os.path.join(self.results_dir, "checkpoint_fl_current.pth")
        torch.save(self.current_checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_name: str = None):
        """
        Load a checkpoint of the client from the file system using PyTorch and return the state_dict.

        Args:
            checkpoint_name (str, optional): Name of the checkpoint file to load. Defaults to None.

        Returns:
            dict: The loaded checkpoint containing the model's state_dict.
        """
        # check if checkpoint_name is provided and exists
        assert checkpoint_name is not None, "Checkpoint name must be provided."
        checkpoint_path = os.path.join(self.results_dir, checkpoint_name)
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint file not found: {checkpoint_path}"

        # load and return checkpoint
        try:
            self.current_checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu")
            )
            return self.current_checkpoint
        except RuntimeError as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
