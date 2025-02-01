import os
import glob
import json
import torch

from src.fed.model import nnUNetv2_fed


class Client:
    def __init__(
        self, client_id: int = None, model_args: dict = {}, fl_args: dict = {}
    ):
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
        self.dataset_json = json.load(
            os.path.join(
                os.getenv("nnUNet_preprocessed"), self.dataset_name, "dataset.json"
            )
        )
        self.results_dir = os.path.join(
            os.getenv("nnUNet_results"),
            self.dataset_name,
            f"{self.model_args['trainer']}_{self.model_args['plan']}_{self.model_args['configuration']}",
            self.model_args["fold"],
        )

    def fed_round(self):
        pass

    def update_model(self):
        pass

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
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            return checkpoint
        except RuntimeError as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
