import os
import json
import logging
import copy
from glob import glob

from methods.fedavg.fedavg import FedAvg
import torch


class IOPFL(FedAvg):
    """
    IOP-FL: Inside-Outside Personalization for Federated Medical Image Segmentation
    Meirui Jiang et al., 2023, IEEE TMI
    https://ieeexplore.ieee.org/document/10086676

    IOP-FL method consists of two major contributions:
    1.) Local Adapted Model for Personalization:
       - next to std FedAvg, each client fits a personalized model
       - personalized model is a mixture of federated trained model and its local history per client
       - validation & evaluation are performed using the personalized model
    2.) Test-Time Routing for Outside Personalization:
       - enhanced generalization on models that were not part of training; not relevant label noise considerations.

    Args:
        clients (list): List of clients participating in federated learning.
        iopfl_alpha (float): Weighting factor for combining federated trained model with personal model history.
    """

    def __init__(
        self,
        clients: list = None,
        iopfl_alpha: float = 0.5,
        fl_strategy_state: dict = None,
    ):
        super().__init__(clients)
        self.experiment_id = None  # to be set when saving state

        self.name = "iopfl"
        self.iopfl_alpha = (
            iopfl_alpha
            if fl_strategy_state is None
            else fl_strategy_state["iopfl_alpha"]
        )
        self.trajectory = {}
        for client in clients:
            if (
                fl_strategy_state is not None
                and fl_strategy_state["trajectory_paths"][str(client.client_id)]
                is not None
            ):
                # load trajectory model from checkpoint
                trajectory_path = fl_strategy_state["trajectory_paths"][
                    str(client.client_id)
                ]
                self.trajectory[client.client_id] = torch.load(trajectory_path)
                logging.info(
                    f"IOP-FL: Loaded personalized trajectory model for client {client.client_id} from {trajectory_path}!"
                )
            else:
                self.trajectory[client.client_id] = None

        self.fl_strategy_state = {
            "trajectory_paths": {
                client.client_id: (
                    fl_strategy_state["trajectory_paths"][str(client.client_id)]
                    if fl_strategy_state is not None
                    else None
                )
                for client in clients
            },
            "iopfl_alpha": self.iopfl_alpha,
        }
        print(f"Initialized IOP-FL with alpha={self.iopfl_alpha}!")

    def compute_trajectory(self, w_local: dict = {}, client_id: int = None):
        """
        Compute personalized model trajectory for current client:
        trajectory = alpha * trajectory + (1 - alpha) * w_local
        """
        logging.info(
            f"IOP-FL: Computing personalized trajectory model for client {client_id}!"
        )
        if self.trajectory[client_id] is None:
            # initialize trajectory with local model weights after first local training
            self.trajectory[client_id] = copy.deepcopy(w_local)
        else:
            # compute trajectory update
            # copy current trajectory
            current_trajectory = copy.deepcopy(self.trajectory[client_id])

            # get addresses of keys
            keys = list(current_trajectory.keys())
            address_key_dict = {}
            for k in keys:
                address = current_trajectory[k].data_ptr()
                if address not in address_key_dict.keys():
                    address_key_dict[address] = [k]
                else:
                    address_key_dict[address].append(k)

            # update trajectory weights per key
            for a in address_key_dict.keys():
                current_trajectory[address_key_dict[a][0]] = (
                    self.iopfl_alpha * current_trajectory[address_key_dict[a][0]]
                    + (1 - self.iopfl_alpha) * w_local[address_key_dict[a][0]]
                )

            # update trajectory
            self.trajectory[client_id] = current_trajectory

    def save_state(self, exp_id: str, client_id: int = None):
        """
        Save the current state of IOPFL method to experiment's cli args file.
        """

        # Save trajectory models as torch checkpoints to experiment directory
        # get exp directory
        results_dir = os.getenv("nnUNet_results") or os.getcwd()
        dataset_id = exp_id.split("_")[0].strip("D")
        exp_dir = glob(
            os.path.join(results_dir, f"Dataset{dataset_id}_*", "*", "*", exp_id)
        )[0]
        assert (
            exp_dir is not None
        ), f"Could not find experiment directory for exp_id {exp_id}!"

        for c_idx, trajectory_model in self.trajectory.items():
            if trajectory_model is not None and c_idx == client_id:
                checkpoint_path = os.path.join(
                    exp_dir, f"iopfl_trajectory_client_{client_id}_checkpoint.pth"
                )
                torch.save(trajectory_model, checkpoint_path)
                self.fl_strategy_state["trajectory_paths"][c_idx] = checkpoint_path

        # set experiment id
        self.experiment_id = exp_id

        args_file = self.save_fl_strategy_state_to_file(self.fl_strategy_state, exp_id)

        logging.info(f"Saved IOPFL state to {args_file}")
