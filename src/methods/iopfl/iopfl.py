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
                logging.info(
                    f"IOP-FL: Loading personalized trajectory model for client {client.client_id} from {trajectory_path}!"
                )
                self.trajectory[client.client_id] = torch.load(
                    trajectory_path, map_location="cpu"
                )
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
        
        target_device = self.clients[0].model.nnunet_trainer.device
        
        if self.trajectory[client_id] is None:
            # initialize trajectory with local model weights after first local training
            # move to CPU to save GPU memory
            trajectory_cpu = copy.deepcopy(w_local)
            trajectory_cpu = self._move_state_to_device(trajectory_cpu, torch.device("cpu"))
            self.trajectory[client_id] = trajectory_cpu
        else:
            # compute trajectory update
            # Explicitly delete old trajectory reference before creating new one
            old_trajectory = self.trajectory[client_id]
            self.trajectory[client_id] = None  # Clear reference
            
            # Move old trajectory and w_local to GPU for computation
            current_trajectory = self._move_state_to_device(
                old_trajectory, target_device
            )
            w_local_gpu = self._move_state_to_device(
                copy.deepcopy(w_local), target_device
            )
            
            # Explicitly delete old_trajectory to free memory
            del old_trajectory
            torch.cuda.empty_cache()

            # get addresses of keys
            keys = list(current_trajectory.keys())
            address_key_dict = {}
            for k in keys:
                address = current_trajectory[k].data_ptr()
                if address not in address_key_dict.keys():
                    address_key_dict[address] = [k]
                else:
                    address_key_dict[address].append(k)

            # update trajectory weights per key (in-place on GPU)
            for a in address_key_dict.keys():
                current_trajectory[address_key_dict[a][0]] = (
                    self.iopfl_alpha * current_trajectory[address_key_dict[a][0]]
                    + (1 - self.iopfl_alpha) * w_local_gpu[address_key_dict[a][0]]
                )

            # Move updated trajectory back to CPU to save GPU memory
            trajectory_cpu = self._move_state_to_device(
                current_trajectory, torch.device("cpu")
            )
            
            # Free GPU memory before storing
            del current_trajectory, w_local_gpu
            torch.cuda.empty_cache()
            
            # Store CPU version
            self.trajectory[client_id] = trajectory_cpu

        logging.info(
            f"IOP-FL: Trajectory stored on CPU for client {client_id} to conserve GPU memory"
        )

    @staticmethod
    def _move_state_to_device(state: dict, device: torch.device):
        # preserve shared storage by moving once per data_ptr
        address_key_dict = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            address = value.data_ptr()
            if address not in address_key_dict:
                address_key_dict[address] = [key]
            else:
                address_key_dict[address].append(key)

        for keys in address_key_dict.values():
            source = state[keys[0]]
            if source.device != device:
                moved = source.to(device)
                for key in keys:
                    state[key] = moved

        return state

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
