import logging
import copy

from methods.fedavg.fedavg import FedAvg

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
    def __init__(self, clients: list = None, iopfl_alpha: float = 0.5):
        super().__init__(clients)
        self.name = "iopfl"

        self.iopfl_alpha = iopfl_alpha  # weight combining federated model and personal model history
        self.trajectory = {client.client_id: None for client in clients}

    def compute_trajectory(self, w_local: dict = {}, client_id: int = None):
        """
        Compute personalized model trajectory for current client:
        trajectory = alpha * trajectory + (1 - alpha) * w_local
        """
        logging.info(f"IOP-FL: Computing personalized trajectory model for client {client_id}!")
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