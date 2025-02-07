import os
import argparse
import logging
from datetime import datetime

from orchestrator import Orchestrator
from client import Client

# def setup_logging(args, experiment_id):
#     # Create logger
#     logger = logging.getLogger("my_logger")
#     logger.setLevel(args.log_level.upper())

#     # Create file handler
#     log_file_name = os.path.join(os.getenv("nnUNet_results"), "logs", f"exp{args.dataset_ids.replace(' ','_')}-{experiment_id}.log")
#     os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
#     file_handler = logging.FileHandler(log_file_name, mode="w")
#     file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

#     # Create console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

#     # Add handlers to logger
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)


def main(args):
    # setup experiment id
    experiment_id = f"fold{args.fold}_clients{args.num_clients}_flrounds{args.num_rounds}_localepochs{args.num_local_epochs}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # # set up logging
    # setup_logging(args, experiment_id)

    # setup clients
    clients = [
        Client(
            client_id=i,
            model_args={
                "dataset_id": args.dataset_ids.split()[i],
                "configuration": args.configuration,
                "fold": args.fold,
                "plan": args.plan,
                "trainer": args.trainer,
                "clean_validation_dataset": (
                    args.clean_validation_dataset.split()[i]
                    if args.clean_validation_dataset
                    else None
                ),
                "experiment_id": f"D{args.dataset_ids.split()[i]}_{experiment_id}",
            },
            fl_args={
                "num_local_epochs": args.num_local_epochs,
                "num_rounds": args.num_rounds,
            },
        )
        for i in range(args.num_clients)
    ]

    # setup orchestrator
    orchestrator = Orchestrator(clients, fl_args={"num_rounds": args.num_rounds})

    # run federated learning
    orchestrator.fl_run()


def check_cli_args(args):
    dataset_ids = args.dataset_ids.split()
    assert (
        len(dataset_ids) == args.num_clients
    ), "Every client needs its dataset! Please provide as many datasets as clients."


if __name__ == "__main__":
    # take CLI arguments
    parser = argparse.ArgumentParser()

    # nnU-Net arguments
    parser.add_argument(
        "--dataset_ids",
        type=str,
        default="",
        help="Dataset ID of nnU-Net dataset to use.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        help="Configuration of nnU-Net to use.",
    )
    parser.add_argument(
        "--fold", type=str, default="0", help="Fold of nnU-Net dataset to use."
    )
    parser.add_argument(
        "--plan", type=str, default="nnUNetPlans", help="Plan of nnU-Net to use."
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        help="Trainer of nnU-Net to use.",
    )

    # FL arguments
    parser.add_argument(
        "--num_clients",
        type=int,
        default=4,
        help="Number of clients to join federated training.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of rounds to run federated training.",
    )
    parser.add_argument(
        "--num_local_epochs",
        type=int,
        default=5,
        help="Number of local epochs to run on each client per fl round.",
    )

    # other arguments
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # noise arguments
    parser.add_argument(
        "--clean_validation_dataset",
        type=str,
        default=None,
        help="Path to clean validation data, to not evaluate on noisy data if training on noisy data.",
    )
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # run sanity checks of user input
    check_cli_args(args)
    # run main function
    main(args)
