import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to PYTHONPATH automatically if it's not there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)
print(f"{src_path=}")
print(f"{sys.path=}")

from orchestrator import Orchestrator
from client import Client


def main(args):
    # setup experiment id
    experiment_id = f"{args.noise_mitigation_method.lower()}_noiseroa{args.noise_ratio}_fold{args.fold}_clients{args.num_clients}_flrounds{args.num_rounds}_localepochs{args.num_local_epochs}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

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
                "noisy_train_folder": (
                    args.noisy_train_folder.split()[i]
                    if args.noisy_train_folder
                    else None
                ),
                "noise_ratio": args.noise_ratio,
            },
            fl_args={
                "num_local_epochs": args.num_local_epochs,
                "num_rounds": args.num_rounds,
            },
        )
        for i in range(args.num_clients)
    ]

    # setup orchestrator
    orchestrator = Orchestrator(
        clients,
        fl_args={
            "num_rounds": args.num_rounds,
            "strategy": args.noise_mitigation_method,
            "feda3i_warmup_rounds": (
                args.feda3i_warmup_rounds
                if args.noise_mitigation_method.lower() == "feda3i"
                else None
            ),
        },
    )

    # run federated learning
    orchestrator.fl_run()


def check_cli_args(args):
    dataset_ids = args.dataset_ids.split()

    # num dataset_ids != num_clients
    assert (
        len(dataset_ids) == args.num_clients
    ), "Every client needs its dataset! Please provide as many datasets as clients."

    # if all clients are partially noise, clean_validation_folder has to be given
    if args.noisy_train_folder:
        assert (
            args.noisy_train_folder is None and args.clean_validation_dataset is None
        ) or (
            args.noisy_train_folder is not None
            and args.clean_validation_dataset is not None
        ), "Arguments --noisy_train_folder and --clean_validation_dataset must be provided together or not at all"


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

    # method arguments
    parser.add_argument(
        "--noise_mitigation_method",
        type=str,
        default="FedAvg",
        help="Method to mitigate segmentation label noise: FedA3I, FedDM, None for vanilla FedAvg FL training.",
    )
    parser.add_argument(
        "--feda3i_warmup_rounds",
        type=int,
        default=5,
        help="Number of warmup rounds for FedA3I.",
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
    parser.add_argument(
        "--noisy_train_folder",
        type=str,
        default=None,
        help="Path to noisy train data, to create clients with partially clean, partially noisy data.",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=None,
        help="Ratio of noisy and clean train data, to create clients with partially clean, partially noisy data.",
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
