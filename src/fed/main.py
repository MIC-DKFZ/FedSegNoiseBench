import argparse
import logging

from src.fed.orchestrator import Orchestrator
from src.fed.client import Client


def main(args):
    # setup clients
    clients = [
        Client(
            client_id=i,
            model_args={
                "dataset_id": args.dataset_ids[i],
                "configuration": args.configuration,
                "fold": args.fold,
                "plan": args.plan,
                "trainer": args.trainer,
                "clean_validation_folder": args.clean_validation_folder,
            },
            fl_args={"num_local_epochs": args.num_local_epochs},
        )
        for i in range(args.num_clients)
    ]

    # setup orchestrator
    orchestrator = Orchestrator(clients, fl_args={"num_rounds": args.num_rounds})

    # run federated learning
    orchestrator.fl_run()


def check_cli_args(args):
    assert len(args.dataset_ids) == len(
        args.num_clients
    ), "Every client needs its dataset! Please provide as many datasets as clients."


if __name__ == "__main__":
    # take CLI arguments
    parser = argparse.ArgumentParser()

    # nnU-Net arguments
    parser.add_argument(
        "--dataset_ids",
        type=list,
        default=[],
        description="Dataset ID of nnU-Net dataset to use.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        description="Configuration of nnU-Net to use.",
    )
    parser.add_argument(
        "--fold", type=str, default="0", description="Fold of nnU-Net dataset to use."
    )
    parser.add_argument(
        "--plan", type=str, default="nnUNetPlans", description="Plan of nnU-Net to use."
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        description="Trainer of nnU-Net to use.",
    )

    # FL arguments
    parser.add_argument(
        "--num_clients",
        type=int,
        default=4,
        description="Number of clients to join federated training.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        description="Number of rounds to run federated training.",
    )
    parser.add_argument(
        "--num_local_epochs",
        type=int,
        default=5,
        description="Number of local epochs to run on each client per fl round.",
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
        "--clean_validation_folder",
        type=str,
        default=None,
        description="Path to clean validation data, to not evaluate on noisy data if training on noisy data.",
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
