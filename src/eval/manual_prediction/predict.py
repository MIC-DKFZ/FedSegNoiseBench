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

from prediction_classes.pred_orchestrator import Prediction_Orchestrator
from prediction_classes.pred_client import Prediction_Client


def main(args):

    # setup clients
    clients = [
        Prediction_Client(
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
                "experiment_id": f"D{args.dataset_ids.split()[i]}_{args.experiment_id}"
            },
        )
        for i in range(args.num_clients)
    ]

    # setup orchestrator
    orchestrator = Prediction_Orchestrator(
        clients,
    )

    # run federated prediction
    orchestrator.fl_predict()


if __name__ == "__main__":
    # take CLI arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_id",
        type=str,
        default="",
        help="ID of experiment to predict for.",
    )

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

    # noise arguments
    parser.add_argument(
        "--clean_validation_dataset",
        type=str,
        default=None,
        help="Path to clean validation data, to not evaluate on noisy data if training on noisy data.",
    )

    # other arguments
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # run main function
    main(args)


