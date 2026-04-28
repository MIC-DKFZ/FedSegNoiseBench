import argparse
import json
import logging
import os
import random
import string
import sys
from datetime import datetime

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from client import Client
from orchestrator import Orchestrator


METHOD_ARG_KEYS = {
    "fedavg": (),
    "feda3i": ("feda3i_warmup_rounds_frac", "feda3i_interw"),
    "feddm": (
        "feddm_gamma_hgd_smoothing",
        "feddm_ratio_cac_pixelselection",
        "feddm_cac_label_correction",
        "feddm_loss",
    ),
    "iopfl": ("iopfl_alpha",),
    "fedcorr": (
        "fedcorr_preproc_rounds_frac",
        "fedcorr_relabel_ratio",
        "fedcorr_relabel_confidence_thres",
        "fedcorr_proxterm_beta",
    ),
    "fedselect": (
        "fedselect_warmup_rounds_frac",
        "fedselect_client_select_ratio",
        "fedselect_sample_select_ratio",
        "fedselect_meta_momentum",
        "fedselect_reward_data_size_frac",
        "fedselect_proxy_batch_size",
    ),
}


def cli_args_to_file(args, experiment_id: str):
    """Log all CLI arguments to a JSON file in the results folder."""

    results_dir = os.getenv("nnUNet_results") or os.getcwd()
    os.makedirs(results_dir, exist_ok=True)

    args_file = os.path.join(results_dir, f"ExperimentArgs_{experiment_id}.json")
    args_dict = vars(args).copy()

    # Ensure JSON serializable: fall back to string representation for unknown types
    with open(args_file, "w") as f:
        json.dump(args_dict, f, indent=2, default=str)


def create_experiment_id(args) -> str:
    random_hash = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{args.noise_mitigation_method.lower()}_noiseroa{args.noise_ratio}_"
        f"fold{args.fold}_clients{args.num_clients}_flrounds{args.num_rounds}_"
        f"localepochs{args.num_local_epochs}_{timestamp}_{random_hash}"
    )


def split_optional_arg(value):
    return value.split() if value else None


def build_clients(args, experiment_id: str):
    dataset_ids = args.dataset_ids.split()
    clean_validation_datasets = split_optional_arg(args.clean_validation_dataset)
    noisy_train_folders = split_optional_arg(args.noisy_train_folder)

    clients = []
    for client_id, dataset_id in enumerate(dataset_ids):
        clients.append(
            Client(
                client_id=client_id,
                model_args={
                    "dataset_id": dataset_id,
                    "configuration": args.configuration,
                    "fold": args.fold,
                    "plan": args.plan,
                    "trainer": args.trainer,
                    "save_every": args.save_every,
                    "oversample_foreground_percent": args.oversample_foreground_percent,
                    "class_sampling_probabilities": args.class_sampling_probabilities,
                    "batch_element_class_probabilities": args.batch_element_class_probabilities,
                    "num_gpus": args.num_gpus,
                    "clean_validation_dataset": (
                        clean_validation_datasets[client_id]
                        if clean_validation_datasets
                        else None
                    ),
                    "experiment_id": f"D{dataset_id}_{experiment_id}",
                    "noisy_train_folder": (
                        noisy_train_folders[client_id] if noisy_train_folders else None
                    ),
                    "noise_ratio": args.noise_ratio,
                },
                fl_args={
                    "num_local_epochs": args.num_local_epochs,
                    "num_rounds": args.num_rounds,
                },
            )
        )
    return clients


def build_fl_args(args):
    strategy_name = args.noise_mitigation_method.lower()
    if strategy_name not in METHOD_ARG_KEYS:
        raise NotImplementedError(
            f"Federated learning strategy {args.noise_mitigation_method} not implemented!"
        )

    fl_args = {
        "num_rounds": args.num_rounds,
        "strategy": args.noise_mitigation_method,
    }
    for key in METHOD_ARG_KEYS[strategy_name]:
        fl_args[key] = getattr(args, key)
    return fl_args


def main(args):
    # setup experiment id
    experiment_id = create_experiment_id(args)

    # # set up logging
    # setup_logging(args, experiment_id)

    # log all cli args to file
    cli_args_to_file(args, experiment_id)

    clients = build_clients(args, experiment_id)

    # setup orchestrator
    orchestrator = Orchestrator(clients, fl_args=build_fl_args(args))

    # run federated learning
    orchestrator.fl_run()


def check_cli_args(args):
    dataset_ids = args.dataset_ids.split()

    # num dataset_ids != num_clients
    assert (
        len(dataset_ids) == args.num_clients
    ), "Every client needs its dataset! Please provide as many datasets as clients."

    if args.clean_validation_dataset:
        assert len(args.clean_validation_dataset.split()) == args.num_clients, (
            "--clean_validation_dataset must provide as many datasets as clients."
        )

    if args.noisy_train_folder:
        assert len(args.noisy_train_folder.split()) == args.num_clients, (
            "--noisy_train_folder must provide as many folders as clients."
        )

    if args.noisy_train_folder:
        assert args.clean_validation_dataset is not None, (
            "Argument --clean_validation_dataset must be provided when --noisy_train_folder is used"
        )

    # save_every should be positive and larger than num_local_epochs
    assert args.save_every > 0 and args.save_every >= min(
        args.num_local_epochs, args.num_rounds
    ), "--save_every must be positive and larger than or equal to min(--num_local_epochs, --num_rounds)"

    if args.fedselect_proxy_batch_size is not None:
        assert (
            args.fedselect_proxy_batch_size > 0
        ), "--fedselect_proxy_batch_size must be a positive integer"


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
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save model checkpoint every n epochs during local training on clients.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for local training on each client.",
    )
    parser.add_argument(
        "--oversample_foreground_percent",
        type=float,
        default=0.33,
        help="Percentage of oversampling foreground during patch sampling in nnU-Net trainer.",
    )
    parser.add_argument(
        "--class_sampling_probabilities",
        type=str,
        default=None,
        help="Class-specific oversampling probabilities for nnUNetTrainer_weightedClassSampling.",
    )
    parser.add_argument(
        "--batch_element_class_probabilities",
        type=str,
        default=None,
        help="Batch element-specific class sampling probabilities for nnUNetTrainer_batchElementClassSampling.",
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

    ##### method arguments
    parser.add_argument(
        "--noise_mitigation_method",
        type=str,
        default="FedAvg",
        help="Method to mitigate segmentation label noise: FedA3I, FedDM, None for vanilla FedAvg FL training.",
    )
    # FedA3I
    parser.add_argument(
        "--feda3i_warmup_rounds_frac",
        type=float,
        default=None,  # 0.1,
        help="Number of warmup rounds for FedA3I.",
    )
    parser.add_argument(
        "--feda3i_interw",
        type=float,
        default=None,  # 0.5,
        help="Interpolation weight between expand and shrink clients for quality-based aggregation.",
    )
    # FedDM
    parser.add_argument(
        "--feddm_gamma_hgd_smoothing",
        type=float,
        default=None,  # 0.99,
        help="Smoothing parameter gamma for HDG in FedDM.",
    )
    parser.add_argument(
        "--feddm_ratio_cac_pixelselection",
        type=float,
        default=None,  # 0.6,
        help="Ratio for class-agnostic pixel selection in FedDM.",
    )
    parser.add_argument(
        "--feddm_cac_label_correction",
        type=str,
        default=None,  # "largest",
        help="Label correction strategy for class-agnostic correction in FedDM: 'smallest' or 'largest'.",
    )
    parser.add_argument(
        "--feddm_loss",
        type=str,
        default=None,  # feddm_focal_loss
        help="Loss function to use in FedDM: 'feddm_nnunets_loss' or 'feddm_focal_loss' (theirs).",
    )
    # IOP-FL
    parser.add_argument(
        "--iopfl_alpha",
        type=float,
        default=None,  # 0.9,
        help="Weight for incorporation of model history in trajectory for IOP-FL.",
    )
    # FedCorr
    parser.add_argument(
        "--fedcorr_preproc_rounds_frac",
        type=float,
        default=None,  # 0.05
        help="Fraction of rounds for preprocessing stage in FedCorr.",
    )
    parser.add_argument(
        "--fedcorr_relabel_ratio",
        type=float,
        default=None,  # 0.5
        help="Ratio of most noisy samples that are relabeled in FedCorr.",
    )
    parser.add_argument(
        "--fedcorr_relabel_confidence_thres",
        type=float,
        default=None,  # 0.5
        help="Confidence threshold for relabeling in FedCorr.",
    )
    parser.add_argument(
        "--fedcorr_proxterm_beta",
        type=float,
        default=None,  # 5.0
        help="Beta parameter for proximal term in FedCorr.",
    )
    # FedSelect
    parser.add_argument(
        "--fedselect_warmup_rounds_frac",
        type=float,
        default=0.1,
        help="Fraction of rounds for warmup phase in FedSelect (default: 0.1).",
    )
    parser.add_argument(
        "--fedselect_client_select_ratio",
        type=float,
        default=0.4,
        help="Ratio of clients to select in each FedSelect round (default: 0.4).",
    )
    parser.add_argument(
        "--fedselect_sample_select_ratio",
        type=float,
        default=0.6,
        help="Ratio of samples to select per client in FedSelect (default: 0.6).",
    )
    parser.add_argument(
        "--fedselect_meta_momentum",
        type=float,
        default=0.9,
        help="Momentum for meta-margin computation in FedSelect (default: 0.9).",
    )
    parser.add_argument(
        "--fedselect_reward_data_size_frac",
        type=float,
        default=0.1,
        help="Fraction of proxy validation/reward dataset size in FedSelect (default: 0.1).",
    )
    parser.add_argument(
        "--fedselect_proxy_batch_size",
        type=int,
        default=None,
        help="Proxy validation batch size in FedSelect (default: None, uses trainer batch size).",
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
