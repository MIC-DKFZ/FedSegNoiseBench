import os
import sys
import argparse
import json

import torch

# Add src to PYTHONPATH automatically if it's not there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)
print(f"{src_path=}")
print(f"{sys.path=}")

from fed.client import Client
from fed.orchestrator import Orchestrator


def get_experiment_args(exp_id: str):
    """
    Get experiment arguments from a previous experiment given its experiment ID.

    Load via exp_id and nnUNet_results path the experiment arguments used in the
    previous experiment and its checkpoint.
    Extract and return all relevant arguments needed to restart the experiment.
    """
    # get nnUNet_results path
    nnunet_results_path = os.getenv("nnUNet_results")

    # find result folder ending with exp_id
    result_folders = [
        root
        for root, dirs, files in os.walk(nnunet_results_path)
        if os.path.basename(root).endswith(exp_id)
    ]
    # fallback to immediate children (compat with previous behavior)
    if not result_folders:
        result_folders = [
            os.path.join(nnunet_results_path, folder)
            for folder in os.listdir(nnunet_results_path)
            if folder.endswith(exp_id)
        ]
    if not result_folders:
        raise FileNotFoundError(
            f"No result folders ending with '{exp_id}' found under {nnunet_results_path}"
        )

    # get experiment cli args file
    exp_cli_args_file = os.path.join(
        nnunet_results_path, f"ExperimentArgs_{exp_id}.json"
    )
    if os.path.exists(exp_cli_args_file):
        exp_cli_args = json.loads(open(exp_cli_args_file, "r").read())

    # load checkpoint_latest.pth from result folders to get args
    checkpoints = [
        torch.load(
            os.path.join(folder, "checkpoint_latest.pth"),
            map_location="cpu",
            weights_only=False,
        )
        for folder in result_folders
    ]

    # get number of clients
    num_clients = len(checkpoints)

    # compact, robust extraction of experiment arguments (normalize strings to lists and provide defaults)
    raw_dataset_ids = exp_cli_args.get("dataset_ids", "")
    dataset_ids = (
        raw_dataset_ids.split()
        if isinstance(raw_dataset_ids, str)
        else list(raw_dataset_ids or [])
    )
    raw_clean_validation_datasets = exp_cli_args.get("clean_validation_dataset", None)
    clean_validation_datasets = (
        raw_clean_validation_datasets.split()
        if isinstance(raw_clean_validation_datasets, str)
        else list(raw_clean_validation_datasets or [])
    )
    raw_noisy_train_folders = exp_cli_args.get("noisy_train_folder", None)
    noisy_train_folders = (
        raw_noisy_train_folders.split()
        if isinstance(raw_noisy_train_folders, str)
        else list(raw_noisy_train_folders or [])
    )
    keys = [
        "configuration",
        "fold",
        "plan",
        "trainer",
        "save_every",
        "noise_ratio",
        "num_rounds",
        "num_local_epochs",
        "noise_mitigation_method",
        "feda3i_warmup_rounds_frac",
        "feda3i_interw",
        "feddm_gamma_hgd_smoothing",
        "feddm_ratio_cac_pixelselection",
        "feddm_cac_label_correction",
        "iopfl_alpha",
    ]
    (
        configuration,
        fold,
        plan,
        trainer,
        save_every,
        noise_ratio,
        num_rounds,
        num_local_epochs,
        noise_mitigation_method,
        feda3i_warmup_rounds_frac,
        feda3i_interw,
        feddm_gamma_hgd_smoothing,
        feddm_ratio_cac_pixelselection,
        feddm_cac_label_correction,
        iopfl_alpha,
    ) = [exp_cli_args.get(k) for k in keys]

    # get last checkpoint epoch to continue training from there
    last_epochs = [ckpt["current_epoch"] for ckpt in checkpoints]
    assert (
        len(set(last_epochs)) == 1
    ), "All clients must have the same last epoch to restart the experiment!"
    start_epoch = last_epochs[0]
    start_fl_round = start_epoch // num_local_epochs - 1

    # return all extracted args
    return (
        dataset_ids,
        configuration,
        fold,
        plan,
        trainer,
        save_every,
        noise_ratio,
        num_clients,
        num_rounds,
        num_local_epochs,
        clean_validation_datasets,
        noisy_train_folders,
        noise_mitigation_method,
        feda3i_warmup_rounds_frac,
        feda3i_interw,
        feddm_gamma_hgd_smoothing,
        feddm_ratio_cac_pixelselection,
        feddm_cac_label_correction,
        iopfl_alpha,
        start_epoch,
        start_fl_round,
    )


def main(args):

    # get args from experiment to restart
    (
        dataset_ids,
        configuration,
        fold,
        plan,
        trainer,
        save_every,
        noise_ratio,
        num_clients,
        num_rounds,
        num_local_epochs,
        clean_validation_datasets,
        noisy_train_folders,
        noise_mitigation_method,
        feda3i_warmup_rounds_frac,
        feda3i_interw,
        feddm_gamma_hgd_smoothing,
        feddm_ratio_cac_pixelselection,
        feddm_cac_label_correction,
        iopfl_alpha,
        start_epoch,
        start_fl_round,
    ) = get_experiment_args(args.exp_id)

    # setup clients
    clients = [
        Client(
            client_id=i,
            model_args={
                "dataset_id": dataset_ids[i],
                "configuration": configuration,
                "fold": fold,
                "plan": plan,
                "trainer": trainer,
                "save_every": save_every,
                "continue_training": True,
                "clean_validation_dataset": (
                    clean_validation_datasets[i] if clean_validation_datasets else None
                ),
                "experiment_id": f"D{dataset_ids[i]}_{args.exp_id}",
                "noisy_train_folder": (
                    noisy_train_folders[i] if noisy_train_folders else None
                ),
                "noise_ratio": noise_ratio,
            },
            fl_args={
                "num_local_epochs": num_local_epochs,
                "num_rounds": num_rounds - start_fl_round,
                "start_epoch": start_epoch,
            },
        )
        for i in range(num_clients)
    ]

    # setup orchestrator
    orchestrator = Orchestrator(
        clients,
        fl_args={
            "num_rounds": num_rounds - start_fl_round,
            "start_fl_round": start_fl_round,
            "strategy": noise_mitigation_method,
            "feda3i_warmup_rounds_frac": (
                feda3i_warmup_rounds_frac
                if noise_mitigation_method.lower() == "feda3i"
                else None
            ),
            "feda3i_interw": (
                feda3i_interw if noise_mitigation_method.lower() == "feda3i" else None
            ),
            "feddm_gamma_hgd_smoothing": (
                feddm_gamma_hgd_smoothing
                if noise_mitigation_method.lower() == "feddm"
                else None
            ),
            "feddm_ratio_cac_pixelselection": (
                feddm_ratio_cac_pixelselection
                if noise_mitigation_method.lower() == "feddm"
                else None
            ),
            "feddm_cac_label_correction": (
                feddm_cac_label_correction
                if noise_mitigation_method.lower() == "feddm"
                else None
            ),
            "iopfl_alpha": (
                iopfl_alpha
                if noise_mitigation_method.lower() == "iopfl"
                else None
            ),
        },
    )

    # run federated learning
    orchestrator.fl_run()


if __name__ == "__main__":
    # take CLI arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_id",
        type=str,
        default="",
        help="Experiment ID of the experiment to restart.",
    )

    args = parser.parse_args()
    main(args)
