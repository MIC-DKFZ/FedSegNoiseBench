import os
from pathlib import Path
import sys
import argparse
import json
import logging
from glob import glob

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
    nnunet_results_path = Path(os.getenv("nnUNet_results"))

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

    # load checkpoints from result folders to get args
    # determine whether to load latest or latest_t-1 checkpoint
    latest_checkpoints = glob(
        str(
            nnunet_results_path
            / "*"
            / "*"
            / "*"
            / f"D*_{exp_id}"
            / "checkpoint_latest.pth"
        )
    )
    latest_t_1_checkpoints = glob(
        str(
            nnunet_results_path
            / "*"
            / "*"
            / "*"
            / f"D*_{exp_id}"
            / "checkpoint_latest_t-1.pth"
        )
    )
    checkpoints = {}
    for i, folder in enumerate(result_folders):
        try:
            checkpoint_path = os.path.join(folder, "checkpoint_latest.pth")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoints[i] = {
                "path": checkpoint_path,
                "checkpoint": checkpoint,
            }
        except Exception as e:
            print(
                f"WARNING: Failed to load checkpoint_latest.pth for client {i} from {folder}"
            )
            print(f"Error: {e}")
            # Try to load checkpoint_latest_t-1.pth as fallback
            t_minus_1_path = os.path.join(folder, "checkpoint_latest_t-1.pth")
            if os.path.exists(t_minus_1_path):
                try:
                    checkpoint = torch.load(
                        t_minus_1_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    checkpoints[i] = {
                        "path": t_minus_1_path,
                        "checkpoint": checkpoint,
                    }
                    print(
                        f"Successfully loaded checkpoint_latest_t-1.pth for client {i} as fallback"
                    )
                except Exception as e2:
                    print(
                        f"ERROR: Also failed to load checkpoint_latest_t-1.pth for client {i}"
                    )
                    print(f"Error: {e2}")
                    raise RuntimeError(
                        f"Could not load any checkpoint for client {i}. "
                        f"Both checkpoint_latest.pth and checkpoint_latest_t-1.pth are corrupted or missing. "
                        f"Please check the checkpoint files or manually repair them."
                    )
            else:
                raise RuntimeError(
                    f"Could not load checkpoint_latest.pth for client {i} and checkpoint_latest_t-1.pth does not exist. "
                    f"Please check the checkpoint files at {folder}."
                )
    last_epochs = [ckpt["checkpoint"]["current_epoch"] for ckpt in checkpoints.values()]
    if not len(set(last_epochs)) == 1:
        # lowest last epoch across clients
        min_last_epoch = min(last_epochs)
        # load latest_t-1 for clients where last epoch > min_last_epoch
        for i, (curr_checkpoint_last_epoch, folder) in enumerate(
            zip(last_epochs, result_folders)
        ):
            if curr_checkpoint_last_epoch > min_last_epoch:
                path = os.path.join(folder, "checkpoint_latest_t-1.pth")
                checkpoints[i]["path"] = path
                checkpoints[i]["checkpoint"] = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )

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
        "oversample_foreground_percent",
        "class_sampling_probabilities",
        "batch_element_class_probabilities",
        "noise_ratio",
        "num_rounds",
        "num_local_epochs",
        "noise_mitigation_method",
        "feda3i_warmup_rounds_frac",
        "feda3i_interw",
        "feddm_gamma_hgd_smoothing",
        "feddm_ratio_cac_pixelselection",
        "feddm_cac_label_correction",
        "feddm_loss",
        "iopfl_alpha",
        "fedcorr_preproc_rounds_frac",
        "fedcorr_relabel_ratio",
        "fedcorr_relabel_confidence_thres",
        "fedcorr_proxterm_beta",
        "fedselect_warmup_rounds_frac",
        "fedselect_client_select_ratio",
        "fedselect_sample_select_ratio",
        "fedselect_meta_momentum",
        "fedselect_reward_data_size_frac",
        "fl_strategy_state",
    ]
    (
        configuration,
        fold,
        plan,
        trainer,
        save_every,
        oversample_foreground_percent,
        class_sampling_probabilities,
        batch_element_class_probabilities,
        noise_ratio,
        num_rounds,
        num_local_epochs,
        noise_mitigation_method,
        feda3i_warmup_rounds_frac,
        feda3i_interw,
        feddm_gamma_hgd_smoothing,
        feddm_ratio_cac_pixelselection,
        feddm_cac_label_correction,
        feddm_loss,
        iopfl_alpha,
        fedcorr_preproc_rounds_frac,
        fedcorr_relabel_ratio,
        fedcorr_relabel_confidence_thres,
        fedcorr_proxterm_beta,
        fedselect_warmup_rounds_frac,
        fedselect_client_select_ratio,
        fedselect_sample_select_ratio,
        fedselect_meta_momentum,
        fedselect_reward_data_size_frac,
        fl_strategy_state,
    ) = [exp_cli_args.get(k) for k in keys]

    # Ensure fl_strategy_state is a dict (handle string-serialized JSON if present)
    if isinstance(fl_strategy_state, str):
        try:
            fl_strategy_state = json.loads(fl_strategy_state)
        except Exception:
            logging.warning(
                f"Failed to parse fl_strategy_state as JSON: {fl_strategy_state}"
            )
            fl_strategy_state = None

    # get last checkpoint epoch to continue training from there
    last_epochs = [ckpt["checkpoint"]["current_epoch"] for ckpt in checkpoints.values()]
    assert (
        len(set(last_epochs)) == 1
    ), "All clients must have the same last epoch to restart the experiment!"
    start_epoch = last_epochs[0]
    start_fl_round = start_epoch // num_local_epochs - 1

    # all checks passed, actually rename "checkpoint_latest_t-1.pth" to "checkpoint_latest.pth" where needed
    for i, ckpt_info in checkpoints.items():
        if os.path.basename(ckpt_info["path"]) == "checkpoint_latest_t-1.pth":
            # backup old latest checkpoint
            latest_path = os.path.join(
                os.path.dirname(ckpt_info["path"]), "checkpoint_latest.pth"
            )
            backup_path = os.path.join(
                os.path.dirname(ckpt_info["path"]), "checkpoint_latest_backup.pth"
            )
            os.replace(latest_path, backup_path)
            print(
                f"Backed up latest checkpoint for client {i} to 'checkpoint_latest_backup.pth'"
            )
            # rename t-1 to latest
            new_path = os.path.join(
                os.path.dirname(ckpt_info["path"]), "checkpoint_latest.pth"
            )
            os.replace(ckpt_info["path"], new_path)
            print(
                f"Renamed checkpoint for client {i} from 'checkpoint_latest_t-1.pth' to 'checkpoint_latest.pth'"
            )

    # return all extracted args
    return (
        dataset_ids,
        configuration,
        fold,
        plan,
        trainer,
        save_every,
        oversample_foreground_percent,
        class_sampling_probabilities,
        batch_element_class_probabilities,
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
        feddm_loss,
        iopfl_alpha,
        fedcorr_preproc_rounds_frac,
        fedcorr_relabel_ratio,
        fedcorr_relabel_confidence_thres,
        fedcorr_proxterm_beta,
        fedselect_warmup_rounds_frac,
        fedselect_client_select_ratio,
        fedselect_sample_select_ratio,
        fedselect_meta_momentum,
        fedselect_reward_data_size_frac,
        start_epoch,
        start_fl_round,
        fl_strategy_state,
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
        oversample_foreground_percent,
        class_sampling_probabilities,
        batch_element_class_probabilities,
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
        feddm_loss,
        iopfl_alpha,
        fedcorr_preproc_rounds_frac,
        fedcorr_relabel_ratio,
        fedcorr_relabel_confidence_thres,
        fedcorr_proxterm_beta,
        fedselect_warmup_rounds_frac,
        fedselect_client_select_ratio,
        fedselect_sample_select_ratio,
        fedselect_meta_momentum,
        fedselect_reward_data_size_frac,
        start_epoch,
        start_fl_round,
        fl_strategy_state,
    ) = get_experiment_args(args.exp_id)

    print(f"Restarting experiment '{args.exp_id}' with the following args:")
    print(
        f"{dataset_ids=}\n{configuration=}\n{fold=}\n{plan=}\n{trainer=}\n{save_every=}\n{oversample_foreground_percent=}\n{class_sampling_probabilities=}\n{batch_element_class_probabilities=}\n{noise_ratio=}\n{num_clients=}\n{num_rounds=}\n{num_local_epochs=}\n{clean_validation_datasets=}\n{noisy_train_folders=}\n{noise_mitigation_method=}\n{feda3i_warmup_rounds_frac=}\n{feda3i_interw=}\n{feddm_gamma_hgd_smoothing=}\n{feddm_ratio_cac_pixelselection=}\n{feddm_cac_label_correction=}\n{feddm_loss=}\n{iopfl_alpha=}\n{fedcorr_preproc_rounds_frac=}\n{fedcorr_relabel_ratio=}\n{fedcorr_relabel_confidence_thres=}\n{fedcorr_proxterm_beta=}\n{fedselect_warmup_rounds_frac=}\n{fedselect_client_select_ratio=}\n{fedselect_sample_select_ratio=}\n{fedselect_meta_momentum=}\n{fedselect_reward_data_size_frac=}\n{start_epoch=}\n{start_fl_round=}\n"
    )

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
                "oversample_foreground_percent": oversample_foreground_percent,
                "class_sampling_probabilities": class_sampling_probabilities,
                "batch_element_class_probabilities": batch_element_class_probabilities,
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
                "num_rounds": num_rounds,  #  - start_fl_round,
                "start_epoch": start_epoch,
            },
        )
        for i in range(num_clients)
    ]

    # setup orchestrator
    orchestrator = Orchestrator(
        clients,
        fl_args={
            "num_rounds": num_rounds,  #  - start_fl_round,
            "start_fl_round": start_fl_round,
            "strategy": noise_mitigation_method,
            # FedA3I
            "feda3i_warmup_rounds_frac": (
                feda3i_warmup_rounds_frac
                if noise_mitigation_method.lower() == "feda3i"
                else None
            ),
            "feda3i_interw": (
                feda3i_interw if noise_mitigation_method.lower() == "feda3i" else None
            ),
            # FedDM
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
            "feddm_loss": (
                feddm_loss if noise_mitigation_method.lower() == "feddm" else None
            ),
            # IOP-FL
            "iopfl_alpha": (
                iopfl_alpha if noise_mitigation_method.lower() == "iopfl" else None
            ),
            # FedCorr
            "fedcorr_preproc_rounds_frac": (
                fedcorr_preproc_rounds_frac
                if noise_mitigation_method.lower() == "fedcorr"
                else None
            ),
            "fedcorr_relabel_ratio": (
                fedcorr_relabel_ratio
                if noise_mitigation_method.lower() == "fedcorr"
                else None
            ),
            "fedcorr_relabel_confidence_thres": (
                fedcorr_relabel_confidence_thres
                if noise_mitigation_method.lower() == "fedcorr"
                else None
            ),
            "fedcorr_proxterm_beta": (
                fedcorr_proxterm_beta
                if noise_mitigation_method.lower() == "fedcorr"
                else None
            ),
            # FedSelect
            "fedselect_warmup_rounds_frac": (
                fedselect_warmup_rounds_frac
                if noise_mitigation_method.lower() == "fedselect"
                else None
            ),
            "fedselect_client_select_ratio": (
                fedselect_client_select_ratio
                if noise_mitigation_method.lower() == "fedselect"
                else None
            ),
            "fedselect_sample_select_ratio": (
                fedselect_sample_select_ratio
                if noise_mitigation_method.lower() == "fedselect"
                else None
            ),
            "fedselect_meta_momentum": (
                fedselect_meta_momentum
                if noise_mitigation_method.lower() == "fedselect"
                else None
            ),
            "fedselect_reward_data_size_frac": (
                fedselect_reward_data_size_frac
                if noise_mitigation_method.lower() == "fedselect"
                else None
            ),
            # FL strategy state implemented for FedCorr, FedA3I, IOPFL, FedSelect
            "fl_strategy_state": (
                fl_strategy_state
                if noise_mitigation_method.lower()
                in ["fedcorr", "feda3i", "iopfl", "fedselect"]
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
