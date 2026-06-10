import os
import argparse
import subprocess
import glob
import json
import numpy as np

def execute_command(command):
    """
    Execute a command.
    """
    print(f"Executing command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")

def average_dicts(dicts):
    keys = dicts[0].keys()  # Get keys from the first dictionary
    averaged_dict = {}
    for key in keys:
        values = [d[key] for d in dicts]  # Extract values for each key
        averaged_dict[key] = float(np.mean(values))  # Compute mean
    return averaged_dict

def aggregate_fingerprints(dataset_ids, fp_aggregation):
    """
    Aggregate fingerprints according to aggregation strategy.
    """
    # load dataset_fingerprint.json from dataset_id's nnUNet_preprocessed folder
    nnunet_preprocessed_folder = os.getenv("nnUNet_preprocessed")
    assert nnunet_preprocessed_folder, "nnUNet_preprocessed folder not found!"
    dataset_fingerprints = []
    dataset_fingerprint_fnames = []
    for dataset_id in dataset_ids.split():
        dataset_fingerprint_fname = glob.glob(
            os.path.join(
                nnunet_preprocessed_folder,
                f"Dataset{dataset_id}_*",
                "dataset_fingerprint.json"
            )
        )
        assert dataset_fingerprint_fname, f"dataset_fingerprint.json not found for dataset_id {dataset_id}!"
        dataset_fingerprints.append(json.load(open(dataset_fingerprint_fname[0])))
        dataset_fingerprint_fnames.append(dataset_fingerprint_fname[0])

    # aggregate fingerprints
    if fp_aggregation == "mean":
        average_fingerprint = {
            "foreground_intensity_properties_per_channel": {
                channel: average_dicts(
                    [fp["foreground_intensity_properties_per_channel"][channel] for fp in dataset_fingerprints]
                )
                for channel in dataset_fingerprints[0]["foreground_intensity_properties_per_channel"]
            },
            "median_relative_size_after_cropping": float(np.mean([
                fp["median_relative_size_after_cropping"] for fp in dataset_fingerprints
            ])),
            "shapes_after_crop": [shapes for fp in dataset_fingerprints for shapes in fp["shapes_after_crop"]],
            "spacings": [spacings for fp in dataset_fingerprints for spacings in fp["spacings"]]
        }
    else:
        raise ValueError(f"Unknown fingerprint aggregation strategy {fp_aggregation}!")

    # save old and new fingerprints
    for dataset_fingerprint_fname in dataset_fingerprint_fnames:
        # rename original fingerprint to dataset_fingerprint_org.json to keep it
        os.rename(dataset_fingerprint_fname, dataset_fingerprint_fname.replace("dataset_fingerprint.json", "dataset_fingerprint_org.json"))
        # save aggregated fingerprint to nnUNet_preprocessed folder as "dataset_fingerprint.json"
        with open(dataset_fingerprint_fname, "w") as f:
            json.dump(average_fingerprint, f)


def main(args):
    """
    Script to extract nnUNet fingerprints from each FL clients nnUNet_raw data,
    and jointly plan the nnUNet configuration and preprocess the FL clients nnUNet_raw data.
    """
    # CLI execution of nnUNetv2_extract_fingerprint
    for dataset_id in args.dataset_ids.split():
        command = [
            "nnUNetv2_extract_fingerprint",
            "-d",
            f"{int(dataset_id)}",
            "-np",
            f"{args.num_processes}",
            "--verify_dataset_integrity"
        ]
        execute_command(command)

    # aggregation of extracted fingerprints
    aggregate_fingerprints(args.dataset_ids, args.fp_aggregation)

    # CLI execution of nnUNetv2_plan_experiment
    for dataset_id in args.dataset_ids.split():
        command = [
            "nnUNetv2_plan_experiment",
            "-d",
            f"{int(dataset_id)}",
            "-pl",
            f"{args.planner}",
        ]
        execute_command(command)

    # CLI execution of nnUNetv2_preprocess
    for dataset_id in args.dataset_ids.split():
        command = [
            "nnUNetv2_preprocess",
            "-d",
            f"{int(dataset_id)}",
            "-plans_name",
            f"{args.plans_name}",
            "-c",
            f"{args.configuration}",
            "np",
            f"{args.num_processes}",
        ]
        execute_command(command)

if __name__=="__main__":
    # take CLI arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_ids",
        type=str,
        default="",
        help="nnUNet_raw Dataset IDs to jointly plan and preprocess.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        help="nnUNet configuration to plan: '', '3d_fullres', '3d_lowres', '2d', '3d_cascade_fullres'.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="ExperimentPlanner",
        help="nnUNet plan to plan preproces from: 'ExperimentPlanner', 'nnUNetPlannerResEncM', 'nnUNetPlannerResEncL', 'nnUNetPlannerResEncXL'.",
    )
    parser.add_argument(
        "--plans_name",
        type=str,
        default="nnUNetPlans",
        help="Name of the plans folder.",
    )
    parser.add_argument(
        "--verify_dataset_integrity",
        action="store_true",
        help="Verify dataset integrity.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=10,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--fp_aggregation",
        type=str,
        default="mean",
        help="Fingerprint aggregation strategy: 'mean'.",
    )
    args = parser.parse_args()

    main(args)