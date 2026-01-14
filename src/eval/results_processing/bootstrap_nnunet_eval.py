import os
import json
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from typing import Dict, List, Union, Tuple
from batchgenerators.utilities.file_and_folder_operations import subfiles, join
from tqdm import tqdm
from nnunetv2.evaluation.evaluate_predictions import (
    compute_metrics,
    compute_tp_fp_fn_tn,
    region_or_label_to_mask,
)
from nnunetv2.imageio.reader_writer_registry import (
    determine_reader_writer_from_file_ending,
)


def bootstrap_evaluate(
    folder_ref: str,
    folder_pred: str,
    labels: Union[Tuple[int, ...], List[int]],
    ignore_label: int = None,
    num_bootstrap_iterations: int = 1000,
    file_ending: str = ".nii.gz",
) -> Dict:
    """
    Perform bootstrap evaluation on predictions.

    Args:
        folder_ref: Path to ground truth segmentations
        folder_pred: Path to predicted segmentations
        labels: List of labels to evaluate
        ignore_label: Label to ignore in evaluation
        num_bootstrap_iterations: Number of bootstrap samples (default 1000)
        file_ending: File extension to match

    Returns:
        Dictionary with bootstrap statistics (mean, std, ci_lower, ci_upper per metric per label)
    """

    # Get file list
    example_file = subfiles(folder_ref, join=True)[0]
    rw = determine_reader_writer_from_file_ending(
        file_ending, example_file, allow_nonmatching_filename=True, verbose=False
    )()

    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = [join(folder_ref, f) for f in files_pred]
    files_pred = [join(folder_pred, f) for f in files_pred]

    num_cases = len(files_pred)
    print(
        f"Bootstrapping on {num_cases} cases with {num_bootstrap_iterations} iterations..."
    )

    # Compute metrics for all cases once
    all_case_metrics = {}
    for ref_file, pred_file in tqdm(zip(files_ref, files_pred), total=num_cases, desc="Computing metrics per sample..."):
        case_name = Path(pred_file).stem
        metrics = compute_metrics(ref_file, pred_file, rw, labels, ignore_label)
        all_case_metrics[case_name] = metrics

    # Extract metric names from first case
    first_case_metrics = next(iter(all_case_metrics.values()))["metrics"]
    metric_names = list(first_case_metrics[labels[0]].keys())

    # Bootstrap sampling
    bootstrap_results = {
        label: {metric: [] for metric in metric_names} for label in labels
    }

    np.random.seed(42)  # for reproducibility
    for iteration in tqdm(range(num_bootstrap_iterations), desc="Bootstrap Sampling"):
        # Sample with replacement
        sampled_indices = np.random.choice(num_cases, size=num_cases, replace=True)
        sampled_cases = [list(all_case_metrics.keys())[i] for i in sampled_indices]

        # Compute mean metrics for this bootstrap sample
        for label in labels:
            for metric in metric_names:
                metric_values = []
                for case_name in sampled_cases:
                    val = all_case_metrics[case_name]["metrics"][label][metric]
                    if not np.isnan(val):
                        metric_values.append(val)

                if metric_values:
                    bootstrap_results[label][metric].append(np.mean(metric_values))

    # Compute statistics (95% CI)
    bootstrap_stats = {
        label: {metric: {} for metric in metric_names} for label in labels
    }
    for label in labels:
        for metric in metric_names:
            values = np.array(bootstrap_results[label][metric])
            bootstrap_stats[label][metric]["mean"] = float(np.mean(values))
            bootstrap_stats[label][metric]["std"] = float(np.std(values))
            bootstrap_stats[label][metric]["ci_lower"] = float(
                np.percentile(values, 2.5)
            )
            bootstrap_stats[label][metric]["ci_upper"] = float(
                np.percentile(values, 97.5)
            )

    print("Bootstrap evaluation complete!")
    print("\nBootstrap Results (95% CI):")
    for label in labels:
        print(f"\n  Label {label}:")
        for metric in metric_names:
            stats = bootstrap_stats[label][metric]
            print(
                f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            )

    # Append stats to bootstrap_results
    bootstrap_results["stats"] = bootstrap_stats

    # write results to file
    results_file = Path(folder_pred) / "bootstrap_evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(bootstrap_results, f, indent=4)
    print(f"\nBootstrap results saved to {results_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_id", type=str, help="Experiment ID of bootstrapped model."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-evaluation even if results exist.", default=False
    )
    args = parser.parse_args()

    # determine and define args
    nnunet_res = Path(os.getenv("nnUNet_results"))
    nnunet_preproc = Path(os.getenv("nnUNet_preprocessed"))
    
    # get exp folder
    exp_folders = glob(str(nnunet_res / "*" / "*" / "*" / f"D*_{args.exp_id}"))
    assert (
        len(exp_folders) > 0
    ), f"No experiment folder found for exp_id {args.exp_id} in nnUNet_results!"
    
    for exp_folder in exp_folders:
        # check if exp_folder already has bootstrap results
        if os.path.exists(Path(exp_folder) / "validation" / "bootstrap_evaluation_results.json") and not args.force:
            print(f"\nBootstrap results already exist for {exp_folder}. Skipping...")
            continue

        print(f"\nEvaluating experiment folder: {exp_folder}")
        # get dataset_id
        dataset_id = os.path.basename(exp_folder).split("_")[0].strip("D")

        # get preproc folder
        preproc_folder = glob(str(nnunet_preproc / f"Dataset{dataset_id}_*"))[0]
        assert (
            len(preproc_folder) > 0
        ), f"No preprocessed folder found for dataset_id {dataset_id} in nnUNet_preprocessed!"

        # load dataset.json
        dataset_json = json.load(open(Path(preproc_folder) / "dataset.json"))


        # set args
        folder_ref = Path(preproc_folder) / "gt_segmentations"
        folder_pred = Path(exp_folder) / "validation"
        labels = [int(l) for l in dataset_json["labels"].values() if int(l) != 0]  # exclude background
        ignore_label = dataset_json["ignore_label"] if "ignore_label" in dataset_json else None
        num_bootstrap_iterations = 1000
        file_ending = dataset_json["file_ending"]

        bootstrap_evaluate(
            folder_ref,
            folder_pred,
            labels=labels,
            ignore_label=ignore_label,
            num_bootstrap_iterations=num_bootstrap_iterations,
            file_ending=file_ending,
        )
