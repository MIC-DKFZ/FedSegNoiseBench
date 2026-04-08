import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from nnunetv2.evaluation.evaluate_predictions import compute_metrics
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_file_ending
from tqdm import tqdm


def _normalize_label_key(label: Union[int, str]) -> str:
    return str(label)


def _load_existing_bootstrap_results(results_file: Path) -> Dict:
    if not results_file.is_file():
        return {}
    with open(results_file, "r") as f:
        return json.load(f)


def _metric_vector_is_complete(
    existing_results: Dict,
    label: Union[int, str],
    metric_name: str,
    expected_iterations: int,
) -> bool:
    label_key = _normalize_label_key(label)
    label_metrics = existing_results.get(label_key, {})
    metric_values = label_metrics.get(metric_name)
    if not isinstance(metric_values, list):
        return False
    if len(metric_values) != expected_iterations:
        return False

    stats = existing_results.get("stats", {}).get(label_key, {}).get(metric_name)
    return isinstance(stats, dict) and all(
        key in stats for key in ("mean", "std", "ci_lower", "ci_upper")
    )


def _infer_missing_metrics(
    existing_results: Dict,
    labels: List[int],
    available_metric_names: List[str],
    expected_iterations: int,
) -> Set[str]:
    missing_metrics: Set[str] = set()
    for label in labels:
        for metric_name in available_metric_names:
            if not _metric_vector_is_complete(
                existing_results, label, metric_name, expected_iterations
            ):
                missing_metrics.add(metric_name)
    return missing_metrics


def _compute_bootstrap_stats(bootstrap_results: Dict[str, Dict[str, List[float]]]) -> Dict:
    bootstrap_stats = {
        label: {metric: {} for metric in metrics}
        for label, metrics in bootstrap_results.items()
    }

    for label, metrics in bootstrap_results.items():
        for metric_name, values in metrics.items():
            values_arr = np.asarray(values, dtype=float)
            if values_arr.size == 0:
                bootstrap_stats[label][metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
                continue

            bootstrap_stats[label][metric_name] = {
                "mean": float(np.nanmean(values_arr)),
                "std": float(np.nanstd(values_arr)),
                "ci_lower": float(np.nanpercentile(values_arr, 2.5)),
                "ci_upper": float(np.nanpercentile(values_arr, 97.5)),
            }

    return bootstrap_stats


def _merge_bootstrap_results(existing_results: Dict, new_results: Dict) -> Dict:
    merged = dict(existing_results) if existing_results else {}

    for label_key, metrics in new_results.items():
        if label_key == "stats":
            continue
        merged.setdefault(label_key, {})
        merged[label_key].update(metrics)

    merged.setdefault("stats", {})
    for label_key, metric_stats in new_results.get("stats", {}).items():
        merged["stats"].setdefault(label_key, {})
        merged["stats"][label_key].update(metric_stats)

    return merged


def _print_bootstrap_stats(bootstrap_stats: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\nBootstrap Results (95% CI):")
    for label_key, metrics in bootstrap_stats.items():
        print(f"\n  Label {label_key}:")
        for metric_name, stats in metrics.items():
            print(
                f"    {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            )


def bootstrap_evaluate(
    folder_ref: str,
    folder_pred: str,
    labels: Union[Tuple[int, ...], List[int]],
    ignore_label: int = None,
    num_bootstrap_iterations: int = 1000,
    file_ending: str = ".nii.gz",
    force: bool = False,
) -> Dict:
    """
    Perform bootstrap evaluation on predictions.

    Default behavior is incremental: if bootstrap results already exist, only
    missing metrics are recomputed and merged into the existing JSON. With
    --force, all metrics are recomputed from scratch.
    """

    labels = [int(label) for label in labels]
    results_file = Path(folder_pred) / "bootstrap_evaluation_results.json"
    existing_results = {} if force else _load_existing_bootstrap_results(results_file)

    example_file = subfiles(folder_ref, join=True)[0]
    rw = determine_reader_writer_from_file_ending(
        file_ending, example_file, allow_nonmatching_filename=True, verbose=False
    )()

    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = [join(folder_ref, f) for f in files_pred]
    files_pred = [join(folder_pred, f) for f in files_pred]

    num_cases = len(files_pred)
    if num_cases == 0:
        raise RuntimeError(f"No prediction files found in {folder_pred}")

    print(
        f"Bootstrapping on {num_cases} cases with {num_bootstrap_iterations} iterations..."
    )

    first_case_metrics = compute_metrics(
        files_ref[0],
        files_pred[0],
        rw,
        labels,
        ignore_label,
    )["metrics"]
    available_metric_names = list(first_case_metrics[labels[0]].keys())

    metrics_to_compute = set(available_metric_names)
    if not force and existing_results:
        metrics_to_compute = _infer_missing_metrics(
            existing_results,
            labels,
            available_metric_names,
            num_bootstrap_iterations,
        )
        if metrics_to_compute:
            print(
                "Existing bootstrap results found. Recomputing only missing metrics: "
                + ", ".join(sorted(metrics_to_compute))
            )
        else:
            print(
                f"\nBootstrap results already complete for {folder_pred}. Skipping..."
            )
            return existing_results
    elif force and results_file.is_file():
        print("Force mode enabled. Recomputing all metrics from scratch.")

    all_case_metrics = {}
    requested_metrics = set(metrics_to_compute) if metrics_to_compute else None
    for ref_file, pred_file in tqdm(
        zip(files_ref, files_pred),
        total=num_cases,
        desc="Computing metrics per sample...",
    ):
        case_name = Path(pred_file).stem
        metrics = compute_metrics(
            ref_file,
            pred_file,
            rw,
            labels,
            ignore_label,
            requested_metrics=requested_metrics,
        )
        all_case_metrics[case_name] = metrics

    bootstrap_results = {
        _normalize_label_key(label): {metric: [] for metric in sorted(metrics_to_compute)}
        for label in labels
    }

    case_names = list(all_case_metrics.keys())
    np.random.seed(42)
    for _ in tqdm(range(num_bootstrap_iterations), desc="Bootstrap Sampling"):
        sampled_indices = np.random.choice(num_cases, size=num_cases, replace=True)
        sampled_cases = [case_names[i] for i in sampled_indices]

        for label in labels:
            label_key = _normalize_label_key(label)
            for metric_name in metrics_to_compute:
                metric_values = []
                for case_name in sampled_cases:
                    val = all_case_metrics[case_name]["metrics"][label].get(metric_name)
                    if val is None:
                        continue
                    if not np.isnan(val):
                        metric_values.append(val)

                if metric_values:
                    bootstrap_results[label_key][metric_name].append(
                        float(np.mean(metric_values))
                    )

    bootstrap_stats = _compute_bootstrap_stats(bootstrap_results)
    _print_bootstrap_stats(bootstrap_stats)

    new_results = dict(bootstrap_results)
    new_results["stats"] = bootstrap_stats
    merged_results = new_results if force else _merge_bootstrap_results(existing_results, new_results)

    with open(results_file, "w") as f:
        json.dump(merged_results, f, indent=4)
    print(f"\nBootstrap results saved to {results_file}")

    return merged_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, help="Experiment ID of bootstrapped model.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full re-evaluation of all bootstrap metrics, even if results exist.",
        default=False,
    )
    args = parser.parse_args()

    nnunet_res = Path(os.getenv("nnUNet_results"))
    nnunet_preproc = Path(os.getenv("nnUNet_preprocessed"))

    exp_folders = glob(str(nnunet_res / "*" / "*" / "*" / f"D*_{args.exp_id}"))
    assert len(exp_folders) > 0, (
        f"No experiment folder found for exp_id {args.exp_id} in nnUNet_results!"
    )

    for exp_folder in exp_folders:
        print(f"\nEvaluating experiment folder: {exp_folder}")
        dataset_id = os.path.basename(exp_folder).split("_")[0].strip("D")

        preproc_folder = glob(str(nnunet_preproc / f"Dataset{dataset_id}_*"))[0]
        assert len(preproc_folder) > 0, (
            f"No preprocessed folder found for dataset_id {dataset_id} in nnUNet_preprocessed!"
        )

        with open(Path(preproc_folder) / "dataset.json", "r") as f:
            dataset_json = json.load(f)

        folder_ref = Path(preproc_folder) / "gt_segmentations"
        folder_pred = Path(exp_folder) / "validation"
        labels = [
            int(l) for l in dataset_json["labels"].values() if int(l) != 0
        ]
        ignore_label = dataset_json.get("ignore_label")
        num_bootstrap_iterations = 1000
        file_ending = dataset_json["file_ending"]

        bootstrap_evaluate(
            folder_ref,
            folder_pred,
            labels=labels,
            ignore_label=ignore_label,
            num_bootstrap_iterations=num_bootstrap_iterations,
            file_ending=file_ending,
            force=args.force,
        )
