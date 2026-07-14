"""
Compute per-sample noise-type decisions from noise-analysis JSON files.

The script consumes the JSON files produced by analyze_noise_clean_noisy.py and
marks each sample as clean/noisy with respect to three noise-sensitive metrics:

- contour noise: HD95 > threshold * median object diagonal for the dataset
- instance noise: foreground-vs-background instance F1 < threshold
- class-confusion noise: class-confusion score > configured threshold(s)

Dataset median object diagonals are measured from the clean/expert masks with
compute_clean_object_diagonals.py and set below.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = "./results/noise_analysis/noise_analysis_results_*.json"
DEFAULT_OUTPUT_DIR = "./results/noise_analysis/noise_type_decisions"
DEFAULT_CONTOUR_FRACTIONS = (0.01, 0.05, 0.1)
DEFAULT_INSTANCE_F1_THRESHOLDS = (0.9, 0.99, 1.0)
DEFAULT_CONFUSION_THRESHOLDS = (0.01, 0.05, 0.1)

DATASET_BY_CLEAN_IDS = {
    "041": "LIDC",
    "300": "RIGA",
    "436": "Gleason",
    "500": "MouseTumor",
    "600": "MMIA",
    "700": "MMIS",
}

DATASET_MEDIAN_OBJECT_DIAGONAL = {
    "LIDC": 13.45362404707371,
    "RIGA": 204.35997645345176,
    "Gleason": 3762.265996740815,
    "MouseTumor": 18.429204661310052,
    "MMIA": 62.47277256164732,
    "MMIS": 79.51469684833486,
}


def is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.lower() in {
        "nan",
        "inf",
        "-inf",
        "infinity",
        "-infinity",
    }:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def to_float_or_none(value: Any) -> float | None:
    return float(value) if is_finite_number(value) else None


def resolve_json_files(input_path: str) -> list[Path]:
    if os.path.isdir(input_path):
        return sorted(Path(input_path).glob("*.json"))
    matches = sorted(Path(p) for p in glob.glob(input_path))
    if matches:
        return matches
    path = Path(input_path)
    if path.is_file():
        return [path]
    return []


def dataset_from_entry(entry: dict[str, Any], source_path: Path) -> str:
    clean_dataset_id = str(entry.get("clean_dataset_id", ""))
    if clean_dataset_id in DATASET_BY_CLEAN_IDS:
        return DATASET_BY_CLEAN_IDS[clean_dataset_id]

    match = re.search(r"clean([0-9]+)", source_path.stem)
    if match:
        first_id = match.group(1).split("-")[0]
        if first_id in DATASET_BY_CLEAN_IDS:
            return DATASET_BY_CLEAN_IDS[first_id]

    return source_path.stem


def compute_confusion_score(overlap_matrix: dict[str, Any], fg_classes: list[int]) -> float | None:
    """
    Compute foreground-to-foreground class confusion from an overlap matrix.

    For each clean foreground class, sum the fractions assigned to *other*
    foreground classes in the noisy mask, then macro-average over clean classes
    that are present in the sample. Transitions involving background are
    excluded in both directions.

    The score is undefined for datasets with fewer than two foreground classes.
    """
    foreground_classes = sorted(
        {int(class_id) for class_id in fg_classes if int(class_id) != 0}
    )
    if len(foreground_classes) < 2:
        return None

    per_source_confusion = []
    for source_class in foreground_classes:
        row = overlap_matrix.get(str(source_class), {})
        row_values = [to_float_or_none(value) for value in row.values()]
        if not any(value is not None and value > 0.0 for value in row_values):
            # The source class is absent from the clean/reference mask.
            continue

        confusion = 0.0
        for target_class in foreground_classes:
            if target_class == source_class:
                continue
            value = to_float_or_none(row.get(str(target_class)))
            if value is not None:
                confusion += value
        per_source_confusion.append(confusion)

    return float(np.mean(per_source_confusion)) if per_source_confusion else None


def sample_hd95(entry: dict[str, Any]) -> float | None:
    overall_hd95 = to_float_or_none(entry.get("overall_metrics", {}).get("mean_hd95"))
    if overall_hd95 is not None:
        # print(f"Using overall mean_hd95={overall_hd95} for sample {entry.get('sample_id')}")
        return overall_hd95

    values = []
    for metrics in entry.get("per_class_metrics", {}).values():
        value = to_float_or_none(metrics.get("hd95"))
        if value is not None:
            # print(f"Using per-class hd95={value} for sample {entry.get('sample_id')}")
            values.append(value)
    return float(np.mean(values)) if values else None


def sample_instance_f1(entry: dict[str, Any]) -> float | None:
    value = (
        entry.get("foreground_vs_background_metrics", {})
        .get("instance_level_prf", {})
        .get("f1")
    )
    return to_float_or_none(value)


def decision_column(prefix: str, comparator: str, threshold: float) -> str:
    return f"{prefix}_{comparator}_{threshold:g}".replace(".", "_")


def metric_summary(
    rows: list[dict[str, Any]],
    value_key: str,
    decision_key: str,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row[value_key] is not None]
    noisy_rows = [row for row in valid_rows if row[decision_key]]
    return {
        "n_total": len(rows),
        "n_valid": len(valid_rows),
        "n_noisy": len(noisy_rows),
        "prevalence_valid": len(noisy_rows) / len(valid_rows) if valid_rows else None,
        "prevalence_total": (
            len(noisy_rows) / len(rows) if rows and valid_rows else None
        ),
    }


def compute_decisions(
    json_files: list[Path],
    contour_fractions: list[float],
    instance_f1_thresholds: list[float],
    confusion_thresholds: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = []

    for json_file in json_files:
        with json_file.open("r") as f:
            data = json.load(f)

        for sample_id, entry in data.items():
            dataset = dataset_from_entry(entry, json_file)
            entries.append((json_file, sample_id, dataset, entry))

    thresholds = {}
    foreground_classes_by_dataset: dict[str, set[int]] = defaultdict(set)
    for _, _, dataset, entry in entries:
        foreground_classes_by_dataset[dataset].update(
            int(class_id)
            for class_id in entry.get("classes", {}).get("fg_classes", [])
            if int(class_id) != 0
        )

    for dataset in sorted({dataset for _, _, dataset, _ in entries}):
        median_diagonal = DATASET_MEDIAN_OBJECT_DIAGONAL.get(dataset)
        thresholds[dataset] = {
            "median_object_bbox_diagonal": median_diagonal,
            "contour_fraction_thresholds": contour_fractions,
            "hd95_contour_thresholds": {
                f"{fraction:g}": (
                    fraction * median_diagonal
                    if median_diagonal is not None
                    else None
                )
                for fraction in contour_fractions
            },
            "instance_fgbg_f1_thresholds": instance_f1_thresholds,
            "confusion_thresholds": confusion_thresholds,
            "class_confusion_foreground_classes": sorted(
                foreground_classes_by_dataset[dataset]
            ),
        }

    rows = []
    for json_file, sample_id, dataset, entry in entries:
        hd95_value = sample_hd95(entry)
        instance_f1_value = sample_instance_f1(entry)
        fg_classes = sorted(foreground_classes_by_dataset[dataset])
        confusion_value = compute_confusion_score(
            entry.get("class_overlap_matrix", {}),
            fg_classes,
        )

        row = {
            "dataset": dataset,
            "sample_id": sample_id,
            "source_json": json_file.name,
            "client_idx": entry.get("client_idx"),
            "clean_dataset_id": entry.get("clean_dataset_id"),
            "noisy_dataset_id": entry.get("noisy_dataset_id"),
            "hd95_value": hd95_value,
            "instance_fgbg_f1_value": instance_f1_value,
            "class_confusion_value": confusion_value,
            "any_noise": False,
        }

        hd95_thresholds = thresholds.get(dataset, {}).get("hd95_contour_thresholds", {})
        for fraction in contour_fractions:
            threshold_key = f"{fraction:g}"
            threshold_value = hd95_thresholds.get(threshold_key)
            row[f"hd95_threshold_frac_{threshold_key}".replace(".", "_")] = threshold_value
            decision_key = decision_column("contour_noisy_frac", "gt", fraction)
            noisy = (
                hd95_value is not None
                and threshold_value is not None
                and hd95_value > threshold_value
            )
            row[decision_key] = noisy
            row["any_noise"] = row["any_noise"] or noisy

        for threshold in instance_f1_thresholds:
            decision_key = decision_column("instance_noisy_f1", "lt", threshold)
            noisy = instance_f1_value is not None and instance_f1_value < threshold
            row[decision_key] = noisy
            row["any_noise"] = row["any_noise"] or noisy

        for threshold in confusion_thresholds:
            key = decision_column("class_confusion_noisy", "gt", threshold)
            noisy = confusion_value is not None and confusion_value > threshold
            row[key] = noisy
            row["any_noise"] = row["any_noise"] or noisy

        rows.append(row)

    metadata = {
        "thresholds_by_dataset": thresholds,
        "contour_fraction_thresholds": contour_fractions,
        "instance_fgbg_f1_thresholds": instance_f1_thresholds,
        "confusion_thresholds": confusion_thresholds,
        "class_confusion_definition": (
            "Macro-average over present clean foreground classes of the fraction "
            "assigned to other foreground classes; background transitions excluded."
        ),
        "summaries_by_dataset": {},
        "summary_all": {},
    }

    rows_by_dataset = defaultdict(list)
    for row in rows:
        rows_by_dataset[row["dataset"]].append(row)

    for dataset, dataset_rows in sorted(rows_by_dataset.items()):
        dataset_summary = {}
        for fraction in contour_fractions:
            decision_key = decision_column("contour_noisy_frac", "gt", fraction)
            dataset_summary[f"contour_frac_gt_{fraction:g}"] = metric_summary(
                dataset_rows,
                "hd95_value",
                decision_key,
            )
        for threshold in instance_f1_thresholds:
            decision_key = decision_column("instance_noisy_f1", "lt", threshold)
            dataset_summary[f"instance_f1_lt_{threshold:g}"] = metric_summary(
                dataset_rows,
                "instance_fgbg_f1_value",
                decision_key,
            )
        for threshold in confusion_thresholds:
            decision_key = decision_column("class_confusion_noisy", "gt", threshold)
            dataset_summary[f"class_confusion_gt_{threshold:g}"] = {
                **metric_summary(
                    dataset_rows,
                    "class_confusion_value",
                    decision_key,
                ),
            }
        metadata["summaries_by_dataset"][dataset] = dataset_summary

    all_rows = rows
    for fraction in contour_fractions:
        decision_key = decision_column("contour_noisy_frac", "gt", fraction)
        metadata["summary_all"][f"contour_frac_gt_{fraction:g}"] = metric_summary(
            all_rows,
            "hd95_value",
            decision_key,
        )
    for threshold in instance_f1_thresholds:
        decision_key = decision_column("instance_noisy_f1", "lt", threshold)
        metadata["summary_all"][f"instance_f1_lt_{threshold:g}"] = metric_summary(
            all_rows,
            "instance_fgbg_f1_value",
            decision_key,
        )
    for threshold in confusion_thresholds:
        decision_key = decision_column("class_confusion_noisy", "gt", threshold)
        metadata["summary_all"][f"class_confusion_gt_{threshold:g}"] = {
            **metric_summary(
                all_rows,
                "class_confusion_value",
                decision_key,
            ),
        }

    return rows, metadata


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-sample contour, instance, and class-confusion noise "
            "decisions from noise-analysis JSON files."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON path/glob/directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--contour-fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONTOUR_FRACTIONS),
        help=(
            "HD95 contour thresholds as fractions of median object diagonal; "
            "values above each threshold are marked noisy."
        ),
    )
    parser.add_argument(
        "--instance-f1-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_INSTANCE_F1_THRESHOLDS),
        help="Instance fg-bg F1 values below each threshold are marked noisy.",
    )
    parser.add_argument(
        "--confusion-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONFUSION_THRESHOLDS),
        help="Class-confusion thresholds; values above each threshold are marked noisy.",
    )
    args = parser.parse_args()

    json_files = resolve_json_files(args.input)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found for --input={args.input}")

    rows, metadata = compute_decisions(
        json_files=json_files,
        contour_fractions=args.contour_fractions,
        instance_f1_thresholds=args.instance_f1_thresholds,
        confusion_thresholds=args.confusion_thresholds,
    )

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "noise_type_decisions_per_sample.csv"
    json_path = output_dir / "noise_type_decisions_summary.json"
    write_csv(rows, csv_path)
    write_json({"metadata": metadata, "samples": rows}, json_path)

    print(f"Loaded {len(json_files)} JSON file(s).")
    print(f"Wrote per-sample decisions to {csv_path}")
    print(f"Wrote thresholds and prevalence summary to {json_path}")
    for dataset, summary in metadata["summaries_by_dataset"].items():
        parts = []
        for fraction in args.contour_fractions:
            key = f"contour_frac_gt_{fraction:g}"
            prevalence = summary[key]["prevalence_valid"]
            value = "n/a" if prevalence is None else f"{prevalence:.3f}"
            parts.append(f"contour>{fraction:g}diag={value}")
        for threshold in args.instance_f1_thresholds:
            key = f"instance_f1_lt_{threshold:g}"
            prevalence = summary[key]["prevalence_valid"]
            value = "n/a" if prevalence is None else f"{prevalence:.3f}"
            parts.append(f"instance<{threshold:g}={value}")
        print(f"{dataset}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
