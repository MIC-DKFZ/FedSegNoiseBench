"""
Compute per-sample noise-type decisions from noise-analysis JSON files.

The script consumes the JSON files produced by analyze_noise_clean_noisy.py and
marks each sample as clean/noisy with respect to three selected noise metrics:

- contour noise: per-sample normalized HD95 (nHD95) > threshold, where
  nHD95_i = HD95_i / D_i and D_i is the diagonal of the bounding box around
  the foreground object in sample i's clean/expert mask. This makes the
  contour-noise decision scale-invariant per case instead of relying on a
  single dataset-wide diagonal.
- instance noise: foreground-vs-background instance F1 < threshold
- instance class-confusion noise: InstanceClsConf > configured threshold(s)

PixelClsConf is retained as a diagnostic but does not contribute to the
combined ``any_noise`` assignment.

Per-sample object bounding-box diagonals (D_i) are measured from the
clean/expert masks with compute_clean_object_diagonals.py, which writes them
to the CSV path in DEFAULT_OBJECT_DIAGONALS_CSV. When a sample has multiple
foreground classes (e.g. nested disc/cup), D_i is taken as the largest
per-class bbox diagonal for that sample, since the union bbox of nested
foreground regions equals the bbox of the outermost region.
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
DEFAULT_OBJECT_DIAGONALS_CSV = (
    "./results/noise_analysis/clean_object_diagonals/clean_object_bbox_diagonals.csv"
)
DEFAULT_CONTOUR_FRACTIONS = (0.05,)
DEFAULT_INSTANCE_F1_THRESHOLDS = (1.0,)
DEFAULT_PIXEL_CLS_CONF_THRESHOLDS = (0.05,)
DEFAULT_INSTANCE_CLS_CONF_THRESHOLDS = (0.0,)

DATASET_BY_CLEAN_IDS = {
    "041": "LIDC",
    "300": "RIGA",
    "436": "Gleason",
    "500": "MouseTumor",
    "600": "MMIA",
    "700": "MMIS",
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


def load_object_bbox_diagonals(csv_path: Path) -> dict[tuple[str, str], float]:
    """
    Load per-sample foreground object bbox diagonals (D_i) from the CSV
    written by compute_clean_object_diagonals.py.

    Samples with multiple foreground classes (e.g. nested disc/cup) get one
    row per class; D_i is taken as the largest per-class bbox diagonal, which
    equals the bbox diagonal of the union of nested foreground regions.
    """
    diagonals_by_sample: dict[tuple[str, str], float] = {}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            diagonal = to_float_or_none(row.get("bbox_diagonal"))
            if diagonal is None:
                continue
            key = (row["dataset"], row["sample_id"])
            existing = diagonals_by_sample.get(key)
            if existing is None or diagonal > existing:
                diagonals_by_sample[key] = diagonal
    return diagonals_by_sample


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


def compute_pixel_cls_conf_from_overlap(
    overlap_matrix: dict[str, Any], fg_classes: list[int]
) -> float | None:
    """
    Compute foreground-to-foreground class confusion from an overlap matrix.

    For each clean foreground class, sum the fractions assigned to *other*
    foreground classes in the noisy mask, then macro-average over clean classes
    that are present in the sample. Transitions involving background are
    excluded in both directions.

    This is retained as a compatibility fallback for analysis JSON files made
    before PixelClsConf was persisted explicitly.
    """
    foreground_classes = sorted(
        {int(class_id) for class_id in fg_classes if int(class_id) != 0}
    )
    if not foreground_classes:
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


def sample_cls_conf(
    entry: dict[str, Any], metric_name: str, fg_classes: list[int]
) -> float | None:
    value = to_float_or_none(
        entry.get("class_confusion_metrics", {})
        .get(metric_name, {})
        .get("score")
    )
    if value is not None:
        return value
    if metric_name == "PixelClsConf":
        return compute_pixel_cls_conf_from_overlap(
            entry.get("class_overlap_matrix", {}), fg_classes
        )
    return None


def sample_instance_cls_conf_coverage(entry: dict[str, Any]) -> float | None:
    return to_float_or_none(
        entry.get("class_confusion_metrics", {})
        .get("InstanceClsConf", {})
        .get("coverage")
    )


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
    object_bbox_diagonals: dict[tuple[str, str], float],
    contour_fractions: list[float],
    instance_f1_thresholds: list[float],
    pixel_cls_conf_thresholds: list[float],
    instance_cls_conf_thresholds: list[float],
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
        thresholds[dataset] = {
            "contour_normalized_hd95_fraction_thresholds": contour_fractions,
            "instance_fgbg_f1_thresholds": instance_f1_thresholds,
            "pixel_cls_conf_thresholds": pixel_cls_conf_thresholds,
            "instance_cls_conf_thresholds": instance_cls_conf_thresholds,
            "class_confusion_foreground_classes": sorted(
                foreground_classes_by_dataset[dataset]
            ),
        }

    rows = []
    for json_file, sample_id, dataset, entry in entries:
        hd95_value = sample_hd95(entry)
        instance_f1_value = sample_instance_f1(entry)
        fg_classes = sorted(foreground_classes_by_dataset[dataset])
        pixel_cls_conf_value = sample_cls_conf(entry, "PixelClsConf", fg_classes)
        instance_cls_conf_value = sample_cls_conf(
            entry, "InstanceClsConf", fg_classes
        )
        instance_cls_conf_coverage = sample_instance_cls_conf_coverage(entry)

        object_bbox_diagonal_value = object_bbox_diagonals.get((dataset, sample_id))
        normalized_hd95_value = (
            hd95_value / object_bbox_diagonal_value
            if hd95_value is not None
            and object_bbox_diagonal_value is not None
            and object_bbox_diagonal_value > 0
            else None
        )

        row = {
            "dataset": dataset,
            "sample_id": sample_id,
            "source_json": json_file.name,
            "client_idx": entry.get("client_idx"),
            "clean_dataset_id": entry.get("clean_dataset_id"),
            "noisy_dataset_id": entry.get("noisy_dataset_id"),
            "hd95_value": hd95_value,
            "object_bbox_diagonal_value": object_bbox_diagonal_value,
            "normalized_hd95_value": normalized_hd95_value,
            "instance_fgbg_f1_value": instance_f1_value,
            "pixel_cls_conf_value": pixel_cls_conf_value,
            "instance_cls_conf_value": instance_cls_conf_value,
            "instance_cls_conf_coverage": instance_cls_conf_coverage,
            "any_noise": False,
        }

        for fraction in contour_fractions:
            decision_key = decision_column("contour_noisy_frac", "gt", fraction)
            noisy = (
                normalized_hd95_value is not None
                and normalized_hd95_value > fraction
            )
            row[decision_key] = noisy
            row["any_noise"] = row["any_noise"] or noisy

        for threshold in instance_f1_thresholds:
            decision_key = decision_column("instance_noisy_f1", "lt", threshold)
            noisy = instance_f1_value is not None and instance_f1_value < threshold
            row[decision_key] = noisy
            row["any_noise"] = row["any_noise"] or noisy

        for threshold in pixel_cls_conf_thresholds:
            key = decision_column("pixel_cls_conf_noisy", "gt", threshold)
            noisy = (
                pixel_cls_conf_value is not None
                and pixel_cls_conf_value > threshold
            )
            row[key] = noisy

        for threshold in instance_cls_conf_thresholds:
            key = decision_column("instance_cls_conf_noisy", "gt", threshold)
            noisy = (
                instance_cls_conf_value is not None
                and instance_cls_conf_value > threshold
            )
            row[key] = noisy
            row["any_noise"] = row["any_noise"] or noisy

        rows.append(row)

    metadata = {
        "thresholds_by_dataset": thresholds,
        "contour_fraction_thresholds": contour_fractions,
        "instance_fgbg_f1_thresholds": instance_f1_thresholds,
        "pixel_cls_conf_thresholds": pixel_cls_conf_thresholds,
        "instance_cls_conf_thresholds": instance_cls_conf_thresholds,
        "contour_normalized_hd95_definition": (
            "nHD95_i = HD95_i / D_i, where D_i is the bbox diagonal of the "
            "foreground object in sample i's clean/expert mask (per-sample, "
            "not a dataset-wide constant). A sample is contour-noisy if "
            "nHD95_i exceeds the configured fraction threshold."
        ),
        "pixel_cls_conf_definition": (
            "Macro-average over present clean foreground classes of the fraction "
            "assigned to other foreground classes; background transitions excluded. "
            "Diagnostic only and excluded from any_noise."
        ),
        "instance_cls_conf_definition": (
            "Fraction of class-agnostically IoU-matched foreground instance pairs "
            "whose class labels differ; unmatched instances are excluded."
        ),
        "instance_cls_conf_coverage_definition": (
            "Matched clean/reference foreground instances divided by all clean/"
            "reference foreground instances. Diagnostic only; not part of "
            "InstanceClsConf and not thresholded as a noise type."
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
                "normalized_hd95_value",
                decision_key,
            )
        for threshold in instance_f1_thresholds:
            decision_key = decision_column("instance_noisy_f1", "lt", threshold)
            dataset_summary[f"instance_f1_lt_{threshold:g}"] = metric_summary(
                dataset_rows,
                "instance_fgbg_f1_value",
                decision_key,
            )
        for threshold in pixel_cls_conf_thresholds:
            decision_key = decision_column("pixel_cls_conf_noisy", "gt", threshold)
            dataset_summary[f"pixel_cls_conf_gt_{threshold:g}"] = metric_summary(
                dataset_rows, "pixel_cls_conf_value", decision_key
            )
        for threshold in instance_cls_conf_thresholds:
            decision_key = decision_column(
                "instance_cls_conf_noisy", "gt", threshold
            )
            dataset_summary[f"instance_cls_conf_gt_{threshold:g}"] = metric_summary(
                dataset_rows, "instance_cls_conf_value", decision_key
            )
        metadata["summaries_by_dataset"][dataset] = dataset_summary

    all_rows = rows
    for fraction in contour_fractions:
        decision_key = decision_column("contour_noisy_frac", "gt", fraction)
        metadata["summary_all"][f"contour_frac_gt_{fraction:g}"] = metric_summary(
            all_rows,
            "normalized_hd95_value",
            decision_key,
        )
    for threshold in instance_f1_thresholds:
        decision_key = decision_column("instance_noisy_f1", "lt", threshold)
        metadata["summary_all"][f"instance_f1_lt_{threshold:g}"] = metric_summary(
            all_rows,
            "instance_fgbg_f1_value",
            decision_key,
        )
    for threshold in pixel_cls_conf_thresholds:
        decision_key = decision_column("pixel_cls_conf_noisy", "gt", threshold)
        metadata["summary_all"][f"pixel_cls_conf_gt_{threshold:g}"] = metric_summary(
            all_rows, "pixel_cls_conf_value", decision_key
        )
    for threshold in instance_cls_conf_thresholds:
        decision_key = decision_column("instance_cls_conf_noisy", "gt", threshold)
        metadata["summary_all"][f"instance_cls_conf_gt_{threshold:g}"] = metric_summary(
            all_rows, "instance_cls_conf_value", decision_key
        )

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
        "--object-diagonals-csv",
        default=DEFAULT_OBJECT_DIAGONALS_CSV,
        help=(
            "CSV of per-sample foreground object bbox diagonals (D_i), written "
            "by compute_clean_object_diagonals.py."
        ),
    )
    parser.add_argument(
        "--contour-fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONTOUR_FRACTIONS),
        help=(
            "Normalized HD95 (HD95_i / D_i) thresholds; samples whose "
            "per-sample normalized HD95 exceeds a threshold are marked noisy."
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
        "--pixel-cls-conf-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_PIXEL_CLS_CONF_THRESHOLDS),
        help="PixelClsConf thresholds; values above each threshold are marked noisy.",
    )
    parser.add_argument(
        "--instance-cls-conf-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_INSTANCE_CLS_CONF_THRESHOLDS),
        help="InstanceClsConf thresholds; values above each threshold are marked noisy.",
    )
    args = parser.parse_args()

    json_files = resolve_json_files(args.input)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found for --input={args.input}")

    object_diagonals_csv = Path(args.object_diagonals_csv)
    if not object_diagonals_csv.is_file():
        raise FileNotFoundError(
            f"No object bbox diagonals CSV found at --object-diagonals-csv="
            f"{object_diagonals_csv}. Run compute_clean_object_diagonals.py first."
        )
    object_bbox_diagonals = load_object_bbox_diagonals(object_diagonals_csv)

    rows, metadata = compute_decisions(
        json_files=json_files,
        object_bbox_diagonals=object_bbox_diagonals,
        contour_fractions=args.contour_fractions,
        instance_f1_thresholds=args.instance_f1_thresholds,
        pixel_cls_conf_thresholds=args.pixel_cls_conf_thresholds,
        instance_cls_conf_thresholds=args.instance_cls_conf_thresholds,
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
            parts.append(f"nHD95>{fraction:g}={value}")
        for threshold in args.instance_f1_thresholds:
            key = f"instance_f1_lt_{threshold:g}"
            prevalence = summary[key]["prevalence_valid"]
            value = "n/a" if prevalence is None else f"{prevalence:.3f}"
            parts.append(f"instance<{threshold:g}={value}")
        for threshold in args.pixel_cls_conf_thresholds:
            key = f"pixel_cls_conf_gt_{threshold:g}"
            prevalence = summary[key]["prevalence_valid"]
            value = "n/a" if prevalence is None else f"{prevalence:.3f}"
            parts.append(f"PixelClsConf>{threshold:g}={value}")
        for threshold in args.instance_cls_conf_thresholds:
            key = f"instance_cls_conf_gt_{threshold:g}"
            prevalence = summary[key]["prevalence_valid"]
            value = "n/a" if prevalence is None else f"{prevalence:.3f}"
            parts.append(f"InstanceClsConf>{threshold:g}={value}")
        print(f"{dataset}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
