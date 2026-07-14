"""
Compute actual clean-object bounding-box diagonals for noise thresholds.

This scans the clean/expert nnU-Net preprocessed datasets and writes per-object
and per-dataset diagonal statistics. One object is one foreground class instance
in one sample, and its diagonal is computed from the class bounding box. For
NIfTI masks, voxel spacing is applied so the units match HD95 from
analyze_noise_clean_noisy.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image


CLEAN_DATASETS = {
    "LIDC": ["041", "042", "043", "044"],
    "RIGA": ["300", "301", "302"],
    "Gleason": ["436", "437", "438"],
    "MouseTumor": ["500", "501", "502", "503", "504"],
    "MMIA": ["600", "601", "602", "603"],
    "MMIS": ["700", "701", "702", "703"],
}
SUPPORTED_ENDINGS = (".nii.gz", ".png", ".tif", ".tiff")
DEFAULT_OUTPUT_DIR = "./results/noise_analysis/clean_object_diagonals"


def detect_file_ending(gt_dir: Path) -> str:
    for ending in SUPPORTED_ENDINGS:
        if list(gt_dir.glob(f"*{ending}")):
            return ending
    raise ValueError(f"No supported label files found in {gt_dir}")


def load_mask(path: Path, file_ending: str) -> tuple[np.ndarray, tuple[float, ...]]:
    if file_ending == ".nii.gz":
        image = nib.load(path)
        mask = np.asarray(image.get_fdata()).astype(np.int32)
        spacing = tuple(float(v) for v in image.header.get_zooms()[: mask.ndim])
        return mask, spacing

    mask = np.asarray(Image.open(path)).astype(np.int32)
    return mask, tuple(1.0 for _ in range(mask.ndim))


def sample_id_from_path(path: Path, file_ending: str) -> str:
    if file_ending == ".nii.gz":
        return path.name[: -len(".nii.gz")]
    return path.stem


def find_dataset_dir(data_dir: Path, dataset_id: str) -> Path | None:
    matches = sorted(data_dir.glob(f"Dataset{dataset_id}_*"))
    return matches[0] if matches else None


def class_bbox_diagonals(
    mask: np.ndarray,
    spacing: tuple[float, ...],
) -> list[dict[str, Any]]:
    records = []
    for class_id in sorted(int(v) for v in np.unique(mask) if int(v) > 0):
        class_mask = mask == class_id
        if not np.any(class_mask):
            continue
        extents = []
        for axis in range(class_mask.ndim):
            other_axes = tuple(i for i in range(class_mask.ndim) if i != axis)
            occupied = np.any(class_mask, axis=other_axes)
            occupied_idx = np.flatnonzero(occupied)
            extents.append(int(occupied_idx[-1] - occupied_idx[0] + 1))
        extents_voxels = np.asarray(extents, dtype=float)
        spacing_arr = np.asarray(spacing[: len(extents_voxels)], dtype=float)
        extents_physical = extents_voxels * spacing_arr
        records.append(
            {
                "class_id": class_id,
                "volume_voxels": int(np.sum(class_mask)),
                "bbox_extents_voxels": extents_voxels.tolist(),
                "bbox_extents_physical": extents_physical.tolist(),
                "bbox_diagonal": float(np.linalg.norm(extents_physical)),
            }
        )
    return records


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n_objects": 0,
            "median_bbox_diagonal": None,
            "mean_bbox_diagonal": None,
            "p05_bbox_diagonal": None,
            "p95_bbox_diagonal": None,
            "min_bbox_diagonal": None,
            "max_bbox_diagonal": None,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "n_objects": int(arr.size),
        "median_bbox_diagonal": float(np.median(arr)),
        "mean_bbox_diagonal": float(np.mean(arr)),
        "p05_bbox_diagonal": float(np.percentile(arr, 5)),
        "p95_bbox_diagonal": float(np.percentile(arr, 95)),
        "min_bbox_diagonal": float(np.min(arr)),
        "max_bbox_diagonal": float(np.max(arr)),
    }


def compute_clean_object_diagonals(
    data_dir: Path,
    clean_datasets: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_rows = []
    summary = {}

    for dataset, dataset_ids in clean_datasets.items():
        dataset_diagonals = []
        for dataset_id in dataset_ids:
            dataset_dir = find_dataset_dir(data_dir, dataset_id)
            if dataset_dir is None:
                print(f"WARNING: No dataset directory found for Dataset{dataset_id}_*")
                continue

            gt_dir = dataset_dir / "gt_segmentations"
            if not gt_dir.is_dir():
                print(f"WARNING: No gt_segmentations directory at {gt_dir}")
                continue

            file_ending = detect_file_ending(gt_dir)
            label_files = sorted(gt_dir.glob(f"*{file_ending}"))
            print(
                f"{dataset} Dataset{dataset_id}: {len(label_files)} labels "
                f"({file_ending})",
                flush=True,
            )

            for label_file in label_files:
                mask, spacing = load_mask(label_file, file_ending)
                sample_id = sample_id_from_path(label_file, file_ending)
                object_records = class_bbox_diagonals(mask, spacing)
                for record in object_records:
                    row = {
                        "dataset": dataset,
                        "dataset_id": dataset_id,
                        "sample_id": sample_id,
                        "file": str(label_file),
                        "spacing": list(spacing),
                        **record,
                    }
                    object_rows.append(row)
                    dataset_diagonals.append(record["bbox_diagonal"])

        summary[dataset] = summarize(dataset_diagonals)
        summary[dataset]["clean_dataset_ids"] = dataset_ids

    return object_rows, summary


def write_object_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_id",
        "sample_id",
        "class_id",
        "volume_voxels",
        "bbox_diagonal",
        "bbox_extents_voxels",
        "bbox_extents_physical",
        "spacing",
        "file",
    ]
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
            "Compute clean/expert foreground-class bounding-box "
            "diagonals for all benchmark datasets."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("nnUNet_preprocessed"),
        help="nnUNet_preprocessed directory containing Dataset*_*/gt_segmentations.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for object-level CSV and summary JSON.",
    )
    args = parser.parse_args()

    if not args.data_dir:
        raise ValueError(
            "Pass --data-dir or set nnUNet_preprocessed to the dataset directory."
        )

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    rows, summary = compute_clean_object_diagonals(
        data_dir,
        CLEAN_DATASETS,
    )
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "clean_object_bbox_diagonals.csv"
    json_path = output_dir / "clean_object_bbox_diagonal_summary.json"

    write_object_csv(rows, csv_path)
    write_json({"summary_by_dataset": summary, "objects": rows}, json_path)

    print(f"Wrote object diagonals to {csv_path}")
    print(f"Wrote summary to {json_path}")
    print("\nHeader constants for compute_noise_type_decisions.py:")
    print("DATASET_MEDIAN_OBJECT_DIAGONAL = {")
    for dataset, stats in summary.items():
        median = stats["median_bbox_diagonal"]
        print(f'    "{dataset}": {median!r},')
    print("}")


if __name__ == "__main__":
    main()
