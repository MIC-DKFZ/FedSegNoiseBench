"""
Comprehensive noise analysis script for comparing clean and noisy segmentation labels.

This script computes extensive metrics to quantify different types of label noise:
1. How noisy is "noisy": Dice, NSD, HD95, relative volume difference
2. Voxel-level precision/recall/F1 (per class and foreground-vs-background)
3. Instance-level precision/recall/F1 via connected-component matching
4. Contour difference noise: boundary-dominated disagreement
5. Missed/additional structures: instance-level errors
6. Swapped class labels: multi-class confusion
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

try:
    from .class_confusion_metrics import (
        compute_instance_cls_conf,
        compute_pixel_cls_conf,
    )
except ImportError:
    from class_confusion_metrics import compute_instance_cls_conf, compute_pixel_cls_conf


def compute_chunked_pairwise_distances(
    coords1: np.ndarray, coords2: np.ndarray, metric: str = "euclidean", chunk_size: int = 1000
) -> np.ndarray:
    """
    Compute pairwise distances in chunks to avoid OOM with large arrays.
    Returns the minimum distance from each point in coords1 to any point in coords2.

    Args:
        coords1: shape (N, D)
        coords2: shape (M, D)
        metric: distance metric (e.g., 'euclidean')
        chunk_size: number of points from coords1 to process per chunk

    Returns:
        array of shape (N,) with minimum distances
    """
    from scipy.spatial.distance import cdist

    N = len(coords1)
    min_distances = np.full(N, np.inf, dtype=np.float64)

    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = coords1[start_idx:end_idx]
        # Compute distances for this chunk to all of coords2
        chunk_distances = cdist(chunk, coords2, metric=metric)
        # Take minimum distance for each point in chunk
        min_distances[start_idx:end_idx] = chunk_distances.min(axis=1)

    return min_distances


def compute_dice(mask1: np.ndarray, mask2: np.ndarray, class_id: int) -> float:
    """Compute Dice coefficient for a specific class."""
    mask1_c = (mask1 == class_id).astype(np.float32)
    mask2_c = (mask2 == class_id).astype(np.float32)

    intersection = np.sum(mask1_c * mask2_c)
    sum_masks = np.sum(mask1_c) + np.sum(mask2_c)

    if sum_masks == 0:
        return np.nan  # both masks have no voxels of this class

    dice = 2.0 * intersection / sum_masks
    return float(dice)


def compute_surface_distances(
    mask1: np.ndarray, mask2: np.ndarray, spacing: Tuple[float, ...] = None
) -> np.ndarray:
    """Compute surface-to-surface distances between two binary masks."""
    if spacing is None:
        spacing = tuple([1.0] * len(mask1.shape))

    # Get surface voxels using morphological gradient
    from scipy.ndimage import binary_erosion, generate_binary_structure

    struct = generate_binary_structure(len(mask1.shape), 1)
    surface1 = mask1 ^ binary_erosion(mask1, struct)
    surface2 = mask2 ^ binary_erosion(mask2, struct)

    # Get coordinates of surface voxels
    coords1 = np.argwhere(surface1)
    coords2 = np.argwhere(surface2)

    if len(coords1) == 0 or len(coords2) == 0:
        return np.array([])

    # Scale by spacing
    coords1_scaled = coords1 * np.array(spacing)
    coords2_scaled = coords2 * np.array(spacing)

    # Compute distances from each surface1 point to closest surface2 point (chunked to avoid OOM)
    distances = compute_chunked_pairwise_distances(
        coords1_scaled, coords2_scaled, metric="euclidean", chunk_size=1000
    )

    return distances


def compute_nsd(
    mask1: np.ndarray,
    mask2: np.ndarray,
    class_id: int,
    tolerance: float = 2.0,
    spacing: Tuple[float, ...] = None,
) -> float:
    """
    Compute Normalized Surface Distance (NSD) for a specific class.

    Args:
        mask1, mask2: Segmentation masks
        class_id: Class to evaluate
        tolerance: Distance tolerance in mm
        spacing: Voxel spacing (default: isotropic 1mm)
    """
    mask1_c = (mask1 == class_id).astype(np.uint8)
    mask2_c = (mask2 == class_id).astype(np.uint8)

    if np.sum(mask1_c) == 0 and np.sum(mask2_c) == 0:
        return np.nan
    if np.sum(mask1_c) == 0 or np.sum(mask2_c) == 0:
        return 0.0

    # Compute surface distances
    dist_1_to_2 = compute_surface_distances(mask1_c, mask2_c, spacing)
    dist_2_to_1 = compute_surface_distances(mask2_c, mask1_c, spacing)

    if len(dist_1_to_2) == 0 or len(dist_2_to_1) == 0:
        return 0.0

    # NSD: fraction of surface points within tolerance
    within_tolerance = (
        np.sum(dist_1_to_2 <= tolerance) + np.sum(dist_2_to_1 <= tolerance)
    ) / (len(dist_1_to_2) + len(dist_2_to_1))

    return float(within_tolerance)


def compute_hd95(
    mask1: np.ndarray,
    mask2: np.ndarray,
    class_id: int,
    spacing: Tuple[float, ...] = None,
) -> float:
    """Compute 95th percentile Hausdorff Distance for a specific class."""
    mask1_c = (mask1 == class_id).astype(np.uint8)
    mask2_c = (mask2 == class_id).astype(np.uint8)

    if np.sum(mask1_c) == 0 and np.sum(mask2_c) == 0:
        return np.nan
    if np.sum(mask1_c) == 0 or np.sum(mask2_c) == 0:
        return np.inf

    dist_1_to_2 = compute_surface_distances(mask1_c, mask2_c, spacing)
    dist_2_to_1 = compute_surface_distances(mask2_c, mask1_c, spacing)

    if len(dist_1_to_2) == 0 or len(dist_2_to_1) == 0:
        return np.inf

    all_distances = np.concatenate([dist_1_to_2, dist_2_to_1])
    hd95 = np.percentile(all_distances, 95)

    return float(hd95)


def compute_relative_volume_diff(
    mask1: np.ndarray, mask2: np.ndarray, class_id: int
) -> float:
    """
    Compute relative volume difference: (|noisy| - |clean|) / |clean|
    Assumes mask1 is clean, mask2 is noisy.
    """
    vol_clean = np.sum(mask1 == class_id)
    vol_noisy = np.sum(mask2 == class_id)

    if vol_clean == 0:
        if vol_noisy == 0:
            return 0.0
        return np.inf

    rel_vol_diff = (vol_noisy - vol_clean) / vol_clean
    return float(rel_vol_diff)


def _compute_precision_recall_f1_from_counts(
    tp: int, fp: int, fn: int, valid_if_no_objects: bool = False
) -> Dict[str, float]:
    """Compute precision/recall/F1 from TP/FP/FN counts."""
    if not valid_if_no_objects and tp == 0 and fp == 0 and fn == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_voxel_prf(binary_clean: np.ndarray, binary_noisy: np.ndarray) -> Dict[str, float]:
    """
    Compute voxel-level precision/recall/F1 for binary masks.

    clean is treated as reference (ground truth) and noisy as prediction.
    """
    clean_bool = binary_clean.astype(bool)
    noisy_bool = binary_noisy.astype(bool)

    tp = int(np.sum(clean_bool & noisy_bool))
    fp = int(np.sum(~clean_bool & noisy_bool))
    fn = int(np.sum(clean_bool & ~noisy_bool))

    scores = _compute_precision_recall_f1_from_counts(tp, fp, fn)
    scores.update({"tp": tp, "fp": fp, "fn": fn})
    return scores


def _connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label connected components in a binary mask."""
    return ndimage.label(binary_mask.astype(np.uint8))


def compute_instance_prf(
    binary_clean: np.ndarray,
    binary_noisy: np.ndarray,
    overlap_iou_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Compute instance-level precision/recall/F1 from connected components.

    Connected components in noisy are treated as predicted instances.
    A predicted instance matches a clean instance if IoU >= overlap_iou_threshold.
    Matching is one-to-one using greedy highest-IoU assignment.
    """
    labeled_clean, n_clean = _connected_components(binary_clean)
    labeled_noisy, n_noisy = _connected_components(binary_noisy)

    if n_clean == 0 and n_noisy == 0:
        scores = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
        scores.update(
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "num_instances_clean": 0,
                "num_instances_noisy": 0,
                "match_iou_threshold": float(overlap_iou_threshold),
            }
        )
        return scores

    # Build IoU candidates only where components overlap in voxels
    iou_candidates = []
    for noisy_idx in range(1, n_noisy + 1):
        noisy_component = labeled_noisy == noisy_idx
        overlapping_clean_labels = np.unique(labeled_clean[noisy_component])
        overlapping_clean_labels = overlapping_clean_labels[overlapping_clean_labels > 0]

        if overlapping_clean_labels.size == 0:
            continue

        noisy_size = np.sum(noisy_component)
        for clean_idx in overlapping_clean_labels:
            clean_component = labeled_clean == clean_idx
            inter = np.sum(noisy_component & clean_component)
            union = noisy_size + np.sum(clean_component) - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou >= overlap_iou_threshold:
                iou_candidates.append((float(iou), int(clean_idx), int(noisy_idx)))

    # Greedy one-to-one matching by highest IoU
    iou_candidates.sort(reverse=True, key=lambda x: x[0])
    matched_clean = set()
    matched_noisy = set()
    tp = 0

    for _, clean_idx, noisy_idx in iou_candidates:
        if clean_idx in matched_clean or noisy_idx in matched_noisy:
            continue
        matched_clean.add(clean_idx)
        matched_noisy.add(noisy_idx)
        tp += 1

    fp = n_noisy - tp
    fn = n_clean - tp

    scores = _compute_precision_recall_f1_from_counts(tp, fp, fn)
    scores.update(
        {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "num_instances_clean": int(n_clean),
            "num_instances_noisy": int(n_noisy),
            "match_iou_threshold": float(overlap_iou_threshold),
        }
    )
    return scores


def compute_connected_components_stats(mask: np.ndarray, class_id: int) -> Dict:
    """
    Compute connected component statistics for a specific class.

    Returns:
        Dictionary with:
        - num_components: number of foreground connected components
        - avg_volume: average volume of connected components
        - volumes: list of volumes of all connected components
    """
    mask_c = (mask == class_id).astype(np.uint8)

    if np.sum(mask_c) == 0:
        return {"num_components": 0, "avg_volume": 0.0, "volumes": []}

    labeled, num_components = ndimage.label(mask_c)

    volumes = []
    for i in range(1, num_components + 1):
        vol = np.sum(labeled == i)
        volumes.append(int(vol))

    avg_volume = np.mean(volumes) if volumes else 0.0

    return {
        "num_components": int(num_components),
        "avg_volume": float(avg_volume),
        "volumes": volumes,
    }


def compute_class_overlap_matrix(
    clean_mask: np.ndarray, noisy_mask: np.ndarray, classes: List[int]
) -> Dict:
    """
    Compute class-to-class overlap matrix.

    For every pair of classes (c, d):
    S_{c→d} = (|mask_clean_c ∩ mask_noisy_d|) / |mask_clean_c|

    This quantifies how much of clean class c is labeled as noisy class d.
    Includes background class (0) and returns a square matrix over all classes.
    """
    overlap_matrix = {}

    for c in classes:
        clean_c = clean_mask == c
        vol_clean_c = np.sum(clean_c)

        overlap_matrix[int(c)] = {}

        for d in classes:
            if vol_clean_c == 0:
                overlap_ratio = 0.0
            else:
                noisy_d = noisy_mask == d
                intersection = np.sum(clean_c & noisy_d)
                overlap_ratio = intersection / vol_clean_c
            overlap_matrix[int(c)][int(d)] = float(overlap_ratio)

    return overlap_matrix


def analyze_sample(
    clean_mask: np.ndarray,
    noisy_mask: np.ndarray,
    spacing: Tuple[float, ...] = None,
    instance_match_iou_threshold: float = 0.1,
) -> Dict:
    """
    Comprehensive analysis of a single sample comparing clean vs noisy masks.

    Returns dictionary with all computed metrics.
    """
    # Get all unique classes (excluding background=0 for some metrics)
    all_classes = sorted(set(np.unique(clean_mask)) | set(np.unique(noisy_mask)))
    fg_classes = [c for c in all_classes if c > 0]

    results = {
        "classes": {
            "all_classes": [int(c) for c in all_classes],
            "fg_classes": [int(c) for c in fg_classes],
            "only_in_clean": [
                int(c) for c in set(np.unique(clean_mask)) - set(np.unique(noisy_mask))
            ],
            "only_in_noisy": [
                int(c) for c in set(np.unique(noisy_mask)) - set(np.unique(clean_mask))
            ],
        },
        "per_class_metrics": {},
        "foreground_vs_background_metrics": {},
        "class_confusion_metrics": {},
        "overall_metrics": {},
    }

    pixel_cls_conf = compute_pixel_cls_conf(
        clean_mask,
        noisy_mask,
        foreground_classes=[int(c) for c in fg_classes],
    )
    instance_cls_conf = compute_instance_cls_conf(
        clean_mask,
        noisy_mask,
        foreground_classes=[int(c) for c in fg_classes],
        overlap_iou_threshold=instance_match_iou_threshold,
    )
    # Individual match records are useful while debugging but unnecessarily
    # inflate the persisted analysis JSON. Counts and coverage fully document
    # the score denominator.
    instance_cls_conf.pop("matches", None)
    results["class_confusion_metrics"] = {
        "PixelClsConf": pixel_cls_conf,
        "InstanceClsConf": instance_cls_conf,
    }

    # Per-class metrics
    for class_id in fg_classes:
        clean_class_binary = clean_mask == class_id
        noisy_class_binary = noisy_mask == class_id

        class_metrics = {
            "dice": compute_dice(clean_mask, noisy_mask, class_id),
            "nsd": compute_nsd(clean_mask, noisy_mask, class_id, spacing=spacing),
            "hd95": compute_hd95(clean_mask, noisy_mask, class_id, spacing=spacing),
            "relative_volume_diff": compute_relative_volume_diff(
                clean_mask, noisy_mask, class_id
            ),
            "volume_clean": int(np.sum(clean_class_binary)),
            "volume_noisy": int(np.sum(noisy_class_binary)),
        }

        class_metrics["voxel_level_prf"] = compute_voxel_prf(
            clean_class_binary, noisy_class_binary
        )
        class_metrics["instance_level_prf"] = compute_instance_prf(
            clean_class_binary,
            noisy_class_binary,
            overlap_iou_threshold=instance_match_iou_threshold,
        )
        class_metrics["PixelClsConf"] = pixel_cls_conf["per_class"].get(
            int(class_id)
        )
        class_metrics["InstanceClsConf"] = instance_cls_conf["per_class"].get(
            int(class_id), {}
        ).get("score")
        class_metrics["InstanceClsConfCoverage"] = instance_cls_conf[
            "per_class"
        ].get(int(class_id), {}).get("coverage")

        # Connected components stats
        cc_clean = compute_connected_components_stats(clean_mask, class_id)
        cc_noisy = compute_connected_components_stats(noisy_mask, class_id)

        class_metrics["cc_clean"] = cc_clean
        class_metrics["cc_noisy"] = cc_noisy
        class_metrics["delta_num_cc"] = (
            cc_noisy["num_components"] - cc_clean["num_components"]
        )
        class_metrics["delta_avg_volume_cc"] = (
            cc_noisy["avg_volume"] - cc_clean["avg_volume"]
        )

        results["per_class_metrics"][int(class_id)] = class_metrics

    # Foreground vs background (all foreground classes collapsed into one foreground class)
    clean_fg_binary = clean_mask > 0
    noisy_fg_binary = noisy_mask > 0
    results["foreground_vs_background_metrics"]["voxel_level_prf"] = compute_voxel_prf(
        clean_fg_binary, noisy_fg_binary
    )
    results["foreground_vs_background_metrics"]["instance_level_prf"] = (
        compute_instance_prf(
            clean_fg_binary,
            noisy_fg_binary,
            overlap_iou_threshold=instance_match_iou_threshold,
        )
    )

    # Class-to-class overlap matrix (for swapped labels analysis)
    results["class_overlap_matrix"] = compute_class_overlap_matrix(
        clean_mask, noisy_mask, all_classes
    )

    # Overall metrics (averaged across all foreground classes)
    if fg_classes:
        valid_dice = [
            results["per_class_metrics"][c]["dice"]
            for c in fg_classes
            if not np.isnan(results["per_class_metrics"][c]["dice"])
        ]
        results["overall_metrics"]["mean_dice"] = (
            float(np.mean(valid_dice)) if valid_dice else np.nan
        )

        valid_nsd = [
            results["per_class_metrics"][c]["nsd"]
            for c in fg_classes
            if not np.isnan(results["per_class_metrics"][c]["nsd"])
        ]
        results["overall_metrics"]["mean_nsd"] = (
            float(np.mean(valid_nsd)) if valid_nsd else np.nan
        )

        valid_hd95 = [
            results["per_class_metrics"][c]["hd95"]
            for c in fg_classes
            if not np.isnan(results["per_class_metrics"][c]["hd95"])
            and not np.isinf(results["per_class_metrics"][c]["hd95"])
        ]
        results["overall_metrics"]["mean_hd95"] = (
            float(np.mean(valid_hd95)) if valid_hd95 else np.nan
        )

        for metric_scope in ["voxel_level_prf", "instance_level_prf"]:
            for score_name in ["precision", "recall", "f1"]:
                valid_scores = [
                    results["per_class_metrics"][c][metric_scope][score_name]
                    for c in fg_classes
                    if not np.isnan(results["per_class_metrics"][c][metric_scope][score_name])
                ]
                results["overall_metrics"][f"mean_{metric_scope}_{score_name}"] = (
                    float(np.mean(valid_scores)) if valid_scores else np.nan
                )

        # Total volumes
        results["overall_metrics"]["total_volume_clean"] = int(np.sum(clean_mask > 0))
        results["overall_metrics"]["total_volume_noisy"] = int(np.sum(noisy_mask > 0))

        # Total number of connected components
        total_cc_clean = sum(
            [
                results["per_class_metrics"][c]["cc_clean"]["num_components"]
                for c in fg_classes
            ]
        )
        total_cc_noisy = sum(
            [
                results["per_class_metrics"][c]["cc_noisy"]["num_components"]
                for c in fg_classes
            ]
        )
        results["overall_metrics"]["total_num_cc_clean"] = int(total_cc_clean)
        results["overall_metrics"]["total_num_cc_noisy"] = int(total_cc_noisy)
        results["overall_metrics"]["delta_total_num_cc"] = int(
            total_cc_noisy - total_cc_clean
        )

    return results


def load_mask(file_path: str, file_ending: str) -> np.ndarray:
    """Load segmentation mask from file."""
    if file_ending == ".nii.gz":
        mask = np.array(nib.load(file_path).get_fdata()).astype(np.int32)
    elif file_ending in [".tif", ".tiff", ".png"]:
        mask = np.array(Image.open(file_path)).astype(np.int32)
    else:
        raise ValueError(f"Unsupported file ending: {file_ending}")
    return mask


def get_spacing_from_nifti(file_path: str) -> Tuple[float, ...]:
    """Get voxel spacing from NIfTI file."""
    try:
        nii = nib.load(file_path)
        spacing = tuple(nii.header.get_zooms())
        return spacing
    except:
        return None


def detect_file_ending(gt_dir: str) -> str:
    """Auto-detect the file ending from files in the directory."""
    supported_endings = ['.nii.gz', '.png', '.tif', '.tiff']
    
    for ending in supported_endings:
        files = glob.glob(f"{gt_dir}/*{ending}")
        if files:
            print(f"Auto-detected file ending: {ending}")
            return ending
    
    raise ValueError(f"Could not auto-detect file ending in {gt_dir}. No files with supported endings found: {supported_endings}")


def main(args):
    """Main analysis function."""

    # get data_dir from nnUNet_preprocessed env var
    data_dir = os.getenv("nnUNet_preprocessed")
    if data_dir is None:
        raise ValueError(
            "Environment variable nnUNet_preprocessed is not set. Please set it to the directory containing the datasets."
        )

    # Parse dataset IDs
    clean_dataset_ids = (
        args.clean_dataset_ids[0].split()
        if isinstance(args.clean_dataset_ids[0], str)
        else args.clean_dataset_ids
    )
    noisy_dataset_ids = (
        args.noisy_dataset_ids[0].split()
        if isinstance(args.noisy_dataset_ids[0], str)
        else args.noisy_dataset_ids
    )

    assert len(clean_dataset_ids) == len(
        noisy_dataset_ids
    ), f"Number of clean ({len(clean_dataset_ids)}) and noisy ({len(noisy_dataset_ids)}) dataset IDs must match"

    print(f"Analyzing {len(clean_dataset_ids)} dataset pairs:")
    for clean_id, noisy_id in zip(clean_dataset_ids, noisy_dataset_ids):
        print(f"  Clean: Dataset{clean_id}_* vs Noisy: Dataset{noisy_id}_*")

    # Prepare output file path
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"noise_analysis_results_clean{'-'.join(clean_dataset_ids)}_noisy{'-'.join(noisy_dataset_ids)}.json",
    )

    # Resume support: load existing results once and skip already processed samples
    persisted_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                persisted_results = json.load(f)
            print(
                f"Found existing results file with {len(persisted_results)} samples. "
                "Resuming and skipping already processed samples."
            )
        except Exception as e:
            print(
                f"WARNING: Could not read existing results file at {output_file} ({e}). "
                "Starting with an empty result set."
            )
            persisted_results = {}

    processed_sample_ids = {
        sample_id
        for sample_id, result in persisted_results.items()
        if result.get("class_confusion_metrics", {})
        .get("InstanceClsConf", {})
        .get("transition_matrix") is not None
    }
    num_outdated = len(persisted_results) - len(processed_sample_ids)
    if num_outdated:
        print(
            f"Recomputing {num_outdated} existing samples that predate "
            "PixelClsConf/InstanceClsConf."
        )

    # Process each dataset pair
    for client_idx, (clean_id, noisy_id) in enumerate(
        zip(clean_dataset_ids, noisy_dataset_ids)
    ):
        print(f"\n{'='*80}")
        print(f"Processing client {client_idx}: Clean={clean_id}, Noisy={noisy_id}")
        print(f"{'='*80}")

        # Find dataset directories
        clean_dataset_pattern = f"{data_dir}/Dataset{clean_id}_*"
        noisy_dataset_pattern = f"{data_dir}/Dataset{noisy_id}_*"

        clean_dataset_dirs = glob.glob(clean_dataset_pattern)
        noisy_dataset_dirs = glob.glob(noisy_dataset_pattern)

        if not clean_dataset_dirs:
            print(f"WARNING: No clean dataset found matching {clean_dataset_pattern}")
            continue
        if not noisy_dataset_dirs:
            print(f"WARNING: No noisy dataset found matching {noisy_dataset_pattern}")
            continue

        clean_dataset_dir = clean_dataset_dirs[0]
        noisy_dataset_dir = noisy_dataset_dirs[0]

        clean_gt_dir = os.path.join(clean_dataset_dir, "gt_segmentations")
        noisy_gt_dir = os.path.join(noisy_dataset_dir, "gt_segmentations")

        print(f"Clean GT dir: {clean_gt_dir}")
        print(f"Noisy GT dir: {noisy_gt_dir}")

        if not os.path.exists(clean_gt_dir):
            print(f"WARNING: Clean GT directory does not exist: {clean_gt_dir}")
            continue
        if not os.path.exists(noisy_gt_dir):
            print(f"WARNING: Noisy GT directory does not exist: {noisy_gt_dir}")
            continue

        # Auto-detect file ending if not provided or set to 'auto'
        file_ending = args.gt_file_ending
        if file_ending == 'auto' or file_ending is None:
            file_ending = detect_file_ending(clean_gt_dir)
        
        print(f"Using file ending: {file_ending}")

        # Get label files
        clean_label_files = sorted(glob.glob(f"{clean_gt_dir}/*{file_ending}"))
        noisy_label_files = sorted(glob.glob(f"{noisy_gt_dir}/*{file_ending}"))

        print(
            f"Found {len(clean_label_files)} clean labels and {len(noisy_label_files)} noisy labels"
        )

        if len(clean_label_files) != len(noisy_label_files):
            print(f"WARNING: Mismatch in number of label files!")

        # Create sample_id to file mapping
        clean_label_map = {}
        for f in clean_label_files:
            sample_id = os.path.basename(f).replace(file_ending, "")
            clean_label_map[sample_id] = f

        noisy_label_map = {}
        for f in noisy_label_files:
            sample_id = os.path.basename(f).replace(file_ending, "")
            noisy_label_map[sample_id] = f

        # Find common sample IDs
        common_sample_ids = sorted(
            set(clean_label_map.keys()) & set(noisy_label_map.keys())
        )
        print(f"Processing {len(common_sample_ids)} common samples")

        # Batch size for incremental saving
        batch_size = 1
        sample_counter = 0
        skipped_counter = 0
        pending_results = {}

        # Process each sample
        for sample_id in tqdm(common_sample_ids, desc=f"Client {client_idx}"):
            if sample_id in processed_sample_ids:
                skipped_counter += 1
                continue

            clean_file = clean_label_map[sample_id]
            noisy_file = noisy_label_map[sample_id]

            # Load masks
            clean_mask = load_mask(clean_file, file_ending)
            noisy_mask = load_mask(noisy_file, file_ending)

            # Get spacing if NIfTI
            spacing = None
            if file_ending == ".nii.gz":
                spacing = get_spacing_from_nifti(clean_file)

            # Analyze
            sample_results = analyze_sample(
                clean_mask,
                noisy_mask,
                spacing=spacing,
                instance_match_iou_threshold=args.instance_match_iou_threshold,
            )
            sample_results["client_idx"] = client_idx
            sample_results["clean_dataset_id"] = clean_id
            sample_results["noisy_dataset_id"] = noisy_id

            pending_results[sample_id] = sample_results
            
            # Free memory immediately after processing
            del clean_mask, noisy_mask
            
            sample_counter += 1
            
            # Periodically save to disk and clear memory
            if sample_counter % batch_size == 0:
                persisted_results.update(pending_results)
                processed_sample_ids.update(pending_results.keys())
                with open(output_file, "w") as f:
                    json.dump(persisted_results, f, indent=2)
                pending_results.clear()
        
        # Save any remaining results after the loop
        if pending_results:
            persisted_results.update(pending_results)
            processed_sample_ids.update(pending_results.keys())
            with open(output_file, "w") as f:
                json.dump(persisted_results, f, indent=2)
            pending_results.clear()

        print(
            f"Client {client_idx}: added {sample_counter} new samples, "
            f"skipped {skipped_counter} already processed samples."
        )

    # Load final results for summary statistics
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = persisted_results
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total samples analyzed: {len(all_results)}")
    print(f"{'='*80}")

    # Print summary statistics
    if all_results:
        print("\nSummary Statistics:")
        print("-" * 80)

        # Aggregate metrics across all samples
        all_mean_dice = [
            r["overall_metrics"]["mean_dice"]
            for r in all_results.values()
            if not np.isnan(r["overall_metrics"].get("mean_dice", np.nan))
        ]
        all_mean_nsd = [
            r["overall_metrics"]["mean_nsd"]
            for r in all_results.values()
            if not np.isnan(r["overall_metrics"].get("mean_nsd", np.nan))
        ]
        all_mean_hd95 = [
            r["overall_metrics"]["mean_hd95"]
            for r in all_results.values()
            if not np.isnan(r["overall_metrics"].get("mean_hd95", np.nan))
        ]

        if all_mean_dice:
            print(
                f"Mean Dice across all samples: {np.mean(all_mean_dice):.4f} ± {np.std(all_mean_dice):.4f}"
            )
        if all_mean_nsd:
            print(
                f"Mean NSD across all samples: {np.mean(all_mean_nsd):.4f} ± {np.std(all_mean_nsd):.4f}"
            )
        if all_mean_hd95:
            print(
                f"Mean HD95 across all samples: {np.mean(all_mean_hd95):.2f} ± {np.std(all_mean_hd95):.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive noise analysis: compare clean vs noisy segmentation labels"
    )
    parser.add_argument(
        "--clean_dataset_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of clean dataset IDs (e.g., '041 042 043' or 041 042 043)",
    )
    parser.add_argument(
        "--noisy_dataset_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of noisy dataset IDs (e.g., '045 046 047' or 045 046 047)",
    )
    parser.add_argument(
        "--gt_file_ending",
        type=str,
        default="auto",
        help="File ending of ground truth masks (default: auto-detect). Options: auto, .nii.gz, .png, .tif, .tiff",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/noise_analysis",
        help="Directory to save analysis results (default: ./results)",
    )
    parser.add_argument(
        "--instance_match_iou_threshold",
        type=float,
        default=0.1,
        help="IoU threshold used to match connected components for instance-level precision/recall/F1 (default: 0.1)",
    )

    args = parser.parse_args()
    main(args)
