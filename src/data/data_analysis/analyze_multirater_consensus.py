"""
Comprehensive multirater-vs-consensus analysis for segmentation labels.

For each sample, this script computes per-class metrics:
- Fleiss' kappa among rater masks
- Voxel-wise entropy among rater masks
- Dice between consensus mask and each rater mask
- HD95 between consensus mask and each rater mask
- Voxel-level precision/recall/F1 between consensus and each rater
- Instance-level precision/recall/F1 between consensus and each rater

Output is a JSON dictionary keyed by sample_id, similar in structure to
analyze_noise_clean_noisy.py.
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

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
	coords1: np.ndarray,
	coords2: np.ndarray,
	metric: str = "euclidean",
	chunk_size: int = 1000,
) -> np.ndarray:
	"""
	Compute pairwise distances in chunks to avoid OOM with large arrays.

	Returns minimum distance from each point in coords1 to any point in coords2.
	"""
	from scipy.spatial.distance import cdist

	n_points = len(coords1)
	min_distances = np.full(n_points, np.inf, dtype=np.float64)

	for start_idx in range(0, n_points, chunk_size):
		end_idx = min(start_idx + chunk_size, n_points)
		chunk = coords1[start_idx:end_idx]
		chunk_distances = cdist(chunk, coords2, metric=metric)
		min_distances[start_idx:end_idx] = chunk_distances.min(axis=1)

	return min_distances


def compute_dice(mask1: np.ndarray, mask2: np.ndarray, class_id: int) -> float:
	"""Compute Dice coefficient for a specific class."""
	mask1_c = (mask1 == class_id).astype(np.float32)
	mask2_c = (mask2 == class_id).astype(np.float32)

	intersection = np.sum(mask1_c * mask2_c)
	sum_masks = np.sum(mask1_c) + np.sum(mask2_c)

	if sum_masks == 0:
		return np.nan

	dice = 2.0 * intersection / sum_masks
	return float(dice)


def compute_surface_distances(
	mask1: np.ndarray,
	mask2: np.ndarray,
	spacing: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
	"""Compute surface-to-surface distances between two binary masks."""
	if spacing is None:
		spacing = tuple([1.0] * len(mask1.shape))

	from scipy.ndimage import binary_erosion, generate_binary_structure

	struct = generate_binary_structure(len(mask1.shape), 1)
	surface1 = mask1 ^ binary_erosion(mask1, struct)
	surface2 = mask2 ^ binary_erosion(mask2, struct)

	coords1 = np.argwhere(surface1)
	coords2 = np.argwhere(surface2)

	if len(coords1) == 0 or len(coords2) == 0:
		return np.array([])

	coords1_scaled = coords1 * np.array(spacing)
	coords2_scaled = coords2 * np.array(spacing)

	distances = compute_chunked_pairwise_distances(
		coords1_scaled, coords2_scaled, metric="euclidean", chunk_size=1000
	)
	return distances


def compute_hd95(
	mask1: np.ndarray,
	mask2: np.ndarray,
	class_id: int,
	spacing: Optional[Tuple[float, ...]] = None,
) -> float:
	"""Compute 95th percentile Hausdorff distance for a specific class."""
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


def _compute_precision_recall_f1_from_counts(
	tp: int,
	fp: int,
	fn: int,
	valid_if_no_objects: bool = False,
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


def compute_voxel_prf(binary_gt: np.ndarray, binary_pred: np.ndarray) -> Dict[str, float]:
	"""Compute voxel-level precision/recall/F1 for binary masks."""
	gt_bool = binary_gt.astype(bool)
	pred_bool = binary_pred.astype(bool)

	tp = int(np.sum(gt_bool & pred_bool))
	fp = int(np.sum(~gt_bool & pred_bool))
	fn = int(np.sum(gt_bool & ~pred_bool))

	scores = _compute_precision_recall_f1_from_counts(tp, fp, fn)
	scores.update({"tp": tp, "fp": fp, "fn": fn})
	return scores


def _connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
	"""Label connected components in a binary mask."""
	return ndimage.label(binary_mask.astype(np.uint8))


def compute_instance_prf(
	binary_gt: np.ndarray,
	binary_pred: np.ndarray,
	overlap_iou_threshold: float = 0.1,
) -> Dict[str, float]:
	"""
	Compute instance-level precision/recall/F1 from connected components.

	Components in pred are treated as predicted instances and gt as reference.
	"""
	labeled_gt, n_gt = _connected_components(binary_gt)
	labeled_pred, n_pred = _connected_components(binary_pred)

	if n_gt == 0 and n_pred == 0:
		scores = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
		scores.update(
			{
				"tp": 0,
				"fp": 0,
				"fn": 0,
				"num_instances_gt": 0,
				"num_instances_pred": 0,
				"match_iou_threshold": float(overlap_iou_threshold),
			}
		)
		return scores

	iou_candidates = []
	for pred_idx in range(1, n_pred + 1):
		pred_component = labeled_pred == pred_idx
		overlapping_gt_labels = np.unique(labeled_gt[pred_component])
		overlapping_gt_labels = overlapping_gt_labels[overlapping_gt_labels > 0]

		if overlapping_gt_labels.size == 0:
			continue

		pred_size = np.sum(pred_component)
		for gt_idx in overlapping_gt_labels:
			gt_component = labeled_gt == gt_idx
			inter = np.sum(pred_component & gt_component)
			union = pred_size + np.sum(gt_component) - inter
			if union <= 0:
				continue
			iou = inter / union
			if iou >= overlap_iou_threshold:
				iou_candidates.append((float(iou), int(gt_idx), int(pred_idx)))

	iou_candidates.sort(reverse=True, key=lambda x: x[0])
	matched_gt = set()
	matched_pred = set()
	tp = 0

	for _, gt_idx, pred_idx in iou_candidates:
		if gt_idx in matched_gt or pred_idx in matched_pred:
			continue
		matched_gt.add(gt_idx)
		matched_pred.add(pred_idx)
		tp += 1

	fp = n_pred - tp
	fn = n_gt - tp

	scores = _compute_precision_recall_f1_from_counts(tp, fp, fn)
	scores.update(
		{
			"tp": int(tp),
			"fp": int(fp),
			"fn": int(fn),
			"num_instances_gt": int(n_gt),
			"num_instances_pred": int(n_pred),
			"match_iou_threshold": float(overlap_iou_threshold),
		}
	)
	return scores


def compute_fleiss_kappa_binary(binary_rater_masks: List[np.ndarray]) -> float:
	"""
	Compute Fleiss' kappa for binary class membership across raters.

	Each voxel is an item; categories are {not-class, class}.
	"""
	n_raters = len(binary_rater_masks)
	if n_raters < 2:
		return np.nan

	n_class = np.zeros(binary_rater_masks[0].shape, dtype=np.uint16)
	for r_mask in binary_rater_masks:
		n_class += r_mask.astype(np.uint16)

	n_not_class = n_raters - n_class

	denom = n_raters * (n_raters - 1)
	p_i = (n_class * (n_class - 1) + n_not_class * (n_not_class - 1)) / denom
	p_bar = np.mean(p_i, dtype=np.float64)

	n_items = int(np.prod(n_class.shape))
	p_class = np.sum(n_class, dtype=np.float64) / (n_items * n_raters)
	p_not_class = 1.0 - p_class
	p_e = p_class**2 + p_not_class**2

	if np.isclose(1.0 - p_e, 0.0):
		return np.nan

	kappa = (p_bar - p_e) / (1.0 - p_e)
	return float(kappa)


def compute_voxelwise_entropy_binary(binary_rater_masks: List[np.ndarray]) -> float:
	"""
	Compute mean voxel-wise binary entropy of class membership across raters.

	Entropy per voxel is computed from p(class) over raters.
	"""
	n_raters = len(binary_rater_masks)
	if n_raters == 0:
		return np.nan

	n_class = np.zeros(binary_rater_masks[0].shape, dtype=np.uint16)
	for r_mask in binary_rater_masks:
		n_class += r_mask.astype(np.uint16)

	p = n_class.astype(np.float64) / float(n_raters)
	eps = 1e-12
	p_clipped = np.clip(p, eps, 1.0 - eps)
	entropy = -(p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))
	return float(np.mean(entropy, dtype=np.float64))


def load_mask(file_path: str, file_ending: str, riga_mode: bool = False) -> np.ndarray:
	"""Load segmentation mask from file.
	
	Args:
		file_path: Path to the mask file
		file_ending: File extension (.nii.gz, .tif, .tiff, .png)
		riga_mode: If True, interpret RGB TIF images with RIGA color mapping:
			- Red channel 255 -> label 2
			- Red channel 120 -> label 1
			- Everything else -> label 0
	"""
	if file_ending == ".nii.gz":
		return np.array(nib.load(file_path).get_fdata()).astype(np.int32)
	if file_ending in [".tif", ".tiff", ".png"]:
		img_array = np.array(Image.open(file_path))
		
		if riga_mode:
			# RIGA masks may be stored either as RGB images with only the red channel
			# used, or as grayscale images containing the same encoded values.
			if img_array.ndim == 3 and img_array.shape[2] >= 3:
				encoded_channel = img_array[:, :, 0]
			elif img_array.ndim == 2:
				encoded_channel = img_array
			else:
				raise ValueError(
					f"Unsupported RIGA image shape {img_array.shape} for {file_path}"
				)

			label_mask = np.zeros_like(encoded_channel, dtype=np.int32)
			label_mask[encoded_channel == 255] = 2
			label_mask[encoded_channel == 120] = 1
			return label_mask
		
		return img_array.astype(np.int32)
	raise ValueError(f"Unsupported file ending: {file_ending}")


def get_spacing_from_nifti(file_path: str) -> Optional[Tuple[float, ...]]:
	"""Get voxel spacing from NIfTI header."""
	try:
		nii = nib.load(file_path)
		return tuple(nii.header.get_zooms())
	except Exception:
		return None


def detect_file_ending(directory: str) -> str:
	"""Auto-detect file ending by scanning supported endings recursively."""
	supported_endings = [".nii.gz", ".png", ".tif", ".tiff"]
	for ending in supported_endings:
		files = glob.glob(f"{directory}/**/*{ending}", recursive=True)
		if files:
			print(f"Auto-detected file ending in {directory}: {ending}")
			return ending
	raise ValueError(
		f"Could not auto-detect file ending in {directory}. "
		f"Supported endings: {supported_endings}"
	)


def strip_known_file_ending(file_name: str, file_ending: str) -> str:
	"""Remove a known file ending from file name."""
	if file_name.endswith(file_ending):
		return file_name[: -len(file_ending)]
	return os.path.splitext(file_name)[0]


def strip_any_supported_file_ending(file_name: str) -> str:
	"""Remove any supported image file ending from a file name."""
	supported_endings = [".nii.gz", ".png", ".tif", ".tiff"]
	for ending in supported_endings:
		if file_name.endswith(ending):
			return file_name[: -len(ending)]
	return os.path.splitext(file_name)[0]


def canonicalize_sample_id(sample_id: str) -> str:
	"""
	Convert sample identifiers from different naming schemes to a comparable form.

	Examples:
	- LIDC_0001-0 -> 0001_0
	- LIDC-IDRI-0001_0 -> 0001_0
	- 0001_0 -> 0001_0
	"""
	sid = sample_id.strip()

	if sid.endswith("_0000"):
		sid = sid[: -len("_0000")]

	match_lidc_pair = re.match(
		r"^(?:LIDC-IDRI-|LIDC_)?(?P<pid>\d+)[_-](?P<nodule>\d+)$",
		sid,
	)
	if match_lidc_pair:
		return f"{match_lidc_pair.group('pid')}_{match_lidc_pair.group('nodule')}"

	match_lidc_single = re.match(r"^(?:LIDC-IDRI-|LIDC_)?(?P<pid>\d+)$", sid)
	if match_lidc_single:
		return match_lidc_single.group("pid")

	return sid


def parse_dataset_ids_argument(dataset_ids_arg: List[str]) -> List[str]:
	"""Parse dataset IDs from CLI argument values."""
	normalized_ids: List[str] = []

	for arg_chunk in dataset_ids_arg:
		tokens = [token for token in re.split(r"[\s,]+", arg_chunk.strip()) if token]
		for token in tokens:
			match = re.match(r"^(?:Dataset)?(?P<id>\d{1,4})$", token, flags=re.IGNORECASE)
			if match is None:
				raise ValueError(
					f"Invalid dataset ID token: '{token}'. "
					"Use IDs like 041 or Dataset041 (comma and space separators are supported)."
				)
			normalized_ids.append(match.group("id").zfill(3))

	if len(normalized_ids) == 0:
		raise ValueError("No valid --dataset_ids provided.")

	return sorted(set(normalized_ids))


def get_nnunet_preprocessed_dir_from_env() -> str:
	"""Get nnUNet preprocessed path from supported environment variable names."""
	env_var_candidates = ["nnUNet_preprocessed", "nnUNet-preprocessed"]

	for env_var in env_var_candidates:
		env_value = os.environ.get(env_var)
		if env_value is None or env_value.strip() == "":
			continue
		if not os.path.isdir(env_value):
			raise ValueError(
				f"Environment variable {env_var} is set, but directory does not exist: {env_value}"
			)
		print(f"Using {env_var}: {env_value}")
		return env_value

	raise ValueError(
		"Could not find nnUNet preprocessed directory from env vars. "
		"Please set nnUNet_preprocessed."
	)


def resolve_dataset_preprocessed_dir(nnunet_preprocessed_dir: str, dataset_id: str) -> str:
	"""Resolve DatasetXXX_* folder inside nnUNet_preprocessed for a dataset ID."""
	pattern = os.path.join(nnunet_preprocessed_dir, f"Dataset{dataset_id}_*")
	matches = sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])

	if len(matches) == 0:
		raise ValueError(
			f"No dataset folder found for ID {dataset_id} in {nnunet_preprocessed_dir}. "
			f"Expected something like Dataset{dataset_id}_*"
		)

	if len(matches) > 1:
		print(
			f"WARNING: Multiple dataset folders found for ID {dataset_id}. "
			f"Using first: {matches[0]}"
		)

	return matches[0]


def collect_sample_ids_from_preprocessed_dataset(dataset_dir: str) -> Set[str]:
	"""
	Collect canonical sample IDs from one nnUNet preprocessed dataset.

	Primary source is gt_segmentations/*. If missing, falls back to splits_final.json.
	"""
	collected_ids: Set[str] = set()

	gt_segmentations_dir = os.path.join(dataset_dir, "gt_segmentations")
	if os.path.isdir(gt_segmentations_dir):
		for f_path in sorted(glob.glob(os.path.join(gt_segmentations_dir, "*"))):
			if not os.path.isfile(f_path):
				continue
			stem = strip_any_supported_file_ending(os.path.basename(f_path))
			canonical_id = canonicalize_sample_id(stem)
			if canonical_id != "":
				collected_ids.add(canonical_id)

		if len(collected_ids) > 0:
			return collected_ids

		print(
			f"WARNING: gt_segmentations exists but no sample IDs were read in {gt_segmentations_dir}. "
			"Falling back to splits_final.json if available."
		)

	splits_path = os.path.join(dataset_dir, "splits_final.json")
	if os.path.isfile(splits_path):
		try:
			with open(splits_path, "r") as f:
				splits = json.load(f)

			for split_entry in splits:
				if not isinstance(split_entry, dict):
					continue
				for key in ["train", "val", "test"]:
					for case_id in split_entry.get(key, []):
						canonical_id = canonicalize_sample_id(str(case_id))
						if canonical_id != "":
							collected_ids.add(canonical_id)
		except Exception as exc:
			raise ValueError(
				f"Failed reading fallback splits file {splits_path}: {exc}"
			) from exc

	if len(collected_ids) == 0:
		raise ValueError(
			f"Could not collect sample IDs for dataset folder {dataset_dir}. "
			"Expected gt_segmentations files and/or splits_final.json."
		)

	return collected_ids


def collect_allowed_sample_ids_from_nnunet_preprocessed(
	nnunet_preprocessed_dir: str,
	dataset_ids: List[str],
) -> Set[str]:
	"""Collect union of canonical sample IDs across requested nnUNet datasets."""
	allowed_sample_ids: Set[str] = set()

	for dataset_id in dataset_ids:
		dataset_dir = resolve_dataset_preprocessed_dir(nnunet_preprocessed_dir, dataset_id)
		dataset_sample_ids = collect_sample_ids_from_preprocessed_dataset(dataset_dir)
		allowed_sample_ids |= dataset_sample_ids
		print(
			f"Dataset {dataset_id}: collected {len(dataset_sample_ids)} sample IDs "
			f"from {dataset_dir}"
		)

	if len(allowed_sample_ids) == 0:
		raise ValueError(
			"Collected 0 sample IDs from requested datasets in nnUNet_preprocessed."
		)

	return allowed_sample_ids


def parse_multirater_sample_and_rater(file_name: str, file_ending: str, riga_mode: bool = False) -> Tuple[Optional[str], Optional[str]]:
	"""
	Parse sample_id and rater_id from a multirater mask file name.

	Supports patterns such as:
	- LIDC-IDRI-0001_0_3_SEG.nii.gz
	- 0001_0_3_SEG.nii.gz
	- <sample>_<rater>_SEG.<ext>
	- GleasonXAI all-raters: <sample>_all_raters_annotator_<rater>_mask.<ext>
	- RIGA mode (riga_mode=True): <sample>-<rater_id>.<ext>  (e.g. image1-1.tif)
	"""
	stem = strip_known_file_ending(os.path.basename(file_name), file_ending)

	lidc_match = re.match(
		r"^(?:LIDC-IDRI-)?(?P<pid>\d+)_(?P<nodule>\d+)_(?P<rater>\d+)_SEG$",
		stem,
	)
	if lidc_match:
		sample_id = canonicalize_sample_id(
			f"{lidc_match.group('pid')}_{lidc_match.group('nodule')}"
		)
		return sample_id, lidc_match.group("rater")

	generic_match = re.match(r"^(?P<sample>.+)_(?P<rater>\d+)_SEG$", stem)
	if generic_match:
		return canonicalize_sample_id(generic_match.group("sample")), generic_match.group("rater")

	gleason_all_raters_match = re.match(
		r"^(?P<sample>.+)_all_raters_annotator_(?P<rater>.+)_mask$",
		stem,
	)
	if gleason_all_raters_match:
		return (
			canonicalize_sample_id(gleason_all_raters_match.group("sample")),
			gleason_all_raters_match.group("rater"),
		)

	if riga_mode:
		# RIGA naming convention: <sample>-<rater_id>  (e.g. image1-1)
		riga_match = re.match(r"^(?P<sample>.+)-(?P<rater>\d+)$", stem)
		if riga_match:
			return canonicalize_sample_id(riga_match.group("sample")), riga_match.group("rater")

	return None, None


def parse_consensus_sample(file_name: str, file_ending: str, riga_mode: bool = False) -> Optional[str]:
	"""
	Parse sample_id from consensus mask file name.

	Supports patterns such as:
	- 0001_0_fused_annotator_majority_SEG.nii.gz
	- LIDC-IDRI-0001_0_fused_random_SEG.nii.gz
	- <sample>_fused_<...>_SEG.<ext>
	- GleasonXAI consensus: <sample>_consensus_staple_mask.<ext>
	- RIGA mode (riga_mode=True): <sample>mask.<ext>  (e.g. image1mask.tif)
	"""
	stem = strip_known_file_ending(os.path.basename(file_name), file_ending)

	lidc_match = re.match(
		r"^(?:LIDC-IDRI-)?(?P<pid>\d+)_(?P<nodule>\d+)_fused(?:_.+)?_SEG$",
		stem,
	)
	if lidc_match:
		return canonicalize_sample_id(
			f"{lidc_match.group('pid')}_{lidc_match.group('nodule')}"
		)

	generic_match = re.match(r"^(?P<sample>.+)_fused(?:_.+)?_SEG$", stem)
	if generic_match:
		return canonicalize_sample_id(generic_match.group("sample"))

	fallback_match = re.match(r"^(?P<sample>.+)_SEG$", stem)
	if fallback_match:
		return canonicalize_sample_id(fallback_match.group("sample"))

	gleason_consensus_match = re.match(r"^(?P<sample>.+)_consensus_staple_mask$", stem)
	if gleason_consensus_match:
		return canonicalize_sample_id(gleason_consensus_match.group("sample"))

	if riga_mode:
		# RIGA naming convention: <sample>mask  (e.g. image1mask)
		riga_match = re.match(r"^(?P<sample>.+)mask$", stem, re.IGNORECASE)
		if riga_match:
			return canonicalize_sample_id(riga_match.group("sample"))

	return None


def _nanmean_std(values: List[float], ignore_inf: bool = False) -> Tuple[float, float]:
	"""Compute nan-aware mean and std with optional inf filtering."""
	arr = np.array(values, dtype=np.float64)
	valid = ~np.isnan(arr)
	if ignore_inf:
		valid = valid & ~np.isinf(arr)
	if not np.any(valid):
		return np.nan, np.nan
	return float(np.mean(arr[valid])), float(np.std(arr[valid]))


def analyze_sample(
	consensus_mask: np.ndarray,
	rater_masks: Dict[str, np.ndarray],
	spacing: Optional[Tuple[float, ...]] = None,
	instance_match_iou_threshold: float = 0.1,
) -> Dict:
	"""Analyze one sample: consensus mask against all rater masks."""
	rater_ids = sorted(rater_masks.keys())

	all_classes_set = set(np.unique(consensus_mask))
	for r_id in rater_ids:
		all_classes_set |= set(np.unique(rater_masks[r_id]))

	all_classes = sorted(int(c) for c in all_classes_set)
	fg_classes = [c for c in all_classes if c > 0]

	all_rater_classes = set()
	for r_id in rater_ids:
		all_rater_classes |= set(np.unique(rater_masks[r_id]))

	results = {
		"raters": {
			"num_raters": int(len(rater_ids)),
			"rater_ids": rater_ids,
		},
		"classes": {
			"all_classes": [int(c) for c in all_classes],
			"fg_classes": [int(c) for c in fg_classes],
			"only_in_consensus": [
				int(c) for c in set(np.unique(consensus_mask)) - all_rater_classes
			],
			"only_in_raters": [
				int(c) for c in all_rater_classes - set(np.unique(consensus_mask))
			],
		},
		"per_class_metrics": {},
		"class_confusion_metrics": {"consensus_vs_raters": {}},
		"overall_metrics": {},
	}

	for r_id in rater_ids:
		pixel_cls_conf = compute_pixel_cls_conf(
			consensus_mask,
			rater_masks[r_id],
			foreground_classes=fg_classes,
		)
		instance_cls_conf = compute_instance_cls_conf(
			consensus_mask,
			rater_masks[r_id],
			foreground_classes=fg_classes,
			overlap_iou_threshold=instance_match_iou_threshold,
		)
		instance_cls_conf.pop("matches", None)
		results["class_confusion_metrics"]["consensus_vs_raters"][r_id] = {
			"PixelClsConf": pixel_cls_conf,
			"InstanceClsConf": instance_cls_conf,
		}

	for class_id in fg_classes:
		consensus_binary = consensus_mask == class_id
		binary_rater_masks = [rater_masks[r_id] == class_id for r_id in rater_ids]

		class_metrics = {
			"fleiss_kappa": compute_fleiss_kappa_binary(binary_rater_masks),
			"voxelwise_entropy": compute_voxelwise_entropy_binary(binary_rater_masks),
			"volume_consensus": int(np.sum(consensus_binary)),
			"consensus_vs_raters": {},
		}

		dice_values = []
		hd95_values = []
		voxel_precision_values = []
		voxel_recall_values = []
		voxel_f1_values = []
		instance_precision_values = []
		instance_recall_values = []
		instance_f1_values = []

		for r_id in rater_ids:
			rater_binary = rater_masks[r_id] == class_id

			dice = compute_dice(consensus_mask, rater_masks[r_id], class_id)
			hd95 = compute_hd95(
				consensus_mask,
				rater_masks[r_id],
				class_id,
				spacing=spacing,
			)
			voxel_prf = compute_voxel_prf(consensus_binary, rater_binary)
			instance_prf = compute_instance_prf(
				consensus_binary,
				rater_binary,
				overlap_iou_threshold=instance_match_iou_threshold,
			)

			class_metrics["consensus_vs_raters"][r_id] = {
				"dice": dice,
				"hd95": hd95,
				"voxel_level_prf": voxel_prf,
				"instance_level_prf": instance_prf,
				"PixelClsConf": results["class_confusion_metrics"][
					"consensus_vs_raters"
				][r_id]["PixelClsConf"]["per_class"].get(int(class_id)),
				"InstanceClsConf": results["class_confusion_metrics"][
					"consensus_vs_raters"
				][r_id]["InstanceClsConf"]["per_class"].get(
					int(class_id), {}
				).get("score"),
				"InstanceClsConfCoverage": results["class_confusion_metrics"][
					"consensus_vs_raters"
				][r_id]["InstanceClsConf"]["per_class"].get(
					int(class_id), {}
				).get("coverage"),
				"volume_rater": int(np.sum(rater_binary)),
			}

			dice_values.append(dice)
			hd95_values.append(hd95)
			voxel_precision_values.append(voxel_prf["precision"])
			voxel_recall_values.append(voxel_prf["recall"])
			voxel_f1_values.append(voxel_prf["f1"])
			instance_precision_values.append(instance_prf["precision"])
			instance_recall_values.append(instance_prf["recall"])
			instance_f1_values.append(instance_prf["f1"])

		class_metrics["mean_dice"], class_metrics["std_dice"] = _nanmean_std(
			dice_values, ignore_inf=False
		)
		class_metrics["mean_hd95"], class_metrics["std_hd95"] = _nanmean_std(
			hd95_values, ignore_inf=True
		)

		(
			class_metrics["mean_voxel_level_precision"],
			class_metrics["std_voxel_level_precision"],
		) = _nanmean_std(voxel_precision_values)
		(
			class_metrics["mean_voxel_level_recall"],
			class_metrics["std_voxel_level_recall"],
		) = _nanmean_std(voxel_recall_values)
		class_metrics["mean_voxel_level_f1"], class_metrics["std_voxel_level_f1"] = _nanmean_std(
			voxel_f1_values
		)

		(
			class_metrics["mean_instance_level_precision"],
			class_metrics["std_instance_level_precision"],
		) = _nanmean_std(instance_precision_values)
		(
			class_metrics["mean_instance_level_recall"],
			class_metrics["std_instance_level_recall"],
		) = _nanmean_std(instance_recall_values)
		(
			class_metrics["mean_instance_level_f1"],
			class_metrics["std_instance_level_f1"],
		) = _nanmean_std(instance_f1_values)

		results["per_class_metrics"][int(class_id)] = class_metrics

	pixel_scores = [
		metrics["PixelClsConf"]["score"]
		for metrics in results["class_confusion_metrics"]["consensus_vs_raters"].values()
		if metrics["PixelClsConf"]["score"] is not None
	]
	instance_scores = [
		metrics["InstanceClsConf"]["score"]
		for metrics in results["class_confusion_metrics"]["consensus_vs_raters"].values()
		if metrics["InstanceClsConf"]["score"] is not None
	]
	instance_coverages = [
		metrics["InstanceClsConf"]["coverage"]
		for metrics in results["class_confusion_metrics"]["consensus_vs_raters"].values()
		if metrics["InstanceClsConf"]["coverage"] is not None
	]
	results["class_confusion_metrics"]["mean_PixelClsConf"] = (
		float(np.mean(pixel_scores)) if pixel_scores else None
	)
	results["class_confusion_metrics"]["mean_InstanceClsConf"] = (
		float(np.mean(instance_scores)) if instance_scores else None
	)
	results["class_confusion_metrics"]["mean_InstanceClsConfCoverage"] = (
		float(np.mean(instance_coverages)) if instance_coverages else None
	)

	if fg_classes:
		valid_fleiss = [
			results["per_class_metrics"][c]["fleiss_kappa"]
			for c in fg_classes
			if not np.isnan(results["per_class_metrics"][c]["fleiss_kappa"])
		]
		results["overall_metrics"]["mean_fleiss_kappa"] = (
			float(np.mean(valid_fleiss)) if valid_fleiss else np.nan
		)

		valid_entropy = [
			results["per_class_metrics"][c]["voxelwise_entropy"]
			for c in fg_classes
			if not np.isnan(results["per_class_metrics"][c]["voxelwise_entropy"])
		]
		results["overall_metrics"]["mean_voxelwise_entropy"] = (
			float(np.mean(valid_entropy)) if valid_entropy else np.nan
		)

		valid_dice = [
			results["per_class_metrics"][c]["mean_dice"]
			for c in fg_classes
			if not np.isnan(results["per_class_metrics"][c]["mean_dice"])
		]
		results["overall_metrics"]["mean_dice"] = (
			float(np.mean(valid_dice)) if valid_dice else np.nan
		)

		valid_hd95 = [
			results["per_class_metrics"][c]["mean_hd95"]
			for c in fg_classes
			if not np.isnan(results["per_class_metrics"][c]["mean_hd95"])
			and not np.isinf(results["per_class_metrics"][c]["mean_hd95"])
		]
		results["overall_metrics"]["mean_hd95"] = (
			float(np.mean(valid_hd95)) if valid_hd95 else np.nan
		)

		for metric_scope in ["voxel_level", "instance_level"]:
			for score_name in ["precision", "recall", "f1"]:
				key = f"mean_{metric_scope}_{score_name}"
				class_key = f"mean_{metric_scope}_{score_name}"
				valid_scores = [
					results["per_class_metrics"][c][class_key]
					for c in fg_classes
					if not np.isnan(results["per_class_metrics"][c][class_key])
				]
				results["overall_metrics"][key] = (
					float(np.mean(valid_scores)) if valid_scores else np.nan
				)

		results["overall_metrics"]["total_volume_consensus_fg"] = int(
			np.sum(consensus_mask > 0)
		)

	return results


def collect_multirater_masks(
	multirater_dir: str,
	file_ending: str,
	riga_mode: bool = False,
) -> Dict[str, Dict[str, str]]:
	"""Collect multirater masks grouped by sample_id and rater_id.

	In riga_mode, the sample key includes the relative subdirectory path
	to avoid collisions across subdatasets that share image names.
	"""
	all_files = sorted(glob.glob(f"{multirater_dir}/**/*{file_ending}", recursive=True))
	grouped: Dict[str, Dict[str, str]] = {}

	for f_path in all_files:
		if not os.path.isfile(f_path):
			continue

		sample_id, rater_id = parse_multirater_sample_and_rater(
			os.path.basename(f_path),
			file_ending,
			riga_mode=riga_mode,
		)

		if riga_mode and sample_id is not None:
			rel_dir = os.path.relpath(os.path.dirname(f_path), multirater_dir).replace(os.sep, "/")
			if rel_dir not in ("", "."):
				sample_id = f"{rel_dir}/{sample_id}"
		if sample_id is None or rater_id is None:
			continue

		if sample_id not in grouped:
			grouped[sample_id] = {}

		if rater_id in grouped[sample_id]:
			print(
				f"WARNING: Duplicate rater file for sample {sample_id}, rater {rater_id}. "
				f"Keeping first: {grouped[sample_id][rater_id]}"
			)
			continue

		grouped[sample_id][rater_id] = f_path

	return grouped


def collect_consensus_masks(consensus_dir: str, file_ending: str, riga_mode: bool = False) -> Dict[str, str]:
	"""Collect consensus masks keyed by sample_id.

	In riga_mode, the sample key includes the relative subdirectory path
	to avoid collisions across subdatasets that share image names.
	"""
	all_files = sorted(glob.glob(f"{consensus_dir}/**/*{file_ending}", recursive=True))
	mapping: Dict[str, str] = {}

	for f_path in all_files:
		if not os.path.isfile(f_path):
			continue

		sample_id = parse_consensus_sample(os.path.basename(f_path), file_ending, riga_mode=riga_mode)
		if sample_id is None:
			continue

		if riga_mode:
			rel_dir = os.path.relpath(os.path.dirname(f_path), consensus_dir).replace(os.sep, "/")
			if rel_dir not in ("", "."):
				sample_id = f"{rel_dir}/{sample_id}"

		if sample_id in mapping:
			print(
				f"WARNING: Duplicate consensus file for sample {sample_id}. "
				f"Keeping first: {mapping[sample_id]}"
			)
			continue

		mapping[sample_id] = f_path

	return mapping


def collect_masks_by_pattern(
	directory: str,
	file_ending: str,
	rater_pattern: str,
	consensus_pattern: str,
	disambiguate_by_path: bool = False,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
	"""
	Collect rater and consensus masks from a unified directory using filename patterns.

	Rater masks: filenames containing rater_pattern (e.g., "Annotator_")
	  - If pattern at start: first token after pattern = rater_id, rest = sample_id
	  - If pattern in middle: text before pattern = sample_id, first token after = rater_id
	  - If sample_id is not present in filename: parent directory name is used as sample_id

	Consensus masks: filenames containing consensus_pattern (e.g., "STAPLE_")
	  - If pattern at start: text after pattern = sample_id
	  - If pattern in middle: text before pattern = sample_id
	  - If sample_id is not present in filename: parent directory name is used as sample_id

	Args:
		directory: Directory to search recursively
		file_ending: File extension to match (e.g., ".nii.gz")
		rater_pattern: String pattern to identify rater masks (e.g., "Annotator_", "_rater_")
		consensus_pattern: String pattern to identify consensus masks (e.g., "STAPLE_", "_consensus")
		disambiguate_by_path: If True, append relative parent directory to sample key
			to avoid collapsing duplicate sample IDs across different subfolders.

	Returns:
		Tuple of (rater_masks_dict, consensus_masks_dict)
	"""
	all_files = sorted(glob.glob(f"{directory}/**/*{file_ending}", recursive=True))
	rater_masks: Dict[str, Dict[str, str]] = {}
	consensus_masks: Dict[str, str] = {}
	rater_file_matches = 0
	consensus_file_matches = 0
	rater_duplicates = 0
	consensus_duplicates = 0
	rater_duplicate_examples: List[str] = []
	consensus_duplicate_examples: List[str] = []

	def build_sample_key(sample_id: str, path: str) -> str:
		if not disambiguate_by_path:
			return sample_id
		rel_dir = os.path.relpath(os.path.dirname(path), directory).replace(os.sep, "/")
		if rel_dir in ["", "."]:
			return sample_id
		return f"{sample_id}@@{rel_dir}"

	for f_path in all_files:
		if not os.path.isfile(f_path):
			continue

		filename = os.path.basename(f_path)
		file_stem = strip_any_supported_file_ending(filename)
		parent_name = os.path.basename(os.path.dirname(f_path))
		parent_sample_id = canonicalize_sample_id(parent_name) if parent_name else None

		# Check if it's a consensus mask
		if consensus_pattern in file_stem:
			consensus_file_matches += 1
			# Extract sample_id
			parts = file_stem.split(consensus_pattern, 1)  # Split only on first occurrence
			if len(parts) >= 2:
				before_pattern = parts[0].strip("_- ")
				after_pattern = parts[1].strip("_- ")

				sample_candidate = before_pattern if before_pattern else after_pattern
				if sample_candidate:
					sample_id = canonicalize_sample_id(sample_candidate)
				else:
					# Filename does not encode sample ID (e.g., singlerater_label_majority.nii.gz)
					sample_id = parent_sample_id

				if sample_id:
					sample_key = build_sample_key(sample_id, f_path)
					if sample_key not in consensus_masks:
						consensus_masks[sample_key] = f_path
					else:
						consensus_duplicates += 1
						if len(consensus_duplicate_examples) < 5:
							consensus_duplicate_examples.append(
								f"sample={sample_key} kept={consensus_masks[sample_key]} skipped={f_path}"
							)

		# Check if it's a rater mask
		elif rater_pattern in file_stem:
			rater_file_matches += 1
			# Extract sample_id and rater_id
			parts = file_stem.split(rater_pattern, 1)  # Split only on first occurrence
			if len(parts) >= 2:
				before_pattern = parts[0].strip("_- ")
				after_pattern = parts[1].strip("_- ")

				sample_candidate: Optional[str] = None
				rater_id: Optional[str] = None

				if before_pattern:
					# Pattern in middle: before = sample_id, first token after = rater_id
					sample_candidate = before_pattern
					# Extract first alphanumeric token as rater_id
					rater_match = re.match(r'^([A-Za-z0-9]+)', after_pattern)
					if rater_match:
						rater_id = rater_match.group(1)
				else:
					# Pattern at start: parse rater_id first, then optional sample suffix
					# Examples:
					# - Annotator_A_M01_0h -> rater=A, sample=M01_0h
					# - label_a1 -> rater=1, sample from parent directory
					rater_and_sample = re.match(r'^([A-Za-z0-9]+)(?:[_\-\s]+(.*))?$', after_pattern)
					if rater_and_sample:
						rater_id = rater_and_sample.group(1)
						rest = (rater_and_sample.group(2) or '').strip("_- ")
						if rest:
							sample_candidate = rest

				if sample_candidate:
					sample_id = canonicalize_sample_id(sample_candidate)
				else:
					# Filename does not encode sample ID (e.g., label_a1.nii.gz)
					sample_id = parent_sample_id

				if sample_id and rater_id:
					sample_key = build_sample_key(sample_id, f_path)
					if sample_key not in rater_masks:
						rater_masks[sample_key] = {}
					if rater_id not in rater_masks[sample_key]:
						rater_masks[sample_key][rater_id] = f_path
					else:
						rater_duplicates += 1
						if len(rater_duplicate_examples) < 5:
							rater_duplicate_examples.append(
								f"sample={sample_key},rater={rater_id} kept={rater_masks[sample_key][rater_id]} skipped={f_path}"
							)

	unique_rater_entries = sum(len(rater_map) for rater_map in rater_masks.values())
	print("Unified-pattern discovery summary:")
	print(f"  Matched rater files:     {rater_file_matches}")
	print(f"  Matched consensus files: {consensus_file_matches}")
	if disambiguate_by_path:
		print("  Sample key mode:         disambiguated by relative directory")
	print(f"  Unique sample IDs (r):   {len(rater_masks)}")
	print(f"  Unique sample IDs (c):   {len(consensus_masks)}")
	print(f"  Unique (sample,rater):   {unique_rater_entries}")
	if rater_duplicates > 0 or consensus_duplicates > 0:
		print("  Duplicate entries skipped due identical logical IDs:")
		print(f"    rater duplicates:     {rater_duplicates}")
		print(f"    consensus duplicates: {consensus_duplicates}")
		for example in rater_duplicate_examples:
			print(f"    e.g. {example}")
		for example in consensus_duplicate_examples:
			print(f"    e.g. {example}")

	return rater_masks, consensus_masks


def main(args):
	"""Main analysis function."""

	def base_sample_id(sample_key: str) -> str:
		"""Strip optional unified disambiguation suffix from sample key."""
		if "@@" in sample_key:
			return sample_key.split("@@", 1)[0]
		return sample_key

	# Determine which mode to use: separate directories or unified directory
	unified_dir = args.unified_dir
	multirater_dir = args.multirater_dir
	consensus_dir = args.consensus_dir

	if unified_dir:
		# Mode: Unified directory with patterns
		if multirater_dir or consensus_dir:
			raise ValueError(
				"Cannot use --unified_dir together with --multirater_dir or --consensus_dir. "
				"Choose one mode: either (--multirater_dir + --consensus_dir) or --unified_dir."
			)
		if not os.path.isdir(unified_dir):
			raise ValueError(f"--unified_dir does not exist: {unified_dir}")
	else:
		# Mode: Separate directories (legacy)
		if args.disambiguate_by_path:
			raise ValueError("--disambiguate_by_path is only supported with --unified_dir")
		if not multirater_dir or not consensus_dir:
			raise ValueError(
				"Must provide either (--multirater_dir + --consensus_dir) or --unified_dir. "
				"For unified directory mode with patterns, use --unified_dir with --rater_pattern and --consensus_pattern."
			)
		if not os.path.isdir(multirater_dir):
			raise ValueError(f"multirater_dir does not exist: {multirater_dir}")
		if not os.path.isdir(consensus_dir):
			raise ValueError(f"consensus_dir does not exist: {consensus_dir}")

	# Dataset ID filtering (optional for unified_dir mode)
	if args.dataset_ids is not None and args.riga_mode:
		# nnUNet preprocessed IDs for RIGA use a different naming scheme
		# (e.g. RIGABinRushed_0554) that does not match the raw RIGA filenames
		# (e.g. image1). Filtering by dataset_ids is therefore incompatible with
		# riga_mode and would always result in 0 samples. Skipping filter.
		print(
			"WARNING: --dataset_ids is ignored in --riga_mode because nnUNet preprocessed "
			"IDs (e.g. RIGABinRushed_0554) do not match raw RIGA filenames (e.g. image1). "
			"All samples found in the provided directories will be analyzed."
		)
		dataset_ids = []
		allowed_sample_ids = None
	elif args.dataset_ids is not None:
		dataset_ids = parse_dataset_ids_argument(args.dataset_ids)
		nnunet_preprocessed_dir = get_nnunet_preprocessed_dir_from_env()
		allowed_sample_ids = collect_allowed_sample_ids_from_nnunet_preprocessed(
			nnunet_preprocessed_dir,
			dataset_ids,
		)
	else:
		if unified_dir is None:
			raise ValueError(
				"--dataset_ids is required when using --multirater_dir and --consensus_dir. "
				"For --unified_dir, dataset_ids is optional."
			)
		dataset_ids = []
		allowed_sample_ids = None  # No filtering

	file_ending = args.file_ending
	if file_ending == "auto" or file_ending is None:
		# Auto-detect from one of the directories
		detect_dir = unified_dir if unified_dir else consensus_dir
		file_ending = detect_file_ending(detect_dir)

	print(f"Using file ending: {file_ending}")
	if allowed_sample_ids is not None:
		print(f"Dataset IDs:   {dataset_ids}")
		print(f"Allowed IDs:   {len(allowed_sample_ids)}")
	elif args.dataset_ids is not None:
		print(f"Dataset IDs:   {dataset_ids if dataset_ids else args.dataset_ids}")
		print("Allowed IDs:   None (filtering disabled)")
	else:
		print(f"Dataset IDs:   None (no filtering)")

	# Collect masks using the appropriate method
	if unified_dir:
		print(f"Unified directory: {unified_dir}")
		print(f"  Rater pattern: {args.rater_pattern}")
		print(f"  Consensus pattern: {args.consensus_pattern}")
		multirater_map, consensus_map = collect_masks_by_pattern(
			unified_dir,
			file_ending,
			args.rater_pattern,
			args.consensus_pattern,
			disambiguate_by_path=args.disambiguate_by_path,
		)
	else:
		print(f"Multirater dir: {multirater_dir}")
		print(f"Consensus dir:  {consensus_dir}")
		multirater_map = collect_multirater_masks(multirater_dir, file_ending, riga_mode=args.riga_mode)
		consensus_map = collect_consensus_masks(consensus_dir, file_ending, riga_mode=args.riga_mode)

	n_multirater_before = len(multirater_map)
	n_consensus_before = len(consensus_map)

	# Apply dataset ID filtering only if allowed_sample_ids is provided
	if allowed_sample_ids is not None:
		multirater_map = {
			sample_id: rater_map
			for sample_id, rater_map in multirater_map.items()
			if base_sample_id(sample_id) in allowed_sample_ids
		}
		consensus_map = {
			sample_id: consensus_path
			for sample_id, consensus_path in consensus_map.items()
			if base_sample_id(sample_id) in allowed_sample_ids
		}
		print(
			f"Found {n_multirater_before} samples in multirater masks "
			f"-> {len(multirater_map)} after dataset-ID filtering"
		)
		print(
			f"Found {n_consensus_before} samples in consensus masks "
			f"-> {len(consensus_map)} after dataset-ID filtering"
		)
	else:
		print(f"Found {n_multirater_before} samples in multirater masks (no filtering)")
		print(f"Found {n_consensus_before} samples in consensus masks (no filtering)")

	common_sample_ids = sorted(set(multirater_map.keys()) & set(consensus_map.keys()))
	print(f"Common samples to analyze: {len(common_sample_ids)}")

	if len(common_sample_ids) == 0:
		raise ValueError(
			"No common sample IDs found between multirater and consensus masks. "
			"Check file naming and directories."
		)

	if args.output_json is not None:
		output_json = args.output_json
	else:
		os.makedirs(args.output_dir, exist_ok=True)
		dataset_tag = "-".join(dataset_ids) if dataset_ids else "all"
		multi_tag = os.path.basename(os.path.normpath(multirater_dir))
		cons_tag = os.path.basename(os.path.normpath(consensus_dir))
		output_json = os.path.join(
			args.output_dir,
			f"multirater_consensus_analysis_ds{dataset_tag}_multi_{multi_tag}_cons_{cons_tag}.json",
		)

	os.makedirs(os.path.dirname(output_json), exist_ok=True)

	persisted_results = {}
	if os.path.exists(output_json):
		try:
			with open(output_json, "r") as f:
				persisted_results = json.load(f)
			removed_from_persisted = 0
			for sample_id in list(persisted_results.keys()):
				if sample_id not in common_sample_ids:
					persisted_results.pop(sample_id)
					removed_from_persisted += 1
			print(
				f"Found existing output JSON with {len(persisted_results)} samples. "
				"Resuming and skipping already processed samples."
			)
			if removed_from_persisted > 0:
				print(
					f"Dropped {removed_from_persisted} persisted samples not in current filtered set."
				)
		except Exception as exc:
			print(
				f"WARNING: Could not read existing output JSON at {output_json} ({exc}). "
				"Starting with empty result set."
			)
			persisted_results = {}

	processed_sample_ids = {
		sample_id
		for sample_id, result in persisted_results.items()
		if result.get("class_confusion_metrics", {})
		.get("consensus_vs_raters", {})
		and all(
			metrics.get("InstanceClsConf", {}).get("transition_matrix") is not None
			for metrics in result["class_confusion_metrics"][
				"consensus_vs_raters"
			].values()
		)
	}
	num_outdated = len(persisted_results) - len(processed_sample_ids)
	if num_outdated:
		print(
			f"Recomputing {num_outdated} existing samples that predate "
			"PixelClsConf/InstanceClsConf."
		)
	all_results = dict(persisted_results)

	n_processed_now = 0
	save_every = max(1, int(args.save_every))

	progress_bar = tqdm(common_sample_ids, desc="Analyzing samples", unit="sample")
	for sample_id in progress_bar:
		if sample_id in processed_sample_ids:
			continue

		consensus_path = consensus_map[sample_id]
		rater_paths = multirater_map[sample_id]
		if len(rater_paths) == 0:
			print(f"WARNING: No rater masks for sample {sample_id}, skipping")
			continue

		consensus_mask = load_mask(consensus_path, file_ending, riga_mode=args.riga_mode)
		spacing = get_spacing_from_nifti(consensus_path) if file_ending == ".nii.gz" else None

		rater_masks: Dict[str, np.ndarray] = {}
		for rater_id, r_path in sorted(rater_paths.items()):
			r_mask = load_mask(r_path, file_ending, riga_mode=args.riga_mode)
			if r_mask.shape != consensus_mask.shape:
				print(
					f"WARNING: Shape mismatch for sample {sample_id}, rater {rater_id}: "
					f"consensus shape {consensus_mask.shape}, rater shape {r_mask.shape}. Skipping rater."
				)
				continue
			rater_masks[rater_id] = r_mask

		if len(rater_masks) == 0:
			print(f"WARNING: No valid rater masks for sample {sample_id}, skipping")
			continue

		sample_results = analyze_sample(
			consensus_mask=consensus_mask,
			rater_masks=rater_masks,
			spacing=spacing,
			instance_match_iou_threshold=args.instance_match_iou_threshold,
		)

		sample_results["paths"] = {
			"consensus": consensus_path,
			"raters": {r_id: rater_paths[r_id] for r_id in sorted(rater_masks.keys())},
		}

		all_results[sample_id] = sample_results
		n_processed_now += 1

		if n_processed_now % save_every == 0:
			with open(output_json, "w") as f:
				json.dump(all_results, f, indent=2)

	with open(output_json, "w") as f:
		json.dump(all_results, f, indent=2)

	print(f"\nSaved analysis to: {output_json}")
	print(f"Total samples in output: {len(all_results)}")

	if all_results:
		overall_fleiss = [
			s["overall_metrics"].get("mean_fleiss_kappa", np.nan)
			for s in all_results.values()
			if "overall_metrics" in s
		]
		overall_entropy = [
			s["overall_metrics"].get("mean_voxelwise_entropy", np.nan)
			for s in all_results.values()
			if "overall_metrics" in s
		]
		overall_dice = [
			s["overall_metrics"].get("mean_dice", np.nan)
			for s in all_results.values()
			if "overall_metrics" in s
		]
		overall_hd95 = [
			s["overall_metrics"].get("mean_hd95", np.nan)
			for s in all_results.values()
			if "overall_metrics" in s
		]

		valid_fleiss = np.array(overall_fleiss, dtype=np.float64)
		valid_fleiss = valid_fleiss[~np.isnan(valid_fleiss)]
		valid_entropy = np.array(overall_entropy, dtype=np.float64)
		valid_entropy = valid_entropy[~np.isnan(valid_entropy)]
		valid_dice = np.array(overall_dice, dtype=np.float64)
		valid_dice = valid_dice[~np.isnan(valid_dice)]
		valid_hd95 = np.array(overall_hd95, dtype=np.float64)
		valid_hd95 = valid_hd95[~np.isnan(valid_hd95) & ~np.isinf(valid_hd95)]

		print("\nSummary across samples:")
		if len(valid_fleiss) > 0:
			print(f"  Mean Fleiss kappa:      {np.mean(valid_fleiss):.4f} ± {np.std(valid_fleiss):.4f}")
		if len(valid_entropy) > 0:
			print(f"  Mean voxelwise entropy: {np.mean(valid_entropy):.4f} ± {np.std(valid_entropy):.4f}")
		if len(valid_dice) > 0:
			print(f"  Mean Dice:              {np.mean(valid_dice):.4f} ± {np.std(valid_dice):.4f}")
		if len(valid_hd95) > 0:
			print(f"  Mean HD95:              {np.mean(valid_hd95):.4f} ± {np.std(valid_hd95):.4f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Analyze multirater masks vs consensus masks with per-class metrics"
	)
	parser.add_argument(
		"--dataset_ids",
		type=str,
		nargs="+",
		default=None,
		help=(
			"nnUNet dataset IDs to define allowed samples (e.g., 041 042 or Dataset041,Dataset042). "
			"Sample IDs are read from $nnUNet_preprocessed. Optional when using --unified_dir (if not provided, all found samples are analyzed)."
		),
	)

	# Option 1: Separate directories for rater and consensus masks
	parser.add_argument(
		"--multirater_dir",
		type=str,
		default=None,
		help="Directory containing multirater masks (supports recursive search). Either use this with --consensus_dir, or use --unified_dir instead.",
	)
	parser.add_argument(
		"--consensus_dir",
		type=str,
		default=None,
		help="Directory containing consensus masks (supports recursive search). Either use this with --multirater_dir, or use --unified_dir instead.",
	)

	# Option 2: Unified directory with pattern-based mask discovery
	parser.add_argument(
		"--unified_dir",
		type=str,
		default=None,
		help="Unified directory containing both rater and consensus masks (supports recursive search). Use this with --rater_pattern and --consensus_pattern instead of --multirater_dir/--consensus_dir.",
	)
	parser.add_argument(
		"--rater_pattern",
		type=str,
		default="Annotator_",
		help="Filename pattern to identify rater masks (default: 'Annotator_'). Example: 'Annotator_', '_rater_', or 'annotator_'. Sample ID is extracted as text before pattern, rater ID as first number after pattern.",
	)
	parser.add_argument(
		"--consensus_pattern",
		type=str,
		default="STAPLE_",
		help="Filename pattern to identify consensus masks (default: 'STAPLE_'). Example: 'STAPLE_', '_consensus', or 'consensus_'. Sample ID is extracted as text before pattern.",
	)
	parser.add_argument(
		"--disambiguate_by_path",
		action="store_true",
		help=(
			"Only for --unified_dir mode. If set, sample keys include relative parent folder "
			"(sample_id@@relative/path) so repeated IDs in different subfolders are treated as distinct samples."
		),
	)
	parser.add_argument(
		"--file_ending",
		type=str,
		default="auto",
		help="Mask file ending (default: auto). Options: auto, .nii.gz, .png, .tif, .tiff",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./results/noise_analysis",
		help="Output directory for JSON when --output_json is not provided",
	)
	parser.add_argument(
		"--output_json",
		type=str,
		default=None,
		help="Optional explicit output JSON path",
	)
	parser.add_argument(
		"--instance_match_iou_threshold",
		type=float,
		default=0.1,
		help="IoU threshold for instance-level matching (default: 0.1)",
	)
	parser.add_argument(
		"--save_every",
		type=int,
		default=20,
		help="Save intermediate JSON every N newly processed samples (default: 20)",
	)
	parser.add_argument(
		"--riga_mode",
		action="store_true",
		help=(
			"Enable RIGA RGB TIF mode. When set, RGB TIF images are interpreted with color mapping: "
			"red channel 255 -> label 2, red channel 120 -> label 1, everything else -> label 0."
		),
	)

	args = parser.parse_args()
	main(args)
