"""Pixel- and instance-level foreground class-confusion metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def compute_pixel_cls_conf(
    reference: np.ndarray,
    prediction: np.ndarray,
    foreground_classes: list[int] | None = None,
) -> dict[str, Any]:
    """Compute foreground-to-other-foreground pixel confusion.

    Background transitions are excluded in both directions. The overall score
    is the macro-average over reference foreground classes present in the
    sample. A present class has score zero when no other foreground class is
    available.
    """
    if foreground_classes is None:
        foreground_classes = sorted(
            int(c)
            for c in (set(np.unique(reference)) | set(np.unique(prediction)))
            if int(c) != 0
        )
    else:
        foreground_classes = sorted(
            {int(c) for c in foreground_classes if int(c) != 0}
        )

    per_class: dict[int, float | None] = {}
    valid_scores = []
    for source_class in foreground_classes:
        source_mask = reference == source_class
        source_size = int(np.sum(source_mask))
        if source_size == 0:
            per_class[source_class] = None
            continue

        other_classes = [c for c in foreground_classes if c != source_class]
        if not other_classes:
            score = 0.0
        else:
            score = float(
                np.sum(source_mask & np.isin(prediction, other_classes))
                / source_size
            )
        per_class[source_class] = score
        valid_scores.append(score)

    return {
        "score": float(np.mean(valid_scores)) if valid_scores else None,
        "per_class": per_class,
    }


def _class_aware_instance_map(
    segmentation: np.ndarray,
    foreground_classes: list[int],
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Return global instance IDs, instance-to-class labels, and sizes."""
    instance_map = np.zeros(segmentation.shape, dtype=np.int32)
    class_by_instance: dict[int, int] = {}
    size_by_instance: dict[int, int] = {}
    next_instance_id = 1

    for class_id in foreground_classes:
        labeled, num_components = ndimage.label(
            (segmentation == class_id).astype(np.uint8)
        )
        for component_id in range(1, num_components + 1):
            component = labeled == component_id
            instance_map[component] = next_instance_id
            class_by_instance[next_instance_id] = int(class_id)
            size_by_instance[next_instance_id] = int(np.sum(component))
            next_instance_id += 1

    return instance_map, class_by_instance, size_by_instance


def compute_instance_cls_conf(
    reference: np.ndarray,
    prediction: np.ndarray,
    foreground_classes: list[int] | None = None,
    overlap_iou_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute semantic class confusion among class-agnostically matched instances.

    Instances are connected components extracted separately for each foreground
    class. Reference and prediction instances are greedily matched one-to-one by
    descending IoU without using their semantic labels. ``score`` is the
    fraction of matched pairs with different foreground labels. Unmatched
    instances do not enter the score. ``coverage`` is diagnostic metadata only.
    """
    if foreground_classes is None:
        foreground_classes = sorted(
            int(c)
            for c in (set(np.unique(reference)) | set(np.unique(prediction)))
            if int(c) != 0
        )
    else:
        foreground_classes = sorted(
            {int(c) for c in foreground_classes if int(c) != 0}
        )

    ref_map, ref_classes, ref_sizes = _class_aware_instance_map(
        reference, foreground_classes
    )
    pred_map, pred_classes, pred_sizes = _class_aware_instance_map(
        prediction, foreground_classes
    )

    candidates = []
    for pred_id, pred_size in pred_sizes.items():
        pred_component = pred_map == pred_id
        overlapping_ref_ids = np.unique(ref_map[pred_component])
        overlapping_ref_ids = overlapping_ref_ids[overlapping_ref_ids > 0]
        for ref_id_raw in overlapping_ref_ids:
            ref_id = int(ref_id_raw)
            intersection = int(np.sum(pred_component & (ref_map == ref_id)))
            union = pred_size + ref_sizes[ref_id] - intersection
            if union <= 0:
                continue
            iou = intersection / union
            if iou >= overlap_iou_threshold:
                candidates.append((float(iou), ref_id, pred_id))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_ref: set[int] = set()
    matched_pred: set[int] = set()
    matches = []
    for iou, ref_id, pred_id in candidates:
        if ref_id in matched_ref or pred_id in matched_pred:
            continue
        matched_ref.add(ref_id)
        matched_pred.add(pred_id)
        matches.append(
            {
                "reference_instance_id": ref_id,
                "prediction_instance_id": pred_id,
                "reference_class": ref_classes[ref_id],
                "prediction_class": pred_classes[pred_id],
                "iou": iou,
            }
        )

    per_class: dict[int, dict[str, Any]] = {}
    for class_id in foreground_classes:
        class_matches = [
            match for match in matches if match["reference_class"] == class_id
        ]
        wrong = sum(
            match["reference_class"] != match["prediction_class"]
            for match in class_matches
        )
        num_ref = sum(label == class_id for label in ref_classes.values())
        num_matched = len(class_matches)
        per_class[class_id] = {
            "score": wrong / num_matched if num_matched else None,
            "coverage": num_matched / num_ref if num_ref else None,
            "num_matched": num_matched,
            "num_wrong_class": int(wrong),
            "num_reference_instances": int(num_ref),
        }

    num_matched = len(matches)
    num_wrong = sum(
        match["reference_class"] != match["prediction_class"] for match in matches
    )
    num_ref = len(ref_classes)
    transition_matrix: dict[int, dict[int, float | None]] = {}
    for source_class in foreground_classes:
        source_matches = [
            match
            for match in matches
            if match["reference_class"] == source_class
        ]
        transition_matrix[source_class] = {
            target_class: (
                float(
                    sum(
                        match["prediction_class"] == target_class
                        for match in source_matches
                    )
                    / len(source_matches)
                )
                if source_matches
                else None
            )
            for target_class in foreground_classes
        }
    return {
        "score": num_wrong / num_matched if num_matched else None,
        "coverage": num_matched / num_ref if num_ref else None,
        "num_matched": num_matched,
        "num_wrong_class": int(num_wrong),
        "num_correct_class": int(num_matched - num_wrong),
        "num_reference_instances": int(num_ref),
        "num_prediction_instances": int(len(pred_classes)),
        "match_iou_threshold": float(overlap_iou_threshold),
        "per_class": per_class,
        "transition_matrix": transition_matrix,
        "matches": matches,
    }
