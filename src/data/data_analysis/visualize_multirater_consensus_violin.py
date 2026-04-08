"""
Visualize per-class multi-rater consensus metrics as single-row violin plots.

Creates one figure with 4 violin subplots plus 1 class-confusion heatmap:
- Fleiss' kappa (class-wise)
- Mean Dice between consensus and raters (class-wise; averaged over raters)
- Mean HD95 between consensus and raters (class-wise; averaged over raters)
- Mean instance-level F1 between consensus and raters (class-wise)
- Class confusion matrix consensus -> raters (averaged over raters and samples)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .analyze_multirater_consensus import load_mask
except ImportError:
    from analyze_multirater_consensus import load_mask


# Fixed class palette (same ordering convention as existing per-class plots)
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

# Central style control for the whole figure.
BASE_FONT_SIZE = 16
TITLE_FONT_SIZE = BASE_FONT_SIZE # + 1
TICK_FONT_SIZE = BASE_FONT_SIZE # - 1
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE # - 1
HEATMAP_CMAP = "Blues"
# DEFAULT_MAX_SAMPLES = 100
DEFAULT_MAX_SAMPLES = None

METRICS_TO_PLOT = [
    (
        "fleiss_kappa",
        "Fleiss' Kappa",
        "Class-wise Fleiss' Kappa\n(Among Raters)",
    ),
    (
        "mean_dice",
        "Dice",
        "Class-wise Dice\n(Consensus vs Raters)",
    ),
    (
        "mean_hd95",
        "HD95 (mm)",
        "Class-wise HD95\n(Consensus vs Raters)",
    ),
    (
        "mean_instance_level_f1",
        "Instance F1",
        "Class-wise Instance F1\n(Consensus vs Raters)",
    ),
]


def is_finite_number(value) -> bool:
    """Return True if value can be interpreted as a finite float."""
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("nan", "infinity", "-infinity", "inf", "-inf"):
            return False
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def to_float_or_nan(value) -> float:
    """Convert numeric-like value to float, otherwise return NaN."""
    return float(value) if is_finite_number(value) else np.nan


def load_multirater_consensus(json_file: str) -> Dict:
    """Load multirater consensus JSON results."""
    with open(json_file, "r") as f:
        return json.load(f)


def limit_results_to_first_n_samples(results: Dict, max_samples: int | None) -> Dict:
    """Keep only the first max_samples entries from the loaded JSON."""
    if max_samples is None or max_samples <= 0:
        return results
    return dict(list(results.items())[:max_samples])


def extract_perclass_dataframe(results: Dict) -> pd.DataFrame:
    """
    Extract one row per sample-class entry with metrics needed for plotting.
    """
    rows = []

    for sample_id, sample_data in results.items():
        if not isinstance(sample_data, dict):
            continue

        per_class_metrics = sample_data.get("per_class_metrics", {})
        if not isinstance(per_class_metrics, dict):
            continue

        for class_id_raw, class_metrics in per_class_metrics.items():
            if not isinstance(class_metrics, dict):
                continue

            try:
                class_id = int(class_id_raw)
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "class_id": class_id,
                    "fleiss_kappa": to_float_or_nan(
                        class_metrics.get("fleiss_kappa", np.nan)
                    ),
                    "mean_dice": to_float_or_nan(class_metrics.get("mean_dice", np.nan)),
                    "mean_hd95": to_float_or_nan(class_metrics.get("mean_hd95", np.nan)),
                    "mean_instance_level_f1": to_float_or_nan(
                        class_metrics.get("mean_instance_level_f1", np.nan)
                    ),
                }
            )

    return pd.DataFrame(rows)


def infer_file_ending(file_path: str) -> str:
    if file_path.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(file_path).suffix.lower()


def infer_riga_mode(file_path: str) -> bool:
    lower_path = str(file_path).lower()
    return "riga" in lower_path and infer_file_ending(file_path) in {".png", ".tif", ".tiff"}


def compute_class_overlap_matrix(
    consensus_mask: np.ndarray, rater_mask: np.ndarray, classes: List[int]
) -> Dict[int, Dict[int, float]]:
    overlap_matrix: Dict[int, Dict[int, float]] = {}

    for src_class in classes:
        consensus_class = consensus_mask == src_class
        consensus_volume = np.sum(consensus_class)
        overlap_matrix[int(src_class)] = {}

        for dst_class in classes:
            if consensus_volume == 0:
                overlap_ratio = 0.0
            else:
                rater_class = rater_mask == dst_class
                overlap_ratio = np.sum(consensus_class & rater_class) / consensus_volume
            overlap_matrix[int(src_class)][int(dst_class)] = float(overlap_ratio)

    return overlap_matrix


def extract_avg_confusion_matrix(results: Dict) -> Tuple[np.ndarray, List[int]]:
    sample_level_matrices: List[Dict[int, Dict[int, float]]] = []
    all_classes = set()

    for sample_data in results.values():
        if not isinstance(sample_data, dict):
            continue

        paths = sample_data.get("paths", {})
        consensus_path = paths.get("consensus")
        rater_paths = paths.get("raters", {})
        if not consensus_path or not isinstance(rater_paths, dict) or not rater_paths:
            continue

        consensus_mask = load_mask(
            consensus_path,
            infer_file_ending(consensus_path),
            riga_mode=infer_riga_mode(consensus_path),
        )

        per_rater_matrices = []
        for rater_path in rater_paths.values():
            rater_mask = load_mask(
                rater_path,
                infer_file_ending(rater_path),
                riga_mode=infer_riga_mode(rater_path),
            )
            if rater_mask.shape != consensus_mask.shape:
                continue

            classes = sorted(
                set(np.unique(consensus_mask)) | set(np.unique(rater_mask))
            )
            all_classes.update(int(c) for c in classes)
            per_rater_matrices.append(
                compute_class_overlap_matrix(consensus_mask, rater_mask, classes)
            )

        if not per_rater_matrices:
            continue

        sample_avg: Dict[int, Dict[int, float]] = {}
        sample_classes = sorted(
            {int(k) for mat in per_rater_matrices for k in mat.keys()}
            | {
                int(dst)
                for mat in per_rater_matrices
                for src in mat.values()
                for dst in src.keys()
            }
        )
        for src_class in sample_classes:
            sample_avg[src_class] = {}
            for dst_class in sample_classes:
                vals = [
                    mat.get(src_class, {}).get(dst_class, 0.0)
                    for mat in per_rater_matrices
                ]
                sample_avg[src_class][dst_class] = float(np.mean(vals))

        sample_level_matrices.append(sample_avg)

    sorted_classes = sorted(all_classes)
    if not sorted_classes:
        return np.zeros((0, 0), dtype=float), []

    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    matrix = np.zeros((len(sorted_classes), len(sorted_classes)), dtype=float)

    for src_class in sorted_classes:
        for dst_class in sorted_classes:
            vals = [
                sample_matrix.get(src_class, {}).get(dst_class, 0.0)
                for sample_matrix in sample_level_matrices
            ]
            matrix[class_to_idx[src_class], class_to_idx[dst_class]] = (
                float(np.mean(vals)) if vals else 0.0
            )

    return matrix, sorted_classes


def _prepare_metric_data_by_class(
    df: pd.DataFrame,
    classes: List[int],
    metric: str,
) -> Tuple[List[int], List[np.ndarray]]:
    """Prepare finite per-class arrays for a metric."""
    valid_classes: List[int] = []
    data_by_class: List[np.ndarray] = []

    for class_id in classes:
        values = (
            df.loc[df["class_id"] == class_id, metric]
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        # Matplotlib's KDE-based violin fails on singletons; duplicate with tiny jitter.
        if len(values) == 1:
            eps = 1e-6 * (1.0 + abs(values[0]))
            values = np.array([values[0] - eps, values[0] + eps], dtype=np.float64)

        valid_classes.append(class_id)
        data_by_class.append(values)

    return valid_classes, data_by_class


def _add_violin_subplot(
    ax,
    df: pd.DataFrame,
    classes: List[int],
    metric: str,
    ylabel: str,
    title: str,
):
    """Add one class-wise violin subplot."""
    valid_classes, data_by_class = _prepare_metric_data_by_class(df, classes, metric)

    if not data_by_class:
        ax.text(
            0.5,
            0.5,
            "No valid data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=BASE_FONT_SIZE,
        )
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        return

    positions = list(range(len(valid_classes)))
    parts = ax.violinplot(
        data_by_class,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )

    # Color violin bodies by class index
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(CLASS_COLORS[idx % len(CLASS_COLORS)])
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    if "cmedians" in parts:
        parts["cmedians"].set_edgecolor("black")
        parts["cmedians"].set_linewidth(1.5)

    if "cmeans" in parts:
        parts["cmeans"].set_edgecolor("red")
        parts["cmeans"].set_linewidth(1.5)
        parts["cmeans"].set_linestyle("--")

    for key in ("cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_edgecolor("black")
            parts[key].set_linewidth(1.0)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Class {c}" for c in valid_classes], fontsize=TICK_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)


def plot_multirater_consensus_violin_row(
    df: pd.DataFrame,
    confusion_matrix: np.ndarray,
    confusion_classes: List[int],
    output_path: str,
):
    """Create a single-row figure with 4 per-class violin subplots and 1 heatmap."""
    classes = sorted(df["class_id"].unique())
    if not classes:
        raise ValueError("No class IDs found in extracted data.")

    fig, axes = plt.subplots(
        1,
        len(METRICS_TO_PLOT) + 1,
        figsize=(27, 5.4),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.2]},
    )

    for ax, (metric, ylabel, title) in zip(axes, METRICS_TO_PLOT):
        _add_violin_subplot(ax, df, classes, metric, ylabel, title)

    ax_cm = axes[-1]
    if confusion_matrix.size == 0:
        ax_cm.text(
            0.5,
            0.5,
            "No valid data",
            transform=ax_cm.transAxes,
            ha="center",
            va="center",
            fontsize=BASE_FONT_SIZE,
        )
        ax_cm.set_title(
            "Class Confusion Matrix\n(Consensus vs Raters)",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax_cm.set_xticks([])
        ax_cm.set_yticks([])
    else:
        im = ax_cm.imshow(confusion_matrix, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)
        ax_cm.set_title(
            "Class Confusion Matrix\n(Consensus vs Raters)",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax_cm.set_xticks(range(len(confusion_classes)))
        ax_cm.set_yticks(range(len(confusion_classes)))
        ax_cm.set_xticklabels([f"C{c}" for c in confusion_classes], fontsize=TICK_FONT_SIZE)
        ax_cm.set_yticklabels([f"C{c}" for c in confusion_classes], fontsize=TICK_FONT_SIZE)
        ax_cm.set_xlabel("Rater classes", fontsize=BASE_FONT_SIZE)
        ax_cm.set_ylabel("Consensus classes", fontsize=BASE_FONT_SIZE)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax_cm.text(
                    j,
                    i,
                    f"{confusion_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_FONT_SIZE,
                    color="#0b1f2a" if confusion_matrix[i, j] < 0.62 else "white",
                )

        cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
        cbar.set_label("Overlap", fontsize=BASE_FONT_SIZE)
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    # fig.suptitle(
    #     "Per-Class Multi-Rater vs Consensus Metrics",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.03,
    # )

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved single-row violin plot to: {output_path}")


def default_output_path(input_json: str) -> str:
    """Derive default output path next to the input JSON."""
    input_path = Path(input_json)
    return str(input_path.parent / "multirater_consensus_perclass_violin.png")


def main(args):
    """Main execution."""
    print(f"Loading multirater consensus results from: {args.input_json}")
    results = load_multirater_consensus(args.input_json)
    if DEFAULT_MAX_SAMPLES is not None and DEFAULT_MAX_SAMPLES > 0:
        results = limit_results_to_first_n_samples(results, DEFAULT_MAX_SAMPLES)

    df = extract_perclass_dataframe(results)
    if df.empty:
        raise ValueError("No per-class metrics found in input JSON.")

    num_samples = df["sample_id"].nunique()
    classes = sorted(df["class_id"].unique())
    print(
        f"Extracted {len(df)} sample-class rows from {num_samples} samples. Classes: {classes}"
    )

    confusion_matrix, confusion_classes = extract_avg_confusion_matrix(results)
    output_path = args.output_png or default_output_path(args.input_json)
    plot_multirater_consensus_violin_row(
        df, confusion_matrix, confusion_classes, output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-class multi-rater consensus metrics as single-row violin plots"
        )
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to multirater_consensus.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help=(
            "Output PNG path (default: next to input JSON as "
            "multirater_consensus_perclass_violin.png)"
        ),
    )

    main(parser.parse_args())
