"""
Visualize per-class multi-rater consensus metrics as single-row violin plots.

Creates one figure with 4 violin subplots (in one row):
- Fleiss' kappa (class-wise)
- Voxel-wise entropy (class-wise)
- Mean Dice between consensus and raters (class-wise; averaged over raters)
- Mean HD95 between consensus and raters (class-wise; averaged over raters)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Fixed class palette (same ordering convention as existing per-class plots)
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]


METRICS_TO_PLOT = [
    (
        "fleiss_kappa",
        "Fleiss' Kappa",
        "Class-wise Fleiss' Kappa\n(Among Raters)",
    ),
    (
        "voxelwise_entropy",
        "Voxel Entropy",
        "Class-wise Voxel Entropy\n(Among Raters)",
    ),
    (
        "mean_dice",
        "Mean Dice",
        "Class-wise Mean Dice\n(Consensus vs Raters)",
    ),
    (
        "mean_hd95",
        "Mean HD95 (mm)",
        "Class-wise Mean HD95\n(Consensus vs Raters)",
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
                    "voxelwise_entropy": to_float_or_nan(
                        class_metrics.get("voxelwise_entropy", np.nan)
                    ),
                    "mean_dice": to_float_or_nan(class_metrics.get("mean_dice", np.nan)),
                    "mean_hd95": to_float_or_nan(class_metrics.get("mean_hd95", np.nan)),
                }
            )

    return pd.DataFrame(rows)


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
            fontsize=10,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
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

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{c}" for c in valid_classes], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)


def plot_multirater_consensus_violin_row(df: pd.DataFrame, output_path: str):
    """Create a single-row figure with 4 per-class violin subplots."""
    classes = sorted(df["class_id"].unique())
    if not classes:
        raise ValueError("No class IDs found in extracted data.")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))

    for ax, (metric, ylabel, title) in zip(axes, METRICS_TO_PLOT):
        _add_violin_subplot(ax, df, classes, metric, ylabel, title)

    fig.suptitle(
        "Per-Class Multi-Rater vs Consensus Metrics",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

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

    df = extract_perclass_dataframe(results)
    if df.empty:
        raise ValueError("No per-class metrics found in input JSON.")

    num_samples = df["sample_id"].nunique()
    classes = sorted(df["class_id"].unique())
    print(
        f"Extracted {len(df)} sample-class rows from {num_samples} samples. Classes: {classes}"
    )

    output_path = args.output_png or default_output_path(args.input_json)
    plot_multirater_consensus_violin_row(df, output_path)


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
