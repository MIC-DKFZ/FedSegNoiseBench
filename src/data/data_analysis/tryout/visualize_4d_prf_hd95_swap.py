"""
Visualize HD95 vs per-class-vs-fg/bg metrics as three 2D scatter subplots.

For a selected evaluation level (voxel or instance/CC), the subplots are:
1) x=mean HD95, y=fg-vs-bg Precision, color=per-class average Precision
2) x=mean HD95, y=fg-vs-bg Recall, color=per-class average Recall
3) x=mean HD95, y=fg-vs-bg F1, color=per-class average F1
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_finite_number(value) -> bool:
    """Check if a value can be interpreted as a finite float."""
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


def compute_swap_score(overlap_matrix: Dict, fg_classes: List[int]) -> float:
    """
    Compute swap score from class overlap matrix.

    For multiclass: average of (1 - diagonal) over foreground classes.
    For binary: average of off-diagonal confusion rates (0->1, 1->0).
    Returns NaN if undefined.
    """
    if len(fg_classes) >= 2:
        per_class_misclassification = []
        for class_id in fg_classes:
            diag = overlap_matrix.get(str(class_id), {}).get(str(class_id), np.nan)
            if is_finite_number(diag):
                per_class_misclassification.append(1.0 - float(diag))
        return (
            float(np.mean(per_class_misclassification))
            if per_class_misclassification
            else np.nan
        )

    off_01 = overlap_matrix.get("0", {}).get("1", np.nan)
    off_10 = overlap_matrix.get("1", {}).get("0", np.nan)
    vals = []
    if is_finite_number(off_01):
        vals.append(float(off_01))
    if is_finite_number(off_10):
        vals.append(float(off_10))
    return float(np.mean(vals)) if vals else np.nan


def resolve_json_files(json_path: str) -> List[str]:
    """Resolve input path/glob/directory into JSON files."""
    if os.path.isdir(json_path):
        return sorted(glob.glob(os.path.join(json_path, "*.json")))

    glob_matches = sorted(glob.glob(json_path))
    if glob_matches:
        return glob_matches

    if os.path.isfile(json_path):
        return [json_path]

    return []


def to_float_or_nan(value) -> float:
    """Convert value to float if finite, otherwise NaN."""
    return float(value) if is_finite_number(value) else np.nan


def load_plot_dataframe(json_path: str) -> pd.DataFrame:
    """Load one/multiple analysis JSON files and extract plotting metrics."""
    json_files = resolve_json_files(json_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found for --json_path={json_path}")

    print(f"Loading {len(json_files)} JSON file(s)")

    rows = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        source_name = os.path.splitext(os.path.basename(json_file))[0]
        for sample_id, entry in data.items():
            overall = entry.get("overall_metrics", {})
            classes = entry.get("classes", {})
            fg_classes = classes.get("fg_classes", [])
            overlap_matrix = entry.get("class_overlap_matrix", {})
            fg_bg = entry.get("foreground_vs_background_metrics", {})

            voxel_fg_bg = fg_bg.get("voxel_level_prf", {})
            cc_fg_bg = fg_bg.get("instance_level_prf", {})

            rows.append(
                {
                    "sample_id": sample_id,
                    "source_json": source_name,
                    "n_fg_classes": len(fg_classes),
                    "swap_score": compute_swap_score(overlap_matrix, fg_classes),
                    "mean_hd95": to_float_or_nan(overall.get("mean_hd95", np.nan)),
                    "voxel_pc_precision": to_float_or_nan(
                        overall.get("mean_voxel_level_prf_precision", np.nan)
                    ),
                    "voxel_pc_recall": to_float_or_nan(
                        overall.get("mean_voxel_level_prf_recall", np.nan)
                    ),
                    "voxel_pc_f1": to_float_or_nan(
                        overall.get("mean_voxel_level_prf_f1", np.nan)
                    ),
                    "cc_pc_precision": to_float_or_nan(
                        overall.get("mean_instance_level_prf_precision", np.nan)
                    ),
                    "cc_pc_recall": to_float_or_nan(
                        overall.get("mean_instance_level_prf_recall", np.nan)
                    ),
                    "cc_pc_f1": to_float_or_nan(
                        overall.get("mean_instance_level_prf_f1", np.nan)
                    ),
                    "voxel_fgbg_precision": to_float_or_nan(
                        voxel_fg_bg.get("precision", np.nan)
                    ),
                    "voxel_fgbg_recall": to_float_or_nan(
                        voxel_fg_bg.get("recall", np.nan)
                    ),
                    "voxel_fgbg_f1": to_float_or_nan(voxel_fg_bg.get("f1", np.nan)),
                    "cc_fgbg_precision": to_float_or_nan(
                        cc_fg_bg.get("precision", np.nan)
                    ),
                    "cc_fgbg_recall": to_float_or_nan(cc_fg_bg.get("recall", np.nan)),
                    "cc_fgbg_f1": to_float_or_nan(cc_fg_bg.get("f1", np.nan)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No sample entries found in the provided JSON file(s).")

    return df


def plot_metric_subplots(
    df: pd.DataFrame,
    output_path: str,
    level: str = "voxel",
    figsize=(20, 7),
    marker_size: int = 42,
):
    """Create three 2D subplots for precision, recall, and F1."""
    if level not in ("voxel", "instance"):
        raise ValueError("level must be one of: 'voxel', 'instance'")

    if level == "voxel":
        level_prefix = "voxel"
        level_title = "Voxel-level"
        level_short = "Voxel"
    else:
        level_prefix = "cc"
        level_title = "Instance/CC-level"
        level_short = "CC"

    subplot_specs = [
        {
            "title": f"Plot 1: {level_title} Precision",
            "x": "mean_hd95",
            "y": f"{level_prefix}_fgbg_precision",
            "c": f"{level_prefix}_pc_precision",
            "xlabel": "Mean HD95",
            "ylabel": f"{level_short} Precision (fg vs bg)",
            "clabel": f"{level_short} Precision (avg classes)",
        },
        {
            "title": f"Plot 2: {level_title} Recall",
            "x": "mean_hd95",
            "y": f"{level_prefix}_fgbg_recall",
            "c": f"{level_prefix}_pc_recall",
            "xlabel": "Mean HD95",
            "ylabel": f"{level_short} Recall (fg vs bg)",
            "clabel": f"{level_short} Recall (avg classes)",
        },
        {
            "title": f"Plot 3: {level_title} F1-score",
            "x": "mean_hd95",
            "y": f"{level_prefix}_fgbg_f1",
            "c": f"{level_prefix}_pc_f1",
            "xlabel": "Mean HD95",
            "ylabel": f"{level_short} F1 (fg vs bg)",
            "clabel": f"{level_short} F1 (avg classes)",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = axes.ravel()

    for axis, spec in zip(axes, subplot_specs):
        x_col, y_col, c_col = spec["x"], spec["y"], spec["c"]

        valid_xyc = df.dropna(subset=[x_col, y_col, c_col])
        valid_multiclass = valid_xyc[valid_xyc["n_fg_classes"] > 1]
        valid_singleclass = valid_xyc[valid_xyc["n_fg_classes"] == 1]

        if len(valid_multiclass) > 0:
            scatter = axis.scatter(
                valid_multiclass[x_col],
                valid_multiclass[y_col],
                c=valid_multiclass[c_col],
                cmap="viridis",
                marker="o",
                s=marker_size,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
            )
            colorbar = fig.colorbar(scatter, ax=axis)
            colorbar.set_label(spec["clabel"], fontsize=9)

        if len(valid_singleclass) > 0:
            axis.scatter(
                valid_singleclass[x_col],
                valid_singleclass[y_col],
                c="gray",
                marker="x",
                s=marker_size,
                alpha=0.8,
                linewidths=0.8,
                label="Single foreground class (no color encoding)",
            )

        axis.set_title(spec["title"], fontsize=11, fontweight="bold", pad=12)
        axis.set_xlabel(spec["xlabel"], fontsize=9)
        axis.set_ylabel(spec["ylabel"], fontsize=9)
        axis.grid(True, alpha=0.25)

        if len(valid_singleclass) > 0:
            axis.legend(loc="best", fontsize=8, framealpha=0.9)

        if len(valid_xyc) == 0:
            axis.text(0.02, 0.95, "No valid data", transform=axis.transAxes, color="red")

    fig.suptitle(
        f"{level_title}: HD95 vs Foreground-vs-Background (color = Per-class Average)",
        fontsize=14,
        fontweight="bold",
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved 3-subplot figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 3 requested 2D subplots: HD95 vs fg-bg with per-class avg as color for Precision/Recall/F1."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./results/noise_analysis/*.json",
        help="JSON file, directory, or glob pattern from analyze_noise_clean_noisy.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/noise_analysis/prf_hd95_perclass_vs_fgbg_3subplots.png",
        help="Output figure path (.png)",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["voxel", "instance"],
        default="voxel",
        help="Metric level to visualize: voxel or instance (CC)",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[20, 7],
        help="Figure size: width height",
    )
    parser.add_argument(
        "--marker_size",
        type=int,
        default=42,
        help="Scatter marker size",
    )
    args = parser.parse_args()

    df = load_plot_dataframe(args.json_path)
    print(f"Loaded {len(df)} samples")

    plot_metric_subplots(
        df=df,
        output_path=args.output,
        level=args.level,
        figsize=tuple(args.figsize),
        marker_size=args.marker_size,
    )


if __name__ == "__main__":
    main()
