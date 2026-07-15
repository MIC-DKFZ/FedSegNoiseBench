"""
Visualize HD95 vs fg-bg F1 with InstanceClsConf on the color axis.

Single plot with:
- X-axis: HD95 in its original millimetre values
- Y-axis: F1 score (foreground-vs-background)
- Color: instance class-confusion score (foreground object relabeling)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from .compute_noise_type_decisions import (
        DATASET_MEDIAN_OBJECT_DIAGONAL,
        dataset_from_entry,
    )
except ImportError:
    from compute_noise_type_decisions import (
        DATASET_MEDIAN_OBJECT_DIAGONAL,
        dataset_from_entry,
    )


BASE_FONT_SIZE = 30
TITLE_FONT_SIZE = BASE_FONT_SIZE
TICK_FONT_SIZE = BASE_FONT_SIZE
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE # - 2
DEFAULT_CONFUSION_CMAP = "Blues"
DEFAULT_SCATTER_FIG_WIDTH = 12.4
# Match the combined height of the two 5.4-inch violin rows on the left.
DEFAULT_SCATTER_FIG_HEIGHT = 10.8
DEFAULT_CONTOUR_THRESHOLD = 0.05
DEFAULT_INSTANCE_F1_THRESHOLD = 1.0
DEFAULT_INSTANCE_CLS_CONF_THRESHOLD = 0.0
THRESHOLD_LINE_COLOR = "#b8b8b8"
THRESHOLD_LINE_WIDTH = 1.8


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


def compute_pixel_cls_conf_score(overlap_matrix: Dict, fg_classes: List[int]) -> float:
    """
    Compute foreground-to-foreground class confusion from an overlap matrix.

    Background transitions are excluded. For a dataset with exactly one
    foreground class, the score is 0 because there is no alternative
    foreground class to swap to. Returns NaN only when no clean foreground
    source is present.
    """
    foreground_classes = sorted({int(c) for c in fg_classes if int(c) != 0})
    if not foreground_classes:
        return np.nan

    if len(foreground_classes) == 1:
        row = overlap_matrix.get(str(foreground_classes[0]), {})
        source_is_present = any(
            is_finite_number(value) and float(value) > 0.0
            for value in row.values()
        )
        return 0.0 if source_is_present else np.nan

    values = []
    for source_class in foreground_classes:
        row = overlap_matrix.get(str(source_class), {})
        if not any(is_finite_number(v) and float(v) > 0.0 for v in row.values()):
            continue
        values.append(
            sum(
                float(row.get(str(target_class), 0.0))
                for target_class in foreground_classes
                if target_class != source_class
                and is_finite_number(row.get(str(target_class), 0.0))
            )
        )
    return float(np.mean(values)) if values else np.nan


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


def load_hd95_f1_confusion_dataframe(json_path: str) -> pd.DataFrame:
    """Load analysis JSON files and extract only metrics needed for the plot."""
    json_files = resolve_json_files(json_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found for --json_path={json_path}")

    print(f"Loading {len(json_files)} JSON file(s)")

    rows = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        source_name = os.path.splitext(os.path.basename(json_file))[0]
        dataset_fg_classes = sorted(
            {
                int(class_id)
                for entry in data.values()
                for class_id in entry.get("classes", {}).get("fg_classes", [])
                if int(class_id) != 0
            }
        )
        for sample_id, entry in data.items():
            dataset = dataset_from_entry(entry, Path(json_file))
            median_object_diagonal = DATASET_MEDIAN_OBJECT_DIAGONAL.get(dataset)
            sample_fg_classes = {
                int(class_id)
                for class_id in entry.get("classes", {}).get("fg_classes", [])
                if int(class_id) != 0
            }
            overall = entry.get("overall_metrics", {})
            overlap_matrix = entry.get("class_overlap_matrix", {})
            cls_conf_metrics = entry.get("class_confusion_metrics", {})
            fg_bg = entry.get("foreground_vs_background_metrics", {})

            voxel_fg_bg = fg_bg.get("voxel_level_prf", {})
            cc_fg_bg = fg_bg.get("instance_level_prf", {})

            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "source_json": source_name,
                    "n_fg_classes": len(sample_fg_classes),
                    "pixel_cls_conf_score": to_float_or_nan(
                        cls_conf_metrics.get("PixelClsConf", {}).get("score")
                    )
                    if cls_conf_metrics.get("PixelClsConf")
                    else compute_pixel_cls_conf_score(
                        overlap_matrix, dataset_fg_classes
                    ),
                    "instance_cls_conf_score": to_float_or_nan(
                        cls_conf_metrics.get("InstanceClsConf", {}).get("score")
                    ),
                    "instance_cls_conf_coverage": to_float_or_nan(
                        cls_conf_metrics.get("InstanceClsConf", {}).get("coverage")
                    ),
                    "mean_hd95": to_float_or_nan(overall.get("mean_hd95", np.nan)),
                    "median_object_bbox_diagonal": to_float_or_nan(
                        median_object_diagonal
                    ),
                    "voxel_fgbg_f1": to_float_or_nan(voxel_fg_bg.get("f1", np.nan)),
                    "cc_fgbg_f1": to_float_or_nan(cc_fg_bg.get("f1", np.nan)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No sample entries found in the provided JSON file(s).")

    df["hd95_fraction_of_median_diagonal"] = (
        df["mean_hd95"] / df["median_object_bbox_diagonal"]
    )

    return df


def _noise_assignment_percentages(
    values: pd.Series,
    threshold: float,
    comparator: str,
) -> tuple[float, float]:
    """Return noisy and clean percentages among finite plotted values."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return np.nan, np.nan
    if comparator == "gt":
        noisy = numeric > threshold
    elif comparator == "lt":
        noisy = numeric < threshold
    else:
        raise ValueError(f"Unsupported comparator: {comparator}")
    noisy_percent = 100.0 * float(noisy.mean())
    return noisy_percent, 100.0 - noisy_percent


def plot_hd95_vs_f1_confusion(
    df: pd.DataFrame,
    output_path: str,
    level: str = "voxel",
    figsize=(DEFAULT_SCATTER_FIG_WIDTH, DEFAULT_SCATTER_FIG_HEIGHT),
    marker_size: int = 220,
    cmap: str = DEFAULT_CONFUSION_CMAP,
    add_labels: bool = False,
    max_points: int | None = None,
    contour_threshold: float = DEFAULT_CONTOUR_THRESHOLD,
    instance_f1_threshold: float = DEFAULT_INSTANCE_F1_THRESHOLD,
    instance_cls_conf_threshold: float = DEFAULT_INSTANCE_CLS_CONF_THRESHOLD,
):
    """
    Create a single 2D scatter plot with:
    - X: HD95 in millimetres
    - Y: F1 score (fg-vs-bg)
    - Color: InstanceClsConf (wrong foreground label among matched objects)
    - Optional: sample_id labels next to each point
    """
    if level not in ("voxel", "instance"):
        raise ValueError("level must be one of: 'voxel', 'instance'")

    if level == "voxel":
        level_prefix = "voxel"
        level_title = "Voxel-level"
        y_axis_label = "fg-bg Voxel F1"
    else:
        level_prefix = "cc"
        level_title = "Instance/CC-level"
        y_axis_label = "fg-bg Instance F1"

    x_col = "mean_hd95"
    y_col = f"{level_prefix}_fgbg_f1"
    color_col = "instance_cls_conf_score"

    valid_data = df.dropna(subset=[x_col, y_col])
    if len(valid_data) == 0:
        raise ValueError(f"No valid data for {level_title} analysis")

    if max_points is not None and max_points > 0:
        valid_data = valid_data.head(max_points).copy()

    print(f"Plotting {len(valid_data)} samples")

    fig, ax = plt.subplots(figsize=figsize)

    multiclass_all = valid_data[valid_data["n_fg_classes"] > 1]
    multiclass_data = multiclass_all.dropna(subset=[color_col])
    multiclass_without_confusion = multiclass_all[
        multiclass_all[color_col].isna()
    ]
    binary_data = valid_data[valid_data["n_fg_classes"] <= 1]

    scatter_for_colorbar = None

    if len(multiclass_data) > 0:
        scatter_for_colorbar = ax.scatter(
            multiclass_data[x_col],
            multiclass_data[y_col],
            s=marker_size,
            c=multiclass_data[color_col],
            cmap=cmap,
            alpha=0.8,
            edgecolors="#404040",
            linewidths=1.0,
            label="Multi-class",
            vmin=0,
            vmax=1,
        )

    if len(multiclass_without_confusion) > 0:
        ax.scatter(
            multiclass_without_confusion[x_col],
            multiclass_without_confusion[y_col],
            s=marker_size,
            facecolors="none",
            edgecolors="#404040",
            linewidths=1.0,
            marker="o",
            alpha=0.8,
        )

    if len(binary_data) > 0:
        ax.scatter(
            binary_data[x_col],
            binary_data[y_col],
            s=marker_size,
            facecolors="none",
            alpha=0.8,
            edgecolors="#404040",
            linewidths=1.0,
            marker="^",
        )

    # Add sample labels if requested
    if add_labels:
        for idx, row in valid_data.iterrows():
            ax.annotate(
                row["sample_id"],
                (row[x_col], row[y_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=ANNOTATION_FONT_SIZE,
                alpha=0.7,
                ha="left",
            )

    # The contour threshold is a dataset-relative fraction, so in raw HD95
    # units every represented dataset has its own threshold position.
    contour_thresholds_mm = sorted(
        {
            float(contour_threshold * diagonal)
            for diagonal in valid_data["median_object_bbox_diagonal"]
            if np.isfinite(diagonal)
        }
    )
    for threshold_mm in contour_thresholds_mm:
        ax.axvline(
            threshold_mm,
            color=THRESHOLD_LINE_COLOR,
            linestyle="--",
            linewidth=THRESHOLD_LINE_WIDTH,
            zorder=4,
        )
    ax.axhline(
        instance_f1_threshold,
        color=THRESHOLD_LINE_COLOR,
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        zorder=4,
    )

    contour_noisy, _ = _noise_assignment_percentages(
        valid_data["hd95_fraction_of_median_diagonal"], contour_threshold, "gt"
    )
    instance_noisy, _ = _noise_assignment_percentages(
        valid_data[y_col], instance_f1_threshold, "lt"
    )
    cls_noisy, _ = _noise_assignment_percentages(
        multiclass_data[color_col], instance_cls_conf_threshold, "gt"
    )
    def format_percent(value: float, bold: bool = False) -> str:
        if not np.isfinite(value):
            return "N/A"
        formatted_value = f"{value:.1f}".replace(".", "{,}")
        if bold:
            return rf"$\mathbf{{{formatted_value}\%}}$"
        return f"{value:.1f}%".replace(".", ",")

    prevalences = {
        "Contour": contour_noisy,
        "Instance": instance_noisy,
        "Confusion": cls_noisy,
    }
    applicable_prevalences = {
        name: value for name, value in prevalences.items() if np.isfinite(value)
    }
    largest_noise_types = set()
    if applicable_prevalences:
        maximum = max(applicable_prevalences.values())
        largest_noise_types = {
            name
            for name, value in applicable_prevalences.items()
            if np.isclose(value, maximum)
        }

    contour_metric_label = r"$\mathrm{Cont}$"
    instance_metric_label = r"$\mathrm{Inst}$"
    confusion_metric_label = r"$\mathrm{Conf}$"
    panel_separator = "      "
    assignment_panel = (
        f"{contour_metric_label}="
        f"{format_percent(contour_noisy, 'Contour' in largest_noise_types)}"
        f"{panel_separator}"
        f"{instance_metric_label}="
        f"{format_percent(instance_noisy, 'Instance' in largest_noise_types)}"
        f"{panel_separator}"
        f"{confusion_metric_label}="
        f"{format_percent(cls_noisy, 'Confusion' in largest_noise_types)}"
    )
    ax.set_xlabel(
        "HD95 [mm]", fontsize=BASE_FONT_SIZE
    )
    ax.set_ylabel(y_axis_label, fontsize=BASE_FONT_SIZE)
    label_suffix = " with labels" if add_labels else ""
    # ax.set_title(
    #     f"HD95 vs F1 vs InstanceClsConf{label_suffix}",
    #     fontsize=TITLE_FONT_SIZE,
    #     fontweight="bold",
    #     pad=15,
    # )

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    current_xmax = ax.get_xlim()[1]
    ax.set_xlim(0.0, current_xmax)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    # Keep the actual scatter axes square while the outer figure can remain wider
    # to accommodate the legend and colorbar.
    ax.set_box_aspect(1)

    if scatter_for_colorbar is not None:
        # Binary-only tasks have no applicable class-confusion metric and
        # therefore get neither color encoding nor a colorbar.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
        cbar.set_label("Instance Class Confusion", fontsize=BASE_FONT_SIZE)
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
        cbar.ax.axhline(
            instance_cls_conf_threshold,
            color=THRESHOLD_LINE_COLOR,
            linestyle="--",
            linewidth=THRESHOLD_LINE_WIDTH,
            clip_on=False,
        )

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    # Reserve a header band above the axes and place the percentage summary
    # there. This is outside the plotting area rather than overlaid on data.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    ax_position = ax.get_position()
    fig.text(
        (ax_position.x0 + ax_position.x1) / 2.0,
        0.92,
        assignment_panel,
        ha="center",
        va="top",
        fontsize=TICK_FONT_SIZE - 5,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    label_info = " (with labels)" if add_labels else " (no labels)"
    print(f"✓ Saved HD95-vs-F1 (confusion-colored) plot{label_info} to {output_path}")

    print(f"\nData summary ({level_title}):")
    print(f"  Total samples: {len(valid_data)}")
    print(f"  Multiclass: {len(multiclass_all)}, Binary: {len(binary_data)}")
    print(
        f"  HD95 range (x-axis): [{valid_data[x_col].min():.3f}, {valid_data[x_col].max():.3f}] mm"
    )
    print(
        f"  F1 score range (y-axis): [{valid_data[y_col].min():.3f}, {valid_data[y_col].max():.3f}]"
    )
    if len(multiclass_data) > 0:
        print(
            "  Instance class-confusion range (color): "
            f"[{multiclass_data[color_col].min():.3f}, "
            f"{multiclass_data[color_col].max():.3f}]"
        )
    else:
        print("  Instance class confusion: N/A (binary segmentation task)")


def _insert_suffix_before_extension(path: str, suffix: str) -> str:
    """Insert a suffix before the file extension. e.g., 'plot.png' + '_voxel' -> 'plot_voxel.png'"""
    if path.endswith(".png"):
        return path[:-4] + suffix + ".png"
    return path + suffix


def main():
    parser = argparse.ArgumentParser(
        description="Plot HD95 (x-axis) vs fg-bg F1 score (y-axis) with confusion score as color."
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
        default="./results/noise_analysis/hd95_vs_f1_confusion.png",
        help="Output figure path template (.png); will create both voxel and instance plots unless --level is specified",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["voxel", "instance", "both"],
        default="instance",
        help="Metric level(s) to visualize: voxel, instance (CC; default), or both",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[DEFAULT_SCATTER_FIG_WIDTH, DEFAULT_SCATTER_FIG_HEIGHT],
        help=(
            "Figure size: width height. Default height matches the two stacked "
            "5.4-inch violin figures together, while the scatter axes are kept square."
        ),
    )
    parser.add_argument(
        "--marker_size",
        type=int,
        default=400,
        help="Marker size for scatter points",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CONFUSION_CMAP,
        help="Colormap for confusion score",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help=(
            "Optional maximum number of valid points to plot, useful for debugging. "
            "Uses the first N valid samples."
        ),
    )
    parser.add_argument(
        "--contour_threshold",
        type=float,
        default=DEFAULT_CONTOUR_THRESHOLD,
        help="Contour threshold as fraction of dataset median object diagonal.",
    )
    parser.add_argument(
        "--instance_f1_threshold",
        type=float,
        default=DEFAULT_INSTANCE_F1_THRESHOLD,
        help="Foreground-background F1 threshold for instance-noise assignment.",
    )
    parser.add_argument(
        "--instance_cls_conf_threshold",
        type=float,
        default=DEFAULT_INSTANCE_CLS_CONF_THRESHOLD,
        help="InstanceClsConf threshold for class-confusion-noise assignment.",
    )
    args = parser.parse_args()

    df = load_hd95_f1_confusion_dataframe(args.json_path)
    print(f"Loaded {len(df)} samples\n")

    levels_to_plot = ["voxel", "instance"] if args.level == "both" else [args.level]

    for level in levels_to_plot:
        for add_labels in [False, True]:
            label_suffix = "_labeled" if add_labels else ""
            output_path = _insert_suffix_before_extension(
                args.output, f"_{level}{label_suffix}"
            )
            plot_hd95_vs_f1_confusion(
                df=df,
                output_path=output_path,
                level=level,
                figsize=tuple(args.figsize),
                marker_size=args.marker_size,
                cmap=args.cmap,
                add_labels=add_labels,
                max_points=args.max_points,
                contour_threshold=args.contour_threshold,
                instance_f1_threshold=args.instance_f1_threshold,
                instance_cls_conf_threshold=args.instance_cls_conf_threshold,
            )
        print()  # blank line between level groups


if __name__ == "__main__":
    main()
