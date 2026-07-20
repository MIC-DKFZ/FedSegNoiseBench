"""
Per-class violin plot visualizations for noise analysis.

Creates violin plots comparing clean and noisy masks for various per-class metrics:
- Per-class Dice coefficient
- Per-class NSD (Normalized Surface Distance)
- Per-class HD95 (Hausdorff Distance 95%)
- Per-class relative volume difference
- Per-class number of connected components (clean vs noisy)
- Per-class average volume of connected components (clean vs noisy)
- Per-class delta of number of connected components
- Per-class delta of average volume of connected components

Also creates a square InstanceClsConf transition-matrix heatmap.

All visualizations are combined into a single figure with subplots.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


def load_noise_analysis(json_file: str) -> Dict:
    """Load noise analysis results from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


# Fixed color palette for classes: blue, orange, green, violet
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

# Central style control for the single-row summary figures.
BASE_FONT_SIZE = 16
TITLE_FONT_SIZE = BASE_FONT_SIZE
TICK_FONT_SIZE = BASE_FONT_SIZE
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE # - 1
HEATMAP_CMAP = "Blues"
REQUESTED_ROW_DEFAULT_WIDTH = 27.0
REQUESTED_ROW_DEFAULT_HEIGHT = 5.4
PIXEL_HD95_DATASETS = {"RIGA", "Gleason"}
CLEAN_DATASET_ID_TO_NAME = {
    "041": "LIDC",
    "300": "RIGA",
    "436": "Gleason",
    "500": "MouseTumor",
    "600": "MMIA",
    "700": "MMIS",
}


def infer_hd95_unit(results: Dict) -> str:
    """Infer whether HD95 values in one noise-analysis JSON are px or mm."""
    dataset_names = {
        CLEAN_DATASET_ID_TO_NAME.get(str(result.get("clean_dataset_id")))
        for result in results.values()
    }
    dataset_names.discard(None)
    if dataset_names and dataset_names <= PIXEL_HD95_DATASETS:
        return "px"
    if dataset_names and dataset_names.isdisjoint(PIXEL_HD95_DATASETS):
        return "mm"
    return "px/mm"


def resolve_requested_row_figsize(
    width: float = REQUESTED_ROW_DEFAULT_WIDTH,
    height: float = REQUESTED_ROW_DEFAULT_HEIGHT,
) -> tuple[float, float]:
    """Resolve figsize for the requested-metrics single-row plot."""
    return (width, height)


def extract_perclass_data(results: Dict) -> pd.DataFrame:
    """
    Extract per-class metrics from all samples.
    
    Returns a DataFrame with one row per sample-class combination.
    """
    data = []
    
    for sample_id, result in results.items():
        per_class_metrics = result.get("per_class_metrics", {})
        
        for class_id, metrics in per_class_metrics.items():
            if metrics is None:
                continue
                
            # Extract instance-level metrics
            instance_level_prf = metrics.get("instance_level_prf", {})
            
            record = {
                "sample_id": sample_id,
                "class_id": int(class_id),
                "dice": metrics.get("dice", np.nan),
                "nsd": metrics.get("nsd", np.nan),
                "hd95": metrics.get("hd95", np.nan),
                "relative_volume_diff": metrics.get("relative_volume_diff", np.nan),
                "volume_clean": metrics.get("volume_clean", np.nan),
                "volume_noisy": metrics.get("volume_noisy", np.nan),
                "num_cc_clean": metrics.get("cc_clean", {}).get("num_components", np.nan),
                "num_cc_noisy": metrics.get("cc_noisy", {}).get("num_components", np.nan),
                "avg_vol_cc_clean": metrics.get("cc_clean", {}).get("avg_volume", np.nan),
                "avg_vol_cc_noisy": metrics.get("cc_noisy", {}).get("avg_volume", np.nan),
                "delta_num_cc": metrics.get("delta_num_cc", np.nan),
                "delta_avg_vol_cc": metrics.get("delta_avg_volume_cc", np.nan),
                "instance_precision": instance_level_prf.get("precision", np.nan),
                "instance_recall": instance_level_prf.get("recall", np.nan),
                "instance_f1": instance_level_prf.get("f1", np.nan),
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    return df


def extract_requested_row_data(results: Dict) -> pd.DataFrame:
    """
    Extract the sample-level metrics used in the requested single-row figure.

    HD95 and Instance F1 intentionally match the scatter plot in
    visualize_hd95_f1_confusion.py:
    - overall_metrics.mean_hd95
    - foreground_vs_background_metrics.instance_level_prf.f1
    """
    data = []

    for sample_id, result in results.items():
        overall_metrics = result.get("overall_metrics", {})
        fg_bg_metrics = result.get("foreground_vs_background_metrics", {})
        fg_bg_instance_prf = fg_bg_metrics.get("instance_level_prf", {})

        mean_hd95 = overall_metrics.get("mean_hd95", np.nan)
        if not np.isfinite(mean_hd95):
            mean_hd95 = np.nan

        record = {
            "sample_id": sample_id,
            "mean_dice": overall_metrics.get("mean_dice", np.nan),
            "mean_hd95": mean_hd95,
            "fgbg_instance_f1": fg_bg_instance_prf.get("f1", np.nan),
        }
        data.append(record)

    return pd.DataFrame(data)


def finite_series_values(series: pd.Series) -> np.ndarray:
    """Return only finite numeric values from a pandas Series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def extract_instance_cls_conf_matrices(
    results: Dict,
) -> Tuple[np.ndarray, List[int]]:
    """Average InstanceClsConf transition matrices from noise-analysis JSON."""
    matrices = []
    all_classes = {0}
    for result in results.values():
        transition_matrix = (
            result.get("class_confusion_metrics", {})
            .get("InstanceClsConf", {})
            .get("transition_matrix", {})
        )
        if not transition_matrix:
            continue
        matrices.append(transition_matrix)
        for source_class, row in transition_matrix.items():
            all_classes.add(int(source_class))
            all_classes.update(int(target_class) for target_class in row)

    sorted_classes = sorted(all_classes)
    matrix = np.zeros((len(sorted_classes), len(sorted_classes)), dtype=float)
    class_to_idx = {class_id: idx for idx, class_id in enumerate(sorted_classes)}
    matrix[class_to_idx[0], class_to_idx[0]] = 1.0

    for source_class in sorted_classes:
        if source_class == 0:
            continue
        valid_rows = []
        for sample_matrix in matrices:
            row = sample_matrix.get(str(source_class), sample_matrix.get(source_class))
            if row is None or not any(value is not None for value in row.values()):
                continue
            valid_rows.append(row)
        for target_class in sorted_classes:
            if target_class == 0:
                continue
            values = [
                row.get(str(target_class), row.get(target_class, 0.0))
                for row in valid_rows
            ]
            finite_values = [float(value) for value in values if value is not None]
            if finite_values:
                matrix[class_to_idx[source_class], class_to_idx[target_class]] = (
                    float(np.mean(finite_values))
                )

    return matrix, sorted_classes


def plot_all_metrics_combined(
    df: pd.DataFrame,
    overlap_matrix: np.ndarray,
    sorted_classes: List[int],
    output_path: str,
    hd95_unit: str = "mm",
):
    """
    Create a comprehensive figure with all 9 metrics as subplots.
    
    Layout: 3x3 grid for metrics + overlap matrix integrated.
    """
    classes = sorted(df["class_id"].unique())
    n_classes = len(classes)
    
    # Create a large figure with subplots
    # Use GridSpec for flexible layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Helper function to add violin plot to subplot
    def add_violinplot(ax, metric, ylabel, title):
        """Add a single violin plot to subplot."""
        data_by_class = [df[df["class_id"] == cls][metric].dropna().values for cls in classes]
        
        parts = ax.violinplot(
            data_by_class,
            positions=range(len(classes)),
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )
        
        # Color the violin bodies
        colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(classes))]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Style the median lines
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.5)
        
        # Style the mean lines (dashed red)
        parts['cmeans'].set_edgecolor('red')
        parts['cmeans'].set_linewidth(1.5)
        parts['cmeans'].set_linestyle('--')
        
        # Style the whiskers and caps
        for partname in ('cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)
        
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f"C{cls}" for cls in classes])
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
    
    # Helper function for comparison plots (side-by-side violins)
    def add_comparison_violinplot(ax, metric_clean, metric_noisy, ylabel, title):
        """Add side-by-side comparison violin plot."""
        positions = []
        data_to_plot = []
        colors_list = []
        
        for i, cls in enumerate(classes):
            clean_data = df[df["class_id"] == cls][metric_clean].dropna().values
            noisy_data = df[df["class_id"] == cls][metric_noisy].dropna().values
            
            pos_clean = i * 2.5 + 0.5
            pos_noisy = i * 2.5 + 1.5
            
            positions.extend([pos_clean, pos_noisy])
            data_to_plot.extend([clean_data, noisy_data])
            colors_list.extend(["#3498db", "#e74c3c"])
        
        parts = ax.violinplot(
            data_to_plot,
            positions=positions,
            widths=0.5,
            showmeans=True,
            showmedians=True,
        )
        
        # Color the violin bodies
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Style the median and mean lines
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cmeans'].set_edgecolor('yellow')
        parts['cmeans'].set_linewidth(1.5)
        parts['cmeans'].set_linestyle('--')
        
        # Style whiskers and caps
        for partname in ('cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)
        
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks([i * 2.5 + 1 for i in range(len(classes))])
        ax.set_xticklabels([f"C{cls}" for cls in classes], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#3498db", alpha=0.6, label="Clean"),
            Patch(facecolor="#e74c3c", alpha=0.6, label="Noisy"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    # Row 0: Basic metrics (Dice, NSD, HD95)
    ax0 = fig.add_subplot(gs[0, 0])
    add_violinplot(ax0, "dice", "Dice", "Per-Class Dice Coefficient")
    
    ax1 = fig.add_subplot(gs[0, 1])
    add_violinplot(ax1, "nsd", "NSD", "Per-Class NSD")
    
    ax2 = fig.add_subplot(gs[0, 2])
    add_violinplot(ax2, "hd95", f"HD95 ({hd95_unit})", "Per-Class HD95")
    
    # Row 1: Relative volume diff, CC comparisons
    ax3 = fig.add_subplot(gs[1, 0])
    add_violinplot(ax3, "relative_volume_diff", "Rel. Vol. Diff", "Relative Volume Difference")
    
    ax4 = fig.add_subplot(gs[1, 1])
    add_comparison_violinplot(
        ax4, "num_cc_clean", "num_cc_noisy", "Num CC", "Num Connected Components: Clean vs Noisy"
    )
    
    ax5 = fig.add_subplot(gs[1, 2])
    add_comparison_violinplot(
        ax5, "avg_vol_cc_clean", "avg_vol_cc_noisy", "Avg Vol (voxels)", "Avg CC Volume: Clean vs Noisy"
    )
    
    # Row 2: Delta metrics + overlap matrix
    ax6 = fig.add_subplot(gs[2, 0])
    add_violinplot(ax6, "delta_num_cc", "Δ Num CC", "Delta Num CC (Noisy - Clean)")
    
    ax7 = fig.add_subplot(gs[2, 1])
    add_violinplot(ax7, "delta_avg_vol_cc", "Δ Avg Vol (voxels)", "Delta Avg Vol CC (Noisy - Clean)")
    
    # InstanceClsConf transition matrix in last position
    ax8 = fig.add_subplot(gs[2, 2])
    im = ax8.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax8.set_xticks(range(len(sorted_classes)))
    ax8.set_yticks(range(len(sorted_classes)))
    ax8.set_xticklabels([f"C{c}" for c in sorted_classes], fontsize=9)
    ax8.set_yticklabels([f"C{c}" for c in sorted_classes], fontsize=9)
    ax8.set_xlabel("Noisy Class", fontsize=10)
    ax8.set_ylabel("Clean Class", fontsize=10)
    ax8.set_title("InstanceClsConf Matrix", fontsize=11, fontweight="bold")
    
    # Add text annotations to overlap matrix
    for i in range(len(sorted_classes)):
        for j in range(len(sorted_classes)):
            text = ax8.text(j, i, f"{overlap_matrix[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
    cbar.set_label("Matched-instance fraction", fontsize=9)
    
    # Main title
    fig.suptitle(
        "Comprehensive Per-Class Noise Analysis: Clean vs Noisy Segmentation Masks",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved comprehensive plot: {output_path}")


def plot_requested_metrics_single_row(
    df: pd.DataFrame,
    overlap_matrix: np.ndarray,
    sorted_classes: List[int],
    output_path: str,
    figsize: tuple[float, float] | None = None,
    keep_ratio: bool = False,
    hd95_unit: str = "mm",
):
    """
    Create a single-row figure with 4 subplots:
    Dice, HD95, Instance F1, and InstanceClsConf matrix.

    The total figure width is fixed so the row length is identical
    across datasets, independent of the number of classes.
    """
    classes = sorted(df["class_id"].unique())

    if figsize is None:
        figsize = (REQUESTED_ROW_DEFAULT_WIDTH, REQUESTED_ROW_DEFAULT_HEIGHT)

    if keep_ratio:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            1,
            7,
            width_ratios=[1, 2 / 3, 1, 2 / 3, 1, 2 / 3, 1.2],
            wspace=0.0,
        )
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[0, 4]),
            fig.add_subplot(gs[0, 6]),
        ]
    else:
        fig, axes = plt.subplots(
            1,
            4,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1, 1, 1.2]},
        )

    def add_perclass_violinplot(ax, metric, ylabel, title):
        data_by_class = [
            finite_series_values(df[df["class_id"] == cls][metric]) for cls in classes
        ]

        parts = ax.violinplot(
            data_by_class,
            positions=range(len(classes)),
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )
        
        # Color the violin bodies
        colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(classes))]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Style the median and mean lines
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cmeans'].set_edgecolor('red')
        parts['cmeans'].set_linewidth(1.5)
        parts['cmeans'].set_linestyle('--')
        
        # Style whiskers and caps
        for partname in ('cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)

        ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f"Class {cls}" for cls in classes], fontsize=TICK_FONT_SIZE)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    add_perclass_violinplot(
        axes[0],
        "dice",
        "Dice",
        "Class-wise Dice\n(Consensus vs Noisy)",
    )
    add_perclass_violinplot(
        axes[1],
        "hd95",
        f"HD95 ({hd95_unit})",
        "Class-wise HD95\n(Consensus vs Noisy)",
    )
    add_perclass_violinplot(
        axes[2],
        "instance_f1",
        "Instance F1",
        "Class-wise Instance F1\n(Consensus vs Noisy)",
    )

    # InstanceClsConf transition matrix among matched foreground objects.
    ax_cm = axes[3]
    im = ax_cm.imshow(overlap_matrix, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="equal")
    ax_cm.set_xticks(range(len(sorted_classes)))
    ax_cm.set_yticks(range(len(sorted_classes)))
    ax_cm.set_xticklabels([f"C{c}" for c in sorted_classes], fontsize=TICK_FONT_SIZE)
    ax_cm.set_yticklabels([f"C{c}" for c in sorted_classes], fontsize=TICK_FONT_SIZE)
    ax_cm.set_xlabel("Noisy Classes", fontsize=BASE_FONT_SIZE)
    ax_cm.set_ylabel("Consensus Classes", fontsize=BASE_FONT_SIZE)
    ax_cm.set_title("InstanceClsConf Matrix\n(Consensus vs Noisy)", fontsize=TITLE_FONT_SIZE, fontweight="bold")

    for i in range(len(sorted_classes)):
        for j in range(len(sorted_classes)):
            ax_cm.text(
                j,
                i,
                f"{overlap_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="#0b1f2a" if overlap_matrix[i, j] < 0.62 else "white",
                fontsize=ANNOTATION_FONT_SIZE,
            )

    cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label("Matched-instance fraction", fontsize=BASE_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved requested metrics single-row plot: {output_path}")


def plot_core_metrics_single_row(
    df: pd.DataFrame,
    overlap_matrix: np.ndarray,
    sorted_classes: List[int],
    output_path: str,
    hd95_unit: str = "mm",
):
    """
    Create a single-row figure with 5 subplots:
    Dice, NSD, HD95, delta V/V, and InstanceClsConf matrix.

    The total figure width is fixed so the row length is identical
    across datasets, independent of the number of classes.
    """
    classes = sorted(df["class_id"].unique())

    # Fixed-size canvas: row length stays constant across datasets
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.2))

    def add_violinplot(ax, metric, ylabel, title):
        data_by_class = [
            df[df["class_id"] == cls][metric].dropna().values for cls in classes
        ]

        parts = ax.violinplot(
            data_by_class,
            positions=range(len(classes)),
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )
        
        # Color the violin bodies
        colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(classes))]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Style the median and mean lines
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cmeans'].set_edgecolor('red')
        parts['cmeans'].set_linewidth(1.5)
        parts['cmeans'].set_linestyle('--')
        
        # Style whiskers and caps
        for partname in ('cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f"C{cls}" for cls in classes])
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=9)

    add_violinplot(axes[0], "dice", "Dice", "Per-Class Dice")
    add_violinplot(axes[1], "nsd", "NSD", "Per-Class NSD")
    add_violinplot(axes[2], "hd95", f"HD95 ({hd95_unit})", "Per-Class HD95")
    add_violinplot(
        axes[3],
        "relative_volume_diff",
        "ΔV / V",
        "Per-Class ΔV / V",
    )

    # InstanceClsConf transition matrix among matched foreground objects.
    ax_cm = axes[4]
    im = ax_cm.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax_cm.set_xticks(range(len(sorted_classes)))
    ax_cm.set_yticks(range(len(sorted_classes)))
    ax_cm.set_xticklabels([f"C{c}" for c in sorted_classes], fontsize=8)
    ax_cm.set_yticklabels([f"C{c}" for c in sorted_classes], fontsize=8)
    ax_cm.set_xlabel("Noisy Class", fontsize=9)
    ax_cm.set_ylabel("Clean Class", fontsize=9)
    ax_cm.set_title("InstanceClsConf Matrix", fontsize=11, fontweight="bold")

    for i in range(len(sorted_classes)):
        for j in range(len(sorted_classes)):
            ax_cm.text(
                j,
                i,
                f"{overlap_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label("Matched-instance fraction", fontsize=8)

    fig.suptitle(
        "Core Per-Class Metrics + InstanceClsConf Matrix",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved single-row core metrics plot: {output_path}")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for per-class metrics."""
    print("\n" + "=" * 80)
    print("PER-CLASS SUMMARY STATISTICS")
    print("=" * 80)
    
    classes = sorted(df["class_id"].unique())
    print(f"\nTotal samples analyzed: {df['sample_id'].nunique()}")
    print(f"Classes found: {classes}")
    
    for cls in classes:
        cls_data = df[df["class_id"] == cls]
        print(f"\n{'─' * 80}")
        print(f"Class {cls}:")
        print(f"{'─' * 80}")
        print(f"  Number of samples: {len(cls_data)}")
        
        metrics = {
            "Dice": "dice",
            "NSD": "nsd",
            "HD95": "hd95",
            "Rel. Volume Diff": "relative_volume_diff",
            "Num CC (Clean)": "num_cc_clean",
            "Num CC (Noisy)": "num_cc_noisy",
            "Delta Num CC": "delta_num_cc",
            "Avg Vol CC (Clean)": "avg_vol_cc_clean",
            "Avg Vol CC (Noisy)": "avg_vol_cc_noisy",
            "Delta Avg Vol CC": "delta_avg_vol_cc",
        }
        
        for display_name, col_name in metrics.items():
            values = cls_data[col_name].dropna()
            if len(values) > 0:
                print(
                    f"  {display_name:20s}: {values.mean():8.4f} ± {values.std():8.4f} "
                    f"[{values.min():8.4f}, {values.max():8.4f}]"
                )
            else:
                print(f"  {display_name:20s}: No data")
    
    print("\n" + "=" * 80)


def main(args):
    """Main execution function."""
    
    # Load data
    print(f"Loading noise analysis results from: {args.input_json}")
    results = load_noise_analysis(args.input_json)
    hd95_unit = infer_hd95_unit(results)
    
    print(f"Processing {len(results)} samples...")
    df = extract_perclass_data(results)
    
    if len(df) == 0:
        print("ERROR: No per-class data found in the results!")
        return
    
    print(f"Extracted {len(df)} class-sample pairs across {df['sample_id'].nunique()} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract overlap matrices
    print("\nExtracting InstanceClsConf transition matrices...")
    overlap_matrix, sorted_classes = extract_instance_cls_conf_matrices(results)
    
    # Generate combined visualization
    print("Generating comprehensive per-class visualization...")
    plot_all_metrics_combined(
        df,
        overlap_matrix,
        sorted_classes,
        os.path.join(args.output_dir, "00_comprehensive_perclass_analysis.png"),
        hd95_unit=hd95_unit,
    )

    print("Generating single-row core metrics visualization...")
    plot_core_metrics_single_row(
        df,
        overlap_matrix,
        sorted_classes,
        os.path.join(args.output_dir, "01_core_metrics_single_row.png"),
        hd95_unit=hd95_unit,
    )

    print("Generating single-row requested metrics (Dice, HD95, Instance F1, Confusion)...")
    requested_row_figsize = resolve_requested_row_figsize(
        width=args.requested_row_width,
        height=args.requested_row_height,
    )
    plot_requested_metrics_single_row(
        df,
        overlap_matrix,
        sorted_classes,
        os.path.join(args.output_dir, "02_requested_metrics_single_row.png"),
        figsize=requested_row_figsize,
        keep_ratio=args.requested_row_keep_ratio,
        hd95_unit=hd95_unit,
    )
    
    # Print statistics
    print_summary_statistics(df)
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-class violin plot visualizations for noise analysis"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the noise_analysis_results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/noise_perclass_boxplots",
        help="Directory to save visualization outputs (default: ./results/noise_perclass_boxplots)",
    )
    parser.add_argument(
        "--requested_row_width",
        type=float,
        default=REQUESTED_ROW_DEFAULT_WIDTH,
        help=(
            "Width of 02_requested_metrics_single_row.png in inches "
            f"(default: {REQUESTED_ROW_DEFAULT_WIDTH}, matching the multirater-consensus figure width)"
        ),
    )
    parser.add_argument(
        "--requested_row_height",
        type=float,
        default=REQUESTED_ROW_DEFAULT_HEIGHT,
        help=(
            "Height of 02_requested_metrics_single_row.png in inches "
            f"(default: {REQUESTED_ROW_DEFAULT_HEIGHT})"
        ),
    )
    parser.add_argument(
        "--requested_row_keep_ratio",
        action="store_true",
        help=(
            "If set, keep the lower row's subplot proportions aligned with the upper "
            "multirater-consensus figure and distribute the remaining width as equal "
            "spacing between panels. If not set, the panels stretch to use all available space."
        ),
    )
    
    args = parser.parse_args()
    main(args)
