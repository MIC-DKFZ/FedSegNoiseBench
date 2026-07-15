"""
Comprehensive visualization script for noise analysis results.

Implements three main visualization types to highlight differences between
noisy and non-noisy labels:

A) Noise magnitude vs disagreement scatter plot
   - X-axis: mean_dice (or signal-to-noise ratio)
   - Y-axis: mean_hd95 (or mean_nsd)
   - Color: noise group (low/mid/high quantiles)
   - Shape: dataset or client

B) Violin/box plots by noise group
   - Distribution of Dice/NSD/HD95 for low vs high noise groups
   - Publication-friendly display

C) 2D noise type map
   - X-axis: boundary disagreement (mean_hd95)
   - Y-axis: missed/additional proxy (delta_total_num_cc or relative volume diff)
   - Color: swap score (off-diagonal mass from class_overlap_matrix)
   - Interpretable quadrants for different noise types
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
from scipy import stats


def load_noise_analysis(json_file: str) -> Dict:
    """Load noise analysis results from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


def is_finite_number(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compute_swap_score(class_overlap_matrix: Dict[str, Dict[str, float]]) -> float:
    """
    Compute foreground-to-other-foreground class swaps.

    Transitions involving background are excluded in both directions.
    """
    foreground_classes = sorted(
        int(class_id) for class_id in class_overlap_matrix if int(class_id) != 0
    )
    if len(foreground_classes) < 2:
        return np.nan

    swap_scores = []
    for source_class in foreground_classes:
        overlaps = class_overlap_matrix.get(str(source_class), {})
        if not any(is_finite_number(value) and float(value) > 0.0 for value in overlaps.values()):
            continue
        swap_scores.append(sum(
            float(overlaps.get(str(target_class), 0.0))
            for target_class in foreground_classes
            if target_class != source_class
            and is_finite_number(overlaps.get(str(target_class), 0.0))
        ))

    return float(np.mean(swap_scores)) if swap_scores else np.nan


def create_dataframe(results: Dict) -> pd.DataFrame:
    """
    Convert noise analysis results to a pandas DataFrame for easier manipulation.

    Returns DataFrame with columns:
    - sample_id, clean_dataset_id, noisy_dataset_id, client_idx
    - mean_dice, mean_nsd, mean_hd95
    - total_volume_clean, total_volume_noisy, delta_total_num_cc
    - swap_score
    - noise_group (assigned based on quantiles)

    Filters out samples with missing or empty overall_metrics.
    """
    data = []

    for sample_id, result in results.items():
        # Skip samples with missing or empty overall_metrics
        overall_metrics = result.get("overall_metrics", {})
        if (
            not overall_metrics
            or "mean_dice" not in overall_metrics
            or overall_metrics.get("mean_dice") is None
        ):
            continue

        record = {
            "sample_id": sample_id,
            "clean_dataset_id": result.get("clean_dataset_id", "unknown"),
            "noisy_dataset_id": result.get("noisy_dataset_id", "unknown"),
            "client_idx": result.get("client_idx", 0),
            "mean_dice": overall_metrics.get("mean_dice", np.nan),
            "mean_nsd": overall_metrics.get("mean_nsd", np.nan),
            "mean_hd95": overall_metrics.get("mean_hd95", np.nan),
            "total_volume_clean": overall_metrics.get("total_volume_clean", 0),
            "total_volume_noisy": overall_metrics.get("total_volume_noisy", 0),
            "delta_total_num_cc": overall_metrics.get("delta_total_num_cc", 0),
            "swap_score": compute_swap_score(result.get("class_overlap_matrix", {})),
        }
        data.append(record)

    df = pd.DataFrame(data)

    # Filter out rows with NaN in critical columns
    df = df.dropna(subset=["mean_dice", "mean_nsd", "mean_hd95"])

    if len(df) == 0:
        raise ValueError(
            "No valid samples found with complete metrics. "
            "Check that the noise analysis file has overall_metrics with mean_dice, mean_nsd, mean_hd95."
        )

    # Assign noise groups based on mean_dice quantiles (lower dice = more noise)
    # Handle edge case of too few unique values
    try:
        df["noise_group"] = pd.qcut(
            df["mean_dice"],
            q=3,
            labels=["High Noise", "Mid Noise", "Low Noise"],
            duplicates="drop",
        )
    except ValueError:
        # If quantiles fail (e.g., too many duplicates), use simple quantile cutoffs
        q33 = df["mean_dice"].quantile(0.33)
        q67 = df["mean_dice"].quantile(0.67)
        df["noise_group"] = pd.cut(
            df["mean_dice"],
            bins=[df["mean_dice"].min() - 0.01, q33, q67, df["mean_dice"].max() + 0.01],
            labels=["High Noise", "Mid Noise", "Low Noise"],
            include_lowest=True,
        )

    # Compute additional derived metrics
    df["relative_volume_diff_abs"] = np.abs(
        (df["total_volume_noisy"] - df["total_volume_clean"])
        / (df["total_volume_clean"] + 1e-6)
    )

    # Boundary disagreement proxy (inverse NSD - higher means worse boundary match)
    df["boundary_disagreement"] = 1.0 - df["mean_nsd"]

    return df


def plot_noise_vs_disagreement(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualization A: Noise magnitude vs disagreement scatter plot.

    X-axis: mean_dice (noise magnitude)
    Y-axis: mean_hd95 (boundary disagreement)
    Color: noise_group (low/mid/high quantiles)
    Shape: client_idx (different datasets)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use actual noise groups in the data (handles variable number of groups)
    noise_groups = sorted(df["noise_group"].unique())
    color_map = {
        "High Noise": "#e74c3c",
        "Mid Noise": "#f39c12",
        "Low Noise": "#2ecc71",
    }
    colors = {g: color_map.get(g, "#95a5a6") for g in noise_groups}

    for noise_group in noise_groups:
        group_data = df[df["noise_group"] == noise_group]
        ax.scatter(
            group_data["mean_dice"],
            group_data["mean_hd95"],
            s=150,
            alpha=0.7,
            color=colors[noise_group],
            label=str(noise_group),
            edgecolors="black",
            linewidth=1.5,
        )

    ax.set_xlabel(
        "Mean Dice Coefficient (Higher = Less Noisy)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        "Mean Hausdorff Distance 95% (Lower = Less Noisy)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Noise Magnitude vs Boundary Disagreement", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="best", framealpha=0.9, title="Noise Level")

    # Invert x-axis so high Dice is on the right
    ax.invert_xaxis()

    plt.tight_layout()
    output_file = os.path.join(output_dir, "01_noise_vs_disagreement_scatter.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_noise_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualization B: Violin/box plots by noise group.

    Shows distribution of Dice, NSD, and HD95 for different noise groups.
    Publication-friendly visualization.
    """
    # Reshape data for better plotting
    metrics_to_plot = [
        ("mean_dice", "Mean Dice Coefficient"),
        ("mean_nsd", "Mean Normalized Surface Distance (NSD)"),
        ("mean_hd95", "Mean Hausdorff Distance 95%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Get actual noise groups in the data
    noise_groups = sorted(df["noise_group"].unique())
    n_groups = len(noise_groups)

    # Color map for noise groups
    color_map = {
        "High Noise": "#e74c3c",
        "Mid Noise": "#f39c12",
        "Low Noise": "#2ecc71",
    }
    colors = [color_map.get(g, "#95a5a6") for g in noise_groups]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Prepare data for violin plot - only include groups that exist
        data_for_plot = [
            df[df["noise_group"] == group][metric].values for group in noise_groups
        ]
        positions = list(range(n_groups))

        # Create violin plot
        try:
            parts = ax.violinplot(
                data_for_plot, positions=positions, showmeans=True, showmedians=True
            )

            # Customize violin colors
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
        except ValueError as e:
            # Fallback if violinplot fails (e.g., single-point distributions)
            for pos, data, color in zip(positions, data_for_plot, colors):
                if len(data) > 0:
                    ax.scatter([pos] * len(data), data, color=color, s=50, alpha=0.6)

        # Add box plot overlay for better visualization
        bp = ax.boxplot(
            data_for_plot,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

        ax.set_xticks(positions)
        ax.set_xticklabels([str(g) for g in noise_groups], fontsize=11)
        ax.set_ylabel(title, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    output_file = os.path.join(output_dir, "02_noise_distributions_violin.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_noise_type_map(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualization C: 2D noise type map with interpretable quadrants.

    X-axis: boundary disagreement (1 - mean_nsd)
    Y-axis: missed/additional proxy (relative_volume_diff_abs)
    Color: swap_score (indicating class swaps)
    Size: mean_hd95 (boundary error magnitude)

    Quadrants:
    - High X, Low Y: contour noise (boundary error without significant volume change)
    - Low X, High Y: missed/additional structures (volume change without boundary error)
    - High X, High Y: complex noise (both types)
    - Low X, Low Y: clean samples
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot with multiple encodings
    scatter = ax.scatter(
        df["boundary_disagreement"],  # X: boundary disagreement
        df["relative_volume_diff_abs"],  # Y: missed/additional proxy
        c=df["swap_score"],  # Color: swap score
        s=df["mean_hd95"] * 3,  # Size: HD95 magnitude
        alpha=0.6,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=df["swap_score"].max(),
        edgecolors="black",
        linewidth=1,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("PixelClsConf", fontsize=11, fontweight="bold")

    # Add quadrant lines
    ax.axhline(
        y=df["relative_volume_diff_abs"].median(),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )
    ax.axvline(
        x=df["boundary_disagreement"].median(),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )

    # Label quadrants
    ax.text(
        0.95 * df["boundary_disagreement"].max(),
        0.95 * df["relative_volume_diff_abs"].max(),
        "Complex Noise\n(Boundary + Volume)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.text(
        0.05 * df["boundary_disagreement"].max(),
        0.95 * df["relative_volume_diff_abs"].max(),
        "Missed/Additional\nStructures",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )
    ax.text(
        0.95 * df["boundary_disagreement"].max(),
        0.05 * df["relative_volume_diff_abs"].max(),
        "Contour Noise\n(Boundary Only)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )
    ax.text(
        0.05 * df["boundary_disagreement"].max(),
        0.05 * df["relative_volume_diff_abs"].max(),
        "Clean Samples",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    ax.set_xlabel(
        "Boundary Disagreement (1 - Mean NSD)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        "Missed/Additional Structures (|ΔV|/V)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "2D Noise Type Map: Classification by Error Type",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add size legend for HD95
    for size in [10, 30, 50]:
        ax.scatter(
            [],
            [],
            s=size * 3,
            c="gray",
            alpha=0.6,
            edgecolors="black",
            label=f"HD95 ≈ {size}",
        )
    ax.legend(
        scatterpoints=1, frameon=True, labelspacing=2, loc="upper left", fontsize=10
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "03_noise_type_map_2d.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_noise_type_map_sampleid(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualization C: 2D noise type map with interpretable quadrants.

    X-axis: boundary disagreement (1 - mean_nsd)
    Y-axis: missed/additional proxy (relative_volume_diff_abs)
    Color: swap_score (indicating class swaps)
    Size: mean_hd95 (boundary error magnitude)

    Quadrants:
    - High X, Low Y: contour noise (boundary error without significant volume change)
    - Low X, High Y: missed/additional structures (volume change without boundary error)
    - High X, High Y: complex noise (both types)
    - Low X, Low Y: clean samples
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot with multiple encodings
    scatter = ax.scatter(
        df["boundary_disagreement"],  # X: boundary disagreement
        df["relative_volume_diff_abs"],  # Y: missed/additional proxy
        c=df["swap_score"],  # Color: swap score
        s=df["mean_hd95"] * 3,  # Size: HD95 magnitude
        alpha=0.6,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=df["swap_score"].max(),
        edgecolors="black",
        linewidth=1,
    )

    # Add sample ID labels next to each point
    for idx, row in df.iterrows():
        ax.annotate(
            row["sample_id"],
            xy=(row["boundary_disagreement"], row["relative_volume_diff_abs"]),
            xytext=(5, 5),  # offset in points
            textcoords="offset points",
            fontsize=7,
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.6),
        )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("PixelClsConf", fontsize=11, fontweight="bold")

    # Add quadrant lines
    ax.axhline(
        y=df["relative_volume_diff_abs"].median(),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )
    ax.axvline(
        x=df["boundary_disagreement"].median(),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )

    # Label quadrants
    ax.text(
        0.95 * df["boundary_disagreement"].max(),
        0.95 * df["relative_volume_diff_abs"].max(),
        "Complex Noise\n(Boundary + Volume)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.text(
        0.05 * df["boundary_disagreement"].max(),
        0.95 * df["relative_volume_diff_abs"].max(),
        "Missed/Additional\nStructures",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )
    ax.text(
        0.95 * df["boundary_disagreement"].max(),
        0.05 * df["relative_volume_diff_abs"].max(),
        "Contour Noise\n(Boundary Only)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )
    ax.text(
        0.05 * df["boundary_disagreement"].max(),
        0.05 * df["relative_volume_diff_abs"].max(),
        "Clean Samples",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    ax.set_xlabel(
        "Boundary Disagreement (1 - Mean NSD)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        "Missed/Additional Structures (|ΔV|/V)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "2D Noise Type Map: Classification by Error Type",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add size legend for HD95
    for size in [10, 30, 50]:
        ax.scatter(
            [],
            [],
            s=size * 3,
            c="gray",
            alpha=0.6,
            edgecolors="black",
            label=f"HD95 ≈ {size}",
        )
    ax.legend(
        scatterpoints=1, frameon=True, labelspacing=2, loc="upper left", fontsize=10
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "03_noise_type_map_2d_w_sampleid.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_noise_type_map_dice(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualization C: 2D noise type map with interpretable quadrants.

    X-axis: boundary disagreement (1 - mean_nsd)
    Y-axis: mean_dice (Dice coefficient)
    Color: swap_score (indicating class swaps)
    Size: mean_hd95 (boundary error magnitude)

    Quadrants:
    - High X, Low Y: contour noise (boundary error without significant overlap)
    - Low X, High Y: good quality (good boundaries and good overlap)
    - High X, High Y: mixed quality (poor boundaries but good overlap)
    - Low X, Low Y: clean samples (good boundaries and good overlap)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot with multiple encodings
    scatter = ax.scatter(
        df["boundary_disagreement"],  # X: boundary disagreement
        df["mean_dice"],  # Y: Dice coefficient
        c=df["swap_score"],  # Color: swap score
        s=df["mean_hd95"] * 3,  # Size: HD95 magnitude
        alpha=0.6,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=df["swap_score"].max(),
        edgecolors="black",
        linewidth=1,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("PixelClsConf", fontsize=11, fontweight="bold")

    # Add quadrant lines
    ax.axhline(
        y=df["mean_dice"].median(),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )
    ax.axvline(
        x=df["boundary_disagreement"].median(),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )

    # Label quadrants
    ax.text(
        0.95 * df["boundary_disagreement"].max(),
        0.95 * df["mean_dice"].max(),
        "Mixed Quality\n(Poor Boundaries,\nGood Overlap)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.text(
        0.05 * df["boundary_disagreement"].max(),
        0.95 * df["mean_dice"].max(),
        "Good Quality\n(Good Boundaries,\nGood Overlap)",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )
    ax.text(
        0.95 * df["boundary_disagreement"].max(),
        0.05 * df["mean_dice"].max(),
        "Contour Noise\n(Poor Boundaries,\nPoor Overlap)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )
    ax.text(
        0.05 * df["boundary_disagreement"].max(),
        0.05 * df["mean_dice"].max(),
        "Clean Samples\n(Good Boundaries,\nGood Overlap)",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    ax.set_xlabel(
        "Boundary Disagreement (1 - Mean NSD)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        "Mean Dice Coefficient (Higher = Better Overlap)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "2D Noise Type Map: Boundary vs Overlap Quality",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add size legend for HD95
    for size in [10, 30, 50]:
        ax.scatter(
            [],
            [],
            s=size * 3,
            c="gray",
            alpha=0.6,
            edgecolors="black",
            label=f"HD95 ≈ {size}",
        )
    ax.legend(
        scatterpoints=1, frameon=True, labelspacing=2, loc="upper left", fontsize=10
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "03_noise_type_map_dice.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_noise_type_map_dice_hd(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualization C: 2D noise type map - Dice vs HD95.

    X-axis: mean_dice (dominant overlap metric, higher = better agreement)
    Y-axis: mean_hd95 (dominant contour-based metric, lower = better boundary match)
    Color: swap_score (PixelClsConf/label swapping)
    Size: relative_volume_diff_abs (volume change magnitude)

    Quadrants:
    - High Dice, Low HD95: clean/good quality samples (high overlap, good boundaries)
    - Low Dice, Low HD95: boundary-dominant noise (good boundaries but poor overlap)
    - High Dice, High HD95: volume-dominant noise (good overlap but poor boundaries)
    - Low Dice, High HD95: complex noise (poor overlap and poor boundaries)
    """
    fig, ax = plt.subplots(figsize=(13, 10))

    # Create scatter plot with multiple encodings
    scatter = ax.scatter(
        df["mean_dice"],  # X: Dice coefficient (overlap metric)
        df["mean_hd95"],  # Y: HD95 (contour-based metric)
        c=df["swap_score"],  # Color: PixelClsConf/swap score
        s=df["relative_volume_diff_abs"] * 500 + 50,  # Size: volume changes
        alpha=0.6,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=df["swap_score"].max(),
        edgecolors="black",
        linewidth=1,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("PixelClsConf", fontsize=11, fontweight="bold")

    # Add quadrant lines at medians
    dice_median = df["mean_dice"].median()
    hd95_median = df["mean_hd95"].median()
    
    ax.axhline(
        y=hd95_median,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )
    ax.axvline(
        x=dice_median,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
    )

    # Label quadrants
    ax.text(
        0.95 * df["mean_dice"].max(),
        0.05 * df["mean_hd95"].max(),
        "Clean/High Quality\n(Good Overlap,\nGood Boundaries)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )
    ax.text(
        0.05 * df["mean_dice"].max(),
        0.05 * df["mean_hd95"].max(),
        "Boundary-Dominant Noise\n(Good Boundaries,\nPoor Overlap)",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )
    ax.text(
        0.95 * df["mean_dice"].max(),
        0.95 * df["mean_hd95"].max(),
        "Volume-Dominant Noise\n(Good Overlap,\nPoor Boundaries)",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )
    ax.text(
        0.05 * df["mean_dice"].max(),
        0.95 * df["mean_hd95"].max(),
        "Complex Noise\n(Poor Overlap,\nPoor Boundaries)",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    )

    ax.set_xlabel(
        "Mean Dice Coefficient (Overlap Metric)\nHigher = Better Agreement →",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        "Mean Hausdorff Distance 95% (Contour Metric)\n← Lower = Better Boundaries",
        fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "2D Noise Type Map: Overlap vs Boundary Quality",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add size legend for volume changes
    vol_sizes = [0.05, 0.15, 0.25]
    for vol_change in vol_sizes:
        ax.scatter(
            [],
            [],
            s=vol_change * 500 + 50,
            c="gray",
            alpha=0.6,
            edgecolors="black",
            label=f"|ΔV|/V ≈ {vol_change:.0%}",
        )
    ax.legend(
        scatterpoints=1, frameon=True, labelspacing=2, loc="upper left", fontsize=10,
        title="Volume Change Magnitude"
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "03_noise_type_map_dice_hd.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_additional_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bonus: Additional complementary visualizations.

    - Correlation heatmap of key metrics
    - Distribution of noise groups
    """
    # Heatmap of correlations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correlation matrix
    correlation_cols = [
        "mean_dice",
        "mean_nsd",
        "mean_hd95",
        "swap_score",
        "relative_volume_diff_abs",
    ]
    corr_matrix = df[correlation_cols].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=axes[0],
        cbar_kws={"label": "Correlation"},
    )
    axes[0].set_title(
        "Correlation Matrix of Noise Metrics", fontsize=12, fontweight="bold"
    )

    # Noise group distribution - with adaptive coloring
    noise_counts = df["noise_group"].value_counts()

    # Create color mapping based on actual group names
    color_map = {
        "High Noise": "#e74c3c",
        "Mid Noise": "#f39c12",
        "Low Noise": "#2ecc71",
    }
    colors_for_bars = [
        color_map.get(str(group), "#95a5a6") for group in noise_counts.index
    ]

    axes[1].bar(
        range(len(noise_counts)),
        noise_counts.values,
        color=colors_for_bars,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1].set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Noise Group", fontsize=11, fontweight="bold")
    axes[1].set_title(
        "Distribution of Samples by Noise Group", fontsize=12, fontweight="bold"
    )
    axes[1].set_xticks(range(len(noise_counts)))
    axes[1].set_xticklabels([str(g) for g in noise_counts.index], fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y", linestyle="--")

    # Add counts on bars
    for i, val in enumerate(noise_counts.values):
        axes[1].text(
            i,
            val + max(noise_counts.values) * 0.02,
            str(val),
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "04_additional_metrics_correlation.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def print_statistics(df: pd.DataFrame) -> None:
    """Print detailed statistics about the noise analysis."""
    print("\n" + "=" * 80)
    print("NOISE ANALYSIS STATISTICS")
    print("=" * 80)

    print(f"\nTotal samples analyzed: {len(df)}")
    print(f"Number of datasets: {df['noisy_dataset_id'].nunique()}")
    print(f"Number of clients: {df['client_idx'].nunique()}")

    print("\nSample distribution by noise group:")
    print(df["noise_group"].value_counts().sort_index())

    print("\n" + "-" * 80)
    print("Metrics by Noise Group:")
    print("-" * 80)

    noise_groups = sorted(df["noise_group"].unique())

    for metric, title in [
        ("mean_dice", "Mean Dice"),
        ("mean_nsd", "Mean NSD"),
        ("mean_hd95", "Mean HD95"),
    ]:
        print(f"\n{title}:")
        for group in noise_groups:
            group_data = df[df["noise_group"] == group][metric]
            if len(group_data) > 0:
                print(
                    f"  {str(group):20s}: {group_data.mean():.4f} ± {group_data.std():.4f} [min: {group_data.min():.4f}, max: {group_data.max():.4f}] (n={len(group_data)})"
                )

    print("\n" + "-" * 80)
    print("Correlation Analysis:")
    print("-" * 80)
    correlation_cols = ["mean_dice", "mean_nsd", "mean_hd95", "swap_score"]
    corr_matrix = df[correlation_cols].corr()
    print(corr_matrix.to_string())

    print("\n" + "-" * 80)
    print("Swap Score Analysis:")
    print("-" * 80)
    print(
        f"Mean swap score: {df['swap_score'].mean():.4f} ± {df['swap_score'].std():.4f}"
    )
    print(
        f"Samples with high swap score (>0.3): {(df['swap_score'] > 0.3).sum()} / {len(df)}"
    )

    print("\n" + "-" * 80)
    print("Volume Change Analysis:")
    print("-" * 80)
    print(
        f"Mean relative volume diff: {df['relative_volume_diff_abs'].mean():.4f} ± {df['relative_volume_diff_abs'].std():.4f}"
    )

    print("\n" + "=" * 80)


def main(args):
    """Main execution function."""

    # Load and process data
    print(f"Loading noise analysis results from: {args.input_json}")
    results = load_noise_analysis(args.input_json)

    print(f"Processing {len(results)} samples...")
    df = create_dataframe(results)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    print("  [A] Noise magnitude vs disagreement scatter plot...")
    plot_noise_vs_disagreement(df, args.output_dir)

    print("  [B] Noise group distribution (violin/box plots)...")
    plot_noise_distributions(df, args.output_dir)

    print("  [C] 2D noise type map (Dice vs HD95)...")
    plot_noise_type_map(df, args.output_dir)
    plot_noise_type_map_sampleid(df, args.output_dir)
    plot_noise_type_map_dice(df, args.output_dir)
    plot_noise_type_map_dice_hd(df, args.output_dir)

    print("  [D] Additional metrics (correlation, distribution)...")
    plot_additional_metrics(df, args.output_dir)

    # Print statistics
    print_statistics(df)

    print(f"\n✓ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive visualization of noise analysis results"
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
        default="./results/noise_visualization",
        help="Directory to save visualization outputs (default: ./results/noise_visualization)",
    )

    args = parser.parse_args()
    main(args)
