"""
Visualize Dice coefficient vs Hausdorff distance with swap score as color.

This script creates a scatter plot where:
- X-axis: Mean Dice coefficient (higher = better segmentation agreement)
- Y-axis: Mean Hausdorff distance 95% (lower = better boundary agreement)
- Color: Swap score (confusion rate from class overlap matrix)
  Swap score measures proportion of misclassified voxels (higher = more label swapping)

Works for datasets with multiple foreground classes; samples with single class are marked separately.
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import argparse


def is_finite_number(x) -> bool:
    """Check if value is a valid finite number."""
    if x is None:
        return False
    if isinstance(x, str):
        return x.lower() not in ("infinity", "nan")
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def compute_swap_score(overlap_matrix: dict, fg_classes: list) -> float:
    """
    Compute foreground-to-other-foreground swaps, excluding background.
    """
    foreground_classes = sorted({int(c) for c in fg_classes if int(c) != 0})
    if len(foreground_classes) < 2:
        return np.nan
    values = []
    for source_class in foreground_classes:
        row = overlap_matrix.get(str(source_class), {})
        if not any(is_finite_number(v) and float(v) > 0.0 for v in row.values()):
            continue
        values.append(sum(
            float(row.get(str(target), 0.0))
            for target in foreground_classes
            if target != source_class and is_finite_number(row.get(str(target), 0.0))
        ))
    return float(np.mean(values)) if values else np.nan


def resolve_json_files(json_path: str):
    """Resolve JSON path to list of files."""
    if os.path.isdir(json_path):
        return sorted(glob.glob(os.path.join(json_path, "*.json")))
    matches = sorted(glob.glob(json_path))
    if matches:
        return matches
    if os.path.isfile(json_path):
        return [json_path]
    return []


def load_noise_analysis_data(json_path: str) -> pd.DataFrame:
    """
    Load noise analysis results and extract Dice, Hausdorff distance, and swap score.
    
    Args:
        json_path: Path to JSON file(s) from updated analyze_noise_clean_noisy.py
        
    Returns:
        DataFrame with columns: sample_id, dice, hd95, swap_score, n_fg_classes
    """
    json_files = resolve_json_files(json_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON file(s) found for json_path={json_path}")

    print(f"Loading {len(json_files)} JSON file(s)")

    rows = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        source_name = os.path.splitext(os.path.basename(json_file))[0]
        for sample_id, entry in data.items():
            overall = entry.get("overall_metrics", {})
            classes_info = entry.get("classes", {})
            fg_classes = classes_info.get("fg_classes", [])
            overlap = entry.get("class_overlap_matrix", {})

            # Extract Dice
            dice = overall.get("mean_dice", np.nan)
            dice = float(dice) if is_finite_number(dice) else np.nan

            # Extract Hausdorff distance 95%
            hd95 = overall.get("mean_hd95", np.nan)
            hd95 = float(hd95) if is_finite_number(hd95) else np.nan

            # Compute swap score
            swap_score = compute_swap_score(overlap, fg_classes)

            # Number of foreground classes
            n_fg_classes = len(fg_classes)

            row = {
                "sample_id": sample_id,
                "source_json": source_name,
                "dice": dice,
                "hd95": hd95,
                "swap_score": swap_score,
                "n_fg_classes": n_fg_classes,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No sample entries found in JSON file(s)")

    return df


def plot_dice_vs_hausdorff(
    df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (12, 9),
    marker_size: int = 80,
    show_labels: bool = False,
):
    """
    Create scatter plot of Dice vs Hausdorff distance with swap score as color.

    Args:
        df: DataFrame with dice, hd95, swap_score, n_fg_classes columns
        output_path: Path to save output PNG
        figsize: Figure size tuple
        marker_size: Size of scatter markers
        show_labels: Whether to annotate sample IDs next to points
    """
    # Separate samples by multiclass vs single-class
    df_multi = df[df["n_fg_classes"] > 1].copy()
    df_single = df[df["n_fg_classes"] == 1].copy()

    fig, ax = plt.subplots(figsize=figsize)

    has_data = False
    cbar = None

    # Plot multiclass samples (with color encoding for swap score)
    if len(df_multi) > 0:
        # Filter to valid values for plotting
        valid_multi = df_multi.dropna(subset=["dice", "hd95"])
        if len(valid_multi) > 0:
            has_data = True
            # Create colormap for swap score
            swap_scores = valid_multi["swap_score"].values
            # For samples without swap score, use NaN (will be gray)
            cmap_norm = Normalize(
                vmin=np.nanmin(swap_scores),
                vmax=np.nanmax(swap_scores),
                clip=False
            )
            cmap = plt.cm.RdYlGn_r  # Red=high swap (bad), Green=low swap (good)

            for idx, row in valid_multi.iterrows():
                swap = row["swap_score"]
                color = (
                    cmap(cmap_norm(swap))
                    if is_finite_number(swap)
                    else (0.7, 0.7, 0.7, 1.0)  # Gray for undefined
                )
                ax.scatter(
                    row["dice"],
                    row["hd95"],
                    s=marker_size,
                    c=[color],
                    edgecolors="black",
                    linewidth=0.8,
                    alpha=0.8,
                )
                
                # Annotate sample ID if requested
                if show_labels:
                    ax.annotate(
                        row["sample_id"],
                        xy=(row["dice"], row["hd95"]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
                    )

            # Add colorbar
            sm = ScalarMappable(cmap=cmap, norm=cmap_norm)
            sm.set_array([])
            cbar = plt.colorbar(
                sm, ax=ax, label="Swap Score (label confusion rate)",
                pad=0.02
            )

    # Plot single-class samples (gray, no swap score)
    if len(df_single) > 0:
        valid_single = df_single.dropna(subset=["dice", "hd95"])
        if len(valid_single) > 0:
            has_data = True
            ax.scatter(
                valid_single["dice"],
                valid_single["hd95"],
                s=marker_size,
                c="gray",
                edgecolors="black",
                linewidth=0.8,
                alpha=0.5,
                marker="^",
                label="Single foreground class (no swap score)",
            )
            
            # Annotate sample IDs if requested
            if show_labels:
                for idx, row in valid_single.iterrows():
                    ax.annotate(
                        row["sample_id"],
                        xy=(row["dice"], row["hd95"]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
                    )

    # Labels and title
    ax.set_xlabel("Mean Dice Coefficient", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Hausdorff Distance (95%)", fontsize=12, fontweight="bold")
    title_suffix = " (with sample labels)" if show_labels else ""
    ax.set_title(
        f"Segmentation Quality: Dice vs Hausdorff Distance{title_suffix}\n(Color = Label Swap Score for Multiclass)",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Only add legend if we have single-class samples
    if len(df_single) > 0:
        ax.legend(loc="best", fontsize=10, framealpha=0.95)

    # Add annotation for interpretation
    if has_data:
        textstr = "Better segmentation: ↑ Dice (right), ↓ HD95 (down)\nSwap score: Green=low confusion, Red=high confusion"
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Dice vs Hausdorff distance with swap score as color."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./results/noise_analysis/*.json",
        help="Path to noise analysis JSON file(s) or directory (supports wildcards)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/noise_analysis/dice_vs_hd95.png",
        help="Output PNG path (base name; _labeled variant also saved)",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[12, 9],
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--marker_size",
        type=int,
        default=80,
        help="Marker size in scatter plot",
    )

    args = parser.parse_args()

    # Load data
    df = load_noise_analysis_data(args.json_path)

    print(f"\nLoaded {len(df)} samples")
    print(f"  - Multiclass samples: {len(df[df['n_fg_classes'] > 1])}")
    print(f"  - Single-class samples: {len(df[df['n_fg_classes'] == 1])}")

    print("\nMetric statistics:")
    print(f"  Dice: {df['dice'].min():.3f} to {df['dice'].max():.3f} (mean: {df['dice'].mean():.3f})")
    print(f"  HD95: {df['hd95'].min():.2f} to {df['hd95'].max():.2f} (mean: {df['hd95'].mean():.2f})")
    valid_swap = df[df["n_fg_classes"] > 1]["swap_score"].dropna()
    if len(valid_swap) > 0:
        print(
            f"  Swap Score: {valid_swap.min():.3f} to {valid_swap.max():.3f} (mean: {valid_swap.mean():.3f})"
        )

    # Create two versions: one without labels, one with labels
    # Generate output path variants
    output_base = args.output
    if output_base.endswith(".png"):
        output_base_no_ext = output_base[:-4]
    else:
        output_base_no_ext = output_base

    output_no_labels = f"{output_base_no_ext}.png"
    output_with_labels = f"{output_base_no_ext}_labeled.png"

    print(f"\nGenerating plots...")
    plot_dice_vs_hausdorff(
        df,
        output_no_labels,
        figsize=tuple(args.figsize),
        marker_size=args.marker_size,
        show_labels=False,
    )

    plot_dice_vs_hausdorff(
        df,
        output_with_labels,
        figsize=tuple(args.figsize),
        marker_size=args.marker_size,
        show_labels=True,
    )


if __name__ == "__main__":
    main()
