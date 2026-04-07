"""
Comparison of ROC client groups against the clean baseline.

For each dataset / algorithm pair, this script computes:

    delta(clean, roc-clean-clients) = Dice(clean baseline) - Dice(roc clean clients)
    delta(clean, roc-noisy-clients) = Dice(clean baseline) - Dice(roc noisy clients)

Dice values are taken from bootstrap_evaluation_results.json across checkpoints,
averaged per client group, then per fold, then per (dataset, algorithm, group).
Folds [0, 1, 2] are included.

Outputs
-------
- results/segmentation_results/roc_clean_vs_noisy_clients_comparison/
    roc_clean_vs_noisy_clients_comparison.csv
  Long-form table: one row per (dataset, algorithm).
- results/segmentation_results/roc_clean_vs_noisy_clients_comparison/
    roc_clean_vs_noisy_clients_comparison.png
  Scatter plot with delta(clean, roc-clean clients) on x-axis and
  delta(clean, roc-noisy clients) on y-axis.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
sheet_id = "1AP_KH1cVSDwgpI1n7qK_VZU0Vi19Wh8vKo4jYWkuIXg"
gid = "332656109"
CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
)

algo_col = "Algo"
raw_dataset_col = "Data"
noise_col = "Noise"
fold_col = "Fold"

TARGET_ALGOS = ["FedAvg", "FedA3I", "IOP-FL", "FedCorr", "FedSelect"]
TARGET_DATASETS = ["LIDC", "RIGA", "Gleason", "MouseTumor", "MMIA", "MMIS"]
INCLUDED_FOLDS = [0, 1, 2]
DEFAULT_NNUNET_RESULTS_ROOTS = [
    Path("/home/m391k/cluster-data/checkpoints/nnUNet_results"),
    Path("/home/m391k/juwels/checkpoints/nnUNet_results"),
]

OUTPUT_DIR = Path(
    "./results/segmentation_results/roc_clean_vs_noisy_clients_comparison"
)

CLEAN_CLIENTS_PER_DATASET: Dict[str, List[int]] = {
    "LIDC": [0, 1],
    "RIGA": [0],
    "Gleason": [0],
    "MouseTumor": [0, 1],
    "MMIA": [0, 1],
    "MMIS": [0, 1],
}

ALGO_COLORS = {
    "FedAvg": "#4C72B0",
    "FedA3I": "#DD8452",
    "IOP-FL": "#55A868",
    "FedCorr": "#C44E52",
    "FedSelect": "#8172B2",
}

classwise_metric = "Dice"


def load_bootstrap_metric_vector(
    bootstrap_file: Path, metric_name: str = classwise_metric
) -> Optional[np.ndarray]:
    """Load bootstrap metric vector and average over classes element-wise."""
    if not bootstrap_file.is_file():
        return None

    try:
        with open(bootstrap_file, "r") as f:
            res = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    class_vectors: List[np.ndarray] = []
    for class_label, metrics in res.items():
        if class_label == "stats" or not isinstance(metrics, dict):
            continue

        metric_vals = metrics.get(metric_name)
        if metric_vals is None:
            continue

        if isinstance(metric_vals, list):
            try:
                arr = np.asarray(metric_vals, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
        else:
            try:
                arr = np.asarray([float(metric_vals)], dtype=float)
            except (TypeError, ValueError):
                continue

        if len(arr) == 0:
            continue
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            class_vectors.append(arr)

    if not class_vectors:
        return None

    max_len = max(len(v) for v in class_vectors)
    padded = [
        np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in class_vectors
    ]
    return np.nanmean(padded, axis=0)


def build_experiment_path_index(nnunet_results_roots: Sequence[Path]) -> List[Path]:
    all_exp_paths_raw: List[str] = []
    for nnunet_results_root in nnunet_results_roots:
        all_exp_paths_raw.extend(
            glob.glob(str(nnunet_results_root / "*" / "*" / "fold_*" / "*"))
        )

    all_exp_paths = sorted({Path(p) for p in all_exp_paths_raw}, key=str)
    return [p for p in all_exp_paths if extract_fold_from_path(p) in INCLUDED_FOLDS]


def resolve_nnunet_results_roots(cli_root: Optional[Path]) -> List[Path]:
    if cli_root is not None:
        return [Path(cli_root)]

    env_root = os.environ.get("nnUNet_results")
    if env_root:
        return [Path(env_root)]

    return DEFAULT_NNUNET_RESULTS_ROOTS.copy()


def extract_fold_from_path(path: Path) -> Optional[int]:
    m = re.search(r"/fold_(\d+)/", str(path))
    return int(m.group(1)) if m else None


def extract_client_id_from_path(path: Path) -> Optional[int]:
    m = re.search(r"fedclient(\d+)", str(path))
    if m:
        return int(m.group(1))
    m = re.search(r"client(\d+)", str(path))
    if m:
        return int(m.group(1))
    return None


def normalize_algorithm(name: str) -> str:
    n = re.sub(r"[\s_\-]+", "", str(name).strip().lower())
    mapping = {
        "fedavg": "FedAvg",
        "feda3i": "FedA3I",
        "iopfl": "IOP-FL",
        "fedcorr": "FedCorr",
        "fedselect": "FedSelect",
    }
    return mapping.get(n, str(name).strip())


def normalize_dataset(name: str) -> str:
    n = str(name).lower()
    if "lidc" in n:
        return "LIDC"
    if "riga" in n:
        return "RIGA"
    if "gleason" in n:
        return "Gleason"
    if "mousetumor" in n or "mouse tumor" in n or "mouse_tumor" in n:
        return "MouseTumor"
    if "mmia" in n:
        return "MMIA"
    if "mmis" in n:
        return "MMIS"
    return str(name).strip()


def classify_noise(noise_str: str) -> Optional[str]:
    s = str(noise_str).strip()
    if re.fullmatch(r"0(?:\.0+)?", s):
        return "clean"
    if re.search(r"(?i)\broc\b", s):
        return "roc"
    return None


def load_sheet() -> pd.DataFrame:
    df = pd.read_csv(CSV_URL)
    print(f"Loaded {len(df)} rows from Google Sheets.")

    for c in [algo_col, raw_dataset_col, noise_col, fold_col]:
        if c not in df.columns:
            raise ValueError(f"Missing expected column '{c}' in sheet.")

    df[algo_col] = df[algo_col].astype(str).str.strip().apply(normalize_algorithm)
    df["dataset"] = df[raw_dataset_col].astype(str).apply(normalize_dataset)
    df["noise_scenario"] = df[noise_col].astype(str).apply(classify_noise)
    df["fold"] = pd.to_numeric(df[fold_col], errors="coerce")
    df["exp_id"] = df["Experiment ID"].astype(str).str.strip()

    df = df[df[algo_col].isin(TARGET_ALGOS)]
    df = df[df["dataset"].isin(TARGET_DATASETS)]
    df = df[df["noise_scenario"].isin(["clean", "roc"])]
    df = df[df["fold"].isin(INCLUDED_FOLDS)]
    df = df[df["exp_id"].notna()]
    df = df[df["exp_id"] != ""]

    print(
        f"Retained {len(df)} rows after filtering "
        f"(algos={TARGET_ALGOS!r}; datasets={TARGET_DATASETS!r}; "
        f"noise=[clean,roc]; folds={INCLUDED_FOLDS!r})."
    )
    return df


def average_vectors(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    max_len = max(len(v) for v in vectors)
    padded = [np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in vectors]
    return np.nanmean(padded, axis=0)


def load_bootstrap_vectors(
    df: pd.DataFrame,
    all_exp_paths: List[Path],
) -> Dict[tuple, np.ndarray]:
    """
    Returns vectors keyed by:
      (algo, dataset, "clean")
      (algo, dataset, "roc_clean_clients")
      (algo, dataset, "roc_noisy_clients")
    """
    bootstrap_cache: Dict[Path, Optional[np.ndarray]] = {}
    result: Dict[tuple, np.ndarray] = {}

    for (algo, dataset, noise_scenario), group_df in df.groupby(
        [algo_col, "dataset", "noise_scenario"]
    ):
        exp_ids = group_df["exp_id"].unique()

        if noise_scenario == "clean":
            per_fold_vectors: Dict[int, List[np.ndarray]] = {}

            for exp_id in exp_ids:
                exp_paths = [p for p in all_exp_paths if exp_id in str(p)]
                for exp_path in exp_paths:
                    fold = extract_fold_from_path(exp_path)
                    if fold is None or fold not in INCLUDED_FOLDS:
                        continue

                    bootstrap_file = (
                        exp_path / "validation" / "bootstrap_evaluation_results.json"
                    )
                    if bootstrap_file not in bootstrap_cache:
                        bootstrap_cache[bootstrap_file] = load_bootstrap_metric_vector(
                            bootstrap_file
                        )
                    vec = bootstrap_cache[bootstrap_file]
                    if vec is None:
                        continue

                    per_fold_vectors.setdefault(fold, []).append(vec)

            fold_means = []
            for fold in sorted(per_fold_vectors):
                fold_mean = average_vectors(per_fold_vectors[fold])
                if fold_mean is not None:
                    fold_means.append(fold_mean)

            cell_vec = average_vectors(fold_means)
            if cell_vec is not None:
                result[(algo, dataset, "clean")] = cell_vec
            else:
                print(f"Warning: no bootstrap vectors found for {algo}/{dataset}/clean")

        elif noise_scenario == "roc":
            per_group_per_fold: Dict[str, Dict[int, List[np.ndarray]]] = {
                "roc_clean_clients": {},
                "roc_noisy_clients": {},
            }
            clean_client_ids = set(CLEAN_CLIENTS_PER_DATASET.get(dataset, []))

            for exp_id in exp_ids:
                exp_paths = [p for p in all_exp_paths if exp_id in str(p)]
                for exp_path in exp_paths:
                    fold = extract_fold_from_path(exp_path)
                    if fold is None or fold not in INCLUDED_FOLDS:
                        continue

                    client_id = extract_client_id_from_path(exp_path)
                    if client_id is None:
                        print(
                            f"Warning: could not infer client id from path: {exp_path}"
                        )
                        continue

                    group_key = (
                        "roc_clean_clients"
                        if client_id in clean_client_ids
                        else "roc_noisy_clients"
                    )

                    bootstrap_file = (
                        exp_path / "validation" / "bootstrap_evaluation_results.json"
                    )
                    if bootstrap_file not in bootstrap_cache:
                        bootstrap_cache[bootstrap_file] = load_bootstrap_metric_vector(
                            bootstrap_file
                        )
                    vec = bootstrap_cache[bootstrap_file]
                    if vec is None:
                        continue

                    per_group_per_fold.setdefault(group_key, {}).setdefault(
                        fold, []
                    ).append(vec)

            for group_key, per_fold_vectors in per_group_per_fold.items():
                fold_means = []
                for fold in sorted(per_fold_vectors):
                    fold_mean = average_vectors(per_fold_vectors[fold])
                    if fold_mean is not None:
                        fold_means.append(fold_mean)

                cell_vec = average_vectors(fold_means)
                if cell_vec is not None:
                    result[(algo, dataset, group_key)] = cell_vec
                else:
                    print(
                        f"Warning: no bootstrap vectors found for "
                        f"{algo}/{dataset}/{group_key}"
                    )

    return result


def aggregate_bootstrap_vectors_to_mean_dice(
    bootstrap_vectors: Dict[tuple, np.ndarray],
) -> Dict[tuple, float]:
    mean_dice: Dict[tuple, float] = {}
    for key, vec in bootstrap_vectors.items():
        if vec is None or len(vec) == 0:
            continue
        v = float(np.nanmean(vec))
        if np.isfinite(v):
            mean_dice[key] = v
    return mean_dice


def build_comparison_table(bootstrap_dice: Dict[tuple, float]) -> pd.DataFrame:
    rows = []
    for dataset in TARGET_DATASETS:
        for algo in TARGET_ALGOS:
            dice_clean = bootstrap_dice.get((algo, dataset, "clean"), np.nan)
            dice_roc_clean = bootstrap_dice.get(
                (algo, dataset, "roc_clean_clients"), np.nan
            )
            dice_roc_noisy = bootstrap_dice.get(
                (algo, dataset, "roc_noisy_clients"), np.nan
            )

            if np.isnan(dice_clean):
                continue

            delta_roc_clean = (
                dice_clean - dice_roc_clean if not np.isnan(dice_roc_clean) else np.nan
            )
            delta_roc_noisy = (
                dice_clean - dice_roc_noisy if not np.isnan(dice_roc_noisy) else np.nan
            )

            rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algo,
                    "dice_clean": dice_clean,
                    "dice_roc_clean_clients": dice_roc_clean,
                    "dice_roc_noisy_clients": dice_roc_noisy,
                    "delta_roc_clean_clients": delta_roc_clean,
                    "delta_roc_noisy_clients": delta_roc_noisy,
                }
            )

    result = pd.DataFrame(rows)
    ds_order = {d: i for i, d in enumerate(TARGET_DATASETS)}
    al_order = {a: i for i, a in enumerate(TARGET_ALGOS)}
    result["_ds"] = result["dataset"].map(ds_order)
    result["_al"] = result["algorithm"].map(al_order)
    result = result.sort_values(["_ds", "_al"]).drop(columns=["_ds", "_al"])
    return result.reset_index(drop=True)


def print_table(result: pd.DataFrame) -> None:
    float_cols = [
        "dice_clean",
        "dice_roc_clean_clients",
        "dice_roc_noisy_clients",
        "delta_roc_clean_clients",
        "delta_roc_noisy_clients",
    ]
    display = result.copy()
    for c in float_cols:
        if c in display.columns:
            display[c] = display[c].map(
                lambda x: (
                    f"{x:.4f}" if not (isinstance(x, float) and np.isnan(x)) else "—"
                )
            )

    print()
    print("=" * 120)
    print("roc client-group comparison: clean baseline vs roc clean/noisy clients")
    print("=" * 120)
    print(display.to_string(index=False))
    print()

    print("=" * 120)
    print("Mean over algorithms per dataset")
    print("=" * 120)
    numeric_cols = float_cols
    summary = result.groupby("dataset")[numeric_cols].mean().reset_index()
    ds_order = {d: i for i, d in enumerate(TARGET_DATASETS)}
    summary["_ds"] = summary["dataset"].map(ds_order)
    summary = summary.sort_values("_ds").drop(columns=["_ds"]).reset_index(drop=True)
    for c in numeric_cols:
        summary[c] = summary[c].map(
            lambda x: f"{x:.4f}" if not (isinstance(x, float) and np.isnan(x)) else "—"
        )
    print(summary.to_string(index=False))
    print()


def plot_comparison_scatter(
    result: pd.DataFrame,
    output_path: Path,
    bootstrap_vectors: Dict[tuple, np.ndarray],
    dpi: int = 200,
) -> None:
    from matplotlib.lines import Line2D

    algo_colors = {
        algo: ALGO_COLORS.get(algo, plt.cm.tab10(i % 10))
        for i, algo in enumerate(TARGET_ALGOS)
    }
    dataset_markers = {
        ds: m for ds, m in zip(TARGET_DATASETS, ["o", "s", "^", "D", "P", "X"])
    }

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    bg_x: List[float] = []
    bg_y: List[float] = []
    mean_points: List[tuple] = []
    method_points: List[tuple] = []

    for dataset in TARGET_DATASETS:
        for algo in TARGET_ALGOS:
            clean_vec = bootstrap_vectors.get((algo, dataset, "clean"))
            roc_clean_vec = bootstrap_vectors.get((algo, dataset, "roc_clean_clients"))
            roc_noisy_vec = bootstrap_vectors.get((algo, dataset, "roc_noisy_clients"))

            if clean_vec is None or roc_clean_vec is None or roc_noisy_vec is None:
                continue
            if (
                len(clean_vec) == 0
                or len(roc_clean_vec) == 0
                or len(roc_noisy_vec) == 0
            ):
                continue

            n = min(len(clean_vec), len(roc_clean_vec), len(roc_noisy_vec))
            if n == 0:
                continue

            clean_arr = np.asarray(clean_vec[:n], dtype=float)
            roc_clean_arr = np.asarray(roc_clean_vec[:n], dtype=float)
            roc_noisy_arr = np.asarray(roc_noisy_vec[:n], dtype=float)

            delta_clean_vec = clean_arr - roc_clean_arr
            delta_noisy_vec = clean_arr - roc_noisy_arr
            valid = np.isfinite(delta_clean_vec) & np.isfinite(delta_noisy_vec)
            if not np.any(valid):
                continue

            x_vals = delta_clean_vec[valid]
            y_vals = delta_noisy_vec[valid]
            bg_x.extend(x_vals.tolist())
            bg_y.extend(y_vals.tolist())

            row = result[(result["dataset"] == dataset) & (result["algorithm"] == algo)]
            if not row.empty:
                x_mean = float(row.iloc[0]["delta_roc_clean_clients"])
                y_mean = float(row.iloc[0]["delta_roc_noisy_clients"])
            else:
                x_mean = float(np.nanmean(x_vals))
                y_mean = float(np.nanmean(y_vals))

            if np.isfinite(x_mean) and np.isfinite(y_mean):
                mean_points.append((x_mean, y_mean, algo, dataset))

    if not mean_points:
        print("No valid points to plot.")
        return

    for algo in TARGET_ALGOS:
        algo_xy = [(x, y) for x, y, a, _ in mean_points if a == algo]
        if not algo_xy:
            continue
        method_points.append(
            (
                float(np.nanmean([p[0] for p in algo_xy])),
                float(np.nanmean([p[1] for p in algo_xy])),
                algo,
            )
        )

    if bg_x and bg_y:
        ax.scatter(
            bg_x,
            bg_y,
            s=10,
            c="lightgray",
            alpha=0.14,
            edgecolors="none",
            zorder=1,
        )

    for x, y, algo, dataset in mean_points:
        ax.scatter(
            x,
            y,
            s=90,
            marker=dataset_markers[dataset],
            c=algo_colors[algo],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            zorder=3,
        )

    for x, y, algo in method_points:
        ax.scatter(
            x,
            y,
            s=300,
            marker="*",
            c=algo_colors[algo],
            edgecolors="black",
            linewidths=1.4,
            alpha=1.0,
            zorder=4,
        )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.35)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.35)

    all_x = list(bg_x) + [p[0] for p in mean_points] + [p[0] for p in method_points]
    all_y = list(bg_y) + [p[1] for p in mean_points] + [p[1] for p in method_points]
    if all_x and all_y:
        lim_min = float(min(min(all_x), min(all_y)))
        lim_max = float(max(max(all_x), max(all_y)))
        span = lim_max - lim_min
        pad = 0.02 * span if span > 0 else 0.01
        lim_low = lim_min - pad
        lim_high = lim_max + pad
        ax.plot(
            [lim_low, lim_high],
            [lim_low, lim_high],
            color="dimgray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_aspect("equal", adjustable="box")

        x_clean = lim_high - 0.11 * span
        y_clean = lim_high - 0.08 * span
        x_noisy = lim_high - 0.18 * span
        y_noisy = lim_high - 0.02 * span

        ax.text(
            x_noisy,
            y_noisy,
            "roc noisy clients more harmful",
            fontsize=9,
            ha="center",
            va="top",
            rotation=45,
            zorder=5,
        )
        ax.text(
            x_clean,
            y_clean,
            "roc clean clients more harmful",
            fontsize=9,
            ha="center",
            va="top",
            rotation=45,
            zorder=5,
        )

    ax.set_xlabel(
        r"$\Delta Dice(\mathrm{clean},\mathrm{roc\ clean\ clients})$",
        fontsize=11,
    )
    ax.set_ylabel(
        r"$\Delta Dice(\mathrm{clean},\mathrm{roc\ noisy\ clients})$",
        fontsize=11,
    )
    ax.set_title(
        "ROC client-group robustness: clean-clients vs noisy-clients",
        fontsize=13,
        pad=14,
    )

    algo_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=algo_colors[a],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=8,
            label=a,
        )
        for a in TARGET_ALGOS
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=dataset_markers[d],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=d,
        )
        for d in TARGET_DATASETS
    ]
    bootstrap_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="lightgray",
        markeredgecolor="lightgray",
        markersize=6,
        label="Bootstrap deltas",
    )
    method_mean_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=12,
        linewidth=0,
        label="Method mean across datasets",
    )
    diagonal_handle = Line2D(
        [0],
        [0],
        color="dimgray",
        linestyle="--",
        linewidth=1.2,
        label="x = y (equal degradation)",
    )

    legend1 = ax.legend(
        handles=[bootstrap_handle, method_mean_handle, diagonal_handle] + algo_handles,
        title="Color = FNLL method",
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=dataset_handles,
        title="Marker = dataset",
        loc="lower right",
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path.resolve()}")


def plot_comparison_paired_dot(
    result: pd.DataFrame,
    output_path: Path,
    dpi: int = 200,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors

    def blend_with(color: str, blend_to: str, alpha: float) -> tuple:
        c1 = np.asarray(mcolors.to_rgb(color), dtype=float)
        c2 = np.asarray(mcolors.to_rgb(blend_to), dtype=float)
        return tuple((1.0 - alpha) * c1 + alpha * c2)

    algo_colors = {
        algo: ALGO_COLORS.get(algo, plt.cm.tab10(i % 10))
        for i, algo in enumerate(TARGET_ALGOS)
    }
    dataset_markers = {
        ds: m for ds, m in zip(TARGET_DATASETS, ["o", "s", "^", "D", "P", "X"])
    }

    roc_clean_points = []
    roc_noisy_points = []
    for _, row in result.iterrows():
        if np.isfinite(row["delta_roc_clean_clients"]):
            roc_clean_points.append(
                (
                    row["algorithm"],
                    row["dataset"],
                    float(row["delta_roc_clean_clients"]),
                )
            )
        if np.isfinite(row["delta_roc_noisy_clients"]):
            roc_noisy_points.append(
                (
                    row["algorithm"],
                    row["dataset"],
                    float(row["delta_roc_noisy_clients"]),
                )
            )

    if not roc_clean_points and not roc_noisy_points:
        print("No valid points to plot.")
        return

    all_y = [p[2] for p in roc_clean_points] + [p[2] for p in roc_noisy_points]
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    y_span = y_max - y_min
    y_pad = 0.08 * y_span if y_span > 0 else 0.05
    y_low = y_min - y_pad
    y_high = y_max + y_pad

    fig, ax = plt.subplots(1, 1, figsize=(11.8, 5.6), dpi=dpi)
    x_offset_clean = -0.18
    x_offset_noisy = +0.18

    for x_sep in np.arange(1.5, len(TARGET_ALGOS) + 0.5, 1.0):
        ax.axvline(x_sep, color="0.90", lw=1.1, zorder=0)

    for i, algo in enumerate(TARGET_ALGOS, start=1):
        clean_color = blend_with(algo_colors[algo], "white", 0.18)
        noisy_color = blend_with(algo_colors[algo], "black", 0.12)

        clean_vals = []
        for _, dataset, y in [p for p in roc_clean_points if p[0] == algo]:
            clean_vals.append(y)
            ax.scatter(
                i + x_offset_clean,
                y,
                s=95,
                marker=dataset_markers[dataset],
                c=[clean_color],
                edgecolors="black",
                linewidths=0.8,
                alpha=0.96,
                zorder=3,
            )
        if clean_vals:
            ax.scatter(
                i + x_offset_clean,
                float(np.nanmean(clean_vals)),
                s=240,
                marker="_",
                c="black",
                linewidths=3.0,
                zorder=4,
            )

        noisy_vals = []
        for _, dataset, y in [p for p in roc_noisy_points if p[0] == algo]:
            noisy_vals.append(y)
            ax.scatter(
                i + x_offset_noisy,
                y,
                s=95,
                marker=dataset_markers[dataset],
                c=[noisy_color],
                edgecolors="black",
                linewidths=0.8,
                alpha=0.96,
                zorder=3,
            )
        if noisy_vals:
            ax.scatter(
                i + x_offset_noisy,
                float(np.nanmean(noisy_vals)),
                s=240,
                marker="_",
                c="black",
                linewidths=3.0,
                zorder=4,
            )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.35)
    ax.set_xlim(0.5, len(TARGET_ALGOS) + 0.5)
    ax.set_ylim(y_low, y_high)

    x_ticks = []
    x_labels = []
    for i in range(1, len(TARGET_ALGOS) + 1):
        x_ticks.extend([i + x_offset_clean, i + x_offset_noisy])
        x_labels.extend(["roc-clean", "roc-noisy"])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=0, ha="center")

    for i, algo in enumerate(TARGET_ALGOS, start=1):
        ax.text(
            i,
            -0.09,
            algo,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
        )

    ax.set_xlabel("roc client group per method", fontsize=11, labelpad=22)
    ax.set_ylabel(r"$\Delta Dice$(clean, roc client group)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=dataset_markers[d],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=d,
        )
        for d in TARGET_DATASETS
    ]
    mean_handle = Line2D(
        [0],
        [0],
        marker="_",
        color="black",
        markersize=16,
        markeredgewidth=3,
        linewidth=0,
        label="Mean across datasets",
    )

    fig.legend(
        handles=dataset_handles + [mean_handle],
        loc="upper right",
        bbox_to_anchor=(1.12, 0.92),
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0.03, 0.10, 0.97, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ROC clean-client and noisy-client groups against the clean "
            "baseline per dataset and algorithm."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="paired_dot",
        help="Generate figure: scatter_plot, paired_dot, no_figure.",
    )
    parser.add_argument(
        "--nnunet-results-root",
        type=Path,
        default=None,
        help=(
            "Root directory of nnUNet results. If not set, uses $nnUNet_results "
            "environment variable or searches these defaults: "
            f"{', '.join(str(p) for p in DEFAULT_NNUNET_RESULTS_ROOTS)}."
        ),
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    nnunet_results_roots = resolve_nnunet_results_roots(args.nnunet_results_root)
    print(
        "Using nnUNet results roots: "
        + ", ".join(str(root) for root in nnunet_results_roots)
    )

    df = load_sheet()

    print("Building checkpoint index...")
    all_exp_paths = build_experiment_path_index(nnunet_results_roots)
    print(f"Found {len(all_exp_paths)} checkpoint paths in folds {INCLUDED_FOLDS}.")

    print("Loading bootstrap Dice vectors...")
    bootstrap_vectors = load_bootstrap_vectors(df, all_exp_paths)
    print(f"Loaded bootstrap vectors for {len(bootstrap_vectors)} cells.")

    bootstrap_dice = aggregate_bootstrap_vectors_to_mean_dice(bootstrap_vectors)
    result = build_comparison_table(bootstrap_dice)
    print_table(result)

    csv_path = out_dir / "roc_clean_vs_noisy_clients_comparison.csv"
    result.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved CSV: {csv_path.resolve()}")

    if args.figure == "scatter_plot":
        fig_path = out_dir / "roc_clean_vs_noisy_clients_comparison.png"
        plot_comparison_scatter(result, fig_path, bootstrap_vectors=bootstrap_vectors)
    elif args.figure == "paired_dot":
        fig_path = out_dir / "roc_clean_vs_noisy_clients_comparison_paired_dot.png"
        plot_comparison_paired_dot(result, fig_path)
    elif args.figure == "no_figure":
        print("No figure generated (as per --figure=no_figure).")
    else:
        print(f"Unknown figure option: {args.figure!r}. No figure generated.")


if __name__ == "__main__":
    main()
