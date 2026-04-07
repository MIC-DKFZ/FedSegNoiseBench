"""
Client-level degradation analysis for the roa(X) setting.

This script keeps the clean baseline aggregated over all clean clients, folds, and
bootstrap samples, and compares it against each individual roa client:

    delta(clean, roa-client_k) = Dice(clean baseline) - Dice(roa client_k)

The goal is not to split roa into clean/noisy clients (which does not exist), but
to visualize how much degradation varies across clients under the same mixed-noise
training condition.

Outputs
-------
- results/segmentation_results/roa_client_level_variability/
    roa_client_level_variability.csv
  One row per (dataset, algorithm, client_id) with client-level deltas.
- results/segmentation_results/roa_client_level_variability/
    roa_client_level_variability.png
  Strip-style plot showing the spread of delta(clean, roa-client) per method.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

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

OUTPUT_DIR = Path("./results/segmentation_results/roa_client_level_variability")

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


def build_experiment_path_index(nnunet_results_root: Path) -> List[Path]:
    all_exp_paths_raw = glob.glob(str(nnunet_results_root / "*" / "*" / "fold_*" / "*"))
    all_exp_paths = [Path(p) for p in all_exp_paths_raw]
    return [p for p in all_exp_paths if extract_fold_from_path(p) in INCLUDED_FOLDS]


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
    if re.search(r"(?i)\broa\b", s):
        return "roa"
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
    df = df[df["noise_scenario"].isin(["clean", "roa"])]
    df = df[df["fold"].isin(INCLUDED_FOLDS)]
    df = df[df["exp_id"].notna()]
    df = df[df["exp_id"] != ""]

    print(
        f"Retained {len(df)} rows after filtering "
        f"(algos={TARGET_ALGOS!r}; datasets={TARGET_DATASETS!r}; "
        f"noise=[clean,roa]; folds={INCLUDED_FOLDS!r})."
    )
    return df


def average_vectors(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    max_len = max(len(v) for v in vectors)
    padded = [
        np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in vectors
    ]
    return np.nanmean(padded, axis=0)


def load_bootstrap_vectors(
    df: pd.DataFrame,
    all_exp_paths: List[Path],
) -> Dict[tuple, np.ndarray]:
    """
    Returns vectors keyed by:
      (algo, dataset, "clean")
      (algo, dataset, "roa_client", client_id)
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

            clean_vec = average_vectors(fold_means)
            if clean_vec is not None:
                result[(algo, dataset, "clean")] = clean_vec
            else:
                print(f"Warning: no bootstrap vectors found for {algo}/{dataset}/clean")

        elif noise_scenario == "roa":
            per_client_per_fold: Dict[int, Dict[int, List[np.ndarray]]] = {}
            for exp_id in exp_ids:
                exp_paths = [p for p in all_exp_paths if exp_id in str(p)]
                for exp_path in exp_paths:
                    fold = extract_fold_from_path(exp_path)
                    if fold is None or fold not in INCLUDED_FOLDS:
                        continue

                    client_id = extract_client_id_from_path(exp_path)
                    if client_id is None:
                        print(f"Warning: could not infer client id from path: {exp_path}")
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

                    per_client_per_fold.setdefault(client_id, {}).setdefault(
                        fold, []
                    ).append(vec)

            for client_id, per_fold_vectors in per_client_per_fold.items():
                fold_means = []
                for fold in sorted(per_fold_vectors):
                    fold_mean = average_vectors(per_fold_vectors[fold])
                    if fold_mean is not None:
                        fold_means.append(fold_mean)

                client_vec = average_vectors(fold_means)
                if client_vec is not None:
                    result[(algo, dataset, "roa_client", client_id)] = client_vec

    return result


def build_client_level_table(bootstrap_vectors: Dict[tuple, np.ndarray]) -> pd.DataFrame:
    rows = []
    for dataset in TARGET_DATASETS:
        for algo in TARGET_ALGOS:
            clean_vec = bootstrap_vectors.get((algo, dataset, "clean"))
            if clean_vec is None or len(clean_vec) == 0:
                continue

            clean_mean = float(np.nanmean(clean_vec))
            for key, roa_vec in bootstrap_vectors.items():
                if len(key) != 4:
                    continue
                key_algo, key_dataset, key_kind, client_id = key
                if (
                    key_algo != algo
                    or key_dataset != dataset
                    or key_kind != "roa_client"
                    or roa_vec is None
                    or len(roa_vec) == 0
                ):
                    continue

                n = min(len(clean_vec), len(roa_vec))
                if n == 0:
                    continue
                clean_arr = np.asarray(clean_vec[:n], dtype=float)
                roa_arr = np.asarray(roa_vec[:n], dtype=float)
                delta_vec = clean_arr - roa_arr
                delta_mean = float(np.nanmean(delta_vec))
                roa_mean = float(np.nanmean(roa_arr))

                rows.append(
                    {
                        "dataset": dataset,
                        "algorithm": algo,
                        "client_id": client_id,
                        "dice_clean": clean_mean,
                        "dice_roa_client": roa_mean,
                        "delta_roa_client": delta_mean,
                        "delta_roa_client_std": float(np.nanstd(delta_vec)),
                        "n_bootstrap": int(n),
                    }
                )

    result = pd.DataFrame(rows)
    ds_order = {d: i for i, d in enumerate(TARGET_DATASETS)}
    al_order = {a: i for i, a in enumerate(TARGET_ALGOS)}
    result["_ds"] = result["dataset"].map(ds_order)
    result["_al"] = result["algorithm"].map(al_order)
    result = result.sort_values(["_ds", "_al", "client_id"]).drop(
        columns=["_ds", "_al"]
    )
    return result.reset_index(drop=True)


def print_summary(result: pd.DataFrame) -> None:
    if result.empty:
        print("No client-level roa data available.")
        return

    print()
    print("=" * 120)
    print("ROA client-level degradation relative to the clean baseline")
    print("=" * 120)
    display = result.copy()
    for c in ["dice_clean", "dice_roa_client", "delta_roa_client", "delta_roa_client_std"]:
        display[c] = display[c].map(
            lambda x: f"{x:.4f}" if not (isinstance(x, float) and np.isnan(x)) else "—"
        )
    print(display.to_string(index=False))
    print()

    print("=" * 120)
    print("Mean and spread over clients per dataset / method")
    print("=" * 120)
    summary = (
        result.groupby(["dataset", "algorithm"])["delta_roa_client"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    for c in ["mean", "std", "min", "max"]:
        summary[c] = summary[c].map(
            lambda x: f"{x:.4f}" if not (isinstance(x, float) and np.isnan(x)) else "—"
        )
    print(summary.to_string(index=False))
    print()


def plot_client_spread(result: pd.DataFrame, output_path: Path, dpi: int = 200) -> None:
    from matplotlib.lines import Line2D

    if result.empty:
        print("No valid points to plot.")
        return

    dataset_markers = {
        ds: m for ds, m in zip(TARGET_DATASETS, ["o", "s", "^", "D", "P", "X"])
    }
    dataset_offsets = {
        ds: off for ds, off in zip(TARGET_DATASETS, np.linspace(-0.24, 0.24, len(TARGET_DATASETS)))
    }

    fig, ax = plt.subplots(figsize=(11.6, 5.8), dpi=dpi)

    all_y = result["delta_roa_client"].to_numpy(dtype=float)
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    y_span = y_max - y_min
    y_pad = 0.10 * y_span if y_span > 0 else 0.03
    y_low = y_min - y_pad
    y_high = y_max + y_pad

    for x_sep in np.arange(1.5, len(TARGET_ALGOS) + 0.5, 1.0):
        ax.axvline(x_sep, color="0.90", lw=1.1, zorder=0)

    for i, algo in enumerate(TARGET_ALGOS, start=1):
        algo_rows = result[result["algorithm"] == algo]
        if algo_rows.empty:
            continue

        method_vals = []
        for dataset in TARGET_DATASETS:
            ds_rows = algo_rows[algo_rows["dataset"] == dataset].sort_values("client_id")
            if ds_rows.empty:
                continue

            base_x = i + dataset_offsets[dataset]
            client_offsets = np.linspace(-0.035, 0.035, max(len(ds_rows), 1))
            for (_, row), client_off in zip(ds_rows.iterrows(), client_offsets):
                x = base_x + client_off
                y = float(row["delta_roa_client"])
                method_vals.append(y)
                ax.scatter(
                    x,
                    y,
                    s=80,
                    marker=dataset_markers[dataset],
                    c=[ALGO_COLORS[algo]],
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=0.88,
                    zorder=3,
                )

            ds_mean = float(ds_rows["delta_roa_client"].mean())
            ax.scatter(
                base_x,
                ds_mean,
                s=180,
                marker="_",
                c="black",
                linewidths=2.6,
                zorder=4,
            )

        if method_vals:
            ax.scatter(
                i,
                float(np.nanmean(method_vals)),
                s=360,
                marker="*",
                c=[ALGO_COLORS[algo]],
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.35)
    ax.set_xlim(0.5, len(TARGET_ALGOS) + 0.5)
    ax.set_ylim(y_low, y_high)
    ax.set_xticks(range(1, len(TARGET_ALGOS) + 1))
    ax.set_xticklabels(TARGET_ALGOS)
    ax.set_xlabel("FNLL method", fontsize=11)
    ax.set_ylabel(r"$\Delta Dice(\mathrm{clean},\mathrm{roa\ client})$", fontsize=11)
    ax.set_title(
        "ROA client-level degradation spread on clean validation",
        fontsize=13,
        pad=12,
    )
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
    dataset_mean_handle = Line2D(
        [0],
        [0],
        marker="_",
        color="black",
        markersize=14,
        markeredgewidth=2.6,
        linewidth=0,
        label="Dataset mean across clients",
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
        label="Method mean across all dataset-clients",
    )

    legend1 = ax.legend(
        handles=[dataset_mean_handle, method_mean_handle],
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=dataset_handles,
        title="Marker = dataset",
        loc="upper right",
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize client-level degradation spread for roa(X) relative to the "
            "clean baseline."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--nnunet-results-root",
        type=Path,
        default=None,
        help=(
            "Root directory of nnUNet results. If not set, uses $nnUNet_results "
            "environment variable or /home/m391k/cluster-data/checkpoints/nnUNet_results."
        ),
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.nnunet_results_root:
        nnunet_results_root = args.nnunet_results_root
    else:
        env_root = os.environ.get("nnUNet_results")
        nnunet_results_root = (
            Path(env_root)
            if env_root
            else Path("/home/m391k/cluster-data/checkpoints/nnUNet_results")
        )

    print(f"Using nnUNet results root: {nnunet_results_root}")

    df = load_sheet()
    print("Building checkpoint index...")
    all_exp_paths = build_experiment_path_index(nnunet_results_root)
    print(f"Found {len(all_exp_paths)} checkpoint paths in folds {INCLUDED_FOLDS}.")

    print("Loading bootstrap Dice vectors...")
    bootstrap_vectors = load_bootstrap_vectors(df, all_exp_paths)
    print(f"Loaded bootstrap vectors for {len(bootstrap_vectors)} cells.")

    result = build_client_level_table(bootstrap_vectors)
    print_summary(result)

    csv_path = out_dir / "roa_client_level_variability.csv"
    result.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved CSV: {csv_path.resolve()}")

    fig_path = out_dir / "roa_client_level_variability.png"
    plot_client_spread(result, fig_path)


if __name__ == "__main__":
    main()
