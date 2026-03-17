import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    from .ranking import (
        DEFAULT_NNUNET_RESULTS_ROOT,
        OUTPUT_DIR,
        algo_col,
        build_experiment_path_index,
        classwise_metric,
        exp_id_col,
        extract_client_id_from_path,
        extract_fold_from_path,
        included_folds,
        load_and_preprocess_results,
        target_algos,
        target_datasets,
    )
except ImportError:
    from ranking import (
        DEFAULT_NNUNET_RESULTS_ROOT,
        OUTPUT_DIR,
        algo_col,
        build_experiment_path_index,
        classwise_metric,
        exp_id_col,
        extract_client_id_from_path,
        extract_fold_from_path,
        included_folds,
        load_and_preprocess_results,
        target_algos,
        target_datasets,
    )


NOISE_ORDER = ["clean", "roa", "roc", "noisy"]
NOISE_LABELS = {
    "clean": "clean",
    "roa": "roa(X)",
    "roc": "roc(X)",
    "noisy": "noisy",
    "all": "overall",
}
ALGO_COLORS = {
    "FedAvg": "#4C72B0",
    "FedA3I": "#DD8452",
    "IOP-FL": "#55A868",
    "FedCorr": "#C44E52",
    "FedSelect": "#8172B2",
}
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "ranking_stability"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "ranking_stability_summary.csv"
DEFAULT_BLOB_SCALE = 2200.0
DEFAULT_DPI = 220


def load_bootstrap_metric_vector(
    bootstrap_file: Path, metric_name: str = classwise_metric
) -> Optional[np.ndarray]:
    """
    Load the bootstrap metric vector for one experiment/client and average it
    over classes element-wise.
    """
    if not bootstrap_file.is_file():
        return None

    with open(bootstrap_file, "r") as f:
        res = json.load(f)

    class_vectors: List[np.ndarray] = []
    for class_label, metrics in res.items():
        if class_label == "stats" or not isinstance(metrics, dict):
            continue

        metric_vals = metrics.get(metric_name, None)
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

        if arr.size == 0:
            continue

        arr = arr.astype(float, copy=False)
        arr[~np.isfinite(arr)] = np.nan
        class_vectors.append(arr)

    if not class_vectors:
        return None

    min_len = min(len(v) for v in class_vectors)
    if min_len == 0:
        return None

    stacked = np.vstack([v[:min_len] for v in class_vectors])
    with np.errstate(invalid="ignore"):
        mean_vec = np.nanmean(stacked, axis=0)

    if np.all(np.isnan(mean_vec)):
        return None

    return mean_vec.astype(float, copy=False)


def average_vectors(vectors: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    valid_vectors: List[np.ndarray] = []
    for vec in vectors:
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        valid_vectors.append(arr)

    if not valid_vectors:
        return None

    min_len = min(len(v) for v in valid_vectors)
    if min_len == 0:
        return None

    stacked = np.vstack([v[:min_len] for v in valid_vectors])
    with np.errstate(invalid="ignore"):
        mean_vec = np.nanmean(stacked, axis=0)

    if np.all(np.isnan(mean_vec)):
        return None

    return mean_vec.astype(float, copy=False)


def collect_client_bootstrap_vectors(
    df: pd.DataFrame, all_exp_paths: List[Path]
) -> pd.DataFrame:
    rows = []
    bootstrap_cache: Dict[Path, Optional[np.ndarray]] = {}

    print(f"Collecting bootstrap vectors for {len(df)} experiment records...")

    records = df[[algo_col, "Dataset_norm", "noise_scenario", exp_id_col]].itertuples(
        index=False, name=None
    )

    for algo, dataset, noise_scenario, exp_id in records:
        exp_paths = [p for p in all_exp_paths if exp_id in str(p)]
        if not exp_paths:
            print(
                f"No checkpoint paths found for Exp_ID {exp_id} ({dataset}/{algo}/{noise_scenario})."
            )
            continue

        exp_paths_sorted = sorted(
            exp_paths,
            key=lambda p: (
                (
                    extract_client_id_from_path(p)
                    if extract_client_id_from_path(p) is not None
                    else -1
                ),
                str(p),
            ),
        )

        for exp_path in exp_paths_sorted:
            fold = extract_fold_from_path(exp_path)
            client_id = extract_client_id_from_path(exp_path)
            if fold is None:
                print(f"Skipping path with missing fold info: {exp_path}")
                continue

            bootstrap_file = exp_path / "validation" / "bootstrap_evaluation_results.json"
            if bootstrap_file not in bootstrap_cache:
                bootstrap_cache[bootstrap_file] = load_bootstrap_metric_vector(
                    bootstrap_file, classwise_metric
                )

            bootstrap_vector = bootstrap_cache[bootstrap_file]
            if bootstrap_vector is None:
                print(f"Missing/invalid bootstrap metrics: {bootstrap_file}")
                continue

            rows.append(
                {
                    "algorithm": algo,
                    "dataset": dataset,
                    "noise_scenario": noise_scenario,
                    "experiment_id": exp_id,
                    "fold": fold,
                    "client_id": client_id,
                    "bootstrap_vector": bootstrap_vector,
                }
            )

    out = pd.DataFrame(rows)
    print(f"Collected {len(out)} client bootstrap-vector rows.")
    return out


def aggregate_bootstrap_vectors(
    df: pd.DataFrame, group_cols: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        mean_vec = average_vectors(group["bootstrap_vector"].tolist())
        if mean_vec is None:
            continue

        row = {col: value for col, value in zip(group_cols, key)}
        row["bootstrap_vector"] = mean_vec
        row["n_members"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


def build_cell_bootstrap_vectors(client_vectors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors ranking.py aggregation, but keeps the full bootstrap vector:
      1. average over clients per experiment/fold
      2. average over folds/experiments per (algo, dataset, noise_scenario)
    """
    if client_vectors_df.empty:
        return pd.DataFrame(
            columns=["algorithm", "dataset", "noise_scenario", "bootstrap_vector"]
        )

    per_experiment = aggregate_bootstrap_vectors(
        client_vectors_df,
        ["algorithm", "dataset", "noise_scenario", "experiment_id", "fold"],
    )
    per_cell = aggregate_bootstrap_vectors(
        per_experiment,
        ["algorithm", "dataset", "noise_scenario"],
    )
    return per_cell


def filter_records_to_included_folds(df: pd.DataFrame) -> pd.DataFrame:
    exp_folds = pd.to_numeric(
        df[exp_id_col].astype(str).str.extract(r"fold(\d+)")[0],
        errors="coerce",
    )
    keep_mask = exp_folds.isin(included_folds)
    filtered_df = df.loc[keep_mask].copy()
    print(
        f"Retained {len(filtered_df)} rows after restricting to folds {included_folds}."
    )
    return filtered_df


def parse_selected_datasets(datasets_arg: Optional[Sequence[str]]) -> List[str]:
    if not datasets_arg:
        return list(target_datasets)

    def _normalize_name(name: str) -> str:
        return re.sub(r"[\s_\-]+", "", str(name).strip().lower())

    canonical_map = {_normalize_name(d): d for d in target_datasets}
    selected: List[str] = []
    unknown: List[str] = []

    for item in datasets_arg:
        parts = [p for p in re.split(r"[,\s]+", str(item).strip()) if p]
        for part in parts:
            key = _normalize_name(part)
            canonical = canonical_map.get(key)
            if canonical is None:
                unknown.append(part)
                continue
            if canonical not in selected:
                selected.append(canonical)

    if unknown:
        raise ValueError(
            "Unknown dataset names in --datasets: "
            f"{unknown}. Allowed values: {target_datasets}"
        )

    if not selected:
        raise ValueError(
            "No valid datasets selected via --datasets. "
            f"Allowed values: {target_datasets}"
        )

    return selected


def filter_records_to_selected_datasets(
    df: pd.DataFrame, selected_datasets: Sequence[str]
) -> pd.DataFrame:
    filtered_df = df[df["Dataset_norm"].isin(selected_datasets)].copy()
    print(
        f"Retained {len(filtered_df)} rows after restricting to datasets: {list(selected_datasets)}"
    )
    return filtered_df


def compute_rank_matrices(
    vectors_by_algorithm: Dict[str, np.ndarray],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    valid: Dict[str, np.ndarray] = {}
    for algo, vec in vectors_by_algorithm.items():
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        valid[algo] = arr

    if not valid:
        return None, None

    min_len = min(len(v) for v in valid.values())
    if min_len == 0:
        return None, None

    score_df = pd.DataFrame({algo: vec[:min_len] for algo, vec in valid.items()})
    score_df = score_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if score_df.empty:
        return None, None

    rank_df = score_df.rank(axis=1, method="min", ascending=False).astype(int)
    return score_df, rank_df


def summarize_rank_group(
    scope: str,
    dataset: str,
    noise_scenario: str,
    group_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    vectors_by_algorithm = {
        row.algorithm: row.bootstrap_vector for row in group_df.itertuples(index=False)
    }
    score_df, rank_df = compute_rank_matrices(vectors_by_algorithm)
    if score_df is None or rank_df is None:
        return []

    n_bootstrap = len(rank_df)
    n_algorithms_present = len(rank_df.columns)
    records: List[Dict[str, object]] = []

    for algo in target_algos:
        available = algo in rank_df.columns
        if available:
            rank_values = rank_df[algo].to_numpy(dtype=float)
            median_rank = float(np.median(rank_values))
            mean_rank = float(np.mean(rank_values))
            rank_ci_low = float(np.quantile(rank_values, 0.025))
            rank_ci_high = float(np.quantile(rank_values, 0.975))
            mean_score = float(score_df[algo].mean())
        else:
            rank_values = np.asarray([], dtype=float)
            median_rank = np.nan
            mean_rank = np.nan
            rank_ci_low = np.nan
            rank_ci_high = np.nan
            mean_score = np.nan

        for rank in range(1, len(target_algos) + 1):
            if available:
                frequency = float(np.mean(rank_values == rank))
            else:
                frequency = 0.0

            records.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "noise_scenario": noise_scenario,
                    "algorithm": algo,
                    "rank": rank,
                    "frequency": frequency,
                    "available": available,
                    "median_rank": median_rank,
                    "mean_rank": mean_rank,
                    "rank_ci_low": rank_ci_low,
                    "rank_ci_high": rank_ci_high,
                    "mean_score": mean_score,
                    "n_bootstrap": n_bootstrap,
                    "n_algorithms_present": n_algorithms_present,
                }
            )

    return records


def build_rank_frequency_summary(
    cell_vectors_df: pd.DataFrame, dataset_order: Sequence[str]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    dataset_order_present = [
        d for d in dataset_order if d in cell_vectors_df["dataset"].unique()
    ]
    for dataset in dataset_order_present:
        for noise_scenario in NOISE_ORDER:
            group_df = cell_vectors_df[
                (cell_vectors_df["dataset"] == dataset)
                & (cell_vectors_df["noise_scenario"] == noise_scenario)
            ]
            if group_df.empty:
                continue
            rows.extend(
                summarize_rank_group(
                    scope="per_dataset",
                    dataset=dataset,
                    noise_scenario=noise_scenario,
                    group_df=group_df,
                )
            )

    per_noise_df = aggregate_bootstrap_vectors(
        cell_vectors_df,
        ["algorithm", "noise_scenario"],
    )
    if not per_noise_df.empty:
        for noise_scenario in NOISE_ORDER:
            group_df = per_noise_df[per_noise_df["noise_scenario"] == noise_scenario]
            if group_df.empty:
                continue
            rows.extend(
                summarize_rank_group(
                    scope="all_datasets",
                    dataset="ALL",
                    noise_scenario=noise_scenario,
                    group_df=group_df,
                )
            )

    overall_df = aggregate_bootstrap_vectors(cell_vectors_df, ["algorithm"])
    if not overall_df.empty:
        rows.extend(
            summarize_rank_group(
                scope="overall",
                dataset="ALL",
                noise_scenario="all",
                group_df=overall_df,
            )
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["_dataset_order"] = out["dataset"].map(
        {d: i for i, d in enumerate(dataset_order_present)}
    ).fillna(len(dataset_order_present))
    out["_noise_order"] = out["noise_scenario"].map(
        {n: i for i, n in enumerate(NOISE_ORDER + ["all"])}
    ).fillna(len(NOISE_ORDER) + 1)
    out["_algo_order"] = out["algorithm"].map(
        {a: i for i, a in enumerate(target_algos)}
    ).fillna(len(target_algos))
    out = out.sort_values(
        ["scope", "_dataset_order", "_noise_order", "_algo_order", "rank"]
    ).drop(columns=["_dataset_order", "_noise_order", "_algo_order"])
    return out


def frequency_legend_handles(blob_scale: float) -> List[Line2D]:
    handles = []
    for freq in [0.25, 0.50, 0.75, 1.00]:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="black",
                markeredgecolor="black",
                alpha=0.65,
                markersize=max(np.sqrt(blob_scale * freq) / 3.0, 3.0),
                label=f"{int(freq * 100)}",
            )
        )
    return handles


def plot_blob_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    title: str,
    algorithm_order: Sequence[str],
    blob_scale: float,
) -> None:
    max_rank = len(algorithm_order)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.5, len(algorithm_order) + 0.5)
    ax.set_ylim(0.5, max_rank + 0.5)
    ax.set_xticks(range(1, len(algorithm_order) + 1))
    ax.set_xticklabels(algorithm_order, rotation=40, ha="right")
    ax.set_yticks(range(1, max_rank + 1))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    if summary_df.empty:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        return

    stats_df = summary_df.drop_duplicates(subset=["algorithm"]).set_index("algorithm")
    available_algorithms = set(
        summary_df.loc[summary_df["available"], "algorithm"].astype(str).tolist()
    )

    for xpos, algo in enumerate(algorithm_order, start=1):
        color = ALGO_COLORS.get(algo, "#4C72B0")
        algo_rows = summary_df[summary_df["algorithm"] == algo]
        if algo not in available_algorithms or algo_rows.empty:
            continue

        for row in algo_rows.itertuples(index=False):
            if row.frequency <= 0:
                continue
            ax.scatter(
                xpos,
                row.rank,
                s=blob_scale * row.frequency,
                color=color,
                alpha=0.65,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

        stat_row = stats_df.loc[algo]
        ax.vlines(
            xpos,
            stat_row["rank_ci_low"],
            stat_row["rank_ci_high"],
            colors="black",
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            xpos,
            stat_row["median_rank"],
            marker="x",
            color="black",
            s=70,
            linewidths=1.6,
            zorder=5,
        )

    for tick in ax.get_xticklabels():
        algo = tick.get_text()
        tick.set_color(ALGO_COLORS.get(algo, "black"))
        if algo not in available_algorithms:
            tick.set_alpha(0.35)


def save_dataset_figures(
    summary_df: pd.DataFrame,
    output_dir: Path,
    blob_scale: float,
    dpi: int,
    dataset_order: Sequence[str],
) -> List[Path]:
    saved_paths: List[Path] = []
    datasets_present = [
        d
        for d in dataset_order
        if (
            (summary_df["scope"] == "per_dataset")
            & (summary_df["dataset"] == d)
        ).any()
    ]

    for dataset in datasets_present:
        fig, axes = plt.subplots(
            1,
            len(NOISE_ORDER),
            figsize=(4.3 * len(NOISE_ORDER), 4.8),
            sharey=True,
        )
        if len(NOISE_ORDER) == 1:
            axes = [axes]

        for ax, noise_scenario in zip(axes, NOISE_ORDER):
            panel_df = summary_df[
                (summary_df["scope"] == "per_dataset")
                & (summary_df["dataset"] == dataset)
                & (summary_df["noise_scenario"] == noise_scenario)
            ]
            plot_blob_panel(
                ax=ax,
                summary_df=panel_df,
                title=NOISE_LABELS.get(noise_scenario, noise_scenario),
                algorithm_order=target_algos,
                blob_scale=blob_scale,
            )
            if ax is axes[0]:
                ax.set_ylabel("Rank")

        fig.suptitle(
            f"{dataset}: bootstrap ranking stability",
            fontsize=14,
            y=1.03,
        )
        fig.legend(
            handles=frequency_legend_handles(blob_scale),
            title="% of bootstrap samples",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
        )
        fig.text(
            0.5,
            0.02,
            "Blob area ∝ frequency at a rank; × = median rank; line = 95% bootstrap interval.",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.90])

        out_path = output_dir / f"ranking_stability_{dataset}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def save_overall_figure(
    summary_df: pd.DataFrame,
    datasets: Sequence[str],
    output_dir: Path,
    blob_scale: float,
    dpi: int,
) -> Path:
    panels = [
        ("all_datasets", "ALL", "clean", "clean"),
        ("all_datasets", "ALL", "roa", "roa(X)"),
        ("all_datasets", "ALL", "roc", "roc(X)"),
        ("all_datasets", "ALL", "noisy", "noisy"),
        ("overall", "ALL", "all", "overall"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.8), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (scope, dataset, noise_scenario, title) in zip(axes, panels):
        panel_df = summary_df[
            (summary_df["scope"] == scope)
            & (summary_df["dataset"] == dataset)
            & (summary_df["noise_scenario"] == noise_scenario)
        ]
        plot_blob_panel(
            ax=ax,
            summary_df=panel_df,
            title=title,
            algorithm_order=target_algos,
            blob_scale=blob_scale,
        )
        if ax is axes[0]:
            ax.set_ylabel("Rank")

    fig.suptitle(
        "Bootstrap ranking stability summarized over all datasets",
        fontsize=14,
        y=1.03,
    )
    fig.legend(
        handles=frequency_legend_handles(blob_scale),
        title="% of bootstrap samples",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.02,
        "Blob area ∝ frequency at a rank; × = median rank; line = 95% bootstrap interval.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])

    out_path = output_dir / f"ranking_stability_datasets_{'_'.join(datasets)}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ranking stability using bootstrap blob plots. "
            "By default, this creates one 5-panel figure summarized over all datasets: "
            "clean, roa(X), roc(X), noisy, and an overall panel across all scenarios. "
            "Ranking follows ranking.py: first average over clients within a fold, "
            "then average over folds, then rank algorithms by Dice. For stability, "
            "the same aggregation is applied element-wise to bootstrap samples."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store ranking-stability plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--blob-scale",
        type=float,
        default=DEFAULT_BLOB_SCALE,
        help=f"Marker area scale for blob plots (default: {DEFAULT_BLOB_SCALE})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output figure DPI (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--also-save-per-dataset-figures",
        action="store_true",
        help="Additionally save one 4-panel figure per dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default="LIDC RIGA Gleason MouseTumor MMIA MMIS".split(),
        help=(
            "Optional list of datasets to include; all others are excluded. "
            f"Allowed: {target_datasets}. "
            "Example: --datasets LIDC RIGA Gleason MouseTumor MMIA MMIS"
        ),
    )
    _env_root = os.environ.get("nnUNet_results")
    parser.add_argument(
        "--nnunet-results-root",
        type=Path,
        default=Path(_env_root) if _env_root else DEFAULT_NNUNET_RESULTS_ROOT,
        help=(
            "Root directory of nnUNet results (default: $nnUNet_results env var, "
            f"fallback: {DEFAULT_NNUNET_RESULTS_ROOT})"
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = parse_selected_datasets(args.datasets)

    df = load_and_preprocess_results()
    df = filter_records_to_included_folds(df)
    df = filter_records_to_selected_datasets(df, selected_datasets)
    all_exp_paths = build_experiment_path_index(Path(args.nnunet_results_root))
    client_vectors_df = collect_client_bootstrap_vectors(df, all_exp_paths)

    if client_vectors_df.empty:
        print("No client bootstrap vectors were collected. Nothing to visualize.")
        return

    cell_vectors_df = build_cell_bootstrap_vectors(client_vectors_df)
    if cell_vectors_df.empty:
        print("No per-cell bootstrap vectors could be aggregated. Nothing to visualize.")
        return

    summary_df = build_rank_frequency_summary(
        cell_vectors_df, dataset_order=selected_datasets
    )
    if summary_df.empty:
        print("No rank-frequency summary could be built. Nothing to visualize.")
        return

    summary_df.to_csv(args.output_dir / f"rank_frequency_summary_{'_'.join(selected_datasets)}.csv", index=False)
    print(f"Saved rank-frequency summary to CSV.")

    overall_path = save_overall_figure(
        summary_df=summary_df,
        datasets=selected_datasets,
        output_dir=args.output_dir,
        blob_scale=args.blob_scale,
        dpi=args.dpi,
    )

    print(f"Saved all-datasets summary figure:\n  - {overall_path.resolve()}")

    if args.also_save_per_dataset_figures:
        dataset_paths = save_dataset_figures(
            summary_df=summary_df,
            output_dir=args.output_dir,
            blob_scale=args.blob_scale,
            dpi=args.dpi,
            dataset_order=selected_datasets,
        )
        print("Saved per-dataset figures:")
        for path in dataset_paths:
            print(f"  - {path.resolve()}")


if __name__ == "__main__":
    main()
