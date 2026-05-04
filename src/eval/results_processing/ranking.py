import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Config (mirrors visualize_results.py)
# -------------------------------------------------------------------
sheet_id = "1AP_KH1cVSDwgpI1n7qK_VZU0Vi19Wh8vKo4jYWkuIXg"
gid = "332656109"
csv_url = (
    f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
)

algo_col = "Algo"
raw_dataset_col = "Data"
noise_col = "Noise"
exp_id_col = "Experiment ID"
classwise_metric = "Dice"

target_algos = ["FedAvg", "FedA3I", "IOP-FL", "FedCorr", "FedSelect"]
target_datasets = ["LIDC", "RIGA", "Gleason", "MouseTumor", "MMIA", "MMIS"]
included_folds = [0, 1, 2]

OUTPUT_DIR = Path("./results/segmentation_results")
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "bootstrap_method_rankings.csv"

DEFAULT_NNUNET_RESULTS_ROOT = Path(
    "/home/m391k/cluster-data/checkpoints/nnUNet_results"
)

dataset_to_numclients = {
    "LIDC": 4,
    "RIGA": 3,
    "Gleason": 3,
    "MouseTumor": 5,
    "MMIA": 4,
    "MMIS": 4,
}

noise_to_scenario = {
    "0": "clean",
    "roa(X)": "roa",
    "roc(X)": "roc",
    "100": "noisy",
}


def normalize_algorithm(name: str) -> str:
    n = str(name).strip().lower()
    n_compact = re.sub(r"[\s_\-]+", "", n)
    if n_compact == "fedavg":
        return "FedAvg"
    if n_compact == "feda3i":
        return "FedA3I"
    if n_compact == "iopfl":
        return "IOP-FL"
    if n_compact == "fedcorr":
        return "FedCorr"
    if n_compact == "fedselect":
        return "FedSelect"
    return str(name).strip()


def normalize_dataset(name: str) -> str:
    n = str(name).lower()
    if "lidc" in n:
        return "LIDC"
    if "riga" in n:
        return "RIGA"
    if "gleason" in n:
        return "Gleason"
    if "mousetumor" in n or "mouse tumor" in n:
        return "MouseTumor"
    if "mmia" in n:
        return "MMIA"
    if "mmis" in n:
        return "MMIS"
    return str(name).strip()


def normalize_noise_val(v: object) -> str:
    s = "" if v is None else str(v).strip()
    if re.fullmatch(r"(?i)0(?:\.0+)?", s):
        return "0"
    if re.fullmatch(r"(?i)100(?:\.0+)?", s):
        return "100"
    if re.search(r"(?i)\broa\b", s) or re.search(r"(?i)roa\s*\(.*\)", s):
        return "roa(X)"
    if re.search(r"(?i)\broc\b", s) or re.search(r"(?i)roc\s*\(.*\)", s):
        return "roc(X)"
    return s


def extract_fold_from_path(path: Path) -> Optional[int]:
    m = re.search(r"/fold_(\d+)/", str(path))
    return int(m.group(1)) if m else None


def extract_client_id_from_path(path: Path) -> Optional[int]:
    # Matches fedclientN (LIDC), flclientN (Gleason), _clientN / clientN (others)
    m = re.search(r"client(\d+)", str(path))
    if m:
        return int(m.group(1))
    # Fallback for RIGA-style paths where the dataset directory IS the client
    # e.g. Dataset303_RIGA-BinRushed_random → use dataset number as surrogate client id
    m2 = re.search(r"Dataset(\d+)_", str(path))
    return int(m2.group(1)) if m2 else None


def compute_bootstrap_metric_vector(
    bootstrap_file: Path, metric_name: str = classwise_metric
) -> Optional[np.ndarray]:
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


def compute_bootstrap_metric_mean(
    bootstrap_file: Path, metric_name: str = classwise_metric
) -> Optional[float]:
    metric_vector = compute_bootstrap_metric_vector(bootstrap_file, metric_name)
    if metric_vector is None:
        return None

    finite_values = metric_vector[np.isfinite(metric_vector)]
    if finite_values.size == 0:
        return None
    return float(np.mean(finite_values))


def load_and_preprocess_results() -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    print(f"Loaded {len(df)} rows from Google Sheets.")

    required_cols = [algo_col, raw_dataset_col, noise_col, exp_id_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in source CSV: {missing_cols}")

    for c in [algo_col, raw_dataset_col, noise_col, exp_id_col]:
        df[c] = df[c].astype(str).str.strip()

    df[algo_col] = df[algo_col].apply(normalize_algorithm)
    df["Dataset_norm"] = df[raw_dataset_col].apply(normalize_dataset)
    df[noise_col] = df[noise_col].apply(normalize_noise_val).astype(str).str.strip()
    df["noise_scenario"] = df[noise_col].map(noise_to_scenario)

    df = df[df[algo_col].isin(target_algos)]
    df = df[df["Dataset_norm"].isin(target_datasets)]
    df = df[df[noise_col].isin(noise_to_scenario.keys())]
    df = df[df["noise_scenario"].notna()]
    df = df[df[exp_id_col].notna()]
    df = df[df[exp_id_col] != ""]
    df = df[df[exp_id_col].str.lower() != "nan"]

    print(f"Retained {len(df)} rows after filtering target methods/datasets/noise.")
    return df


def build_experiment_path_index(nnunet_results_root: Path) -> List[Path]:
    all_exp_paths_raw = glob.glob(str(nnunet_results_root / "*" / "*" / "fold_*" / "*"))
    all_exp_paths = [Path(p) for p in all_exp_paths_raw]
    print(
        f"Found {len(all_exp_paths)} total checkpoint directories under {nnunet_results_root}."
    )

    filtered_exp_paths = [
        p for p in all_exp_paths if extract_fold_from_path(p) in included_folds
    ]
    print(
        f"Filtered to {len(filtered_exp_paths)} checkpoint directories in folds {included_folds}."
    )
    return filtered_exp_paths


def collect_client_bootstrap_scores(
    df: pd.DataFrame, all_exp_paths: List[Path]
) -> pd.DataFrame:
    rows = []
    bootstrap_mean_cache: Dict[Path, Optional[float]] = {}

    print(f"Collecting client bootstrap scores for {len(df)} experiment records...")

    records = df[
        [algo_col, "Dataset_norm", noise_col, "noise_scenario", exp_id_col]
    ].itertuples(index=False, name=None)

    for algo, dataset, noise_bucket, noise_scenario, exp_id in records:
        # Mirror visualize_results.py: substring match against the full path string
        exp_paths = [p for p in all_exp_paths if exp_id in str(p)]
        if not exp_paths:
            print(
                f"No checkpoint paths found for Exp_ID {exp_id} ({dataset}/{algo}/{noise_scenario})."
            )
            continue

        expected_clients = dataset_to_numclients.get(dataset, None)
        if expected_clients is not None and len(exp_paths) != expected_clients:
            print(
                f"Warning: Exp_ID {exp_id} ({dataset}/{algo}/{noise_scenario}): expected {expected_clients} clients, found {len(exp_paths)} — using available paths."
            )

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

            bootstrap_file = (
                exp_path / "validation" / "bootstrap_evaluation_results.json"
            )

            if bootstrap_file not in bootstrap_mean_cache:
                bootstrap_mean_cache[bootstrap_file] = compute_bootstrap_metric_mean(
                    bootstrap_file, classwise_metric
                )

            bootstrap_mean = bootstrap_mean_cache[bootstrap_file]
            if bootstrap_mean is None:
                print(f"Missing/invalid bootstrap metrics: {bootstrap_file}")
                continue

            rows.append(
                {
                    "algorithm": algo,
                    "dataset": dataset,
                    "noise_bucket": noise_bucket,
                    "noise_scenario": noise_scenario,
                    "experiment_id": exp_id,
                    "fold": fold,
                    "client_id": client_id,
                    "bootstrap_mean_dice": bootstrap_mean,
                }
            )

    out = pd.DataFrame(rows)
    print(f"Collected {len(out)} client bootstrap score rows.")
    return out


def build_ranking_table(client_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-step aggregation → pivot table with algorithms as columns and
    (dataset, noise_scenario) as rows, plus SUM summary rows at the bottom.

    Step 1: average bootstrap Dice over all clients within each (algo, dataset,
            noise_scenario, fold).
    Step 2: average those per-fold means over folds 0/1/2 → one Dice value per
            (algo, dataset, noise_scenario).
    Step 3: rank algorithms within each (dataset, noise_scenario) group.
    Step 4: pivot so each algorithm becomes a column.
    Step 5: append SUM rows = mean rank across datasets for each noise scenario
            and overall.
    """
    algo_order = [
        a for a in target_algos if a in client_scores_df["algorithm"].unique()
    ]
    dataset_order = [
        d for d in target_datasets if d in client_scores_df["dataset"].unique()
    ]
    noise_order = ["clean", "roa", "roc", "noisy"]

    if client_scores_df.empty:
        return pd.DataFrame(columns=["dataset", "noise_scenario"] + algo_order)

    # Step 1: average over clients per (algo, dataset, noise_scenario, fold)
    per_experiment = (
        client_scores_df.groupby(
            ["algorithm", "dataset", "noise_scenario", "experiment_id", "fold"],
            as_index=False,
        )["bootstrap_mean_dice"]
        .mean()
        .rename(columns={"bootstrap_mean_dice": "exp_mean_dice"})
    )

    # Step 2: average over folds → one value per (algo, dataset, noise_scenario)
    per_cell = (
        per_experiment.groupby(
            ["algorithm", "dataset", "noise_scenario"], as_index=False
        )["exp_mean_dice"]
        .mean()
        .rename(columns={"exp_mean_dice": "mean_dice"})
    )

    # Step 3: rank within each (dataset, noise_scenario) — higher Dice = rank 1
    per_cell["rank"] = (
        per_cell.groupby(["dataset", "noise_scenario"])["mean_dice"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Step 4: pivot ranks → wide table  (rows = dataset×noise, cols = algorithms)
    pivot = per_cell.pivot_table(
        index=["dataset", "noise_scenario"],
        columns="algorithm",
        values="rank",
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # Enforce row and column ordering
    noise_present = [n for n in noise_order if n in pivot["noise_scenario"].unique()]
    pivot["_ds_order"] = pivot["dataset"].map(
        {d: i for i, d in enumerate(dataset_order)}
    )
    pivot["_ns_order"] = pivot["noise_scenario"].map(
        {n: i for i, n in enumerate(noise_present)}
    )
    pivot = pivot.sort_values(["_ds_order", "_ns_order"]).drop(
        columns=["_ds_order", "_ns_order"]
    )

    # Reorder algorithm columns
    algo_cols = [a for a in algo_order if a in pivot.columns]
    pivot = pivot[["dataset", "noise_scenario"] + algo_cols]

    # Step 5: SUM rows = mean rank across all datasets for each noise scenario
    # (uses the per-cell rank values from the pivot, so each dataset contributes equally)
    rank_wide = pivot.set_index(["dataset", "noise_scenario"])[algo_cols]

    sum_rows = []

    # SUM (all)
    row_all = rank_wide.mean()
    row_all.name = ("SUM (all)", "")
    sum_rows.append(row_all)

    # SUM per noise scenario
    noise_label_map = {
        "clean": "SUM (clean)",
        "noisy": "SUM (noisy)",
        "roa": "SUM (roa(X))",
        "roc": "SUM (roc(X))",
    }
    for ns in noise_present:
        subset = (
            rank_wide.xs(ns, level="noise_scenario")
            if ns in rank_wide.index.get_level_values("noise_scenario")
            else None
        )
        if subset is None or subset.empty:
            continue
        row = subset.mean()
        row.name = (noise_label_map.get(ns, f"SUM ({ns})"), "")
        sum_rows.append(row)

    sum_df = pd.DataFrame(sum_rows)
    sum_df.index = pd.MultiIndex.from_tuples(sum_df.index)
    sum_df = sum_df.reset_index()
    sum_df.columns = ["dataset", "noise_scenario"] + algo_cols

    result = pd.concat([pivot, sum_df], ignore_index=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank FL methods using mean bootstrap Dice from checkpoint results "
            "(folds 0/1/2). Outputs a pivot table: rows = dataset × noise scenario, "
            "columns = algorithms, values = rank. SUM rows show mean rank across datasets."
        )
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT_CSV})",
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
    nnunet_results_root = args.nnunet_results_root

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess_results()
    all_exp_paths = build_experiment_path_index(Path(nnunet_results_root))
    client_scores_df = collect_client_bootstrap_scores(df, all_exp_paths)

    if client_scores_df.empty:
        print("No client bootstrap scores were collected. CSV not written.")
        return

    ranking_df = build_ranking_table(client_scores_df)
    ranking_df.to_csv(args.output_csv, index=False)
    print(f"Saved ranking table to {args.output_csv.resolve()}")

    # Print the table to terminal
    print()
    print(ranking_df.to_string(index=False))


if __name__ == "__main__":
    main()
