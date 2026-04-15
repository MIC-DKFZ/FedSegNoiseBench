"""
Comparison of partially noisy training scenarios roa(p) and roc(p).

For matched X, computes the effective noise ratio p and the per-dataset / per-algorithm
absolute and relative performance differences with respect to the clean baseline:

    delta^{v}_{abs}         = Dice(clean) - Dice(v)
    delta^{v}_{rel,eff}     = (Dice(clean) - Dice(v)) / p^{v}_{eff}(X)

where v ∈ {roa, roc} and p^{v}_{eff}(X) is defined as:
    - roa(p): p^{roa}_{eff}   = X  (X fraction of each client's annotations are noisy)
    - roc(p): p^{roc}_{eff}   = (Σ_{k ∈ K_n} n_k) / (Σ_{k ∈ K} n_k)
                                   with K_n = noisy clients, K = all clients

The Dice values are computed from bootstrap_evaluation_results.json across checkpoints,
averaged per client, then per fold, then per (dataset, algorithm, noise_scenario).
Folds [0, 1, 2] are included.

Outputs
-------
- results/segmentation_results/partial_noise_comparison/partial_noise_comparison.csv
  Long-form table: one row per (dataset, algorithm) with all computed quantities.
- results/segmentation_results/partial_noise_comparison/partial_noise_comparison.png
    Scatter plot with effective delta(clean, roa) on x-axis and effective
    delta(clean, roc) on y-axis. Bootstrap deltas are light-gray background
    points; mean deltas are foreground markers (color = algorithm, marker = dataset).
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
mean_dice_col = "Mean(D_val)"

TARGET_ALGOS = ["FedAvg", "FedA3I", "IOP-FL", "FedCorr", "FedSelect"]
TARGET_DATASETS = ["LIDC", "RIGA", "Gleason", "MouseTumor", "MMIA", "MMIS"]
INCLUDED_FOLDS = [0, 1, 2]
DEFAULT_NNUNET_RESULTS_ROOTS = [
    Path("/home/m391k/cluster-data/checkpoints/nnUNet_results"),
    Path("/home/m391k/juwels/checkpoints/nnUNet_results"),
]

OUTPUT_DIR = Path("./results/segmentation_results/partial_noise_comparison")

# ---------------------------------------------------------------------------
# Effective noise ratio p^{roa}_{eff}
# ---------------------------------------------------------------------------
# For roa(50): all datasets use 50 % noisy annotations per client → p_eff = 0.50
P_ROA_EFF: float = 0.50
P_NOISY_EFF: float = 1.00

# ---------------------------------------------------------------------------
# Effective noise ratio p^{roc}_{eff} per dataset
#
# Derived from actual nnUNet training-set sizes on disk:
#
#   LIDC  roc(50):  clients [D041,D042 (clean), D047,D048 (noisy)]
#                   n_clean = 1281+218=1499,  n_noisy = 475+147=622,  total=2121
#   RIGA  roc(66,6): clients [D300 (clean), D304,D305 (noisy)]
#                   n_clean = 195,  n_noisy = 94+460=554,  total=749
#   Gleason roc(66,6): clients [D436 (clean), D440,D441 (noisy)]
#                   n_clean = 159,  n_noisy = 158+159=317,  total=476
#   MouseTumor roc(60): clients [D500,D501 (clean), D507,D508,D509 (noisy)]
#                   n_clean = 178+127=305,  n_noisy = 67+48+32=147,  total=452
#   MMIA  roc(50):  clients [D600,D601 (clean), D606,D607 (noisy)]
#                   n_clean = 291+171=462,  n_noisy = 980+64=1044,  total=1506
#   MMIS  roc(50):  clients [D700,D701 (clean), D706,D707 (noisy)]
#                   n_clean = 34+29=63,  n_noisy = 27+30=57,  total=120
# ---------------------------------------------------------------------------
P_ROC_EFF: dict[str, float] = {
    "LIDC": 622 / 2121,  # ≈ 0.293
    "RIGA": 554 / 749,  # ≈ 0.740
    "Gleason": 317 / 476,  # ≈ 0.666
    "MouseTumor": 147 / 452,  # ≈ 0.325
    "MMIA": 1044 / 1506,  # ≈ 0.693
    "MMIS": 57 / 120,  # ≈ 0.475
}

# Nominal X for roc per dataset (from Google Sheets Noise column)
ROC_NOMINAL_X: dict[str, str] = {
    "LIDC": "50",
    "RIGA": "66.6",
    "Gleason": "66.6",
    "MouseTumor": "60",
    "MMIA": "50",
    "MMIS": "50",
}

ALGO_COLORS = {
    "FedAvg": "#4C72B0",
    "FedA3I": "#DD8452",
    "IOP-FL": "#55A868",
    "FedCorr": "#C44E52",
    "FedSelect": "#8172B2",
}

DELTA_MODE_COLUMNS = {
    "rel_eff": {
        "roa": "delta_roa_rel_eff",
        "roc": "delta_roc_rel_eff",
        "noisy": "delta_noisy_rel_eff",
    },
    "abs": {
        "roa": "delta_roa_abs",
        "roc": "delta_roc_abs",
        "noisy": "delta_noisy_abs",
    },
}

DELTA_MODE_LABELS = {
    "rel_eff": r"$\Delta Dice_{\mathrm{eff,rel}}$",
    "abs": r"$\Delta Dice$",
}

DELTA_MODE_TITLES = {
    "rel_eff": "effective-relative Dice drop",
    "abs": "absolute Dice drop",
}

SCENARIO_LABELS = {
    "roa": "roa(p)",
    "roc": "roc(p)",
    "noisy": "noisy(100%)",
}

DATASET_OFFSETS = {
    dataset: offset
    for dataset, offset in zip(
        TARGET_DATASETS,
        np.linspace(-0.32, 0.32, len(TARGET_DATASETS)),
    )
}

DATASET_TICK_LABELS = {
    "LIDC": "LIDC",
    "RIGA": "RIGA",
    "Gleason": "GleasonHD",
    "MouseTumor": "MouseT",
    "MMIA": "MMIA",
    "MMIS": "MMIS",
}

SEPARATE_DOT_FIGSIZE = (28.0, 7.6)
SEPARATE_DOT_TITLE_SIZE = 22
SEPARATE_DOT_PANEL_TITLE_SIZE = 22
SEPARATE_DOT_LABEL_SIZE = 24
SEPARATE_DOT_TICK_SIZE = 19
SEPARATE_DOT_LEGEND_SIZE = 20
SEPARATE_DOT_MARKER_SIZE = 200
SEPARATE_DOT_BOOTSTRAP_MARKER_SIZE = 15
SEPARATE_DOT_MEAN_LINEWIDTH = 3.5
SEPARATE_DOT_AXIS_LINEWIDTH = 1.2


def get_effective_noise_ratio(dataset: str, scenario: str) -> float:
    if scenario == "roa":
        return P_ROA_EFF
    if scenario == "roc":
        return P_ROC_EFF.get(dataset, np.nan)
    if scenario == "noisy":
        return P_NOISY_EFF
    return np.nan


def compute_bootstrap_delta_vector(
    bootstrap_vectors: Dict[tuple, np.ndarray],
    algo: str,
    dataset: str,
    scenario: str,
    delta_mode: str,
) -> np.ndarray:
    clean_vec = bootstrap_vectors.get((algo, dataset, "clean"))
    scenario_vec = bootstrap_vectors.get((algo, dataset, scenario))
    if clean_vec is None or scenario_vec is None:
        return np.asarray([], dtype=float)
    if len(clean_vec) == 0 or len(scenario_vec) == 0:
        return np.asarray([], dtype=float)

    n = min(len(clean_vec), len(scenario_vec))
    clean_arr = np.asarray(clean_vec[:n], dtype=float)
    scenario_arr = np.asarray(scenario_vec[:n], dtype=float)
    delta = clean_arr - scenario_arr

    if delta_mode == "rel_eff":
        p_eff = get_effective_noise_ratio(dataset, scenario)
        if not np.isfinite(p_eff) or p_eff <= 0:
            return np.asarray([], dtype=float)
        delta = delta / p_eff

    return delta[np.isfinite(delta)]

# ---------------------------------------------------------------------------
# Bootstrap loading (reused from visualize_ranking.py)
# ---------------------------------------------------------------------------

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

        if len(arr) == 0:
            continue
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            class_vectors.append(arr)

    if not class_vectors:
        return None

    # Pad to max length with NaN, then average
    max_len = max(len(v) for v in class_vectors)
    padded = [
        np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in class_vectors
    ]
    result_arr = np.nanmean(padded, axis=0)
    return result_arr


def build_experiment_path_index(nnunet_results_roots: Sequence[Path]) -> List[Path]:
    """Index all experiment paths by fold across one or more roots."""
    all_exp_paths_raw: List[str] = []
    for nnunet_results_root in nnunet_results_roots:
        all_exp_paths_raw.extend(
            glob.glob(str(nnunet_results_root / "*" / "*" / "fold_*" / "*"))
        )

    all_exp_paths = sorted({Path(p) for p in all_exp_paths_raw}, key=str)
    filtered_exp_paths = [
        p for p in all_exp_paths if extract_fold_from_path(p) in INCLUDED_FOLDS
    ]
    return filtered_exp_paths


def resolve_nnunet_results_roots(cli_root: Optional[Path]) -> List[Path]:
    if cli_root is not None:
        return [Path(cli_root)]

    env_root = os.environ.get("nnUNet_results")
    if env_root:
        return [Path(env_root)]

    return DEFAULT_NNUNET_RESULTS_ROOTS.copy()


def get_exp_paths_with_bootstrap(
    all_exp_paths: List[Path], exp_id: str
) -> List[Path]:
    return [
        p
        for p in all_exp_paths
        if exp_id in str(p)
        and (p / "validation" / "bootstrap_evaluation_results.json").is_file()
    ]


def extract_fold_from_path(path: Path) -> Optional[int]:
    m = re.search(r"/fold_(\d+)/", str(path))
    return int(m.group(1)) if m else None


def extract_client_id_from_path(path: Path) -> Optional[int]:
    m = re.search(r"client(\d+)", str(path))
    if m:
        return int(m.group(1))
    m2 = re.search(r"Dataset(\d+)_", str(path))
    return int(m2.group(1)) if m2 else None


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


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


def parse_mean_dice(value: object) -> float | None:
    """Parse Mean(D_val) which may use comma as decimal separator."""
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except ValueError:
        return None


def classify_noise(noise_str: str) -> str | None:
    """Return canonical noise scenario: 'clean', 'roa', 'roc', 'noisy', or None."""
    s = str(noise_str).strip()
    if re.fullmatch(r"0(?:\.0+)?", s):
        return "clean"
    if re.fullmatch(r"100(?:\.0+)?", s):
        return "noisy"
    if re.search(r"(?i)\broa\b", s):
        return "roa"
    if re.search(r"(?i)\broc\b", s):
        return "roc"
    return None


# ---------------------------------------------------------------------------
# Data loading and bootstrap processing
# ---------------------------------------------------------------------------


def load_sheet() -> pd.DataFrame:
    """Load and preprocess the Google Sheets data."""
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

    # Keep only target algos / datasets / known noise scenarios / included folds
    df = df[df[algo_col].isin(TARGET_ALGOS)]
    df = df[df["dataset"].isin(TARGET_DATASETS)]
    df = df[df["noise_scenario"].isin(["clean", "roa", "roc", "noisy"])]
    df = df[df["fold"].isin(INCLUDED_FOLDS)]
    df = df[df["exp_id"].notna()]
    df = df[df["exp_id"] != ""]

    print(
        f"Retained {len(df)} rows after filtering "
        f"(algos={TARGET_ALGOS!r}; datasets={TARGET_DATASETS!r}; "
        f"noise=[clean,roa,roc,noisy]; folds={INCLUDED_FOLDS!r})."
    )
    return df


def load_bootstrap_dice_per_cell(
    df: pd.DataFrame,
    all_exp_paths: List[Path],
    nnunet_results_root: Path,
) -> Dict[tuple, np.ndarray]:
    """
    For each (algorithm, dataset, noise_scenario), compute bootstrap-derived
    vectors by:
    1. Finding checkpoint paths for that experiment
    2. Loading bootstrap vectors per client
    3. Averaging over clients element-wise → per-fold vectors
    4. Averaging over folds element-wise → final cell vector

    Returns: {(algo, dataset, noise_scenario): bootstrap_vector}
    """
    bootstrap_cache: Dict[Path, Optional[np.ndarray]] = {}
    result = {}

    for (algo, dataset, noise_scenario), group_df in df.groupby(
        [algo_col, "dataset", "noise_scenario"]
    ):
        exp_ids = group_df["exp_id"].unique()

        # Collect all bootstrap vectors for this cell
        per_fold_vectors: Dict[int, List[np.ndarray]] = {}

        for exp_id in exp_ids:
            # Find checkpoint paths matching this exp_id
            exp_paths = get_exp_paths_with_bootstrap(all_exp_paths, exp_id)
            if not exp_paths:
                continue

            for exp_path in exp_paths:
                fold = extract_fold_from_path(exp_path)
                if fold is None or fold not in INCLUDED_FOLDS:
                    continue

                bootstrap_file = (
                    exp_path / "validation" / "bootstrap_evaluation_results.json"
                )

                if bootstrap_file not in bootstrap_cache:
                    bootstrap_cache[bootstrap_file] = load_bootstrap_metric_vector(
                        bootstrap_file, classwise_metric
                    )

                vec = bootstrap_cache[bootstrap_file]
                if vec is not None:
                    if fold not in per_fold_vectors:
                        per_fold_vectors[fold] = []
                    per_fold_vectors[fold].append(vec)

        if not per_fold_vectors:
            print(
                f"Warning: no bootstrap vectors found for "
                f"{algo}/{dataset}/{noise_scenario}"
            )
            continue

        # Average over folds (element-wise)
        fold_means = []
        for fold in sorted(per_fold_vectors.keys()):
            vectors = per_fold_vectors[fold]
            # Average over clients (vectors) for this fold (element-wise)
            max_len = max(len(v) for v in vectors)
            padded = [
                np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
                for v in vectors
            ]
            fold_mean = np.nanmean(padded, axis=0)
            fold_means.append(fold_mean)

        # Average over folds (element-wise)
        if fold_means:
            max_len = max(len(v) for v in fold_means)
            padded = [
                np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
                for v in fold_means
            ]
            cell_vec = np.nanmean(padded, axis=0)
            result[(algo, dataset, noise_scenario)] = cell_vec

    return result


def aggregate_bootstrap_vectors_to_mean_dice(
    bootstrap_vectors: Dict[tuple, np.ndarray],
) -> Dict[tuple, float]:
    """Collapse bootstrap vectors to scalar mean Dice values per cell."""
    mean_dice: Dict[tuple, float] = {}
    for key, vec in bootstrap_vectors.items():
        if vec is None or len(vec) == 0:
            continue
        v = float(np.nanmean(vec))
        if np.isfinite(v):
            mean_dice[key] = v
    return mean_dice


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def build_comparison_table(bootstrap_dice: Dict[tuple, float]) -> pd.DataFrame:
    """
    For each (dataset, algorithm) compute delta_abs and delta_rel_eff
    based on bootstrap-derived Dice values.

    Keys in bootstrap_dice: (algo, dataset, noise_scenario)
    Values: mean Dice
    """
    rows = []
    for dataset in TARGET_DATASETS:
        for algo in TARGET_ALGOS:
            key_clean = (algo, dataset, "clean")
            key_roa = (algo, dataset, "roa")
            key_roc = (algo, dataset, "roc")
            key_noisy = (algo, dataset, "noisy")

            dice_clean = bootstrap_dice.get(key_clean, np.nan)
            dice_roa = bootstrap_dice.get(key_roa, np.nan)
            dice_roc = bootstrap_dice.get(key_roc, np.nan)
            dice_noisy = bootstrap_dice.get(key_noisy, np.nan)

            if np.isnan(dice_clean):
                continue

            p_roa = P_ROA_EFF
            p_roc = P_ROC_EFF.get(dataset, np.nan)
            p_noisy = P_NOISY_EFF
            roc_nom_x = ROC_NOMINAL_X.get(dataset, "?")

            delta_roa_abs = (
                (dice_clean - dice_roa) if not np.isnan(dice_roa) else np.nan
            )
            delta_roc_abs = (
                (dice_clean - dice_roc) if not np.isnan(dice_roc) else np.nan
            )
            delta_noisy_abs = (
                (dice_clean - dice_noisy) if not np.isnan(dice_noisy) else np.nan
            )

            delta_roa_rel = (
                (delta_roa_abs / p_roa)
                if (not np.isnan(delta_roa_abs) and p_roa > 0)
                else np.nan
            )
            delta_roc_rel = (
                (delta_roc_abs / p_roc)
                if (not np.isnan(delta_roc_abs) and not np.isnan(p_roc) and p_roc > 0)
                else np.nan
            )
            delta_noisy_rel = (
                (delta_noisy_abs / p_noisy)
                if (not np.isnan(delta_noisy_abs) and p_noisy > 0)
                else np.nan
            )

            rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algo,
                    "dice_clean": dice_clean,
                    "dice_roa": dice_roa,
                    "dice_roc": dice_roc,
                    "dice_noisy": dice_noisy,
                    "p_roa_eff": p_roa,
                    "p_roc_eff": p_roc,
                    "p_noisy_eff": p_noisy,
                    "roc_nominal_X_pct": roc_nom_x,
                    "delta_roa_abs": delta_roa_abs,
                    "delta_roc_abs": delta_roc_abs,
                    "delta_noisy_abs": delta_noisy_abs,
                    "delta_roa_rel_eff": delta_roa_rel,
                    "delta_roc_rel_eff": delta_roc_rel,
                    "delta_noisy_rel_eff": delta_noisy_rel,
                }
            )

    result = pd.DataFrame(rows)

    # Enforce canonical ordering
    ds_order = {d: i for i, d in enumerate(TARGET_DATASETS)}
    al_order = {a: i for i, a in enumerate(TARGET_ALGOS)}
    result["_ds"] = result["dataset"].map(ds_order)
    result["_al"] = result["algorithm"].map(al_order)
    result = result.sort_values(["_ds", "_al"]).drop(columns=["_ds", "_al"])
    result = result.reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_table(result: pd.DataFrame) -> None:
    float_cols = [
        "dice_clean",
        "dice_roa",
        "dice_roc",
        "dice_noisy",
        "p_roa_eff",
        "p_roc_eff",
        "p_noisy_eff",
        "delta_roa_abs",
        "delta_roc_abs",
        "delta_noisy_abs",
        "delta_roa_rel_eff",
        "delta_roc_rel_eff",
        "delta_noisy_rel_eff",
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
    print("Noise scenario comparison: roa(50%), roc(p), and noisy(100%)")
    print("=" * 120)
    print(display.to_string(index=False))
    print()

    # Per-dataset summary: mean over algorithms
    print("=" * 120)
    print("Mean over algorithms per dataset")
    print("=" * 120)
    numeric_cols = [
        "dice_clean",
        "dice_roa",
        "dice_roc",
        "dice_noisy",
        "p_roa_eff",
        "p_roc_eff",
        "p_noisy_eff",
        "delta_roa_abs",
        "delta_roc_abs",
        "delta_noisy_abs",
        "delta_roa_rel_eff",
        "delta_roc_rel_eff",
        "delta_noisy_rel_eff",
    ]
    summary = result.groupby("dataset")[numeric_cols].mean().reset_index()
    # Restore dataset order
    ds_order = {d: i for i, d in enumerate(TARGET_DATASETS)}
    summary["_ds"] = summary["dataset"].map(ds_order)
    summary = summary.sort_values("_ds").drop(columns=["_ds"]).reset_index(drop=True)
    for c in numeric_cols:
        if c in summary.columns:
            summary[c] = summary[c].map(
                lambda x: (
                    f"{x:.4f}" if not (isinstance(x, float) and np.isnan(x)) else "—"
                )
            )
    print(summary.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_comparison_scatter(
    result: pd.DataFrame,
    output_path: Path,
    bootstrap_vectors: Dict[tuple, np.ndarray],
    dpi: int = 200,
) -> None:
    """
    Scatter plot of effective deltas:
      x-axis: effective delta(clean, roa)
      y-axis: effective delta(clean, roc)

    Background points (light gray): bootstrap sample-level effective deltas.
    Foreground points: mean effective deltas, with
      - color = FNLL method (algorithm)
      - marker shape = dataset
    """
    from matplotlib.lines import Line2D

    algo_colors = {
        algo: ALGO_COLORS.get(algo, plt.cm.tab10(i % 10))
        for i, algo in enumerate(TARGET_ALGOS)
    }
    dataset_markers = {
        ds: m
        for ds, m in zip(
            TARGET_DATASETS,
            ["o", "s", "^", "D", "P", "X"],
        )
    }
    dataset_offsets = {
        dataset: offset
        for dataset, offset in zip(
            TARGET_DATASETS,
            np.linspace(-0.32, 0.32, len(TARGET_DATASETS)),
        )
    }
    dataset_tick_labels = {
        "LIDC": "LIDC",
        "RIGA": "RIGA",
        "Gleason": "Gleason",
        "MouseTumor": "MouseT",
        "MMIA": "MMIA",
        "MMIS": "MMIS",
    }
    dataset_offsets = {
        dataset: offset
        for dataset, offset in zip(
            TARGET_DATASETS,
            np.linspace(-0.32, 0.32, len(TARGET_DATASETS)),
        )
    }
    dataset_tick_labels = {
        "LIDC": "LIDC",
        "RIGA": "RIGA",
        "Gleason": "Gleason",
        "MouseTumor": "MouseT",
        "MMIA": "MMIA",
        "MMIS": "MMIS",
    }

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    bg_x: List[float] = []
    bg_y: List[float] = []
    mean_points: List[tuple] = []  # (x, y, algo, dataset)
    method_points: List[tuple] = []  # (x, y, algo)

    for dataset in TARGET_DATASETS:
        p_roa = P_ROA_EFF
        p_roc = P_ROC_EFF.get(dataset, np.nan)
        if not np.isfinite(p_roc) or p_roc <= 0:
            continue

        for algo in TARGET_ALGOS:
            clean_vec = bootstrap_vectors.get((algo, dataset, "clean"))
            roa_vec = bootstrap_vectors.get((algo, dataset, "roa"))
            roc_vec = bootstrap_vectors.get((algo, dataset, "roc"))

            if clean_vec is None or roa_vec is None or roc_vec is None:
                continue
            if len(clean_vec) == 0 or len(roa_vec) == 0 or len(roc_vec) == 0:
                continue

            n = min(len(clean_vec), len(roa_vec), len(roc_vec))
            if n == 0:
                continue

            clean_arr = np.asarray(clean_vec[:n], dtype=float)
            roa_arr = np.asarray(roa_vec[:n], dtype=float)
            roc_arr = np.asarray(roc_vec[:n], dtype=float)

            delta_roa_eff_vec = (clean_arr - roa_arr) / p_roa
            delta_roc_eff_vec = (clean_arr - roc_arr) / p_roc

            valid = np.isfinite(delta_roa_eff_vec) & np.isfinite(delta_roc_eff_vec)
            if not np.any(valid):
                continue

            x_vals = delta_roa_eff_vec[valid]
            y_vals = delta_roc_eff_vec[valid]

            bg_x.extend(x_vals.tolist())
            bg_y.extend(y_vals.tolist())

            row = result[(result["dataset"] == dataset) & (result["algorithm"] == algo)]
            if not row.empty:
                x_mean = float(row.iloc[0]["delta_roa_rel_eff"])
                y_mean = float(row.iloc[0]["delta_roc_rel_eff"])
            else:
                x_mean = float(np.nanmean(x_vals))
                y_mean = float(np.nanmean(y_vals))

            if np.isfinite(x_mean) and np.isfinite(y_mean):
                mean_points.append((x_mean, y_mean, algo, dataset))

    if not mean_points:
        print("No valid points to plot.")
        return

    # Per-method mean across datasets
    for algo in TARGET_ALGOS:
        algo_xy = [(x, y) for x, y, a, _ in mean_points if a == algo]
        if not algo_xy:
            continue
        xs = np.asarray([p[0] for p in algo_xy], dtype=float)
        ys = np.asarray([p[1] for p in algo_xy], dtype=float)
        x_m = float(np.nanmean(xs))
        y_m = float(np.nanmean(ys))
        if np.isfinite(x_m) and np.isfinite(y_m):
            method_points.append((x_m, y_m, algo))

    # Background bootstrap points
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

    # Mean points (color=algorithm, marker=dataset)
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

    # Method means across datasets (larger marker)
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

    # Diagonal reference x=y (equal sensitivity)
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

        # Annotate regions mirrored across diagonal
        # "roa more harmful" below diagonal (original position)
        x_roa = lim_high - 0.08 * span
        y_roa = lim_high - 0.06 * span
        # "roc more harmful" mirrored across y=x (swap x and y), pushed higher
        x_roc = lim_high - 0.13 * span
        y_roc = lim_high - 0.02 * span

        ax.text(
            x_roc,
            y_roc,
            "roc more harmful",
            fontsize=9,
            ha="center",
            va="top",
            rotation=45,
            # bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
            zorder=5,
        )
        ax.text(
            x_roa,
            y_roa,
            "roa more harmful",
            fontsize=9,
            ha="center",
            va="top",
            rotation=45,
            # bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
            zorder=5,
        )

    ax.set_xlabel(
        r"$\Delta Dice_{eff,rel}(\mathrm{clean},\mathrm{{roa}})$",
        fontsize=11,
    )
    ax.set_ylabel(
        r"$\Delta Dice_{eff,rel}(\mathrm{clean},\mathrm{roc})$",
        fontsize=11,
    )
    ax.set_title(
        "Robustness of FNLL methods to partially noisy training scenarios: roa and roc",
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
        label="x = y (equal sensitivity)",
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
    bootstrap_vectors: Dict[tuple, np.ndarray],
    dpi: int = 200,
) -> None:
    """
    Single-panel paired-dot plot of effective-noise-normalized degradation.

    For each FNLL method on the x-axis, roa and roc are shown side-by-side:
        Δ_eff(clean, roa) = (Dice(clean) - Dice(roa)) / p_roa_eff
        Δ_eff(clean, roc) = (Dice(clean) - Dice(roc)) / p_roc_eff

    Each small point is one dataset-level mean for one FNLL method.
    Marker shape encodes dataset; color encodes FNLL method, with
    lighter shade for roa and darker shade for roc.
    The thick horizontal black marker shows the mean across datasets per method
    for each scenario (roa and roc separately).
    """
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors

    def blend_with(color: str, blend_to: str, alpha: float) -> tuple:
        """Blend color with target color by alpha in [0,1]."""
        c1 = np.asarray(mcolors.to_rgb(color), dtype=float)
        c2 = np.asarray(mcolors.to_rgb(blend_to), dtype=float)
        return tuple((1.0 - alpha) * c1 + alpha * c2)

    algo_colors = {
        algo: ALGO_COLORS.get(algo, plt.cm.tab10(i % 10))
        for i, algo in enumerate(TARGET_ALGOS)
    }
    dataset_markers = {
        ds: m
        for ds, m in zip(
            TARGET_DATASETS,
            ["o", "s", "^", "D", "P", "X"],
        )
    }

    # Collect plotting data
    roa_points = []  # (algo, dataset, y)
    roc_points = []  # (algo, dataset, y)

    for dataset in TARGET_DATASETS:
        p_roa = P_ROA_EFF
        p_roc = P_ROC_EFF.get(dataset, np.nan)

        for algo in TARGET_ALGOS:
            clean_vec = bootstrap_vectors.get((algo, dataset, "clean"))
            roa_vec = bootstrap_vectors.get((algo, dataset, "roa"))
            roc_vec = bootstrap_vectors.get((algo, dataset, "roc"))

            # Prefer scalar means from result table if available
            row = result[(result["dataset"] == dataset) & (result["algorithm"] == algo)]

            if not row.empty:
                delta_roa_eff = float(row.iloc[0]["delta_roa_rel_eff"])
                delta_roc_eff = float(row.iloc[0]["delta_roc_rel_eff"])
            else:
                delta_roa_eff = np.nan
                delta_roc_eff = np.nan

                if (
                    clean_vec is not None
                    and roa_vec is not None
                    and len(clean_vec) > 0
                    and len(roa_vec) > 0
                ):
                    n = min(len(clean_vec), len(roa_vec))
                    clean_arr = np.asarray(clean_vec[:n], dtype=float)
                    roa_arr = np.asarray(roa_vec[:n], dtype=float)
                    tmp = (clean_arr - roa_arr) / p_roa
                    if np.any(np.isfinite(tmp)):
                        delta_roa_eff = float(np.nanmean(tmp))

                if (
                    clean_vec is not None
                    and roc_vec is not None
                    and len(clean_vec) > 0
                    and len(roc_vec) > 0
                    and np.isfinite(p_roc)
                    and p_roc > 0
                ):
                    n = min(len(clean_vec), len(roc_vec))
                    clean_arr = np.asarray(clean_vec[:n], dtype=float)
                    roc_arr = np.asarray(roc_vec[:n], dtype=float)
                    tmp = (clean_arr - roc_arr) / p_roc
                    if np.any(np.isfinite(tmp)):
                        delta_roc_eff = float(np.nanmean(tmp))

            if np.isfinite(delta_roa_eff):
                roa_points.append((algo, dataset, delta_roa_eff))
            if np.isfinite(delta_roc_eff):
                roc_points.append((algo, dataset, delta_roc_eff))

    if not roa_points and not roc_points:
        print("No valid points to plot.")
        return

    # y-limits shared across both panels
    all_y = [p[2] for p in roa_points] + [p[2] for p in roc_points]
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    y_span = y_max - y_min
    y_pad = 0.08 * y_span if y_span > 0 else 0.05
    y_low = y_min - y_pad
    y_high = y_max + y_pad

    fig, ax = plt.subplots(1, 1, figsize=(11.8, 5.6), dpi=dpi)

    # roa and roc side-by-side around each method x-position
    x_offset_roa = -0.18
    x_offset_roc = +0.18

    # Vertical separators only between methods (not between roa/roc within method)
    for x_sep in np.arange(1.5, len(TARGET_ALGOS) + 0.5, 1.0):
        ax.axvline(x_sep, color="0.90", lw=1.1, zorder=0)

    for i, algo in enumerate(TARGET_ALGOS, start=1):

        # Nicer shade pair per method: lighter for roa, deeper for roc
        roa_color = blend_with(algo_colors[algo], "white", 0.18)
        roc_color = blend_with(algo_colors[algo], "black", 0.12)

        # roa points + mean
        roa_vals = []
        for _, dataset, y in [p for p in roa_points if p[0] == algo]:
            roa_vals.append(y)
            ax.scatter(
                i + x_offset_roa,
                y,
                s=95,
                marker=dataset_markers[dataset],
                c=[roa_color],
                edgecolors="black",
                linewidths=0.8,
                alpha=0.96,
                zorder=3,
            )
        if roa_vals:
            y_mean = float(np.nanmean(roa_vals))
            ax.scatter(
                i + x_offset_roa,
                y_mean,
                s=240,
                marker="_",
                c="black",
                linewidths=3.0,
                zorder=4,
            )

        # roc points + mean
        roc_vals = []
        for _, dataset, y in [p for p in roc_points if p[0] == algo]:
            roc_vals.append(y)
            ax.scatter(
                i + x_offset_roc,
                y,
                s=95,
                marker=dataset_markers[dataset],
                c=[roc_color],
                edgecolors="black",
                linewidths=0.8,
                alpha=0.96,
                zorder=3,
            )
        if roc_vals:
            y_mean = float(np.nanmean(roc_vals))
            ax.scatter(
                i + x_offset_roc,
                y_mean,
                s=240,
                marker="_",
                c="black",
                linewidths=3.0,
                zorder=4,
            )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.35)
    ax.set_xlim(0.5, len(TARGET_ALGOS) + 0.5)
    ax.set_ylim(y_low, y_high)
    # Minor x-tick descriptions per pair
    x_ticks = []
    x_labels = []
    for i in range(1, len(TARGET_ALGOS) + 1):
        x_ticks.extend([i + x_offset_roa, i + x_offset_roc])
        x_labels.extend(["roa", "roc"])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=0, ha="center")

    # Method labels centered under each roa/roc pair
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

    ax.set_xlabel("Partial noisy scenario per method", fontsize=11, labelpad=22)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.set_ylabel(
        # r"$\Delta Dice_{\mathrm{eff}} = \left(\mathrm{Dice(clean)} - \mathrm{Dice(partial)}\right)/p^{partial}_{\mathrm{eff}}$",
        r"$\Delta Dice_{\mathrm{eff,rel}}$(clean, partial noisy)",
        fontsize=11,
    )

    # Legends
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
        color="black",
        linewidth=3,
        label="Mean across datasets",
    )

    fig.legend(
        handles=dataset_handles + [mean_handle],
        # title="Marker = dataset",
        loc="upper right",
        # bbox_to_anchor=(1.12, 0.92),
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0.03, 0.10, 0.97, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path.resolve()}")


def plot_single_scenario_dot(
    result: pd.DataFrame,
    output_path: Path,
    scenario: str,
    delta_mode: str = "rel_eff",
    dpi: int = 200,
) -> None:
    """
    Single-panel dot plot for one partial-noise scenario only.

    delta_mode='rel_eff' uses delta_*_rel_eff columns.
    delta_mode='abs' uses delta_*_abs columns.
    """
    if delta_mode not in DELTA_MODE_COLUMNS:
        raise ValueError("delta_mode must be one of: 'rel_eff', 'abs'")
    if scenario not in SCENARIO_LABELS:
        raise ValueError("scenario must be one of: 'roa', 'roc', 'noisy'")

    from matplotlib.lines import Line2D

    metric_col = DELTA_MODE_COLUMNS[delta_mode][scenario]
    scenario_label = SCENARIO_LABELS[scenario]
    y_label = DELTA_MODE_LABELS[delta_mode]
    delta_title = DELTA_MODE_TITLES[delta_mode]

    algo_colors = {
        algo: ALGO_COLORS.get(algo, plt.cm.tab10(i % 10))
        for i, algo in enumerate(TARGET_ALGOS)
    }
    dataset_markers = {
        ds: m
        for ds, m in zip(
            TARGET_DATASETS,
            ["o", "s", "^", "D", "P", "X"],
        )
    }

    points = []
    for dataset in TARGET_DATASETS:
        for algo in TARGET_ALGOS:
            row = result[(result["dataset"] == dataset) & (result["algorithm"] == algo)]
            if row.empty:
                continue
            y = float(row.iloc[0][metric_col])
            if np.isfinite(y):
                points.append((algo, dataset, y))

    if not points:
        print(f"No valid points to plot for scenario '{scenario}'.")
        return

    all_y = [p[2] for p in points]
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    y_span = y_max - y_min
    y_pad = 0.08 * y_span if y_span > 0 else 0.05
    y_low = y_min - y_pad
    y_high = y_max + y_pad

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.4), dpi=dpi)

    for x_sep in np.arange(1.5, len(TARGET_ALGOS) + 0.5, 1.0):
        ax.axvline(x_sep, color="0.90", lw=1.1, zorder=0)

    for i, algo in enumerate(TARGET_ALGOS, start=1):
        vals = []
        for _, dataset, y in [p for p in points if p[0] == algo]:
            vals.append(y)
            ax.scatter(
                i,
                y,
                s=95,
                marker=dataset_markers[dataset],
                c=[algo_colors[algo]],
                edgecolors="black",
                linewidths=0.8,
                alpha=0.96,
                zorder=3,
            )
        if vals:
            y_mean = float(np.nanmean(vals))
            ax.hlines(
                y_mean,
                i + min(DATASET_OFFSETS.values()),
                i + max(DATASET_OFFSETS.values()),
                colors="black",
                linewidth=3.0,
                zorder=4,
            )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.35)
    ax.set_xlim(0.5, len(TARGET_ALGOS) + 0.5)
    ax.set_ylim(y_low, y_high)
    ax.set_xticks(range(1, len(TARGET_ALGOS) + 1))
    ax.set_xticklabels(TARGET_ALGOS, rotation=0, ha="center")
    ax.tick_params(axis="x", which="major", pad=6, length=0)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    # ax.set_xlabel("FNLL method", fontsize=11)
    ax.set_ylabel(
        rf"{y_label}(clean, {scenario})",
        fontsize=11,
    )
    ax.set_title(
        f"{delta_title.capitalize()} for noisy training scenario: {scenario_label}",
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
        bbox_to_anchor=(1.12, 0.94),
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path.resolve()}")


def _plot_single_scenario_dot_on_axis(
    ax,
    result: pd.DataFrame,
    bootstrap_vectors: Dict[tuple, np.ndarray],
    scenario: str,
    delta_mode: str,
) -> None:
    """Render one single-scenario dot plot onto an existing axis."""
    from matplotlib.lines import Line2D

    if delta_mode not in DELTA_MODE_COLUMNS:
        raise ValueError("delta_mode must be one of: 'rel_eff', 'abs'")
    if scenario not in SCENARIO_LABELS:
        raise ValueError("scenario must be one of: 'roa', 'roc', 'noisy'")

    metric_col = DELTA_MODE_COLUMNS[delta_mode][scenario]
    scenario_label = SCENARIO_LABELS[scenario]
    y_label = DELTA_MODE_LABELS[delta_mode]

    algo_colors = {
        algo: ALGO_COLORS.get(algo, plt.cm.tab10(i % 10))
        for i, algo in enumerate(TARGET_ALGOS)
    }
    dataset_markers = {
        ds: m
        for ds, m in zip(
            TARGET_DATASETS,
            ["o", "s", "^", "D", "P", "X"],
        )
    }

    points = []
    bootstrap_points = []
    for dataset in TARGET_DATASETS:
        for algo in TARGET_ALGOS:
            bootstrap_deltas = compute_bootstrap_delta_vector(
                bootstrap_vectors,
                algo,
                dataset,
                scenario,
                delta_mode,
            )
            if bootstrap_deltas.size > 0:
                bootstrap_points.append((algo, dataset, bootstrap_deltas))

            row = result[(result["dataset"] == dataset) & (result["algorithm"] == algo)]
            if row.empty:
                continue
            y = float(row.iloc[0][metric_col])
            if np.isfinite(y):
                points.append((algo, dataset, y))

    if not points and not bootstrap_points:
        ax.text(0.5, 0.5, f"No valid {scenario_label} points", ha="center", va="center", transform=ax.transAxes)
        return

    all_y = [p[2] for p in points]
    for _, _, values in bootstrap_points:
        all_y.extend(values.tolist())
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    y_span = y_max - y_min
    y_pad = 0.08 * y_span if y_span > 0 else 0.05
    y_low = y_min - y_pad
    y_high = y_max + y_pad

    for x_sep in np.arange(1.5, len(TARGET_ALGOS) + 0.5, 1.0):
        ax.axvline(x_sep, color="0.90", lw=1.1, zorder=0)

    for i, algo in enumerate(TARGET_ALGOS, start=1):
        for _, dataset, bootstrap_deltas in [p for p in bootstrap_points if p[0] == algo]:
            x = i + DATASET_OFFSETS[dataset]
            ax.scatter(
                np.full(bootstrap_deltas.shape, x, dtype=float),
                bootstrap_deltas,
                s=SEPARATE_DOT_BOOTSTRAP_MARKER_SIZE,
                c="lightgray",
                alpha=0.13,
                edgecolors="none",
                zorder=1,
            )

    for i, algo in enumerate(TARGET_ALGOS, start=1):
        vals = []
        for _, dataset, y in [p for p in points if p[0] == algo]:
            vals.append(y)
            x = i + DATASET_OFFSETS[dataset]
            ax.scatter(
                x,
                y,
                s=SEPARATE_DOT_MARKER_SIZE,
                marker=dataset_markers[dataset],
                c=[algo_colors[algo]],
                edgecolors="black",
                linewidths=1.1,
                alpha=0.96,
                zorder=3,
            )
        if vals:
            y_mean = float(np.nanmean(vals))
            ax.hlines(
                y_mean,
                i + min(DATASET_OFFSETS.values()),
                i + max(DATASET_OFFSETS.values()),
                colors="black",
                linewidth=SEPARATE_DOT_MEAN_LINEWIDTH,
                zorder=4,
            )

    ax.axhline(0, color="black", linewidth=1.3, linestyle="-", alpha=0.38)
    ax.set_xlim(0.35, len(TARGET_ALGOS) + 0.65)
    ax.set_ylim(-0.1, y_high)
    ax.set_xticks(range(1, len(TARGET_ALGOS) + 1))
    ax.set_xticklabels(TARGET_ALGOS, rotation=0, ha="center", fontsize=SEPARATE_DOT_TICK_SIZE)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=SEPARATE_DOT_TICK_SIZE,
        width=SEPARATE_DOT_AXIS_LINEWIDTH,
    )
    ax.tick_params(axis="x", which="major", pad=8, length=0)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(SEPARATE_DOT_AXIS_LINEWIDTH)

    # ax.set_xlabel("FNLL method", fontsize=SEPARATE_DOT_LABEL_SIZE)
    ax.set_ylabel(
        rf"{y_label}(clean, $\mathbf{{{scenario}}}$)",
        fontsize=SEPARATE_DOT_LABEL_SIZE,
    )
    # ax.set_title(
    #     f"Noise scenario: {scenario_label}",
    #     fontsize=SEPARATE_DOT_PANEL_TITLE_SIZE,
    #     fontweight="bold",
    #     pad=14,
    # )
    ax.grid(axis="y", alpha=0.34, linestyle="--", linewidth=1.0)
    ax.grid(axis="x", alpha=0.22, linestyle=":", linewidth=0.9)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=dataset_markers[d],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=15,
            linewidth=0,
            label=DATASET_TICK_LABELS.get(d, d),
        )
        for d in TARGET_DATASETS
    ]
    mean_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=SEPARATE_DOT_MEAN_LINEWIDTH,
        label="Mean across datasets",
    )
    bootstrap_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="lightgray",
        markeredgecolor="lightgray",
        markersize=7,
        label="Bootstrap deltas",
    )
    return dataset_handles, mean_handle, bootstrap_handle


def plot_separate_scenarios_side_by_side(
    result: pd.DataFrame,
    output_path: Path,
    bootstrap_vectors: Dict[tuple, np.ndarray],
    delta_mode: str = "rel_eff",
    dpi: int = 200,
) -> None:
    """Create one figure with roa, roc, and noisy shown in side-by-side panels."""
    if delta_mode not in DELTA_MODE_COLUMNS:
        raise ValueError("delta_mode must be one of: 'rel_eff', 'abs'")

    scenarios = ["roa", "roc", "noisy"]
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=SEPARATE_DOT_FIGSIZE,
        dpi=dpi,
        sharey=False,
    )
    legend_handles = None

    for ax, scenario in zip(axes, scenarios):
        handles = _plot_single_scenario_dot_on_axis(
            ax,
            result,
            bootstrap_vectors,
            scenario,
            delta_mode,
        )
        if handles is not None:
            legend_handles = handles

    y_lims = [ax.get_ylim() for ax in axes]
    y_low = min(lim[0] for lim in y_lims)
    y_high = max(lim[1] for lim in y_lims)
    for ax in axes:
        ax.set_ylim(y_low, y_high)

    # fig.suptitle(
    #     f"Robustness of FNLL methods to noisy training scenarios "
    #     f"({DELTA_MODE_TITLES[delta_mode]})",
    #     fontsize=SEPARATE_DOT_TITLE_SIZE,
    #     y=0.985,
    # )

    fig.tight_layout(rect=[0.02, 0.03, 0.905, 0.93])
    if legend_handles is not None:
        dataset_handles, mean_handle, bootstrap_handle = legend_handles
        right_ax_pos = axes[-1].get_position()
        fig.legend(
            handles=dataset_handles,  # + [mean_handle, bootstrap_handle],
            loc="upper left",
            bbox_to_anchor=(right_ax_pos.x1 + 0.006, right_ax_pos.y1),
            bbox_transform=fig.transFigure,
            borderaxespad=0.0,
            framealpha=0.95,
            fontsize=SEPARATE_DOT_LEGEND_SIZE,
            title_fontsize=SEPARATE_DOT_LEGEND_SIZE,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare roa(p), roc(p), and noisy training scenarios by computing\n"
            "  delta_abs  = Dice(clean) - Dice(v)  [from bootstrap values]\n"
            "  delta_rel_eff = delta_abs / p_eff\n"
            "per dataset and algorithm."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help=(
            "Generate figure: scatter_plot, paired_dot, separate_dot "
            "(one figure with roa, roc, and noisy side by side), no_figure."
        ),
    )
    parser.add_argument(
        "--delta-mode",
        type=str,
        choices=sorted(DELTA_MODE_COLUMNS),
        default="rel_eff",
        help=(
            "Metric shown in separate_dot plots: "
            "'rel_eff' = (Dice(clean)-Dice(scenario))/p_eff, "
            "'abs' = Dice(clean)-Dice(scenario)."
        ),
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

    # ------------------------------------------------------------------
    # 1. Load sheet to get structure and experiment IDs
    # ------------------------------------------------------------------
    df = load_sheet()

    # ------------------------------------------------------------------
    # 2. Build checkpoint index
    # ------------------------------------------------------------------
    print(f"Building checkpoint index...")
    all_exp_paths = build_experiment_path_index(nnunet_results_roots)
    print(f"Found {len(all_exp_paths)} checkpoint paths in folds {INCLUDED_FOLDS}.")

    # ------------------------------------------------------------------
    # 3. Load bootstrap Dice values
    # ------------------------------------------------------------------
    print(f"Loading bootstrap Dice vectors...")
    bootstrap_vectors = load_bootstrap_dice_per_cell(
        df, all_exp_paths, nnunet_results_roots
    )
    print(
        f"Loaded bootstrap vectors for {len(bootstrap_vectors)} (algo, dataset, noise) cells."
    )

    bootstrap_dice = aggregate_bootstrap_vectors_to_mean_dice(bootstrap_vectors)

    # ------------------------------------------------------------------
    # 4. Print p_eff values for reference
    # ------------------------------------------------------------------
    print()
    print("Effective noise ratios used:")
    print(f"  p^{{roa}}_{{eff}} = {P_ROA_EFF:.4f}  (same for all datasets)")
    print(f"  p^{{noisy}}_{{eff}} = {P_NOISY_EFF:.4f}  (same for all datasets)")
    for ds in TARGET_DATASETS:
        p_roc = P_ROC_EFF.get(ds, float("nan"))
        roc_nom = ROC_NOMINAL_X.get(ds, "?")
        print(f"  p^{{roc}}_{{eff}}({ds:12s}, nominal roc({roc_nom}%)) = {p_roc:.4f}")
    print()

    # ------------------------------------------------------------------
    # 5. Build comparison table
    # ------------------------------------------------------------------
    result = build_comparison_table(bootstrap_dice)

    # ------------------------------------------------------------------
    # 6. Print
    # ------------------------------------------------------------------
    print_table(result)

    # ------------------------------------------------------------------
    # 7. Save CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / "partial_noise_comparison.csv"
    result.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved CSV: {csv_path.resolve()}")

    # ------------------------------------------------------------------
    # 8. Figure
    # ------------------------------------------------------------------
    if args.figure == "scatter_plot":
        fig_path = out_dir / "partial_noise_comparison.png"
        plot_comparison_scatter(result, fig_path, bootstrap_vectors=bootstrap_vectors)
    elif args.figure == "paired_dot":
        fig_path = out_dir / "partial_noise_comparison_paired_dot.png"
        plot_comparison_paired_dot(
            result, fig_path, bootstrap_vectors=bootstrap_vectors
        )
    elif args.figure == "separate_dot":
        fig_path = out_dir / f"robustness_analysis_clean_noise_scenarios_separate_dot_{args.delta_mode}.png"
        plot_separate_scenarios_side_by_side(
            result=result,
            output_path=fig_path,
            bootstrap_vectors=bootstrap_vectors,
            delta_mode=args.delta_mode,
        )
    elif args.figure == "no_figure":
        print("No figure generated (as per --figure=no_figure).")
    else:
        print(f"Unknown figure option: {args.figure!r}. No figure generated.")


if __name__ == "__main__":
    main()
