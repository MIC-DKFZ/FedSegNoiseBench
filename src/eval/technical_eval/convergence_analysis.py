"""Collect and analyze validation-Dice convergence curves from nnU-Net logs.

Only experiments listed in the segmentation-results Google Sheet are included.
An experiment can have several client checkpoint directories and each directory
can contain several training logs after restarts.  Logs belonging to the same
checkpoint directory are merged by epoch before the CSV is written.

Client curves are then averaged into one curve per sheet experiment. For every
dataset/algorithm combination, the analysis reports when each experiment first
reaches 85%, 90%, and 95% of its own final validation Dice and summarizes the
threshold epochs across experiments. It also reports raw, duration-normalized,
and final-Dice-normalized Dice AUC. One convergence plot is created per dataset.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHEET_ID = "1AP_KH1cVSDwgpI1n7qK_VZU0Vi19Wh8vKo4jYWkuIXg"
SHEET_GID = "332656109"
SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    f"?format=csv&gid={SHEET_GID}"
)

RESULTS_ROOTS = (
    Path("/home/m391k/cluster-data/checkpoints/nnUNet_results"),
    Path("/home/m391k/juwels/checkpoints/nnUNet_results"),
)
OUTPUT_CSV = Path("results/segmentation_results/convergence_dice_curves.csv")
PER_EXPERIMENT_OUTPUT_CSV = Path(
    "results/segmentation_results/convergence_thresholds_per_experiment.csv"
)
SUMMARY_OUTPUT_CSV = Path(
    "results/segmentation_results/convergence_thresholds_summary.csv"
)
AUC_SUMMARY_OUTPUT_CSV = Path(
    "results/segmentation_results/convergence_auc_summary.csv"
)
LATEX_TABLE_OUTPUT = Path(
    "results/segmentation_results/convergence_speed_table.tex"
)
PLOT_DIR = Path("results/segmentation_results/convergence_plots")

THRESHOLDS = (("85%", 0.85), ("90%", 0.90), ("95%", 0.95))
ALGORITHM_ORDER = ("FedAvg", "FedA3I", "IOP-FL", "FedCorr", "FedSelect")
DATASET_ORDER = ("LIDC", "RIGA", "Gleason", "MouseTumor", "MMIS", "MMIA")
DATASET_LABELS = {
    "LIDC": "LIDC",
    "RIGA": "RIGA",
    "Gleason": "GleasonHD",
    "MouseTumor": "MouseT",
    "MMIS": "MMIS",
    "MMIA": "MMIA",
}
ALGORITHM_COLORS = {
    "FedAvg": "#4C72B0",
    "FedA3I": "#DD8452",
    "IOP-FL": "#55A868",
    "FedCorr": "#C44E52",
    "FedSelect": "#8172B2",
}

ALGO_COL = "Algo"
DATASET_COL = "Data"
NOISE_COL = "Noise"
EXPERIMENT_ID_COL = "Experiment ID"
TARGET_ALGOS = {"FedAvg", "FedA3I", "IOP-FL", "FedCorr", "FedSelect"}
TARGET_DATASETS = {"LIDC", "RIGA", "Gleason", "MouseTumor", "MMIA", "MMIS"}
INCLUDED_FOLDS = {0, 1, 2}

EPOCH_RE = re.compile(r"\bEpoch\s+(\d+)\s*$")
BEST_DICE_RE = re.compile(
    r"Yayy!\s+New best EMA pseudo Dice:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
FOLD_RE = re.compile(r"(?:^|[_/])fold[_-]?(\d+)(?:[_/]|$)", re.IGNORECASE)
CLIENT_RE = re.compile(r"client[_-]?(\d+)", re.IGNORECASE)

CSV_COLUMNS = [
    "experiment_id",
    "algorithm",
    "dataset",
    "noise",
    "fold",
    "client_id",
    "epoch",
    "best_ema_pseudo_dice",
    "results_root",
    "checkpoint_dir",
    "source_logs",
]


@dataclass(frozen=True)
class DiceAchievement:
    epoch: int
    score: float
    source_log: Path


def normalize_algorithm(value: object) -> str:
    raw = str(value).strip()
    compact = re.sub(r"[\s_-]+", "", raw.lower())
    return {
        "fedavg": "FedAvg",
        "feda3i": "FedA3I",
        "iopfl": "IOP-FL",
        "fedcorr": "FedCorr",
        "fedselect": "FedSelect",
    }.get(compact, raw)


def normalize_dataset(value: object) -> str:
    raw = str(value).strip()
    lower = raw.lower()
    if "lidc" in lower:
        return "LIDC"
    if "riga" in lower:
        return "RIGA"
    if "gleason" in lower:
        return "Gleason"
    if "mousetumor" in lower or "mouse tumor" in lower:
        return "MouseTumor"
    if "mmia" in lower:
        return "MMIA"
    if "mmis" in lower:
        return "MMIS"
    return raw


def normalize_noise(value: object) -> str | None:
    raw = str(value).strip()
    if re.fullmatch(r"0(?:\.0+)?", raw):
        return "clean"
    if re.fullmatch(r"100(?:\.0+)?", raw):
        return "noisy"
    if re.search(r"\broa\b", raw, re.IGNORECASE):
        return "roa"
    if re.search(r"\broc\b", raw, re.IGNORECASE):
        return "roc"
    return None


def extract_fold(value: object) -> int | None:
    match = FOLD_RE.search(str(value))
    return int(match.group(1)) if match else None


def load_sheet_records(sheet_source: str | Path = SHEET_CSV_URL) -> pd.DataFrame:
    """Load the same sheet subset used by the segmentation-results scripts."""
    df = pd.read_csv(sheet_source)
    required = [ALGO_COL, DATASET_COL, NOISE_COL, EXPERIMENT_ID_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Google Sheet is missing required columns: {missing}")

    records = df[required].copy()
    records["algorithm"] = records[ALGO_COL].map(normalize_algorithm)
    records["dataset"] = records[DATASET_COL].map(normalize_dataset)
    records["noise"] = records[NOISE_COL].map(normalize_noise)
    records["experiment_id"] = records[EXPERIMENT_ID_COL].astype(str).str.strip()
    records["fold"] = records["experiment_id"].map(extract_fold)

    records = records[
        records["algorithm"].isin(TARGET_ALGOS)
        & records["dataset"].isin(TARGET_DATASETS)
        & records["noise"].notna()
        & records["fold"].isin(INCLUDED_FOLDS)
        & records["experiment_id"].ne("")
        & records["experiment_id"].str.lower().ne("nan")
    ]
    return records[
        ["experiment_id", "algorithm", "dataset", "noise", "fold"]
    ].drop_duplicates()


def build_checkpoint_index(results_roots: Sequence[Path]) -> dict[str, list[Path]]:
    """Index experiment checkpoint directories by their directory name."""
    index: dict[str, list[Path]] = defaultdict(list)
    for root in results_roots:
        if not root.is_dir():
            print(f"Warning: results root does not exist: {root}")
            continue
        for checkpoint_dir in root.glob("*/*/fold_*/*"):
            if checkpoint_dir.is_dir():
                index[checkpoint_dir.name].append(checkpoint_dir)
    return index


def find_checkpoint_dirs(
    experiment_id: str, checkpoint_index: dict[str, list[Path]]
) -> list[Path]:
    """Match exactly first and retain substring matching for legacy sheet IDs."""
    exact = checkpoint_index.get(experiment_id, [])
    if exact:
        return sorted(set(exact), key=str)
    return sorted(
        {
            path
            for directory_name, paths in checkpoint_index.items()
            if experiment_id in directory_name
            for path in paths
        },
        key=str,
    )


def parse_training_log(log_path: Path) -> list[DiceAchievement]:
    """Extract new-best scores and the latest epoch marker preceding each one."""
    achievements: list[DiceAchievement] = []
    current_epoch: int | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            epoch_match = EPOCH_RE.search(line.rstrip())
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            score_match = BEST_DICE_RE.search(line)
            if not score_match:
                continue
            if current_epoch is None:
                print(
                    f"Warning: ignoring Dice achievement without an epoch in "
                    f"{log_path}:{line_number}"
                )
                continue
            achievements.append(
                DiceAchievement(current_epoch, float(score_match.group(1)), log_path)
            )
    return achievements


def merge_restart_logs(log_paths: Iterable[Path]) -> list[tuple[int, float, str]]:
    """Merge restart logs into a strictly improving best-Dice curve.

    Duplicate epochs can occur at restart boundaries. Their maximum score is
    used and all contributing log names are retained. Achievements below an
    already-observed best are discarded because they result from a restarted
    logger rather than a new best for the complete experiment.
    """
    by_epoch: dict[int, list[DiceAchievement]] = defaultdict(list)
    for log_path in sorted(set(log_paths), key=str):
        for achievement in parse_training_log(log_path):
            by_epoch[achievement.epoch].append(achievement)

    merged: list[tuple[int, float, str]] = []
    previous_best = float("-inf")
    for epoch in sorted(by_epoch):
        events = by_epoch[epoch]
        score = max(event.score for event in events)
        if score <= previous_best:
            continue
        source_logs = ";".join(
            sorted({event.source_log.name for event in events if event.score == score})
        )
        merged.append((epoch, score, source_logs))
        previous_best = score
    return merged


def find_results_root(checkpoint_dir: Path, results_roots: Sequence[Path]) -> Path:
    for root in results_roots:
        try:
            checkpoint_dir.relative_to(root)
            return root
        except ValueError:
            pass
    raise ValueError(f"{checkpoint_dir} is not below a configured results root")


def extract_client_id(checkpoint_dir: Path) -> str:
    for part in checkpoint_dir.parts:
        match = CLIENT_RE.search(part)
        if match:
            return match.group(1)
    # Some datasets encode sites only in the DatasetNNN directory. Keeping the
    # dataset directory name still gives every client curve a stable identity.
    return checkpoint_dir.parts[-4]


def collect_curves(
    sheet_records: pd.DataFrame, results_roots: Sequence[Path]
) -> pd.DataFrame:
    checkpoint_index = build_checkpoint_index(results_roots)
    rows: list[dict[str, object]] = []
    missing_experiments: list[str] = []
    checkpoints_without_achievements = 0

    for record in sheet_records.itertuples(index=False):
        checkpoint_dirs = find_checkpoint_dirs(record.experiment_id, checkpoint_index)
        if not checkpoint_dirs:
            missing_experiments.append(record.experiment_id)
            continue

        for checkpoint_dir in checkpoint_dirs:
            log_paths = list(checkpoint_dir.glob("training_log_*.txt"))
            curve = merge_restart_logs(log_paths)
            if not curve:
                checkpoints_without_achievements += 1
                continue

            results_root = find_results_root(checkpoint_dir, results_roots)
            for epoch, score, source_logs in curve:
                rows.append(
                    {
                        "experiment_id": record.experiment_id,
                        "algorithm": record.algorithm,
                        "dataset": record.dataset,
                        "noise": record.noise,
                        "fold": int(record.fold),
                        "client_id": extract_client_id(checkpoint_dir),
                        "epoch": epoch,
                        "best_ema_pseudo_dice": score,
                        "results_root": str(results_root),
                        "checkpoint_dir": str(checkpoint_dir),
                        "source_logs": source_logs,
                    }
                )

    if missing_experiments:
        print(
            f"Warning: no checkpoint directory found for "
            f"{len(set(missing_experiments))} sheet experiments."
        )
    if checkpoints_without_achievements:
        print(
            "Warning: no parseable best-Dice achievements found in "
            f"{checkpoints_without_achievements} checkpoint directories."
        )

    curves = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if not curves.empty:
        curves = curves.sort_values(
            ["algorithm", "dataset", "noise", "experiment_id", "client_id", "epoch"]
        ).reset_index(drop=True)
    return curves


EXPERIMENT_GROUP_COLUMNS = [
    "experiment_id",
    "algorithm",
    "dataset",
    "noise",
    "fold",
]


def build_experiment_curves(client_curves: pd.DataFrame) -> pd.DataFrame:
    """Average stepwise best-Dice client curves within each experiment.

    A client's best score remains valid until it improves, so sparse achievement
    rows are forward-filled. Clients that finish earlier retain their final best
    score until the last epoch observed for that experiment.
    """
    required = set(EXPERIMENT_GROUP_COLUMNS + ["checkpoint_dir", "epoch", "best_ema_pseudo_dice"])
    missing = sorted(required.difference(client_curves.columns))
    if missing:
        raise ValueError(f"Convergence curves CSV is missing columns: {missing}")

    experiment_frames: list[pd.DataFrame] = []
    for group_values, experiment in client_curves.groupby(
        EXPERIMENT_GROUP_COLUMNS, sort=False, dropna=False
    ):
        max_epoch = int(experiment["epoch"].max())
        epoch_index = pd.RangeIndex(0, max_epoch + 1, name="epoch")
        client_series: list[pd.Series] = []

        for _, client in experiment.groupby("checkpoint_dir", sort=False):
            scores = (
                client.groupby("epoch")["best_ema_pseudo_dice"]
                .max()
                .sort_index()
                .reindex(epoch_index)
                .ffill()
            )
            client_series.append(scores)

        mean_curve = pd.concat(client_series, axis=1).mean(axis=1, skipna=True)
        frame = mean_curve.rename("dice").reset_index()
        for column, value in zip(EXPERIMENT_GROUP_COLUMNS, group_values):
            frame[column] = value
        experiment_frames.append(frame)

    columns = EXPERIMENT_GROUP_COLUMNS + ["epoch", "dice"]
    if not experiment_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(experiment_frames, ignore_index=True)[columns]


def analyze_threshold_epochs(
    experiment_curves: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate threshold epochs and Dice AUC convergence metrics.

    ``dice_auc`` has units of Dice x epoch. ``normalized_dice_auc`` divides by
    the observed training duration and is therefore the mean Dice over training.
    ``relative_dice_auc`` additionally divides by the experiment's final Dice;
    it measures how quickly the curve approaches its own final plateau and is
    the AUC metric most directly comparable to the relative epoch thresholds.
    """
    rows: list[dict[str, object]] = []
    for group_values, curve in experiment_curves.groupby(
        EXPERIMENT_GROUP_COLUMNS, sort=False, dropna=False
    ):
        curve = curve.dropna(subset=["dice"]).sort_values("epoch")
        if curve.empty:
            continue
        final_dice = float(curve.iloc[-1]["dice"])
        first_epoch = int(curve.iloc[0]["epoch"])
        final_epoch = int(curve.iloc[-1]["epoch"])
        duration = final_epoch - first_epoch
        dice_auc = float(np.trapz(curve["dice"], curve["epoch"]))
        normalized_dice_auc = dice_auc / duration if duration > 0 else np.nan
        relative_dice_auc = (
            normalized_dice_auc / final_dice
            if final_dice > 0 and np.isfinite(normalized_dice_auc)
            else np.nan
        )
        row = dict(zip(EXPERIMENT_GROUP_COLUMNS, group_values))
        row["final_validation_dice"] = final_dice
        row["first_epoch"] = first_epoch
        row["final_epoch"] = final_epoch
        row["dice_auc"] = dice_auc
        row["normalized_dice_auc"] = normalized_dice_auc
        row["relative_dice_auc"] = relative_dice_auc

        for label, fraction in THRESHOLDS:
            column = f"epoch_{label.replace('.', '_').replace('%', 'pct')}"
            reached = curve[curve["dice"] >= fraction * final_dice]
            row[column] = int(reached.iloc[0]["epoch"]) if not reached.empty else pd.NA
        rows.append(row)

    per_experiment = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    if not per_experiment.empty:
        for (dataset, algorithm), group in per_experiment.groupby(
            ["dataset", "algorithm"], sort=False
        ):
            for label, fraction in THRESHOLDS:
                column = f"epoch_{label.replace('.', '_').replace('%', 'pct')}"
                epochs = pd.to_numeric(group[column], errors="coerce").dropna()
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "threshold": label,
                        "threshold_fraction": fraction,
                        "mean_epoch": float(epochs.mean()) if not epochs.empty else np.nan,
                        "std_epoch": float(epochs.std(ddof=1)) if len(epochs) > 1 else 0.0,
                        "median_epoch": float(epochs.median()) if not epochs.empty else np.nan,
                        "min_epoch": int(epochs.min()) if not epochs.empty else pd.NA,
                        "max_epoch": int(epochs.max()) if not epochs.empty else pd.NA,
                        "n_experiments": int(len(epochs)),
                    }
                )
    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "dataset",
            "algorithm",
            "threshold",
            "threshold_fraction",
            "mean_epoch",
            "std_epoch",
            "median_epoch",
            "min_epoch",
            "max_epoch",
            "n_experiments",
        ],
    )
    auc_summary_rows: list[dict[str, object]] = []
    if not per_experiment.empty:
        for (dataset, algorithm), group in per_experiment.groupby(
            ["dataset", "algorithm"], sort=False
        ):
            auc_summary_rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "mean_dice_auc": float(group["dice_auc"].mean()),
                    "std_dice_auc": float(group["dice_auc"].std(ddof=1)),
                    "mean_normalized_dice_auc": float(
                        group["normalized_dice_auc"].mean()
                    ),
                    "std_normalized_dice_auc": float(
                        group["normalized_dice_auc"].std(ddof=1)
                    ),
                    "mean_relative_dice_auc": float(
                        group["relative_dice_auc"].mean()
                    ),
                    "std_relative_dice_auc": float(
                        group["relative_dice_auc"].std(ddof=1)
                    ),
                    "mean_final_validation_dice": float(
                        group["final_validation_dice"].mean()
                    ),
                    "mean_final_epoch": float(group["final_epoch"].mean()),
                    "n_experiments": int(len(group)),
                }
            )
    auc_summary = pd.DataFrame(
        auc_summary_rows,
        columns=[
            "dataset",
            "algorithm",
            "mean_dice_auc",
            "std_dice_auc",
            "mean_normalized_dice_auc",
            "std_normalized_dice_auc",
            "mean_relative_dice_auc",
            "std_relative_dice_auc",
            "mean_final_validation_dice",
            "mean_final_epoch",
            "n_experiments",
        ],
    )
    if not per_experiment.empty:
        algorithm_rank = {name: rank for rank, name in enumerate(ALGORITHM_ORDER)}
        dataset_rank = {name: rank for rank, name in enumerate(DATASET_ORDER)}
        per_experiment["_algorithm_rank"] = per_experiment["algorithm"].map(
            algorithm_rank
        )
        per_experiment["_dataset_rank"] = per_experiment["dataset"].map(dataset_rank)
        per_experiment = (
            per_experiment.sort_values(
                ["_dataset_rank", "_algorithm_rank", "noise", "fold", "experiment_id"]
            )
            .drop(columns=["_dataset_rank", "_algorithm_rank"])
            .reset_index(drop=True)
        )
    if not summary.empty:
        algorithm_rank = {name: rank for rank, name in enumerate(ALGORITHM_ORDER)}
        dataset_rank = {name: rank for rank, name in enumerate(DATASET_ORDER)}
        threshold_rank = {label: rank for rank, (label, _) in enumerate(THRESHOLDS)}
        summary["_algorithm_rank"] = summary["algorithm"].map(algorithm_rank)
        summary["_dataset_rank"] = summary["dataset"].map(dataset_rank)
        summary["_threshold_rank"] = summary["threshold"].map(threshold_rank)
        summary = (
            summary.sort_values(
                ["_dataset_rank", "_algorithm_rank", "_threshold_rank"]
            )
            .drop(columns=["_dataset_rank", "_algorithm_rank", "_threshold_rank"])
            .reset_index(drop=True)
        )
    if not auc_summary.empty:
        algorithm_rank = {name: rank for rank, name in enumerate(ALGORITHM_ORDER)}
        dataset_rank = {name: rank for rank, name in enumerate(DATASET_ORDER)}
        auc_summary["_algorithm_rank"] = auc_summary["algorithm"].map(algorithm_rank)
        auc_summary["_dataset_rank"] = auc_summary["dataset"].map(dataset_rank)
        auc_summary = (
            auc_summary.sort_values(["_dataset_rank", "_algorithm_rank"])
            .drop(columns=["_dataset_rank", "_algorithm_rank"])
            .reset_index(drop=True)
        )
    return per_experiment, summary, auc_summary


def average_algorithm_curve(curves: Sequence[pd.DataFrame]) -> pd.Series:
    """Average experiment curves, retaining final values of shorter runs."""
    max_epoch = max(int(curve["epoch"].max()) for curve in curves)
    epoch_index = pd.RangeIndex(0, max_epoch + 1, name="epoch")
    series = []
    for curve in curves:
        experiment_series = curve.set_index("epoch")["dice"].reindex(epoch_index).ffill()
        series.append(experiment_series)
    return pd.concat(series, axis=1).mean(axis=1, skipna=True)


def plot_all_datasets_convergence(
    experiment_curves: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    """Plot all dataset convergence panels with one shared algorithm legend."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    available_datasets = set(experiment_curves["dataset"].dropna().unique())
    datasets = [dataset for dataset in DATASET_ORDER if dataset in available_datasets]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5), constrained_layout=True)
    axes_flat = axes.ravel()

    for subplot_index, dataset in enumerate(datasets):
        ax = axes_flat[subplot_index]
        dataset_curves = experiment_curves[experiment_curves["dataset"] == dataset]

        for algorithm in ALGORITHM_ORDER:
            algorithm_data = dataset_curves[dataset_curves["algorithm"] == algorithm]
            if algorithm_data.empty:
                continue
            color = ALGORITHM_COLORS[algorithm]
            curves = [
                curve.sort_values("epoch")
                for _, curve in algorithm_data.groupby("experiment_id", sort=False)
            ]
            for curve in curves:
                ax.plot(
                    curve["epoch"],
                    curve["dice"],
                    color=color,
                    linewidth=0.7,
                    alpha=0.18,
                    zorder=1,
                )

            mean_curve = average_algorithm_curve(curves)
            valid_mean = mean_curve.dropna()
            ax.plot(
                valid_mean.index,
                valid_mean.values,
                color=color,
                linewidth=3.2,
                label=algorithm,
                zorder=3,
            )
        if subplot_index == len(datasets) - 1:
            ax.legend(
                title="Algorithm",
                loc="lower right",
                fontsize=15,
                title_fontsize=16,
                frameon=True,
                fancybox=True,
                framealpha=0.92,
                borderpad=0.7,
            )
        ax.set_title(DATASET_LABELS[dataset], fontsize=18, fontweight="semibold")
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Best EMA pseudo Dice", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        finite_dice = dataset_curves["dice"].dropna()
        if not finite_dice.empty:
            dice_min = float(finite_dice.min())
            dice_max = float(finite_dice.max())
            dice_span = dice_max - dice_min
            # Keep every curve visible while avoiding a mostly empty [0, 1]
            # axis for datasets whose Dice values occupy a narrow range.
            padding = max(0.01, 0.04 * dice_span)
            ax.set_ylim(
                bottom=max(0.0, dice_min - padding),
                top=min(1.0, dice_max + padding),
            )
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    for ax in axes_flat[len(datasets) :]:
        ax.set_visible(False)
    output_path = plot_dir / "all_datasets_convergence.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [output_path]


def _format_ranked_value(
    mean: float,
    std: float,
    rank: int,
    decimals: int,
) -> str:
    value = rf"{mean:.{decimals}f} \pm {std:.{decimals}f}"
    if rank == 1:
        return rf"\best{{{value}}}"
    if rank == 2:
        return rf"\secondbest{{{value}}}"
    return rf"${value}$"


def _rank_distinct(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Rank values while giving exact ties the same formatting rank."""
    distinct = sorted(values.dropna().unique(), reverse=higher_is_better)
    rank_by_value = {value: rank + 1 for rank, value in enumerate(distinct)}
    return values.map(rank_by_value)


def write_convergence_latex_table(
    threshold_summary: pd.DataFrame,
    auc_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write dataset blocks with metrics as rows and methods as columns."""
    selected_thresholds = threshold_summary[
        threshold_summary["threshold"].eq("95%")
    ]
    threshold_wide = selected_thresholds.pivot(
        index=["dataset", "algorithm"],
        columns="threshold",
        values=["mean_epoch", "std_epoch"],
    )
    threshold_wide.columns = [
        f"{metric}_{threshold}" for metric, threshold in threshold_wide.columns
    ]
    table = auc_summary.merge(threshold_wide.reset_index(), on=["dataset", "algorithm"])
    algorithm_rank = {name: rank for rank, name in enumerate(ALGORITHM_ORDER)}
    dataset_rank = {name: rank for rank, name in enumerate(DATASET_ORDER)}
    table["_algorithm_rank"] = table["algorithm"].map(algorithm_rank)
    table["_dataset_rank"] = table["dataset"].map(dataset_rank)
    table = table.sort_values(["_dataset_rank", "_algorithm_rank"]).drop(
        columns=["_dataset_rank", "_algorithm_rank"]
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Convergence speed per dataset and method. Values are reported as mean $\pm$ standard deviation over experiments. Dice AUC denotes relative Dice AUC normalized by training duration and final Dice. Higher AUC and fewer rounds to the milestone indicate faster convergence. Best results per dataset and metric are bold; second-best distinct results are underlined. Exact ties share the same formatting.}",
        r"\label{tab:convergence_speed}",
        "",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        "",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textit{Dataset} & \textit{Metric} & \textbf{FedAvg} & \textbf{FedA3I} & \textbf{IOP-FL} & \textbf{FedCorr} & \textbf{FedSelect} \\",
        r"\midrule",
        "",
    ]

    dataset_groups = list(table.groupby("dataset", sort=False))
    for dataset_index, (dataset, group) in enumerate(dataset_groups):
        group = group.copy()
        group["auc_rank"] = _rank_distinct(group["mean_relative_dice_auc"], True)
        group["epoch_95_rank"] = _rank_distinct(group["mean_epoch_95%"], False)
        group = group.set_index("algorithm")
        metric_specs = [
            ("Round@95\\% $\\downarrow$", "mean_epoch_95%", "std_epoch_95%", "epoch_95_rank", 1),
            ("Dice AUC $\\uparrow$", "mean_relative_dice_auc", "std_relative_dice_auc", "auc_rank", 3),
        ]
        for metric_index, (label, mean_col, std_col, rank_col, decimals) in enumerate(
            metric_specs
        ):
            dataset_cell = (
                rf"\multirow{{2}}{{*}}{{{DATASET_LABELS[dataset]}}}"
                if metric_index == 0
                else ""
            )
            values = [
                _format_ranked_value(
                    group.loc[algorithm, mean_col],
                    group.loc[algorithm, std_col],
                    group.loc[algorithm, rank_col],
                    decimals,
                )
                for algorithm in ALGORITHM_ORDER
            ]
            lines.append(
                f"{dataset_cell} & {label} & " + " & ".join(values) + r" \\"
            )
        if dataset_index < len(dataset_groups) - 1:
            lines.append(r"\midrule")
            lines.append("")

    lines.extend(
        [
            "",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    curves: pd.DataFrame,
    per_experiment_output: Path,
    summary_output: Path,
    auc_summary_output: Path,
    latex_table_output: Path,
    plot_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    experiment_curves = build_experiment_curves(curves)
    per_experiment, summary, auc_summary = analyze_threshold_epochs(experiment_curves)
    per_experiment_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    auc_summary_output.parent.mkdir(parents=True, exist_ok=True)
    per_experiment.to_csv(per_experiment_output, index=False)
    summary.to_csv(summary_output, index=False)
    auc_summary.to_csv(auc_summary_output, index=False)
    write_convergence_latex_table(summary, auc_summary, latex_table_output)
    plots = plot_all_datasets_convergence(experiment_curves, plot_dir)
    return per_experiment, summary, auc_summary, plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        action="append",
        type=Path,
        dest="results_roots",
        help="nnUNet_results root; repeat to use multiple roots (defaults to both configured roots).",
    )
    parser.add_argument(
        "--sheet-csv",
        default=SHEET_CSV_URL,
        help="Google Sheet CSV URL or a local CSV file (useful for offline runs).",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--reuse-curves",
        action="store_true",
        help="Skip log collection and analyze the existing --output curves CSV.",
    )
    parser.add_argument(
        "--per-experiment-output", type=Path, default=PER_EXPERIMENT_OUTPUT_CSV
    )
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT_CSV)
    parser.add_argument(
        "--auc-summary-output", type=Path, default=AUC_SUMMARY_OUTPUT_CSV
    )
    parser.add_argument("--latex-table-output", type=Path, default=LATEX_TABLE_OUTPUT)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_curves:
        if not args.output.is_file():
            raise FileNotFoundError(f"Cannot reuse missing curves CSV: {args.output}")
        curves = pd.read_csv(args.output)
        print(f"Loaded {len(curves)} existing Dice achievements from {args.output}.")
    else:
        results_roots = tuple(args.results_roots or RESULTS_ROOTS)
        sheet_records = load_sheet_records(args.sheet_csv)
        print(f"Loaded {len(sheet_records)} valid experiment records from the sheet.")
        curves = collect_curves(sheet_records, results_roots)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(args.output, index=False)
        print(
            f"Saved {len(curves)} Dice achievements from "
            f"{curves['experiment_id'].nunique() if not curves.empty else 0} experiments "
            f"to {args.output}."
        )

    per_experiment, summary, auc_summary, plots = run_analysis(
        curves,
        args.per_experiment_output,
        args.summary_output,
        args.auc_summary_output,
        args.latex_table_output,
        args.plot_dir,
    )
    print(
        f"Saved threshold epochs for {len(per_experiment)} experiments to "
        f"{args.per_experiment_output}."
    )
    print(f"Saved {len(summary)} threshold summaries to {args.summary_output}.")
    print(f"Saved {len(auc_summary)} Dice AUC summaries to {args.auc_summary_output}.")
    print(f"Saved convergence LaTeX table to {args.latex_table_output}.")
    print(f"Saved {len(plots)} dataset convergence plots to {args.plot_dir}.")


if __name__ == "__main__":
    main()
