import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob
import numpy as np

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
sheet_id = "1AP_KH1cVSDwgpI1n7qK_VZU0Vi19Wh8vKo4jYWkuIXg"
gid = "332656109"  # use appropriate gid for the sheet tab (0 is usually the first)
csv_url = (
    f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
)

algo_col = "Algo"
raw_dataset_col = "Data"
noise_col = "Noise"
metric_col = "Mean(D_val)"

target_algos = ["FedAvg", "FedA3I", "IOP-FL", "FedCorr", "FedSelect"]
target_datasets = ["LIDC", "RIGA", "Gleason", "MouseTumor", "MMIA", "MMIS"]
noise_order = ["0", "roa(p)", "roc(p)", "100"]  # plotting order

# Include only results from these folds (fold numbers: 0, 1, 2, 3, 4)
included_folds = [0, 1, 2]

OUTPUT_DIR = Path("./results/segmentation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# for reading results from experiment's summary.json
nnUNet_results_roots = [
    Path("/home/m391k/cluster-data/checkpoints/nnUNet_results"),
    Path("/home/m391k/juwels/checkpoints/nnUNet_results"),
]
nnUNet_results_all_exps_raw = []
for nnUNet_results in nnUNet_results_roots:
    nnUNet_results_all_exps_raw.extend(
        glob.glob(str(nnUNet_results / "*" / "*" / "fold_*" / "*"))
    )

nnUNet_results_all_exps_raw = sorted(set(nnUNet_results_all_exps_raw))
# Filter by included_folds
nnUNet_results_all_exps = [
    p
    for p in nnUNet_results_all_exps_raw
    if any(f"/fold_{fold}/" in p for fold in included_folds)
]
dataset_to_numclients = {
    "LIDC": 4,
    "RIGA": 3,
    "Gleason": 3,
    "MouseTumor": 5,
    "MMIA": 4,
    "MMIS": 4,
}
# Clean clients per dataset (for distinguishing in roc(p) plots)
clean_clients_per_dataset = {
    "LIDC": [0, 1],
    "RIGA": [0],
    "Gleason": [0],
    "MouseTumor": [0, 1],
    "MMIA": [0, 1],
    "MMIS": [0, 1],
}
classwise_metric = "Dice"  # metric to extract per class
class_line_marker = {
    1: {
        "linestyle": "-",
        "marker": "o",
    },
    2: {
        "linestyle": "--",
        "marker": "s",
    },
    3: {
        "linestyle": ":",
        "marker": "v",
    },
}

# for boxplots
boxplot_clean_color = "#4c72b0"
boxplot_noisy_color = "#dd8452"
SUPPORTED_CLASSWISE_METRICS = ("Dice", "HD95", "InstanceF1", "ClassConfusion")
BOUNDED_ZERO_ONE_METRICS = {"Dice", "InstanceF1", "ClassConfusion"}
BOXPLOT_TITLE_FONTSIZE = 16
BOXPLOT_LABEL_FONTSIZE = 14
BOXPLOT_TICK_FONTSIZE = 14
BOXPLOT_DATASET_FONTSIZE = 14
BOXPLOT_LEGEND_FONTSIZE = 14


def get_exp_paths_with_bootstrap(exp_id: str) -> list[str]:
    return [
        p
        for p in nnUNet_results_all_exps
        if exp_id in p
        and (Path(p) / "validation" / "bootstrap_evaluation_results.json").is_file()
    ]


def parse_selected_datasets(datasets_arg):
    if not datasets_arg:
        return list(target_datasets)

    def _normalize_name(name: str) -> str:
        return re.sub(r"[\s_\-]+", "", str(name).strip().lower())

    canonical_map = {_normalize_name(d): d for d in target_datasets}
    selected = []
    unknown = []

    for item in datasets_arg:
        parts = [p for p in re.split(r"[,\s]+", str(item).strip()) if p]
        for part in parts:
            canonical = canonical_map.get(_normalize_name(part))
            if canonical is None:
                unknown.append(part)
                continue
            if canonical not in selected:
                selected.append(canonical)

    if unknown:
        raise ValueError(
            f"Unknown dataset names in --datasets: {unknown}. Allowed values: {target_datasets}"
        )
    if not selected:
        raise ValueError(
            f"No valid datasets selected via --datasets. Allowed values: {target_datasets}"
        )
    return selected


def metric_slug(metric_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", metric_name.strip().lower()).strip("_")


def apply_metric_axis_limits(ax):
    if classwise_metric in BOUNDED_ZERO_ONE_METRICS:
        ax.set_ylim(0.0, 1.0)
    elif classwise_metric == "HD95":
        ax.set_yscale("log")


def extract_finite_metric_values(raw_values):
    """Return a list of finite floats from a scalar or list-like metric entry."""
    if raw_values is None:
        return []
    if isinstance(raw_values, list):
        seq = raw_values
    else:
        seq = [raw_values]

    finite_values = []
    for value in seq:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            finite_values.append(value)
    return finite_values


# -------------------------------------------------------------------
# Load and pre-process
# -------------------------------------------------------------------
df = pd.read_csv(csv_url)
print(f"Loaded {len(df)} rows from Google Sheets.")

# Basic cleaning for algorithm and dataset columns
for c in [algo_col, raw_dataset_col]:
    df[c] = df[c].astype(str).str.strip()


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


df[algo_col] = df[algo_col].apply(normalize_algorithm)

# Clean metric column: replace comma decimal separators with dot, remove percent signs/spaces, and convert to float.
# This ensures values like "0,01" become 0.01 and allows mean() to average across folds.
df[metric_col] = (
    df[metric_col]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .str.replace("%", "", regex=False)
    .str.strip()
)
df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")


# Normalize noise values into canonical buckets:
# - exact 0/100 -> "0"/"100"
# - any roa(...) or contains "roa" -> "roa(p)"
# - any roc(...) or contains "roc" -> "roc(p)"
def _normalize_noise_val(v: object) -> str:
    s = "" if v is None else str(v).strip()
    if re.fullmatch(r"(?i)0(?:\.0+)?", s):
        return "0"
    if re.fullmatch(r"(?i)100(?:\.0+)?", s):
        return "100"
    if re.search(r"(?i)\broa\b", s) or re.search(r"(?i)roa\s*\(.*\)", s):
        return "roa(p)"
    if re.search(r"(?i)\broc\b", s) or re.search(r"(?i)roc\s*\(.*\)", s):
        return "roc(p)"
    return s  # fallback: keep as-is


df[noise_col] = df[noise_col].apply(_normalize_noise_val)


# Map the long dataset descriptor to the short dataset label
def normalize_dataset(name: str) -> str:
    n = name.lower()
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
    return name  # fallback: raw


df["Dataset_norm"] = df[raw_dataset_col].apply(normalize_dataset)

# Standardize noise to strings such as "0", "100", "roa(p)", "roc(p)"
df[noise_col] = df[noise_col].astype(str).str.strip()


# -------------------------------------------------------------------
# Plotting function per algorithm
# -------------------------------------------------------------------
def plot_algorithm(df_all: pd.DataFrame, algo_name: str):
    df_algo = df_all[df_all[algo_col] == algo_name]

    if df_algo.empty:
        print(f"{algo_name}: no data")
        return

    # Prepare single figure with all datasets overlaid
    fig, ax = plt.subplots(figsize=(6 + len(target_datasets) * 1.5, 5))

    x = list(range(len(noise_order)))
    for i, ds in enumerate(target_datasets):
        df_ds = df_algo[df_algo["Dataset_norm"] == ds]

        if df_ds.empty:
            print(f"{algo_name} - {ds}: no data")
            continue

        # Reindex to canonical noise_order so x-axis is consistent across datasets
        s = df_ds.groupby(noise_col)[metric_col].mean().reindex(noise_order)
        if s.dropna().empty:
            print(f"{algo_name} - {ds}: no matching noise buckets")
            continue

        y = s.values
        color = plt.cm.tab10(i % 10)
        ax.plot(x, y, marker="o", linestyle="-", color=color, label=ds)
        print(f"{algo_name} - {ds}:\n{s.dropna()}\n")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, rotation=45, ha="right")
    ax.set_xlabel("Noise")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{algo_name}: Mean(D_val) per dataset & noise (overlay)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Dataset", loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"algo_{algo_name}_dataset_metrics.png", dpi=200)


def plot_dataset(df_all: pd.DataFrame, dataset_name: str):
    df_ds = df_all[df_all["Dataset_norm"] == dataset_name]

    if df_ds.empty:
        print(f"{dataset_name}: no data")
        return

    # Prepare single figure with all algorithms overlaid
    fig, ax = plt.subplots(figsize=(6 + len(target_algos) * 1.5, 5))

    x = list(range(len(noise_order)))
    for i, algo in enumerate(target_algos):
        df_algo = df_ds[df_ds[algo_col] == algo]

        if df_algo.empty:
            print(f"{dataset_name} - {algo}: no data")
            continue

        # Reindex to canonical noise_order so x-axis is consistent across algorithms
        s = df_algo.groupby(noise_col)[metric_col].mean().reindex(noise_order)
        if s.dropna().empty:
            print(f"{dataset_name} - {algo}: no matching noise buckets")
            continue

        y = s.values
        color = plt.cm.tab10(i % 10)
        ax.plot(x, y, marker="o", linestyle="-", color=color, label=algo)
        print(f"{dataset_name} - {algo}:\n{s.dropna()}\n")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, rotation=45, ha="right")
    ax.set_xlabel("Noise")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{dataset_name}: Mean(D_val) per algorithm & noise (overlay)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Algorithm", loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"dataset_{dataset_name}_algo_metrics.png", dpi=200)


def plot_dataset_per_class(df_all: pd.DataFrame, dataset_name: str):
    df_ds = df_all[df_all["Dataset_norm"] == dataset_name]

    if df_ds.empty:
        print(f"{dataset_name} (per class): no data")
        return

    # Prepare single figure with all algorithms overlaid
    fig, ax = plt.subplots(figsize=(6 + len(target_algos) * 1.5, 5))

    x = list(range(len(noise_order)))
    for i, algo in enumerate(target_algos):
        df_algo = df_ds[df_ds[algo_col] == algo]

        if df_algo.empty:
            print(f"{dataset_name} - {algo}: no data")
            continue

        # get experiment_ids
        exp_ids = df_algo["Experiment ID"].unique().tolist()
        class_metric_cols = []
        for exp_id in exp_ids:
            if pd.isna(exp_id):
                print(f"Skipping missing Exp_ID for {dataset_name} - {algo}")
                continue
            # get experiment path
            exp_paths = [p for p in nnUNet_results_all_exps if exp_id in p]
            assert (
                len(exp_paths) == dataset_to_numclients[dataset_name]
            ), f"Did not find expected number of experiment paths for Exp_ID {exp_id}. Expected {dataset_to_numclients[dataset_name]}, found {len(exp_paths)}"
            # find and load validation/summary.json from all clients and average
            client_classwise_metrics = []
            for exp_path in exp_paths:
                results_summary = json.load(
                    open(exp_path + "/validation/summary.json", "r")
                )
                client_classwise_metrics.append(results_summary["mean"])
            # average across clients
            classwise_metrics = {}
            for class_label in client_classwise_metrics[0].keys():
                classwise_metrics[class_label] = {}
                for metric_name in client_classwise_metrics[0][class_label].keys():
                    classwise_metrics[class_label][metric_name] = sum(
                        cm[class_label][metric_name] for cm in client_classwise_metrics
                    ) / len(client_classwise_metrics)
            # add classwise metrics to df_algo
            for class_label, metrics in classwise_metrics.items():
                class_metric_col = f"{classwise_metric}_clientavg_class{class_label}"
                if class_metric_col not in df_algo.columns:
                    df_algo[class_metric_col] = None
                df_algo.loc[df_algo["Experiment ID"] == exp_id, class_metric_col] = (
                    metrics[classwise_metric]
                )
                (
                    class_metric_cols.append(class_metric_col)
                    if class_metric_col not in class_metric_cols
                    else None
                )

        # plotting color
        color = plt.cm.tab10(i % 10)
        # Reindex to canonical noise_order so x-axis is consistent across algorithms
        for i, class_metric_col in enumerate(class_metric_cols):
            s = df_algo.groupby(noise_col)[class_metric_col].mean().reindex(noise_order)
            if s.dropna().empty:
                print(f"{dataset_name} - {algo}: no matching noise buckets")
                continue

            y = s.values
            linestyle = class_line_marker[i + 1]["linestyle"]
            marker = class_line_marker[i + 1]["marker"]
            ax.plot(
                x,
                y,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=f"{algo} - {class_metric_col}",
            )
            print(f"{dataset_name} - {algo} - {class_metric_col}:\n{s.dropna()}\n")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, rotation=45, ha="right")
    ax.set_xlabel("Noise")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{dataset_name}: Mean(D_val) per algorithm & noise (overlay)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Algorithm", loc="best")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"dataset_{dataset_name}_algo_classwise_metrics.png", dpi=200
    )


def plot_classwise_boxplots_clean_noisy(df_all: pd.DataFrame):
    """
    Create boxplots with hierarchical structure:
    - 6 columns for datasets
    - Within each dataset: C sub-columns for each class
    - Within each class: clean and noisy boxplots
    - Within each noise state: boxplot of all methods
    """

    noise_states = {"0": "clean", "100": "noisy"}
    # data[dataset][algorithm][class][noise_state] = [bootstrap_dice_values]
    data = {ds: {algo: {} for algo in target_algos} for ds in target_datasets}

    for ds in target_datasets:
        df_ds = df_all[df_all["Dataset_norm"] == ds]
        if df_ds.empty:
            print(f"{ds}: no data for boxplots")
            continue

        for algo in target_algos:
            df_algo = df_ds[df_ds[algo_col] == algo]
            if df_algo.empty:
                continue
            # only keep clean/noisy buckets
            df_algo = df_algo[df_algo[noise_col].isin(noise_states.keys())]

            exp_ids = df_algo["Experiment ID"].dropna().unique().tolist()
            for exp_id in exp_ids:
                exp_rows = df_algo[df_algo["Experiment ID"] == exp_id]
                exp_paths = get_exp_paths_with_bootstrap(exp_id)
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(
                        f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients with bootstrap results, found {len(exp_paths)}"
                    )
                    continue
                if len(exp_paths) == 0:
                    print(
                        f"No experiment paths with bootstrap results found for Exp_ID {exp_id} ({ds}/{algo})"
                    )
                    continue

                client_classwise_metrics = []
                for exp_path in exp_paths:
                    summary_file = Path(exp_path) / "validation" / "summary.json"
                    if not summary_file.is_file():
                        print(f"Missing summary.json at {summary_file}")
                        continue
                    with open(summary_file, "r") as f:
                        results_summary = json.load(f)
                    client_classwise_metrics.append(results_summary["mean"])
                if not client_classwise_metrics:
                    continue

                # average across clients
                classwise_metrics = {}
                first = client_classwise_metrics[0]
                for class_label in first.keys():
                    classwise_metrics[class_label] = {}
                    for metric_name in first[class_label].keys():
                        classwise_metrics[class_label][metric_name] = sum(
                            cm[class_label][metric_name]
                            for cm in client_classwise_metrics
                        ) / len(client_classwise_metrics)

                # assign to buckets: data[ds][class][noise_state].append(dice_value)
                for noise_val in exp_rows[noise_col].unique():
                    state_key = noise_states[noise_val]
                    for class_label, metrics in classwise_metrics.items():
                        if class_label not in data[ds][algo]:
                            data[ds][algo][class_label] = {"clean": [], "noisy": []}
                        val = metrics.get(classwise_metric, None)
                        if val is not None:
                            data[ds][algo][class_label][state_key].append(val)

    # Build boxplot structure (per dataset → class → noise state → algorithm)
    algo_colors = {algo: plt.cm.tab10(i % 10) for i, algo in enumerate(target_algos)}
    positions = []
    labels = []
    box_data = []
    colors = []
    meta_info = []
    dataset_boundaries = []
    class_boundaries = []
    noise_boundaries = []
    label_positions = []  # positions for clean/noisy labels
    label_texts = []  # "clean" or "noisy"

    pos = 0.0
    gap_algo = 0.1  # gap between algorithms within the same state
    gap_noise = 0.05  # gap between clean/noisy within a class
    gap_class = 0.0  # gap between classes within a dataset
    gap_dataset = 0.0  # gap between datasets
    bplot_width = 0.1

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: (
                int(x)
                if isinstance(x, (int, str))
                and str(x).replace("(", "").replace(")", "").split(",")[0].isdigit()
                else str(x)
            ),
        )

        for cl in class_labels_sorted:
            class_has_data = False
            class_positions = []
            clean_positions = []
            noisy_positions = []
            for state_key in ["clean", "noisy"]:
                state_has_data = False
                state_positions = []
                for algo in target_algos:
                    vals = data[ds].get(algo, {}).get(cl, {}).get(state_key, [])
                    if not vals:
                        continue
                    positions.append(pos)
                    box_data.append(vals)
                    colors.append(algo_colors[algo])
                    meta_info.append(
                        {
                            "ds": ds,
                            "class": cl,
                            "state": state_key,
                            "algo": algo,
                            "n": len(vals),
                        }
                    )
                    state_positions.append(pos)
                    class_positions.append(pos)
                    pos += gap_algo
                    state_has_data = True
                    dataset_has_data = True
                    class_has_data = True
                if state_has_data:
                    if state_key == "clean":
                        clean_positions = state_positions
                    else:
                        noisy_positions = state_positions
                    # Add label position at center of state group
                    label_positions.append(np.mean(state_positions))
                    label_texts.append(state_key)
                    pos += gap_noise
            if class_has_data:
                if clean_positions and noisy_positions:
                    noise_sep = (max(clean_positions) + min(noisy_positions)) / 2.0
                    noise_boundaries.append(noise_sep)
                class_boundaries.append(max(class_positions))
                pos += gap_class

        if dataset_has_data:
            dataset_boundaries.append((dataset_start, positions[-1], ds))
            pos += gap_dataset

    if not box_data:
        print("No boxplot data collected.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(max(13, len(target_datasets) * 2.7), 7.5))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=bplot_width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Add vertical lines
    # Dotted lines between clean and noisy
    for boundary in noise_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Dashed lines between classes
    for boundary in class_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add dataset labels and separators
    for ds_start, ds_end, ds_name in dataset_boundaries:
        ds_center = (ds_start + ds_end) / 2.0
        ax.text(
            ds_center,
            -0.11,
            ds_name,
            ha="center",
            va="top",
            fontsize=BOXPLOT_DATASET_FONTSIZE,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(
            ds_end + gap_dataset / 2.0,
            color="gray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(label_positions)
    ax.set_xticklabels(
        label_texts, rotation=0, ha="center", fontsize=BOXPLOT_TICK_FONTSIZE
    )
    ax.set_ylabel(classwise_metric, fontsize=BOXPLOT_LABEL_FONTSIZE)
    # Tighten in-axes horizontal margins
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    apply_metric_axis_limits(ax)
    ax.set_title(
        "Class-wise Dice: clean vs noisy per dataset/class/algorithm",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    ax.scatter(
        positions,
        means,
        marker="D",
        c=colors,
        edgecolors="k",
        zorder=4,
        s=30,
        linewidths=0.6,
        label=None,
    )
    # Print means per boxplot
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(
            f"{meta['ds']} | class {meta['class']} | {meta['state']} | {meta['algo']}: "
            f"mean={m:.4f} (n={meta['n']})"
        )

    # Legend by algorithm color
    algo_patches = [
        plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a)
        for a in target_algos
    ]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(
        OUTPUT_DIR / "classwise_boxplots_clean_vs_noisy.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"Saved boxplot to {OUTPUT_DIR / 'classwise_boxplots_clean_vs_noisy.png'}")


def plot_classwise_boxplots_clean_noisy_bootstrapping(df_all: pd.DataFrame):
    """
    Create boxplots with hierarchical structure:
    - 6 columns for datasets
    - Within each dataset: C sub-columns for each class
    - Within each class: clean and noisy boxplots
    - Within each noise state: boxplot of all methods from bootstrapping results
    """

    noise_states = {"0": "clean", "100": "noisy"}
    # data[dataset][class][noise_state] = [dice_values_across_methods]
    data = {ds: {algo: {} for algo in target_algos} for ds in target_datasets}

    for ds in target_datasets:
        df_ds = df_all[df_all["Dataset_norm"] == ds]
        if df_ds.empty:
            print(f"{ds}: no data for boxplots")
            continue

        for algo in target_algos:
            df_algo = df_ds[df_ds[algo_col] == algo]
            if df_algo.empty:
                continue
            # only keep clean/noisy buckets
            df_algo = df_algo[df_algo[noise_col].isin(noise_states.keys())]

            exp_ids = df_algo["Experiment ID"].dropna().unique().tolist()
            for exp_id in exp_ids:
                exp_rows = df_algo[df_algo["Experiment ID"] == exp_id]
                exp_paths = get_exp_paths_with_bootstrap(exp_id)
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(
                        f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients with bootstrap results, found {len(exp_paths)}"
                    )
                    continue
                if len(exp_paths) == 0:
                    print(
                        f"No experiment paths with bootstrap results found for Exp_ID {exp_id} ({ds}/{algo})"
                    )
                    continue

                client_classwise_metrics = []
                for exp_path in exp_paths:
                    bootstrap_file = (
                        Path(exp_path)
                        / "validation"
                        / "bootstrap_evaluation_results.json"
                    )
                    if not bootstrap_file.is_file():
                        print(
                            f"Missing bootstrap_evaluation_results.json at {bootstrap_file}"
                        )
                        continue
                    with open(bootstrap_file, "r") as f:
                        results_summary = json.load(f)
                    client_classwise_metrics.append(results_summary)
                if not client_classwise_metrics:
                    continue

                # # average across clients
                # classwise_metrics = {}
                # first = client_classwise_metrics[0]
                # for class_label in first.keys():
                #     classwise_metrics[class_label] = {}
                #     for metric_name in first[class_label].keys():
                #         classwise_metrics[class_label][metric_name] = sum(
                #             cm[class_label][metric_name] for cm in client_classwise_metrics
                #         ) / len(client_classwise_metrics)

                # assign to buckets: data[ds][algo][class][noise_state].extend(bootstrap_values)
                for noise_val in exp_rows[noise_col].unique():
                    state_key = noise_states[noise_val]
                    for client_bootstrap_res in client_classwise_metrics:
                        for class_label, metrics in client_bootstrap_res.items():
                            if class_label == "stats":
                                continue
                            if class_label not in data[ds][algo]:
                                data[ds][algo][class_label] = {"clean": [], "noisy": []}
                            raw_vals = metrics.get(classwise_metric, None)
                            finite_vals = extract_finite_metric_values(raw_vals)
                            if raw_vals is None:
                                continue
                            if not finite_vals:
                                print(
                                    f"No finite values for {classwise_metric}: "
                                    f"{ds}/{algo}/{state_key}/class {class_label}"
                                )
                                continue
                            data[ds][algo][class_label][state_key].extend(finite_vals)

    # Build boxplot structure (per dataset → class → noise state → algorithm)
    algo_colors = {algo: plt.cm.tab10(i % 10) for i, algo in enumerate(target_algos)}
    positions = []
    labels = []
    box_data = []
    colors = []
    meta_info = []
    dataset_boundaries = []
    class_boundaries = []
    noise_boundaries = []
    label_positions = []  # positions for clean/noisy labels
    label_texts = []  # "clean" or "noisy"

    pos = 0.0
    gap_algo = 0.1  # gap between algorithms within the same state
    gap_noise = 0.05  # gap between clean/noisy within a class
    gap_class = 0.0  # gap between classes within a dataset
    gap_dataset = 0.0  # gap between datasets
    bplot_width = 0.1

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: (
                int(x)
                if isinstance(x, (int, str))
                and str(x).replace("(", "").replace(")", "").split(",")[0].isdigit()
                else str(x)
            ),
        )

        for cl in class_labels_sorted:
            class_has_data = False
            class_positions = []
            clean_positions = []
            noisy_positions = []
            for state_key in ["clean", "noisy"]:
                state_has_data = False
                state_positions = []
                for algo in target_algos:
                    vals = data[ds].get(algo, {}).get(cl, {}).get(state_key, [])
                    if not vals:
                        continue
                    positions.append(pos)
                    box_data.append(vals)
                    colors.append(algo_colors[algo])
                    meta_info.append(
                        {
                            "ds": ds,
                            "class": cl,
                            "state": state_key,
                            "algo": algo,
                            "n": len(vals),
                        }
                    )
                    state_positions.append(pos)
                    class_positions.append(pos)
                    pos += gap_algo
                    state_has_data = True
                    dataset_has_data = True
                    class_has_data = True
                if state_has_data:
                    if state_key == "clean":
                        clean_positions = state_positions
                    else:
                        noisy_positions = state_positions
                    # Add label position at center of state group
                    label_positions.append(np.mean(state_positions))
                    label_texts.append(state_key)
                    pos += gap_noise
            if class_has_data:
                if clean_positions and noisy_positions:
                    noise_sep = (max(clean_positions) + min(noisy_positions)) / 2.0
                    noise_boundaries.append(noise_sep)
                class_boundaries.append(max(class_positions))
                pos += gap_class

        if dataset_has_data:
            dataset_boundaries.append((dataset_start, positions[-1], ds))
            pos += gap_dataset

    if not box_data:
        print("No bootstrapping boxplot data collected.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(target_datasets) * 2.5), 7))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=bplot_width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Add vertical lines
    # Dotted lines between clean and noisy
    for boundary in noise_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Dashed lines between classes
    for boundary in class_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add dataset labels and separators
    for ds_start, ds_end, ds_name in dataset_boundaries:
        ds_center = (ds_start + ds_end) / 2.0
        ax.text(
            ds_center,
            -0.15,
            ds_name,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(
            ds_end + gap_dataset / 2.0,
            color="gray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=0, ha="center", fontsize=BOXPLOT_LABEL_FONTSIZE)
    ax.set_ylabel(classwise_metric, fontsize=BOXPLOT_LABEL_FONTSIZE)
    # Tighten in-axes horizontal margins
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    apply_metric_axis_limits(ax)
    ax.set_title(
        "Class-wise Dice (bootstrapping): clean vs noisy per dataset/class/algorithm",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    ax.scatter(
        positions,
        means,
        marker="D",
        c=colors,
        edgecolors="k",
        zorder=4,
        s=30,
        linewidths=0.6,
        label=None,
    )
    # Print means per boxplot
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(
            f"{meta['ds']} | class {meta['class']} | {meta['state']} | {meta['algo']}: "
            f"mean={m:.4f} (n={meta['n']})"
        )

    # Legend by algorithm color
    algo_patches = [
        plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a)
        for a in target_algos
    ]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(
        OUTPUT_DIR / "classwise_boxplots_clean_vs_noisy_bootstrapping.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(
        f"Saved boxplot to {OUTPUT_DIR / 'classwise_boxplots_clean_vs_noisy_bootstrapping.png'}"
    )


def plot_classwise_boxplots_clean_noiseratioall_noisy_bootstrapping(
    df_all: pd.DataFrame,
):
    """
    Create boxplots with hierarchical structure:
    - 6 columns for datasets
    - Within each dataset: C sub-columns for each class
    - Within each class: clean, noiseratioall, and noisy boxplots
    - Within each noise state: boxplot of all methods from bootstrapping results
    """

    noise_states = {"0": "clean", "roa(p)": "roa(p)", "100": "noisy"}
    # data[dataset][class][noise_state] = [dice_values_across_methods]
    data = {ds: {algo: {} for algo in target_algos} for ds in target_datasets}

    for ds in target_datasets:
        df_ds = df_all[df_all["Dataset_norm"] == ds]
        if df_ds.empty:
            print(f"{ds}: no data for boxplots")
            continue

        for algo in target_algos:
            df_algo = df_ds[df_ds[algo_col] == algo]
            if df_algo.empty:
                continue
            # only keep clean/noisy buckets
            df_algo = df_algo[df_algo[noise_col].isin(noise_states.keys())]

            exp_ids = df_algo["Experiment ID"].dropna().unique().tolist()
            for exp_id in exp_ids:
                exp_rows = df_algo[df_algo["Experiment ID"] == exp_id]
                exp_paths = get_exp_paths_with_bootstrap(exp_id)
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(
                        f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients with bootstrap results, found {len(exp_paths)}"
                    )
                    continue
                if len(exp_paths) == 0:
                    print(
                        f"No experiment paths with bootstrap results found for Exp_ID {exp_id} ({ds}/{algo})"
                    )
                    continue

                client_classwise_metrics = []
                for exp_path in exp_paths:
                    bootstrap_file = (
                        Path(exp_path)
                        / "validation"
                        / "bootstrap_evaluation_results.json"
                    )
                    if not bootstrap_file.is_file():
                        print(
                            f"Missing bootstrap_evaluation_results.json at {bootstrap_file}"
                        )
                        continue
                    with open(bootstrap_file, "r") as f:
                        results_summary = json.load(f)
                    client_classwise_metrics.append(results_summary)
                if not client_classwise_metrics:
                    continue

                # assign to buckets: data[ds][algo][class][noise_state].extend(bootstrap_values)
                for noise_val in exp_rows[noise_col].unique():
                    state_key = noise_states[noise_val]
                    for client_bootstrap_res in client_classwise_metrics:
                        for class_label, metrics in client_bootstrap_res.items():
                            if class_label == "stats":
                                continue
                            if class_label not in data[ds][algo]:
                                # data[ds][algo][class_label] = {"clean": [], "noisy": []}
                                data[ds][algo][class_label] = {
                                    noise_state: []
                                    for noise_state in noise_states.values()
                                }
                            vals = metrics.get(classwise_metric, None)
                            if vals is None:
                                continue
                            if isinstance(vals, list):
                                data[ds][algo][class_label][state_key].extend(vals)
                            else:
                                data[ds][algo][class_label][state_key].append(vals)

    # Build boxplot structure (per dataset → class → noise state → algorithm)
    algo_colors = {algo: plt.cm.tab10(i % 10) for i, algo in enumerate(target_algos)}
    positions = []
    labels = []
    box_data = []
    colors = []
    meta_info = []
    dataset_boundaries = []
    class_boundaries = []
    noise_boundaries = []
    label_positions = []  # positions for clean/noisy labels
    label_texts = []  # "clean" or "noisy"

    pos = 0.0
    gap_algo = 0.1  # gap between algorithms within the same state
    gap_noise = 0.05  # gap between clean/noisy within a class
    gap_class = 0.0  # gap between classes within a dataset
    gap_dataset = 0.0  # gap between datasets
    bplot_width = 0.1

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: (
                int(x)
                if isinstance(x, (int, str))
                and str(x).replace("(", "").replace(")", "").split(",")[0].isdigit()
                else str(x)
            ),
        )

        for cl in class_labels_sorted:
            class_has_data = False
            class_positions = []
            state_groups = {}  # Track positions for each state
            for state_key in ["clean", "roa(p)", "noisy"]:
                state_has_data = False
                state_positions = []
                for algo in target_algos:
                    vals = data[ds].get(algo, {}).get(cl, {}).get(state_key, [])
                    if not vals:
                        continue
                    positions.append(pos)
                    box_data.append(vals)
                    colors.append(algo_colors[algo])
                    meta_info.append(
                        {
                            "ds": ds,
                            "class": cl,
                            "state": state_key,
                            "algo": algo,
                            "n": len(vals),
                        }
                    )
                    state_positions.append(pos)
                    class_positions.append(pos)
                    pos += gap_algo
                    state_has_data = True
                    dataset_has_data = True
                    class_has_data = True
                if state_has_data:
                    state_groups[state_key] = state_positions
                    # Add label position at center of state group
                    label_positions.append(np.mean(state_positions))
                    label_texts.append(state_key)
                    pos += gap_noise

            # Add noise boundaries between consecutive states
            if class_has_data:
                state_keys_present = ["clean", "roa(p)", "noisy"]
                for i in range(len(state_keys_present) - 1):
                    curr_state = state_keys_present[i]
                    next_state = state_keys_present[i + 1]
                    if curr_state in state_groups and next_state in state_groups:
                        noise_sep = (
                            max(state_groups[curr_state])
                            + min(state_groups[next_state])
                        ) / 2.0
                        noise_boundaries.append(noise_sep)

                class_boundaries.append(max(class_positions))
                pos += gap_class

        if dataset_has_data:
            dataset_boundaries.append((dataset_start, positions[-1], ds))
            pos += gap_dataset

    if not box_data:
        print("No bootstrapping boxplot data collected.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(target_datasets) * 2.5), 7))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=bplot_width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c, meta in zip(bp["boxes"], colors, meta_info):
        patch.set_facecolor(c)
        # Highlight roa(p) by making clean and noisy more transparent
        if meta["state"] in ["clean", "noisy"]:
            patch.set_alpha(0.35)
        else:  # roa(p)
            patch.set_alpha(0.85)

    # Apply color intensity to medians based on state
    for median, meta in zip(bp["medians"], meta_info):
        median.set_linewidth(1.5)
        if meta["state"] in ["clean", "noisy"]:
            median.set_color("gray")
            median.set_alpha(0.5)
        else:  # roa(p)
            median.set_color("black")
            median.set_alpha(1.0)

    # Apply color intensity to whiskers and caps based on state
    for i, (whisker, meta) in enumerate(
        zip(bp["whiskers"], [meta_info[i // 2] for i in range(len(bp["whiskers"]))])
    ):
        if meta["state"] in ["clean", "noisy"]:
            whisker.set_color("gray")
            whisker.set_alpha(0.5)
        else:  # roa(p)
            whisker.set_color("black")
            whisker.set_alpha(1.0)

    for i, (cap, meta) in enumerate(
        zip(bp["caps"], [meta_info[i // 2] for i in range(len(bp["caps"]))])
    ):
        if meta["state"] in ["clean", "noisy"]:
            cap.set_color("gray")
            cap.set_alpha(0.5)
        else:  # roa(p)
            cap.set_color("black")
            cap.set_alpha(1.0)

    # Add vertical lines
    # Dotted lines between clean and noisy
    for boundary in noise_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Dashed lines between classes
    for boundary in class_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add dataset labels and separators
    for ds_start, ds_end, ds_name in dataset_boundaries:
        ds_center = (ds_start + ds_end) / 2.0
        ax.text(
            ds_center,
            -0.15,
            ds_name,
            ha="center",
            va="top",
            fontsize=BOXPLOT_LABEL_FONTSIZE,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(
            ds_end + gap_dataset / 2.0,
            color="gray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=0, ha="center", fontsize=BOXPLOT_LABEL_FONTSIZE)
    ax.set_ylabel(classwise_metric, fontsize=BOXPLOT_LABEL_FONTSIZE)
    # Tighten in-axes horizontal margins
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    apply_metric_axis_limits(ax)
    ax.set_title(
        "Class-wise Dice (bootstrapping): clean vs roa(p) vs noisy per dataset/class/algorithm",
        fontsize=BOXPLOT_LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    # Apply different alpha for mean markers too
    mean_alphas = [
        0.5 if meta["state"] in ["clean", "noisy"] else 1.0 for meta in meta_info
    ]
    for i, (pos, mean, color, alpha) in enumerate(
        zip(positions, means, colors, mean_alphas)
    ):
        ax.scatter(
            pos,
            mean,
            marker="D",
            c=[color],
            edgecolors="k",
            zorder=4,
            s=30,
            linewidths=0.6,
            alpha=alpha,
        )
    # Print means per boxplot
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(
            f"{meta['ds']} | class {meta['class']} | {meta['state']} | {meta['algo']}: "
            f"mean={m:.4f} (n={meta['n']})"
        )

    # Legend by algorithm color
    algo_patches = [
        plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a)
        for a in target_algos
    ]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=BOXPLOT_LABEL_FONTSIZE)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(
        OUTPUT_DIR / "classwise_boxplots_clean_vs_roa(p)_vs_noisy_bootstrapping.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(
        f"Saved boxplot to {OUTPUT_DIR / 'classwise_boxplots_clean_vs_roa(p)_vs_noisy_bootstrapping.png'}"
    )


def plot_classwise_boxplots_clean_roc_noisy_bootstrapping(df_all: pd.DataFrame):
    """
    Create boxplots with hierarchical structure:
    - 6 columns for datasets
    - Within each dataset: C sub-columns for each class
    - Within each class: clean, roc(p), and noisy boxplots
    - Within each noise state: boxplot of all methods from bootstrapping results
    """

    noise_states = {"0": "clean", "roc(p)": "roc(p)", "100": "noisy"}
    # data[dataset][class][noise_state] = [dice_values_across_methods]
    data = {ds: {algo: {} for algo in target_algos} for ds in target_datasets}

    for ds in target_datasets:
        df_ds = df_all[df_all["Dataset_norm"] == ds]
        if df_ds.empty:
            print(f"{ds}: no data for boxplots")
            continue

        for algo in target_algos:
            df_algo = df_ds[df_ds[algo_col] == algo]
            if df_algo.empty:
                continue
            # only keep clean/noisy buckets
            df_algo = df_algo[df_algo[noise_col].isin(noise_states.keys())]

            exp_ids = df_algo["Experiment ID"].dropna().unique().tolist()
            for exp_id in exp_ids:
                exp_rows = df_algo[df_algo["Experiment ID"] == exp_id]
                exp_paths = get_exp_paths_with_bootstrap(exp_id)
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(
                        f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients with bootstrap results, found {len(exp_paths)}"
                    )
                    continue
                if len(exp_paths) == 0:
                    print(
                        f"No experiment paths with bootstrap results found for Exp_ID {exp_id} ({ds}/{algo})"
                    )
                    continue

                client_classwise_metrics = []
                for exp_path in exp_paths:
                    bootstrap_file = (
                        Path(exp_path)
                        / "validation"
                        / "bootstrap_evaluation_results.json"
                    )
                    if not bootstrap_file.is_file():
                        print(
                            f"Missing bootstrap_evaluation_results.json at {bootstrap_file}"
                        )
                        continue
                    with open(bootstrap_file, "r") as f:
                        results_summary = json.load(f)
                    client_classwise_metrics.append(results_summary)
                if not client_classwise_metrics:
                    continue

                # assign to buckets: data[ds][algo][class][noise_state].extend(bootstrap_values)
                for noise_val in exp_rows[noise_col].unique():
                    state_key = noise_states[noise_val]
                    for client_bootstrap_res in client_classwise_metrics:
                        for class_label, metrics in client_bootstrap_res.items():
                            if class_label == "stats":
                                continue
                            if class_label not in data[ds][algo]:
                                data[ds][algo][class_label] = {
                                    noise_state: []
                                    for noise_state in noise_states.values()
                                }
                            vals = metrics.get(classwise_metric, None)
                            if vals is None:
                                continue
                            if isinstance(vals, list):
                                data[ds][algo][class_label][state_key].extend(vals)
                            else:
                                data[ds][algo][class_label][state_key].append(vals)

    # Build boxplot structure (per dataset → class → noise state → algorithm)
    algo_colors = {algo: plt.cm.tab10(i % 10) for i, algo in enumerate(target_algos)}
    positions = []
    labels = []
    box_data = []
    colors = []
    meta_info = []
    dataset_boundaries = []
    class_boundaries = []
    noise_boundaries = []
    label_positions = []  # positions for clean/noisy labels
    label_texts = []  # "clean" or "noisy"

    pos = 0.0
    gap_algo = 0.1  # gap between algorithms within the same state
    gap_noise = 0.05  # gap between clean/noisy within a class
    gap_class = 0.0  # gap between classes within a dataset
    gap_dataset = 0.0  # gap between datasets
    bplot_width = 0.1

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: (
                int(x)
                if isinstance(x, (int, str))
                and str(x).replace("(", "").replace(")", "").split(",")[0].isdigit()
                else str(x)
            ),
        )

        for cl in class_labels_sorted:
            class_has_data = False
            class_positions = []
            state_groups = {}  # Track positions for each state
            for state_key in ["clean", "roc(p)", "noisy"]:
                state_has_data = False
                state_positions = []
                for algo in target_algos:
                    vals = data[ds].get(algo, {}).get(cl, {}).get(state_key, [])
                    if not vals:
                        continue
                    positions.append(pos)
                    box_data.append(vals)
                    colors.append(algo_colors[algo])
                    meta_info.append(
                        {
                            "ds": ds,
                            "class": cl,
                            "state": state_key,
                            "algo": algo,
                            "n": len(vals),
                        }
                    )
                    state_positions.append(pos)
                    class_positions.append(pos)
                    pos += gap_algo
                    state_has_data = True
                    dataset_has_data = True
                    class_has_data = True
                if state_has_data:
                    state_groups[state_key] = state_positions
                    # Add label position at center of state group
                    label_positions.append(np.mean(state_positions))
                    label_texts.append(state_key)
                    pos += gap_noise

            # Add noise boundaries between consecutive states
            if class_has_data:
                state_keys_present = ["clean", "roc(p)", "noisy"]
                for i in range(len(state_keys_present) - 1):
                    curr_state = state_keys_present[i]
                    next_state = state_keys_present[i + 1]
                    if curr_state in state_groups and next_state in state_groups:
                        noise_sep = (
                            max(state_groups[curr_state])
                            + min(state_groups[next_state])
                        ) / 2.0
                        noise_boundaries.append(noise_sep)

                class_boundaries.append(max(class_positions))
                pos += gap_class

        if dataset_has_data:
            dataset_boundaries.append((dataset_start, positions[-1], ds))
            pos += gap_dataset

    if not box_data:
        print("No bootstrapping boxplot data collected.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(target_datasets) * 2.5), 7))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=bplot_width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c, meta in zip(bp["boxes"], colors, meta_info):
        patch.set_facecolor(c)
        # Highlight roc(p) by making clean and noisy more transparent
        if meta["state"] in ["clean", "noisy"]:
            patch.set_alpha(0.35)
        else:  # roc(p)
            patch.set_alpha(0.85)

    # Apply color intensity to medians based on state
    for median, meta in zip(bp["medians"], meta_info):
        median.set_linewidth(1.5)
        if meta["state"] in ["clean", "noisy"]:
            median.set_color("gray")
            median.set_alpha(0.5)
        else:  # roc(p)
            median.set_color("black")
            median.set_alpha(1.0)

    # Apply color intensity to whiskers and caps based on state
    for i, (whisker, meta) in enumerate(
        zip(bp["whiskers"], [meta_info[i // 2] for i in range(len(bp["whiskers"]))])
    ):
        if meta["state"] in ["clean", "noisy"]:
            whisker.set_color("gray")
            whisker.set_alpha(0.5)
        else:  # roc(p)
            whisker.set_color("black")
            whisker.set_alpha(1.0)

    for i, (cap, meta) in enumerate(
        zip(bp["caps"], [meta_info[i // 2] for i in range(len(bp["caps"]))])
    ):
        if meta["state"] in ["clean", "noisy"]:
            cap.set_color("gray")
            cap.set_alpha(0.5)
        else:  # roc(p)
            cap.set_color("black")
            cap.set_alpha(1.0)

    # Add vertical lines
    # Dotted lines between clean and noisy
    for boundary in noise_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Dashed lines between classes
    for boundary in class_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add dataset labels and separators
    for ds_start, ds_end, ds_name in dataset_boundaries:
        ds_center = (ds_start + ds_end) / 2.0
        ax.text(
            ds_center,
            -0.15,
            ds_name,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(
            ds_end + gap_dataset / 2.0,
            color="gray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=0, ha="center", fontsize=BOXPLOT_LABEL_FONTSIZE)
    ax.set_ylabel(classwise_metric, fontsize=BOXPLOT_LABEL_FONTSIZE)
    # Tighten in-axes horizontal margins
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    apply_metric_axis_limits(ax)
    ax.set_title(
        "Class-wise Dice (bootstrapping): clean vs roc(p) vs noisy per dataset/class/algorithm",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    # Apply different alpha for mean markers too
    mean_alphas = [
        0.5 if meta["state"] in ["clean", "noisy"] else 1.0 for meta in meta_info
    ]
    for i, (pos, mean, color, alpha) in enumerate(
        zip(positions, means, colors, mean_alphas)
    ):
        ax.scatter(
            pos,
            mean,
            marker="D",
            c=[color],
            edgecolors="k",
            zorder=4,
            s=30,
            linewidths=0.6,
            alpha=alpha,
        )
    # Print means per boxplot
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(
            f"{meta['ds']} | class {meta['class']} | {meta['state']} | {meta['algo']}: "
            f"mean={m:.4f} (n={meta['n']})"
        )

    # Legend by algorithm color
    algo_patches = [
        plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a)
        for a in target_algos
    ]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(
        OUTPUT_DIR / "classwise_boxplots_clean_vs_roc(p)_vs_noisy_bootstrapping.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(
        f"Saved boxplot to {OUTPUT_DIR / 'classwise_boxplots_clean_vs_roc(p)_vs_noisy_bootstrapping.png'}"
    )


def plot_boxplots_clean_roa_roc_noisy_bootstrapping(
    df_all: pd.DataFrame, classwise=False
):
    """
    Create boxplots with hierarchical structure:
    - 6 columns for datasets
    - Within each dataset: C sub-columns for each class (if classwise=True) or just one column (if classwise=False)
    - Within each class: clean, roa(p), roc(p), and noisy boxplots
    - Within each noise state: boxplot of all methods from bootstrapping results
    """
    noise_states = {
        "0": "clean",
        "roa(p)": "roa(p)",
        "roc(p)": "roc(p)",
        "100": "noisy",
    }
    # data[dataset][algo][class][noise_state] = [dice_values]
    data = {ds: {algo: {} for algo in target_algos} for ds in target_datasets}

    for ds in target_datasets:
        df_ds = df_all[df_all["Dataset_norm"] == ds]
        if df_ds.empty:
            print(f"{ds}: no data for boxplots")
            continue

        for algo in target_algos:
            df_algo = df_ds[df_ds[algo_col] == algo]
            if df_algo.empty:
                continue
            # only keep desired buckets
            df_algo = df_algo[df_algo[noise_col].isin(noise_states.keys())]

            exp_ids = df_algo["Experiment ID"].dropna().unique().tolist()
            for exp_id in exp_ids:
                exp_rows = df_algo[df_algo["Experiment ID"] == exp_id]
                exp_paths = get_exp_paths_with_bootstrap(exp_id)
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(
                        f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients with bootstrap results, found {len(exp_paths)}"
                    )
                    continue
                if len(exp_paths) == 0:
                    print(
                        f"No experiment paths with bootstrap results found for Exp_ID {exp_id} ({ds}/{algo})"
                    )
                    continue

                client_classwise_metrics = []
                for exp_path in exp_paths:
                    bootstrap_file = (
                        Path(exp_path)
                        / "validation"
                        / "bootstrap_evaluation_results.json"
                    )
                    if not bootstrap_file.is_file():
                        print(
                            f"Missing bootstrap_evaluation_results.json at {bootstrap_file}"
                        )
                        continue
                    with open(bootstrap_file, "r") as f:
                        results_summary = json.load(f)
                    client_classwise_metrics.append(results_summary)
                if not client_classwise_metrics:
                    continue

                for noise_val in exp_rows[noise_col].unique():
                    state_key = noise_states[noise_val]
                    for client_bootstrap_res in client_classwise_metrics:
                        for class_label, metrics in client_bootstrap_res.items():
                            if class_label == "stats":
                                continue
                            if class_label not in data[ds][algo]:
                                data[ds][algo][class_label] = {
                                    noise_state: []
                                    for noise_state in noise_states.values()
                                }
                            vals = metrics.get(classwise_metric, None)
                            if vals is None:
                                continue
                            if isinstance(vals, list):
                                data[ds][algo][class_label][state_key].extend(vals)
                            else:
                                data[ds][algo][class_label][state_key].append(vals)

    # Optionally aggregate across classes
    if not classwise:
        for ds in target_datasets:
            for algo in target_algos:
                if not data.get(ds) or algo not in data[ds]:
                    continue
                agg = {noise_state: [] for noise_state in noise_states.values()}
                for cl in data[ds][algo].keys():
                    for state_key, vals in data[ds][algo][cl].items():
                        agg[state_key].extend(vals)
                data[ds][algo] = {"all": agg}

    # Build boxplot structure (per dataset → class → noise state → algorithm)
    algo_colors = {algo: plt.cm.tab10(i % 10) for i, algo in enumerate(target_algos)}
    positions = []
    box_data = []
    colors = []
    meta_info = []
    dataset_boundaries = []
    class_boundaries = []
    noise_boundaries = []
    label_positions = []
    label_texts = []

    pos = 0.0
    gap_algo = 0.1
    gap_noise = 0.05
    gap_class = 0.0
    gap_dataset = 0.0
    bplot_width = 0.1

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: (
                int(x)
                if isinstance(x, (int, str))
                and str(x).replace("(", "").replace(")", "").split(",")[0].isdigit()
                else str(x)
            ),
        )

        for cl in class_labels_sorted:
            class_has_data = False
            class_positions = []
            state_groups = {}
            for state_key in ["clean", "roa(p)", "roc(p)", "noisy"]:
                state_has_data = False
                state_positions = []
                for algo in target_algos:
                    vals = data[ds].get(algo, {}).get(cl, {}).get(state_key, [])
                    if not vals:
                        print(f"No data for {ds}/{algo}/class {cl}/{state_key}, skipping...")
                        continue
                    arr = np.asarray(vals, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        print(
                            f"No finite plot values for {ds}/{algo}/class {cl}/{state_key}, skipping..."
                        )
                        continue
                    positions.append(pos)
                    box_data.append(arr.tolist())
                    colors.append(algo_colors[algo])
                    meta_info.append(
                        {
                            "ds": ds,
                            "class": cl,
                            "state": state_key,
                            "algo": algo,
                            "n": int(arr.size),
                        }
                    )
                    state_positions.append(pos)
                    class_positions.append(pos)
                    pos += gap_algo
                    state_has_data = True
                    dataset_has_data = True
                    class_has_data = True
                if state_has_data:
                    state_groups[state_key] = state_positions
                    label_positions.append(np.mean(state_positions))
                    label_texts.append(state_key)
                    pos += gap_noise

            if class_has_data:
                state_keys_present = ["clean", "roa(p)", "roc(p)", "noisy"]
                for i in range(len(state_keys_present) - 1):
                    curr_state = state_keys_present[i]
                    next_state = state_keys_present[i + 1]
                    if curr_state in state_groups and next_state in state_groups:
                        noise_sep = (
                            max(state_groups[curr_state])
                            + min(state_groups[next_state])
                        ) / 2.0
                        noise_boundaries.append(noise_sep)

                class_boundaries.append(max(class_positions))
                pos += gap_class

        if dataset_has_data:
            dataset_boundaries.append((dataset_start, positions[-1], ds))
            pos += gap_dataset

    if not box_data:
        print("No bootstrapping boxplot data collected.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(target_datasets) * 2.5), 7))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=bplot_width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c, meta in zip(bp["boxes"], colors, meta_info):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    for median, meta in zip(bp["medians"], meta_info):
        median.set_linewidth(1.5)
        median.set_color("black")
        median.set_alpha(1.0)

    for i, (whisker, meta) in enumerate(
        zip(bp["whiskers"], [meta_info[i // 2] for i in range(len(bp["whiskers"]))])
    ):
        whisker.set_color("black")
        whisker.set_alpha(1.0)

    for i, (cap, meta) in enumerate(
        zip(bp["caps"], [meta_info[i // 2] for i in range(len(bp["caps"]))])
    ):
        cap.set_color("black")
        cap.set_alpha(1.0)

    # Add vertical lines
    for boundary in noise_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    for boundary in class_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add dataset labels and separators
    for ds_start, ds_end, ds_name in dataset_boundaries:
        ds_center = (ds_start + ds_end) / 2.0
        ax.text(
            ds_center,
            -0.06,
            ds_name,
            ha="center",
            va="top",
            fontsize=BOXPLOT_LABEL_FONTSIZE,
            # fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(
            ds_end + gap_dataset / 2.0,
            color="gray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=0, ha="center", fontsize=BOXPLOT_LABEL_FONTSIZE-2)
    ax.set_ylabel(classwise_metric, fontsize=BOXPLOT_LABEL_FONTSIZE)
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    apply_metric_axis_limits(ax)
    # prefix = "Class-wise" if classwise else "Overall"
    # suffix = "per dataset/class/algorithm" if classwise else "per dataset/algorithm"
    # ax.set_title(
    #     f"{prefix} {classwise_metric} (bootstrapping): clean vs roa(p) vs roc(p) vs noisy {suffix}",
    #     fontsize=BOXPLOT_TITLE_FONTSIZE,
    #     fontweight="bold",
    # )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=BOXPLOT_TICK_FONTSIZE)

    # Overlay mean markers
    means = []
    for vals in box_data:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        means.append(float(np.mean(arr)) if arr.size else np.nan)
    for pos_i, mean, color in zip(positions, means, colors):
        ax.scatter(
            pos_i,
            mean,
            marker="D",
            c=[color],
            edgecolors="k",
            zorder=4,
            s=30,
            linewidths=0.6,
            alpha=1.0,
        )

    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(
            f"{meta['ds']} | class {meta['class']} | {meta['state']} | {meta['algo']}: "
            f"mean={m:.4f} (n={meta['n']})"
        )

    algo_patches = [
        plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a)
        for a in target_algos
    ]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=BOXPLOT_LEGEND_FONTSIZE)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    out_name = (
        f"classwise_boxplots_clean_vs_roa(p)_vs_roc(p)_vs_noisy_bootstrapping_{metric_slug(classwise_metric)}.png"
        if classwise
        else f"boxplots_clean_vs_roa(p)_vs_roc(p)_vs_noisy_bootstrapping_{metric_slug(classwise_metric)}.png"
    )
    fig.savefig(OUTPUT_DIR / out_name, dpi=200, bbox_inches="tight")
    print(f"Saved boxplot to {OUTPUT_DIR / out_name}")


def plot_boxplots_roa_per_client_bootstrapping(df_all: pd.DataFrame, classwise=False):
    """
    Create boxplots comparing individual FL clients for roa(p) experiments:
    - 6 columns for datasets
    - Within each dataset: C sub-columns for each class (if classwise=True) or just one column (if classwise=False)
    - Within each class: client boxplots for each algorithm
    - Each boxplot represents one client's performance on that algorithm/dataset/class
    """
    noise_state = "roa(p)"
    # data[dataset][algo][class][client_id] = [metric_values]
    data = {ds: {algo: {} for algo in target_algos} for ds in target_datasets}

    for ds in target_datasets:
        df_ds = df_all[df_all["Dataset_norm"] == ds]
        if df_ds.empty:
            print(f"{ds}: no data for per-client roa(p) boxplots")
            continue

        for algo in target_algos:
            df_algo = df_ds[df_ds[algo_col] == algo]
            if df_algo.empty:
                continue
            # only keep roa(p) experiments
            df_algo = df_algo[df_algo[noise_col] == noise_state]

            exp_ids = df_algo["Experiment ID"].dropna().unique().tolist()
            for exp_id in exp_ids:
                exp_paths = get_exp_paths_with_bootstrap(exp_id)
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(
                        f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients with bootstrap results, found {len(exp_paths)}"
                    )
                    continue
                if len(exp_paths) == 0:
                    print(
                        f"No experiment paths with bootstrap results found for Exp_ID {exp_id} ({ds}/{algo})"
                    )
                    continue

                # Process each client
                for client_idx, exp_path in enumerate(exp_paths):
                    bootstrap_file = (
                        Path(exp_path)
                        / "validation"
                        / "bootstrap_evaluation_results.json"
                    )
                    if not bootstrap_file.is_file():
                        print(
                            f"Missing bootstrap_evaluation_results.json at {bootstrap_file}"
                        )
                        continue
                    with open(bootstrap_file, "r") as f:
                        results_summary = json.load(f)

                    # Extract per-class metrics for this client
                    for class_label, metrics in results_summary.items():
                        if class_label == "stats":
                            continue
                        if class_label not in data[ds][algo]:
                            data[ds][algo][class_label] = {}

                        # Use client_idx as key to track individual clients
                        if client_idx not in data[ds][algo][class_label]:
                            data[ds][algo][class_label][client_idx] = []

                        raw_vals = metrics.get(classwise_metric, None)
                        finite_vals = extract_finite_metric_values(raw_vals)
                        if raw_vals is None:
                            continue
                        if not finite_vals:
                            print(
                                f"No finite values for {classwise_metric}: "
                                f"{ds}/{algo}/{noise_state}/class {class_label}/client {client_idx}"
                            )
                            continue
                        data[ds][algo][class_label][client_idx].extend(finite_vals)

    # Optionally aggregate across classes
    if not classwise:
        for ds in target_datasets:
            for algo in target_algos:
                if not data.get(ds) or algo not in data[ds]:
                    continue
                agg = {}
                for cl in data[ds][algo].keys():
                    for client_id, vals in data[ds][algo][cl].items():
                        if client_id not in agg:
                            agg[client_id] = []
                        agg[client_id].extend(vals)
                data[ds][algo] = {"all": agg}

    # Build boxplot structure (per dataset → class → algorithm → client)
    algo_colors = {algo: plt.cm.tab10(i % 10) for i, algo in enumerate(target_algos)}
    positions = []
    box_data = []
    colors = []
    meta_info = []
    dataset_boundaries = []
    class_boundaries = []
    algo_boundaries = []
    client_label_positions = []
    client_label_texts = []

    pos = 0.0
    gap_client = 0.08  # gap between clients within algorithm
    gap_algo = 0.15  # gap between algorithms within class
    gap_class = 0.0  # gap between classes within dataset
    gap_dataset = 0.0  # gap between datasets
    bplot_width = 0.06

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: (
                int(x)
                if isinstance(x, (int, str))
                and str(x).replace("(", "").replace(")", "").split(",")[0].isdigit()
                else str(x)
            ),
        )

        for cl in class_labels_sorted:
            class_has_data = False
            class_positions = []

            for algo in target_algos:
                algo_has_data = False
                algo_positions = []
                algo_label_pos = None

                # Get clients for this algo/class
                clients_dict = data[ds].get(algo, {}).get(cl, {})
                if not clients_dict:
                    continue

                client_ids_sorted = sorted(clients_dict.keys())

                for client_id in client_ids_sorted:
                    vals = clients_dict[client_id]
                    if not vals:
                        continue

                    positions.append(pos)
                    box_data.append(vals)
                    colors.append(algo_colors[algo])
                    meta_info.append(
                        {
                            "ds": ds,
                            "class": cl,
                            "algo": algo,
                            "client": client_id,
                            "n": len(vals),
                        }
                    )
                    # Track label position and text for this client
                    client_label_positions.append(pos)
                    client_label_texts.append(f"FL client {client_id}")
                    algo_positions.append(pos)
                    class_positions.append(pos)
                    pos += gap_client
                    algo_has_data = True
                    dataset_has_data = True
                    class_has_data = True

                if algo_has_data:
                    if algo_label_pos is None:
                        algo_label_pos = np.mean(algo_positions)
                    algo_boundaries.append(max(algo_positions) + gap_client / 2.0)
                    pos += gap_algo

            if class_has_data:
                class_boundaries.append(max(class_positions))
                pos += gap_class

        if dataset_has_data:
            dataset_boundaries.append((dataset_start, positions[-1], ds))
            pos += gap_dataset

    if not box_data:
        print("No per-client roa(p) boxplot data collected.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(max(16, len(target_datasets) * 4), 7))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=bplot_width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
        median.set_alpha(1.0)

    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_alpha(1.0)

    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_alpha(1.0)

    # Add vertical lines between algorithms
    for boundary in algo_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Dashed lines between classes
    for boundary in class_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add dataset labels and separators
    for ds_start, ds_end, ds_name in dataset_boundaries:
        ds_center = (ds_start + ds_end) / 2.0
        ax.text(
            ds_center,
            -0.15,
            ds_name,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(
            ds_end + gap_dataset / 2.0,
            color="gray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(client_label_positions)
    ax.set_xticklabels(client_label_texts, rotation=45, ha="right", fontsize=BOXPLOT_LABEL_FONTSIZE)
    ax.set_ylabel(classwise_metric, fontsize=BOXPLOT_LABEL_FONTSIZE)
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    apply_metric_axis_limits(ax)
    prefix = "Class-wise" if classwise else "Overall"
    suffix = (
        "per dataset/class/algorithm/client"
        if classwise
        else "per dataset/algorithm/client"
    )
    ax.set_title(
        f"{prefix} {classwise_metric} (bootstrapping) - roa(p): Per-client performance {suffix}",
        fontsize=BOXPLOT_LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    for pos_i, mean, color in zip(positions, means, colors):
        ax.scatter(
            pos_i,
            mean,
            marker="D",
            c=[color],
            edgecolors="k",
            zorder=4,
            s=25,
            linewidths=0.6,
            alpha=1.0,
        )

    # Print statistics per client/algo/class
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(
            f"{meta['ds']} | class {meta['class']} | {meta['algo']} | client {meta['client']}: "
            f"mean={m:.4f} (n={meta['n']})"
        )

    # Legend by algorithm color
    algo_patches = [
        plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a)
        for a in target_algos
    ]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    out_name = (
        f"classwise_boxplots_roa(p)_per_client_bootstrapping_{metric_slug(classwise_metric)}.png"
        if classwise
        else f"boxplots_roa(p)_per_client_bootstrapping_{metric_slug(classwise_metric)}.png"
    )
    fig.savefig(OUTPUT_DIR / out_name, dpi=200, bbox_inches="tight")
    print(f"Saved per-client roa(p) boxplot to {OUTPUT_DIR / out_name}")


def plot_boxplots_roc_per_client_bootstrapping(df_all: pd.DataFrame, classwise=False):
    """
    Create boxplots comparing individual FL clients for roc(p) experiments:
    - 6 columns for datasets
    - Within each dataset: C sub-columns for each class (if classwise=True) or just one column (if classwise=False)
    - Within each class: client boxplots for each algorithm
    - Each boxplot represents one client's performance on that algorithm/dataset/class
    """
    noise_state = "roc(p)"
    # data[dataset][algo][class][client_id] = [metric_values]
    data = {ds: {algo: {} for algo in target_algos} for ds in target_datasets}

    for ds in target_datasets:
        df_ds = df_all[df_all["Dataset_norm"] == ds]
        if df_ds.empty:
            print(f"{ds}: no data for per-client roc(p) boxplots")
            continue

        for algo in target_algos:
            df_algo = df_ds[df_ds[algo_col] == algo]
            if df_algo.empty:
                continue
            # only keep roc(p) experiments
            df_algo = df_algo[df_algo[noise_col] == noise_state]

            exp_ids = df_algo["Experiment ID"].dropna().unique().tolist()
            for exp_id in exp_ids:
                exp_paths = get_exp_paths_with_bootstrap(exp_id)
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(
                        f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients with bootstrap results, found {len(exp_paths)}"
                    )
                    continue
                if len(exp_paths) == 0:
                    print(
                        f"No experiment paths with bootstrap results found for Exp_ID {exp_id} ({ds}/{algo})"
                    )
                    continue

                # Process each client
                for client_idx, exp_path in enumerate(exp_paths):
                    bootstrap_file = (
                        Path(exp_path)
                        / "validation"
                        / "bootstrap_evaluation_results.json"
                    )
                    if not bootstrap_file.is_file():
                        print(
                            f"Missing bootstrap_evaluation_results.json at {bootstrap_file}"
                        )
                        continue
                    with open(bootstrap_file, "r") as f:
                        results_summary = json.load(f)

                    # Extract per-class metrics for this client
                    for class_label, metrics in results_summary.items():
                        if class_label == "stats":
                            continue
                        if class_label not in data[ds][algo]:
                            data[ds][algo][class_label] = {}

                        # Use client_idx as key to track individual clients
                        if client_idx not in data[ds][algo][class_label]:
                            data[ds][algo][class_label][client_idx] = []

                        raw_vals = metrics.get(classwise_metric, None)
                        finite_vals = extract_finite_metric_values(raw_vals)
                        if raw_vals is None:
                            continue
                        if not finite_vals:
                            print(
                                f"No finite values for {classwise_metric}: "
                                f"{ds}/{algo}/{noise_state}/class {class_label}/client {client_idx}"
                            )
                            continue
                        data[ds][algo][class_label][client_idx].extend(finite_vals)

    # Optionally aggregate across classes
    if not classwise:
        for ds in target_datasets:
            for algo in target_algos:
                if not data.get(ds) or algo not in data[ds]:
                    continue
                agg = {}
                for cl in data[ds][algo].keys():
                    for client_id, vals in data[ds][algo][cl].items():
                        if client_id not in agg:
                            agg[client_id] = []
                        agg[client_id].extend(vals)
                data[ds][algo] = {"all": agg}

    # Build boxplot structure (per dataset → class → algorithm → client)
    algo_colors = {algo: plt.cm.tab10(i % 10) for i, algo in enumerate(target_algos)}
    positions = []
    box_data = []
    colors = []
    meta_info = []
    dataset_boundaries = []
    class_boundaries = []
    algo_boundaries = []
    client_label_positions = []
    client_label_texts = []

    pos = 0.0
    gap_client = 0.08  # gap between clients within algorithm
    gap_algo = 0.15  # gap between algorithms within class
    gap_class = 0.0  # gap between classes within dataset
    gap_dataset = 0.0  # gap between datasets
    bplot_width = 0.06

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: (
                int(x)
                if isinstance(x, (int, str))
                and str(x).replace("(", "").replace(")", "").split(",")[0].isdigit()
                else str(x)
            ),
        )

        for cl in class_labels_sorted:
            class_has_data = False
            class_positions = []

            for algo in target_algos:
                algo_has_data = False
                algo_positions = []
                algo_label_pos = None

                # Get clients for this algo/class
                clients_dict = data[ds].get(algo, {}).get(cl, {})
                if not clients_dict:
                    continue

                client_ids_sorted = sorted(clients_dict.keys())

                for client_id in client_ids_sorted:
                    vals = clients_dict[client_id]
                    if not vals:
                        continue

                    positions.append(pos)
                    box_data.append(vals)
                    colors.append(algo_colors[algo])
                    meta_info.append(
                        {
                            "ds": ds,
                            "class": cl,
                            "algo": algo,
                            "client": client_id,
                            "n": len(vals),
                        }
                    )
                    # Track label position and text for this client
                    client_label_positions.append(pos)
                    client_label_texts.append(f"FL client {client_id}")
                    algo_positions.append(pos)
                    class_positions.append(pos)
                    pos += gap_client
                    algo_has_data = True
                    dataset_has_data = True
                    class_has_data = True

                if algo_has_data:
                    if algo_label_pos is None:
                        algo_label_pos = np.mean(algo_positions)
                    algo_boundaries.append(max(algo_positions) + gap_client / 2.0)
                    pos += gap_algo

            if class_has_data:
                class_boundaries.append(max(class_positions))
                pos += gap_class

        if dataset_has_data:
            dataset_boundaries.append((dataset_start, positions[-1], ds))
            pos += gap_dataset

    if not box_data:
        print("No per-client roc(p) boxplot data collected.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(max(16, len(target_datasets) * 4), 7))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=bplot_width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # Apply hatching to noisy clients in roc(p) plot
    for patch, meta in zip(bp["boxes"], meta_info):
        ds = meta["ds"]
        client = meta["client"]
        clean_clients = clean_clients_per_dataset.get(ds, [])
        # If client is not in clean clients list, it's noisy - apply hatching
        if client not in clean_clients:
            patch.set_hatch("//")

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
        median.set_alpha(1.0)

    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_alpha(1.0)

    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_alpha(1.0)

    # Add vertical lines between algorithms
    for boundary in algo_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Dashed lines between classes
    for boundary in class_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add dataset labels and separators
    for ds_start, ds_end, ds_name in dataset_boundaries:
        ds_center = (ds_start + ds_end) / 2.0
        ax.text(
            ds_center,
            -0.15,
            ds_name,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(
            ds_end + gap_dataset / 2.0,
            color="gray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(client_label_positions)
    ax.set_xticklabels(client_label_texts, rotation=45, ha="right", fontsize=BOXPLOT_LABEL_FONTSIZE)
    ax.set_ylabel(classwise_metric, fontsize=BOXPLOT_LABEL_FONTSIZE)
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    apply_metric_axis_limits(ax)
    prefix = "Class-wise" if classwise else "Overall"
    suffix = (
        "per dataset/algorithm/client" if classwise else "per dataset/algorithm/client"
    )
    ax.set_title(
        f"{prefix} {classwise_metric} (bootstrapping) - roc(p): Per-client performance {suffix}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    for pos_i, mean, color in zip(positions, means, colors):
        ax.scatter(
            pos_i,
            mean,
            marker="D",
            c=[color],
            edgecolors="k",
            zorder=4,
            s=25,
            linewidths=0.6,
            alpha=1.0,
        )

    # Print statistics per client/algo/class
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(
            f"{meta['ds']} | class {meta['class']} | {meta['algo']} | client {meta['client']}: "
            f"mean={m:.4f} (n={meta['n']})"
        )

    # Legend by algorithm color
    algo_patches = [
        plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a)
        for a in target_algos
    ]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    out_name = (
        f"classwise_boxplots_roc(p)_per_client_bootstrapping_{metric_slug(classwise_metric)}.png"
        if classwise
        else f"boxplots_roc(p)_per_client_bootstrapping_{metric_slug(classwise_metric)}.png"
    )
    fig.savefig(OUTPUT_DIR / out_name, dpi=200, bbox_inches="tight")
    print(f"Saved per-client roc(p) boxplot to {OUTPUT_DIR / out_name}")


def main():
    global classwise_metric, target_datasets

    parser = argparse.ArgumentParser(
        description=(
            "Visualize bootstrap-based segmentation result boxplots for selected "
            "datasets and metrics."
        )
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=SUPPORTED_CLASSWISE_METRICS,
        default=classwise_metric,
        help=(
            f"Bootstrap metric to visualize (default: {classwise_metric}). "
            f"Choices: {SUPPORTED_CLASSWISE_METRICS}"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(target_datasets),
        help=(
            "Optional list of datasets to include; all others are excluded. "
            f"Allowed: {target_datasets}"
        ),
    )
    parser.add_argument(
        "--classwise",
        action="store_true",
        help="If set, keep class-wise boxplots instead of aggregating over classes.",
    )
    args = parser.parse_args()

    classwise_metric = args.metric
    target_datasets = parse_selected_datasets(args.datasets)

    df_selected = df[df["Dataset_norm"].isin(target_datasets)].copy()
    print(
        f"Using metric '{classwise_metric}' for datasets: {target_datasets}. "
        f"Retained {len(df_selected)} rows."
    )

    plot_boxplots_clean_roa_roc_noisy_bootstrapping(
        df_selected, classwise=args.classwise
    )
    # plot_boxplots_roa_per_client_bootstrapping(df_selected, classwise=args.classwise)
    # plot_boxplots_roc_per_client_bootstrapping(df_selected, classwise=args.classwise)


if __name__ == "__main__":
    main()
