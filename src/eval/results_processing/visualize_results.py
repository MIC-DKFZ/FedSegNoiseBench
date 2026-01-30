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
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

algo_col = "Algo"
raw_dataset_col = "Data"
noise_col = "Noise"
metric_col = "Mean(D_val)"

target_algos = ["FedAvg", "FedA3I", "IOP-FL", "FedCorr"]
target_datasets = ["LIDC", "RIGA", "Gleason", "MouseTumor", "MMIA", "MMIS"]
noise_order = ["0", "roa(X)", "roc(X)", "100"]  # plotting order

OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# for reading results from experiment's summary.json
nnUNet_results = Path("/home/m391k/cluster-data/checkpoints/nnUNet_results")
nnUNet_results_all_exps = glob.glob(str(nnUNet_results / "*" / "*" / "fold_*" / "*"))
dataset_to_numclients = {
    "LIDC": 4,
    "RIGA": 3,
    "Gleason": 3,
    "MouseTumor": 5,
    "MMIA": 4,
    "MMIS": 4,
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

# -------------------------------------------------------------------
# Load and pre-process
# -------------------------------------------------------------------
df = pd.read_csv(csv_url)
print(f"Loaded {len(df)} rows from Google Sheets.")

# Basic cleaning for algorithm and dataset columns
for c in [algo_col, raw_dataset_col]:
    df[c] = df[c].astype(str).str.strip()

# Clean metric column: replace comma decimal separators with dot, remove percent signs/spaces, and convert to float.
# This ensures values like "0,01" become 0.01 and allows mean() to average across folds.
df[metric_col] = df[metric_col].astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False).str.strip()
df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

# Normalize noise values into canonical buckets:
# - exact 0/100 -> "0"/"100"
# - any roa(...) or contains "roa" -> "roa(X)"
# - any roc(...) or contains "roc" -> "roc(X)"
def _normalize_noise_val(v: object) -> str:
    s = "" if v is None else str(v).strip()
    if re.fullmatch(r'(?i)0(?:\.0+)?', s):
        return "0"
    if re.fullmatch(r'(?i)100(?:\.0+)?', s):
        return "100"
    if re.search(r'(?i)\broa\b', s) or re.search(r'(?i)roa\s*\(.*\)', s):
        return "roa(X)"
    if re.search(r'(?i)\broc\b', s) or re.search(r'(?i)roc\s*\(.*\)', s):
        return "roc(X)"
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

# Standardize noise to strings such as "0", "100", "roa(X)", "roc(X)"
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
        ax.plot(x, y, marker='o', linestyle='-', color=color, label=ds)
        print(f"{algo_name} - {ds}:\n{s.dropna()}\n")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, rotation=45, ha="right")
    ax.set_xlabel("Noise")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{algo_name}: Mean(D_val) per dataset & noise (overlay)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
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
        ax.plot(x, y, marker='o', linestyle='-', color=color, label=algo)
        print(f"{dataset_name} - {algo}:\n{s.dropna()}\n")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, rotation=45, ha="right")
    ax.set_xlabel("Noise")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{dataset_name}: Mean(D_val) per algorithm & noise (overlay)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
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
            assert len(exp_paths) == dataset_to_numclients[dataset_name], f"Did not find expected number of experiment paths for Exp_ID {exp_id}. Expected {dataset_to_numclients[dataset_name]}, found {len(exp_paths)}"
            # find and load validation/summary.json from all clients and average
            client_classwise_metrics = []
            for exp_path in exp_paths:
                results_summary = json.load(open(exp_path + "/validation/summary.json", 'r'))
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
                df_algo.loc[df_algo["Experiment ID"] == exp_id, class_metric_col] = metrics[classwise_metric]
                class_metric_cols.append(class_metric_col) if class_metric_col not in class_metric_cols else None

        # plotting color
        color = plt.cm.tab10(i % 10)
        # Reindex to canonical noise_order so x-axis is consistent across algorithms
        for i, class_metric_col in enumerate(class_metric_cols):
            s = df_algo.groupby(noise_col)[class_metric_col].mean().reindex(noise_order)
            if s.dropna().empty:
                print(f"{dataset_name} - {algo}: no matching noise buckets")
                continue
            
            y = s.values
            linestyle = class_line_marker[i+1]["linestyle"]
            marker = class_line_marker[i+1]["marker"]
            ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, label=f"{algo} - {class_metric_col}")
            print(f"{dataset_name} - {algo} - {class_metric_col}:\n{s.dropna()}\n")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, rotation=45, ha="right")
    ax.set_xlabel("Noise")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{dataset_name}: Mean(D_val) per algorithm & noise (overlay)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title="Algorithm", loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"dataset_{dataset_name}_algo_classwise_metrics.png", dpi=200)


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
                exp_paths = [p for p in nnUNet_results_all_exps if exp_id in p]
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients, found {len(exp_paths)}")
                    continue
                if len(exp_paths) == 0:
                    print(f"No exp paths found for Exp_ID {exp_id} ({ds}/{algo})")
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
                            cm[class_label][metric_name] for cm in client_classwise_metrics
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
    label_texts = []      # "clean" or "noisy"

    pos = 0.0
    gap_algo = 0.1    # gap between algorithms within the same state
    gap_noise = 0.05   # gap between clean/noisy within a class
    gap_class = 0.0   # gap between classes within a dataset
    gap_dataset = 0.0 # gap between datasets
    bplot_width = 0.1

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).replace('(', '').replace(')', '').split(',')[0].isdigit() else str(x)
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
                    meta_info.append({"ds": ds, "class": cl, "state": state_key, "algo": algo, "n": len(vals)})
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
    fig, ax = plt.subplots(figsize=(max(12, len(target_datasets) * 2.5), 7))

    bp = ax.boxplot(box_data, positions=positions, widths=bplot_width, patch_artist=True, showfliers=False)

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    for median in bp['medians']:
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
        ax.text(ds_center, -0.15, ds_name, ha='center', va='top', fontsize=11, fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.axvline(ds_end + gap_dataset / 2.0, color="gray", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel(classwise_metric, fontsize=11)
    # Tighten in-axes horizontal margins
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Class-wise Dice: clean vs noisy per dataset/class/algorithm", fontsize=12, fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    ax.scatter(positions, means, marker='D', c=colors, edgecolors='k', zorder=4, s=30, linewidths=0.6, label=None)
    # Print means per boxplot
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(f"{meta['ds']} | class {meta['class']} | {meta['state']} | {meta['algo']}: "
              f"mean={m:.4f} (n={meta['n']})")

    # Legend by algorithm color
    algo_patches = [plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a) for a in target_algos]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(OUTPUT_DIR / "classwise_boxplots_clean_vs_noisy.png", dpi=200, bbox_inches='tight')
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
                exp_paths = [p for p in nnUNet_results_all_exps if exp_id in p]
                expected_clients = dataset_to_numclients.get(ds, None)
                if expected_clients and len(exp_paths) != expected_clients:
                    print(f"Skipping Exp_ID {exp_id} for {ds}/{algo}: expected {expected_clients} clients, found {len(exp_paths)}")
                    continue
                if len(exp_paths) == 0:
                    print(f"No exp paths found for Exp_ID {exp_id} ({ds}/{algo})")
                    continue

                client_classwise_metrics = []
                for exp_path in exp_paths:
                    bootstrap_file = Path(exp_path) / "validation" / "bootstrap_evaluation_results.json"
                    if not bootstrap_file.is_file():
                        print(f"Missing bootstrap_evaluation_results.json at {bootstrap_file}")
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
    label_texts = []      # "clean" or "noisy"

    pos = 0.0
    gap_algo = 0.1    # gap between algorithms within the same state
    gap_noise = 0.05   # gap between clean/noisy within a class
    gap_class = 0.0   # gap between classes within a dataset
    gap_dataset = 0.0 # gap between datasets
    bplot_width = 0.1

    for ds in target_datasets:
        if not data.get(ds):
            continue

        dataset_start = pos
        dataset_has_data = False
        class_labels_sorted = sorted(
            {cl for algo_data in data[ds].values() for cl in algo_data.keys()},
            key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).replace('(', '').replace(')', '').split(',')[0].isdigit() else str(x)
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
                    meta_info.append({"ds": ds, "class": cl, "state": state_key, "algo": algo, "n": len(vals)})
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

    bp = ax.boxplot(box_data, positions=positions, widths=bplot_width, patch_artist=True, showfliers=False)

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    for median in bp['medians']:
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
        ax.text(ds_center, -0.15, ds_name, ha='center', va='top', fontsize=11, fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.axvline(ds_end + gap_dataset / 2.0, color="gray", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel(classwise_metric, fontsize=11)
    # Tighten in-axes horizontal margins
    xmin = min(positions) - bplot_width
    xmax = max(positions) + bplot_width
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Class-wise Dice (bootstrapping): clean vs noisy per dataset/class/algorithm", fontsize=12, fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay mean markers
    means = [np.mean(vals) if len(vals) else np.nan for vals in box_data]
    ax.scatter(positions, means, marker='D', c=colors, edgecolors='k', zorder=4, s=30, linewidths=0.6, label=None)
    # Print means per boxplot
    for vals, m, meta in zip(box_data, means, meta_info):
        if np.isnan(m):
            continue
        print(f"{meta['ds']} | class {meta['class']} | {meta['state']} | {meta['algo']}: "
              f"mean={m:.4f} (n={meta['n']})")

    # Legend by algorithm color
    algo_patches = [plt.matplotlib.patches.Patch(facecolor=algo_colors[a], alpha=0.75, label=a) for a in target_algos]
    ax.legend(handles=algo_patches, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(OUTPUT_DIR / "classwise_boxplots_clean_vs_noisy_bootstrapping.png", dpi=200, bbox_inches='tight')
    print(f"Saved boxplot to {OUTPUT_DIR / 'classwise_boxplots_clean_vs_noisy_bootstrapping.png'}")

# -------------------------------------------------------------------
# Generate plots
# -------------------------------------------------------------------

# # plot per algorithm
# for algo in target_algos:
#     plot_algorithm(df, algo)

# # plot per dataset
# for ds in target_datasets:
#     plot_dataset(df, ds)

# # plot per dataset and per class label
# for ds in target_datasets:
#     plot_dataset_per_class(df, ds)

# # boxplots clean vs noisy per dataset/class across methods
# plot_classwise_boxplots_clean_noisy(df)

# boxplots clean vs noisy per dataset/class across methods from bootstrapping
plot_classwise_boxplots_clean_noisy_bootstrapping(df)
