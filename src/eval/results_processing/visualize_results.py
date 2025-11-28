import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

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

target_algos = ["FedAvg", "FedA3I", "IOP-FL"]
target_datasets = ["LIDC", "RIGA", "Gleason", "MouseTumor", "MMIA"]
noise_order = ["0", "roa(X)", "roc(X)", "100"]  # plotting order

OUTPUT_DIR = Path("./visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# -------------------------------------------------------------------
# Generate plots
# -------------------------------------------------------------------

# plot per algorithm
for algo in target_algos:
    plot_algorithm(df, algo)
# plt.show()

# plot per dataset
for ds in target_datasets:
    plot_dataset(df, ds)
# plt.show()