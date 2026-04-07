import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

try:
    from sklearn.cluster import HDBSCAN
except Exception:
    try:
        from hdbscan import HDBSCAN  # type: ignore
    except Exception as e:
        raise ImportError(
            "HDBSCAN is not available. Install either scikit-learn>=1.3 or the 'hdbscan' package."
        ) from e

# ---------- CONFIG ----------
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean041-042-043-044_noisy045-046-047-048.json"
# OUT_PNG = "./results/noise_analysis/LIDC_041-048/noise_type_scatter_LIDC.png"
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean300-301-302_noisy303-304-305.json"
# OUT_PNG = "./results/noise_analysis/RIGA_300-305/noise_type_scatter_RIGA.png"
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean436-437-438_noisy439-440-441.json"
# OUT_PNG = "./results/noise_analysis/Gleason_436-441/noise_type_scatter_Gleason.png"
JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean500-501-502-503-504_noisy505-506-507-508-509.json"
OUT_PNG = "./results/noise_analysis/MouseTumor_500-509/noise_type_scatter_MouseTumor.png"
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean700-701-702-703_noisy704-705-706-707.json"
# OUT_PNG = "./results/noise_analysis/MMIS_700-707/noise_type_scatter_MMIS.png"

MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = None
RANDOM_STATE = 0

def infer_dataset_name(sample_id: str) -> str:
    return sample_id.split("_")[0]


def is_finite_number(x) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return x.lower() not in ("infinity", "nan")
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def resolve_json_files(json_path: str):
    if os.path.isdir(json_path):
        return sorted(glob.glob(os.path.join(json_path, "*.json")))
    matches = sorted(glob.glob(json_path))
    if matches:
        return matches
    if os.path.isfile(json_path):
        return [json_path]
    return []


# ---------- LOAD + FEATURE EXTRACT (initial engineered features only) ----------
json_files = resolve_json_files(JSON_PATH)
if not json_files:
    raise FileNotFoundError(f"No JSON file(s) found for JSON_PATH={JSON_PATH}")

print(f"Loading {len(json_files)} JSON file(s) from JSON_PATH")

rows = []
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)

    source_name = os.path.splitext(os.path.basename(json_file))[0]
    for sample_id, entry in data.items():
        overall = entry.get("overall_metrics", {})
        classes = entry.get("classes", {})
        fg_classes = classes.get("fg_classes", [])
        per_class = entry.get("per_class_metrics", {})
        overlap = entry.get("class_overlap_matrix", {})

        # B: boundary
        B = overall.get("mean_hd95", np.nan)
        B = float(B) if is_finite_number(B) else np.nan

        # V: extent (abs + signed)
        rel_abs, rel_signed = [], []
        for c in fg_classes:
            cm = per_class.get(str(c), {})
            rvd = cm.get("relative_volume_diff", np.nan)
            if is_finite_number(rvd):
                rel_abs.append(abs(float(rvd)))
                rel_signed.append(float(rvd))
        V = float(np.mean(rel_abs)) if len(rel_abs) else np.nan
        V_signed = float(np.mean(rel_signed)) if len(rel_signed) else np.nan

        # I: components mismatch
        I = overall.get("delta_total_num_cc", np.nan)
        I = abs(float(I)) if is_finite_number(I) else np.nan

        # S: confusion feature that works for binary and multiclass
        if len(fg_classes) >= 2:
            ss = []
            for c in fg_classes:
                diag = overlap.get(str(c), {}).get(str(c), np.nan)
                if is_finite_number(diag):
                    ss.append(1.0 - float(diag))
            S = float(np.mean(ss)) if len(ss) else np.nan
        else:
            o01 = overlap.get("0", {}).get("1", np.nan)
            o10 = overlap.get("1", {}).get("0", np.nan)
            vals = []
            if is_finite_number(o01):
                vals.append(float(o01))
            if is_finite_number(o10):
                vals.append(float(o10))
            S = float(np.mean(vals)) if len(vals) else np.nan

        row = {
            "sample_id": sample_id,
            "dataset": infer_dataset_name(sample_id),
            "source_json": source_name,
            "global_sample_id": f"{source_name}::{sample_id}",
            "B_hd95": B,
            "V_abs": V,
            "V_signed": V_signed,
            "I_abs_dCC": I,
            "S_conf": S,
        }
        rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise ValueError("No sample entries found in JSON file(s)")

meta_cols = ["sample_id", "dataset", "source_json", "global_sample_id"]
feature_cols = ["B_hd95", "V_abs", "V_signed", "I_abs_dCC", "S_conf"]

# Ensure purely numeric feature matrix
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace missing with per-dataset medians (fallback to global median)
global_medians = df[feature_cols].median(numeric_only=True)
for col in feature_cols:
    df[col] = df.groupby("dataset")[col].transform(lambda s: s.fillna(s.median()))
    df[col] = df[col].fillna(global_medians.get(col, 0.0)).fillna(0.0)

# ---------- SCALE (per dataset, then global) ----------
# Per-dataset robust scaling prevents dataset-specific numeric ranges dominating
scaled_parts = []
for dname, g in df.groupby("dataset", sort=False):
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
    Xg = scaler.fit_transform(g[feature_cols].values)
    scaled_parts.append(pd.DataFrame(Xg, index=g.index, columns=feature_cols))
X_scaled = pd.concat(scaled_parts).loc[df.index]
df_scaled = pd.concat([df[meta_cols], X_scaled], axis=1)

# ---------- CLUSTER ----------
X = df_scaled[feature_cols].values
hdbscan = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
df_scaled["cluster"] = hdbscan.fit_predict(X)

# ---------- PCA (for plotting) ----------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
XY = pca.fit_transform(X)
df_scaled["PC1"] = XY[:, 0]
df_scaled["PC2"] = XY[:, 1]
df_scaled["cluster_name"] = df_scaled["cluster"].apply(
    lambda c: "noise/outlier" if c == -1 else f"cluster_{c}"
)

# ---------- PLOT ----------
plt.figure(figsize=(12, 9))
sc = plt.scatter(df_scaled["PC1"], df_scaled["PC2"], c=df_scaled["cluster"], s=55, alpha=0.8,
                 edgecolors="black", linewidth=0.4)

# annotate cluster names near their PCA means
for k, g in df_scaled.groupby("cluster"):
    plt.text(g["PC1"].mean(), g["PC2"].mean(),
             "noise/outlier" if k == -1 else f"cluster_{k}",
             fontsize=9, weight="bold")

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("HDBSCAN clustering on initial noise features + PCA visualization")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=400)
print(f"Saved: {OUT_PNG}")

# Optional: save a table for inspection
out_csv = OUT_PNG.replace(".png", ".csv")
save_cols = [
    "sample_id",
    "global_sample_id",
    "dataset",
    "source_json",
    "cluster",
    "cluster_name",
    "PC1",
    "PC2",
] + feature_cols
df_scaled[save_cols].to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")