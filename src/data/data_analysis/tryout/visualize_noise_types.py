import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean041-042-043-044_noisy045-046-047-048.json"
OUT_PNG = "./results/noise_analysis/LIDC_041-048/noise_type_scatter_LIDC.png"
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean300-301-302_noisy303-304-305.json"
# OUT_PNG = "./results/noise_analysis/RIGA_300-305/noise_type_scatter_RIGA.png"
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean436-437-438_noisy439-440-441.json"
# OUT_PNG = "./results/noise_analysis/Gleason_436-441/noise_type_scatter_Gleason.png"
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean500-501-502-503-504_noisy505-506-507-508-509.json"
# OUT_PNG = "./results/noise_analysis/MouseTumor_500-509/noise_type_scatter_MouseTumor.png"
# JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean700-701-702-703_noisy704-705-706-707.json"
# OUT_PNG = "./results/noise_analysis/MMIS_700-707/noise_type_scatter_MMIS.png"


# ---------- DATASET NAME INFERENCE ----------
def dataset_from_paths(json_path: str, out_png: str) -> str:
    """
    Try to infer a stable dataset name even if sample_ids don't contain underscores.
    Priority:
      1) parent folder name of OUT_PNG like ".../MouseTumor_500-509/..."
      2) a token in OUT_PNG filename like "..._MouseTumor.png"
      3) fallback: "dataset"
    """
    parent = os.path.basename(os.path.dirname(out_png))
    if parent:
        return parent

    base = os.path.basename(out_png)
    m = re.search(r"_scatter_(.+?)\.png$", base)
    if m:
        return m.group(1)

    return "dataset"

DEFAULT_DATASET = dataset_from_paths(JSON_PATH, OUT_PNG)

def infer_dataset_name(sample_id: str) -> str:
    """
    If IDs look like 'LIDC_...' keep prefix.
    If there's no underscore (MouseTumor-style: ds1M010h), use DEFAULT_DATASET.
    """
    if "_" in sample_id:
        return sample_id.split("_")[0]
    return DEFAULT_DATASET

# ---------- HELPERS ----------
def is_finite_number(x) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return x.lower() not in {"infinity", "nan"}
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank within group; NaNs filled with median for stable ranking."""
    finite = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(finite.values) if np.any(np.isfinite(finite.values)) else 0.0
    filled = finite.fillna(med)
    return filled.rank(pct=True, method="average")

# ---------- LOAD ----------
with open(JSON_PATH, "r") as f:
    data = json.load(f)

rows = []
for sample_id, entry in data.items():
    overall = entry.get("overall_metrics", {})
    classes = entry.get("classes", {})
    fg_classes = classes.get("fg_classes", [])
    per_class = entry.get("per_class_metrics", {})
    overlap = entry.get("class_overlap_matrix", {})

    # Boundary disagreement B: mean_hd95 (ignore non-finite)
    B = overall.get("mean_hd95", np.nan)
    B = float(B) if is_finite_number(B) else np.nan

    # Extent disagreement V: mean abs(relative_volume_diff) over foreground classes
    rel_diffs = []
    for c in fg_classes:
        cm = per_class.get(str(c), {})
        rvd = cm.get("relative_volume_diff", np.nan)
        if is_finite_number(rvd):
            rel_diffs.append(abs(float(rvd)))
    V = float(np.mean(rel_diffs)) if len(rel_diffs) else np.nan

    # Instance disagreement I: abs(delta_total_num_cc)
    I = overall.get("delta_total_num_cc", np.nan)
    I = abs(float(I)) if is_finite_number(I) else np.nan

    # Swap disagreement S: foreground-to-other-foreground transitions only.
    if len(fg_classes) >= 2:
        swap_scores = []
        for source_class in fg_classes:
            row_map = overlap.get(str(source_class), {})
            if not any(
                is_finite_number(value) and float(value) > 0.0
                for value in row_map.values()
            ):
                continue
            swap_scores.append(sum(
                float(row_map.get(str(target_class), 0.0))
                for target_class in fg_classes
                if target_class != source_class
                and is_finite_number(row_map.get(str(target_class), 0.0))
            ))
        S = float(np.mean(swap_scores)) if len(swap_scores) else np.nan
    else:
        S = np.nan

    rows.append({
        "sample_id": sample_id,
        "dataset": infer_dataset_name(sample_id),
        "client_idx": entry.get("client_idx", np.nan),
        "B_hd95": B,
        "V_abs_relvol": V,
        "I_abs_dCC": I,
        "S_swap": S,
    })

df = pd.DataFrame(rows)

# ---------- NORMALIZE PER DATASET (percentiles) ----------
df["B_p"] = df.groupby("dataset")["B_hd95"].transform(pct_rank)
df["V_p"] = df.groupby("dataset")["V_abs_relvol"].transform(pct_rank)
df["I_p"] = df.groupby("dataset")["I_abs_dCC"].transform(pct_rank)
df["S_p"] = df.groupby("dataset")["S_swap"].transform(pct_rank)

# Missed/additional composite M: equal weight of volume and CC disagreement percentiles
df["M_p"] = 0.5 * df["V_p"] + 0.5 * df["I_p"]

# ---------- PLOT ----------
plt.figure(figsize=(12, 9))
sc = plt.scatter(
    df["B_p"], df["M_p"],
    c=df["S_p"],
    s=50, alpha=0.7,
    edgecolors="black", linewidth=0.5
)

for _, row in df.iterrows():
    plt.annotate(
        row["sample_id"],
        (row["B_p"], row["M_p"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=2,
        alpha=0.75,
        ha="left",
        va="bottom",
    )

plt.xlabel("Boundary disagreement (percentile of mean HD95 within dataset)")
plt.ylabel("Missed/additional disagreement (percentile of |ΔV| and |ΔCC| within dataset)")
cbar = plt.colorbar(sc)
cbar.set_label("PixelClsConf (percentile of foreground-class transitions)")
plt.title("Sample-level label-noise map: contour vs missed/additional vs swapped labels")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=500)
print(f"Saved: {OUT_PNG}")
