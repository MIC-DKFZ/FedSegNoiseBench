import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
# Pick ONE dataset JSON at a time
JSON_PATH = "./results/noise_analysis/noise_analysis_results_clean500-501-502-503-504_noisy505-506-507-508-509.json"
OUT_PNG   = "./results/noise_analysis/MouseTumor_500-509/noise_4D_map_MouseTumor.png"

# If your dataset naming differs, adapt this.
def infer_dataset_name(sample_id: str) -> str:
    return sample_id.split("_")[0]

# ---------------- HELPERS ----------------
def is_finite_number(x) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return x.lower() not in ("infinity", "nan", "none")
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def safe_float(x, default=np.nan):
    return float(x) if is_finite_number(x) else default

def robust_z(x: pd.Series, eps=1e-12) -> pd.Series:
    """
    Robust z-score using median and IQR.
    Falls back gracefully if IQR is ~0.
    """
    s = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(s.values)
    q25 = np.nanpercentile(s.values, 25)
    q75 = np.nanpercentile(s.values, 75)
    iqr = (q75 - q25)
    scale = iqr if (np.isfinite(iqr) and iqr > eps) else (np.nanstd(s.values) + eps)
    return (s - med) / scale

def compute_confusion_evidence(overlap: dict, fg_classes: list) -> float:
    """
    Foreground-to-other-foreground confusion, excluding background transitions.
    """
    if len(fg_classes) < 2:
        return np.nan
    values = []
    for source_class in fg_classes:
        row_map = overlap.get(str(source_class), {})
        if not any(
            is_finite_number(value) and float(value) > 0.0
            for value in row_map.values()
        ):
            continue
        values.append(sum(
            float(row_map.get(str(target_class), 0.0))
            for target_class in fg_classes
            if target_class != source_class
            and is_finite_number(row_map.get(str(target_class), 0.0))
        ))
    return float(np.mean(values)) if values else np.nan

# ---------------- LOAD ----------------
with open(JSON_PATH, "r") as f:
    data = json.load(f)

rows = []
for sample_id, entry in data.items():
    overall = entry.get("overall_metrics", {})
    classes = entry.get("classes", {})
    fg_classes = classes.get("fg_classes", [])
    per_class = entry.get("per_class_metrics", {})
    overlap = entry.get("class_overlap_matrix", {})

    # B: boundary evidence
    B = safe_float(overall.get("mean_hd95", np.nan))

    # V: extent evidence (abs) + signed
    rel_diffs_abs = []
    rel_diffs_signed = []
    for c in fg_classes:
        cm = per_class.get(str(c), {})
        rvd = cm.get("relative_volume_diff", np.nan)
        if is_finite_number(rvd):
            r = float(rvd)
            rel_diffs_abs.append(abs(r))
            rel_diffs_signed.append(r)

    V = float(np.mean(rel_diffs_abs)) if rel_diffs_abs else np.nan
    V_signed = float(np.mean(rel_diffs_signed)) if rel_diffs_signed else np.nan

    # I: instance evidence
    I = safe_float(overall.get("delta_total_num_cc", np.nan))
    I = abs(I) if np.isfinite(I) else np.nan

    # S: PixelClsConf evidence (undefined for binary datasets)
    S = compute_confusion_evidence(overlap, fg_classes)

    rows.append({
        "sample_id": sample_id,
        "dataset": infer_dataset_name(sample_id),
        "client_idx": entry.get("client_idx", np.nan),
        "B_hd95": B,
        "V_abs_rvd": V,
        "V_signed_rvd": V_signed,
        "I_abs_dCC": I,
        "S_conf": S,
    })

df = pd.DataFrame(rows)

# ---------------- PICK DATASET (if JSON contains multiple) ----------------
# Most of your JSONs are single-dataset already, but this keeps it robust.
datasets = sorted(df["dataset"].dropna().unique().tolist())
if len(datasets) == 0:
    raise RuntimeError("No dataset names inferred. Check infer_dataset_name().")
if len(datasets) > 1:
    # Plot each dataset separately by default (save one PNG per dataset)
    print("Multiple datasets found:", datasets)

# For the requested behavior: plot ALL samples of EACH dataset found in the JSON.
for ds in datasets:
    d = df[df["dataset"] == ds].copy()

    # ---------------- ROBUST NORMALIZATION (per dataset) ----------------
    d["B_rz"] = robust_z(d["B_hd95"])
    d["V_rz"] = robust_z(d["V_abs_rvd"])
    d["I_rz"] = robust_z(d["I_abs_dCC"])
    d["S_rz"] = robust_z(d["S_conf"])

    # ---------------- 4D MAPPING TO 2D+COLOR+SIZE ----------------
    # x: Boundary (B)
    x = d["B_rz"]
    # y: Extent (V)
    y = d["V_rz"]
    # color: Confusion (S)
    c = d["S_rz"]

    # size: Instance (I)  -> map robust z to a stable size range
    # Use clipped z and then scale.
    i_clipped = np.clip(d["I_rz"].fillna(0.0).values, -2.0, 4.0)
    # Convert to [0..1] then to size
    i01 = (i_clipped - (-2.0)) / (4.0 - (-2.0))
    sizes = 20 + 180 * i01  # 20..200

    # marker to show missing vs additional based on signed volume diff
    # missing: V_signed < 0 -> triangle down, additional: V_signed > 0 -> triangle up, near 0 -> circle
    vs = d["V_signed_rvd"].fillna(0.0).values
    missing = vs < -0.01
    additional = vs > 0.01
    neutral = ~(missing | additional)

    plt.figure(figsize=(12, 9))

    # Plot in three layers so markers differ
    # (All share same colormap/colorbar)
    sc_neu = plt.scatter(x[neutral], y[neutral], c=c[neutral], s=sizes[neutral],
                         alpha=0.75, edgecolors="black", linewidth=0.4, marker="o")
    plt.scatter(x[missing], y[missing], c=c[missing], s=sizes[missing],
                alpha=0.75, edgecolors="black", linewidth=0.4, marker="v")
    plt.scatter(x[additional], y[additional], c=c[additional], s=sizes[additional],
                alpha=0.75, edgecolors="black", linewidth=0.4, marker="^")

    # Optional tiny labels (can get crowded)
    for _, r in d.iterrows():
        plt.annotate(r["sample_id"], (r["B_rz"], r["V_rz"]),
                     xytext=(3, 3), textcoords="offset points",
                     fontsize=4, alpha=0.7)

    plt.axhline(0, linewidth=0.6)
    plt.axvline(0, linewidth=0.6)
    plt.grid(True, linewidth=0.3, alpha=0.5)

    plt.xlabel("Boundary evidence B (robust z of mean HD95)")
    plt.ylabel("Extent evidence V (robust z of mean |relative volume diff|)")
    cb = plt.colorbar(sc_neu)
    cb.set_label("Confusion evidence S (robust z of foreground-class transitions)")

    plt.title(
        f"{ds}: 4D noise map\n"
        f"x=B (boundary), y=V (extent), color=S (confusion), size=I (instance); "
        f"marker=missing/additional by sign(V_signed)"
    )

    out = OUT_PNG
    # If multiple datasets, suffix output so you don’t overwrite
    if len(datasets) > 1:
        stem, ext = OUT_PNG.rsplit(".", 1)
        out = f"{stem}_{ds}.{ext}"

    plt.tight_layout()
    plt.savefig(out, dpi=500)
    plt.close()
    print(f"Saved: {out}")
