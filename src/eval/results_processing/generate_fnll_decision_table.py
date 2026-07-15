import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ranking import OUTPUT_DIR
except ImportError:
    from ranking import OUTPUT_DIR


ROWS = [
    ("Contour", "HD95", "HD95"),
    ("Instance", "F1", "FgBgInstanceF1"),
    ("Pixel confusion", "PixClsConf", "PixelClsConf"),
    ("Instance confusion", "InstClsConf", "InstanceClsConf"),
    ("General", "Dice", "Dice"),
]
SCENARIOS = ["clean", "roa", "roc", "noisy", "overall"]
METRICS = {m for _, _, m in ROWS}
BASELINE = "FedAvg"

def truthy(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def tex(x):
    return str(x).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

def parse_overrides(items):
    out = {}
    for item in items or []:
        metric, sep, path = item.partition("=")
        if not sep or metric not in METRICS:
            raise ValueError(f"--rank-csv must be Metric=/path.csv; metrics={sorted(METRICS)}")
        out[metric] = Path(path)
    return out

def rank_csv(metric, ranking_dir, overrides):
    if metric in overrides:
        if not overrides[metric].is_file():
            raise FileNotFoundError(overrides[metric])
        return overrides[metric]
    token = metric.lower()
    matches = list(ranking_dir.glob(f"rank_frequency_summary_{token}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No rank_frequency_summary CSV for {metric} in {ranking_dir}")

    def score(path):
        suffix = re.sub(rf"^rank_frequency_summary_{re.escape(token)}_", "", path.stem)
        return len([p for p in suffix.split("_") if p]), path.stat().st_mtime

    return sorted(matches, key=score)[-1]

def best_method(path, scenario):
    df = pd.read_csv(path)
    for col in ["mean_rank", "median_rank", "rank", "frequency"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df["available"] = df["available"].map(truthy)
    if scenario == "overall":
        keep = (df.scope == "overall") & (df.dataset == "ALL") & (df.noise_scenario == "all")
    else:
        keep = (
            (df.scope == "all_datasets")
            & (df.dataset == "ALL")
            & (df.noise_scenario == scenario)
        )
    df = df[keep & df.available & df.mean_rank.notna()].copy()
    if df.empty:
        return "", np.nan
    df = df.sort_values(
        ["mean_rank", "median_rank", "rank", "frequency", "algorithm"],
        ascending=[True, True, True, False, True],
    ).drop_duplicates("algorithm")
    return str(df.iloc[0].algorithm), float(df.iloc[0].mean_rank)

def significance(stats, metric, scenario, method):
    if not method or method == BASELINE:
        return False, False, np.nan, np.nan
    scope = "pooled_dataset_x_scenario_per_metric" if scenario == "overall" else "per_metric_per_scenario"
    stats_scenario = "ALL" if scenario == "overall" else scenario
    rows = stats[
        (stats.scope == scope)
        & (stats.metric == metric)
        & (stats.noise_scenario == stats_scenario)
        & (stats.method == method)
    ]
    if rows.empty:
        return False, False, np.nan, np.nan
    r = rows.iloc[0]
    return truthy(r.reject_uncorrected_alpha), truthy(r.reject_holm_alpha), r.p_value, r.p_value_holm

def latex_cell(method, uncorrected, holm):
    if not method:
        return r"\textemdash{}"
    value = tex(method)
    if holm:
        return rf"\textbf{{{value}}}"
    return rf"\underline{{{value}}}" if uncorrected else value

def collect_records(paths, stats):
    rows = []
    for noise_type, metric_label, metric in ROWS:
        for scenario in SCENARIOS:
            method, mean_rank = best_method(paths[metric], scenario)
            uncorrected, holm, p, p_holm = significance(stats, metric, scenario, method)
            rows.append(
                dict(
                    noise_type=noise_type,
                    metric_label=metric_label,
                    metric=metric,
                    scenario=scenario,
                    algorithm=method,
                    mean_rank=mean_rank,
                    significant_uncorrected=uncorrected,
                    significant_holm=holm,
                    p_value=p,
                    p_value_holm=p_holm,
                    rank_csv=paths[metric],
                )
            )
    return pd.DataFrame(rows)

def render(records, caption, label, include_note=True):
    by_cell = {(r.metric, r.scenario): r for r in records.itertuples()}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{tex(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Noise type (metric)} & \multicolumn{5}{c}{\textbf{Client-noise scenario}} \\",
        r"\cmidrule(lr){2-6}",
        r" & \textbf{clean} & \textbf{\textit{roa}} & \textbf{\textit{roc}} & \textbf{noisy} & \textbf{overall} \\",
        r"\midrule",
    ]
    for i, (noise_type, metric_label, metric) in enumerate(ROWS):
        if i == len(ROWS) - 1:
            lines.append(r"\midrule")
        cells = []
        for scenario in SCENARIOS:
            r = by_cell.get((metric, scenario))
            cells.append(
                latex_cell(r.algorithm, r.significant_uncorrected, r.significant_holm)
                if r is not None
                else r"\textemdash{}"
            )
        lines.append(rf"\textbf{{{tex(noise_type)}}} ({tex(metric_label)}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if include_note:
        lines += [
            r"\vspace{0.35em}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Recommended methods are selected by the lowest mean bootstrap rank for each metric and scenario. "
            r"Underlined entries indicate uncorrected $p<0.05$ versus FedAvg; "
            r"bold entries indicate Holm--Bonferroni corrected significance.",
            r"\end{minipage}",
        ]
    return "\n".join(lines + [r"\end{table}", ""])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking-dir", type=Path, default=OUTPUT_DIR / "ranking_stability")
    p.add_argument("--rank-csv", action="append")
    p.add_argument("--stats-csv", type=Path, default=OUTPUT_DIR / "fnll_vs_fedavg_wilcoxon_holm.csv")
    p.add_argument("--output-tex", type=Path, default=OUTPUT_DIR / "fnll_decision_guide_table.tex")
    p.add_argument("--output-csv", type=Path, default=OUTPUT_DIR / "fnll_decision_guide_table.csv")
    p.add_argument("--label", default="tab:fnll_decision_guide")
    p.add_argument(
        "--caption",
        default="Decision guide for selecting FNLL methods according to the dominant segmentation label-noise type and client-noise scenario.",
    )
    p.add_argument("--no-note", action="store_true")
    args = p.parse_args()

    overrides = parse_overrides(args.rank_csv)
    paths = {metric: rank_csv(metric, args.ranking_dir, overrides) for metric in METRICS}
    records = collect_records(paths, pd.read_csv(args.stats_csv))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_csv, index=False)
    args.output_tex.write_text(render(records, args.caption, args.label, not args.no_note))

    print("\n".join(f"{metric}: {path}" for metric, path in sorted(paths.items())))
    print(f"Saved {args.output_tex}")
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
