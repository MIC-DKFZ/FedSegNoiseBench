import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    from .ranking import OUTPUT_DIR, algo_col, exp_id_col, target_datasets
    from .visualize_ranking import (
        LOWER_IS_BETTER_METRICS,
        NOISE_ORDER,
        SUPPORTED_METRICS,
        build_experiment_path_index_for_roots,
        filter_records_to_included_folds,
        load_and_preprocess_results,
        parse_selected_datasets,
        resolve_nnunet_results_roots,
    )
except ImportError:
    from ranking import OUTPUT_DIR, algo_col, exp_id_col, target_datasets
    from visualize_ranking import (
        LOWER_IS_BETTER_METRICS,
        NOISE_ORDER,
        SUPPORTED_METRICS,
        build_experiment_path_index_for_roots,
        filter_records_to_included_folds,
        load_and_preprocess_results,
        parse_selected_datasets,
        resolve_nnunet_results_roots,
    )


BASELINE = "FedAvg"
METHODS = ["FedSelect", "IOP-FL", "FedCorr", "FedA3I"]
LATEX_METHOD_ORDER = ["FedA3I", "IOP-FL", "FedCorr", "FedSelect"]
DEFAULT_METRICS = ["Dice", "HD95", "FgBgInstanceF1", "ClassConfusion"]
DEFAULT_OUTPUT = OUTPUT_DIR / "fnll_vs_fedavg_wilcoxon_holm.csv"
LATEX_METRIC_ORDER = ["Dice", "HD95", "FgBgInstanceF1", "ClassConfusion"]
LATEX_METRIC_LABELS = {
    "Dice": "Dice",
    "HD95": "HD95",
    "FgBgInstanceF1": "F1",
    "ClassConfusion": "ClsConf",
}
LATEX_SCENARIO_ORDER = ["clean", "roa", "roc", "noisy", "ALL"]
LATEX_SCOPES = {
    "per_metric_per_scenario": "Per-metric, per-scenario Wilcoxon signed-rank tests against FedAvg.",
    "pooled_dataset_x_scenario_per_metric": "Pooled dataset-by-scenario Wilcoxon signed-rank tests against FedAvg.",
}
PAIR_COLUMNS = [
    "case_id",
    "method_score",
    "baseline_score",
    "dataset",
    "noise_scenario",
    "pair_id",
    "improvement",
]


def parse_metric_dataset_overrides(items):
    out = {}
    for item in items or []:
        metric, sep, raw = item.replace(":", "=", 1).partition("=")
        if not sep or metric not in SUPPORTED_METRICS:
            raise ValueError(f"Use Metric=DatasetA,DatasetB. Metrics: {SUPPORTED_METRICS}")
        out[metric] = parse_selected_datasets([raw])
    return out


def holm(rows, alpha):
    finite = sorted(
        [(i, float(r["p_value"])) for i, r in enumerate(rows) if np.isfinite(r["p_value"])],
        key=lambda x: x[1],
    )
    adjusted, rejected = [np.nan] * len(rows), [False] * len(rows)
    running_max, stopped, m = 0.0, False, len(finite)
    for rank, (idx, p) in enumerate(finite, start=1):
        factor = m - rank + 1
        running_max = max(running_max, min(1.0, factor * p))
        adjusted[idx] = running_max
        if not stopped and p <= alpha / factor:
            rejected[idx] = True
        else:
            stopped = True
    for row, p_adj, reject in zip(rows, adjusted, rejected):
        row.update(correction="Holm-Bonferroni", p_value_holm=p_adj, reject_holm_alpha=reject, alpha=alpha)
    return rows


def case_id(path):
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def mean_metric(case_metrics, metric):
    vals = []
    for label_metrics in case_metrics.values():
        if not isinstance(label_metrics, dict) or metric not in label_metrics:
            continue
        try:
            val = float(label_metrics[metric])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else np.nan


def read_summary(summary_file, metric):
    with open(summary_file) as f:
        summary = json.load(f)
    rows = []
    for case in summary.get("metric_per_case", []):
        score = mean_metric(case.get("metrics", {}), metric)
        if np.isfinite(score):
            rows.append(
                {
                    "case_id": case_id(case.get("reference_file") or case.get("prediction_file", "")),
                    "score": score,
                }
            )
    return rows


def collect_case_scores(sheet_df, exp_paths, metric):
    cache, rows = {}, []
    for algo, dataset, scenario, exp_id in sheet_df[[algo_col, "Dataset_norm", "noise_scenario", exp_id_col]].itertuples(index=False):
        matches = [p for p in exp_paths if str(exp_id) in str(p) and (p / "validation" / "summary_reran.json").is_file()]
        if not matches:
            print(f"No summary_reran.json for {exp_id} ({dataset}/{algo}/{scenario})")
            continue
        for exp_path in matches:
            summary_file = exp_path / "validation" / "summary_reran.json"
            cache.setdefault(summary_file, read_summary(summary_file, metric))
            for r in cache[summary_file]:
                rows.append(
                    {
                        "metric": metric,
                        "algorithm": algo,
                        "dataset": dataset,
                        "noise_scenario": scenario,
                        "case_id": r["case_id"],
                        "score": r["score"],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["metric", "algorithm", "dataset", "noise_scenario", "case_id", "score"])
    return pd.DataFrame(rows).groupby(
        ["metric", "algorithm", "dataset", "noise_scenario", "case_id"],
        as_index=False,
    )["score"].mean()


def directional(metric, method_score, baseline_score):
    return baseline_score - method_score if metric in LOWER_IS_BETTER_METRICS else method_score - baseline_score


def paired_cases(scores, metric, method, dataset, scenario):
    subset = scores[
        (scores.metric == metric)
        & (scores.dataset == dataset)
        & (scores.noise_scenario == scenario)
    ]
    method_scores = subset[subset.algorithm == method][["case_id", "score"]].rename(columns={"score": "method_score"})
    baseline = subset[subset.algorithm == BASELINE][["case_id", "score"]].rename(columns={"score": "baseline_score"})
    paired = method_scores.merge(baseline, on="case_id", how="inner")
    paired["dataset"], paired["noise_scenario"] = dataset, scenario
    paired["pair_id"] = dataset + ":" + scenario + ":" + paired["case_id"].astype(str)
    paired["improvement"] = [directional(metric, m, b) for m, b in zip(paired.method_score, paired.baseline_score)]
    return paired[PAIR_COLUMNS]


def wilcoxon_greater(improvements):
    arr = np.asarray(improvements, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan, np.nan, "too few paired observations"
    if np.allclose(arr, 0):
        return 0.0, 1.0, "all paired improvements are zero"
    try:
        res = wilcoxon(arr, alternative="greater", zero_method="wilcox", method="auto")
    except TypeError:
        res = wilcoxon(arr, alternative="greater", zero_method="wilcox")
    except ValueError as exc:
        return np.nan, np.nan, str(exc)
    return float(res.statistic), float(res.pvalue), ""


def format_delta(value):
    if not np.isfinite(value):
        return ""
    return f"{float(value):+.4f}"


def format_p_value(value):
    if not np.isfinite(value):
        return ""
    value = float(value)
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_uncorrected_p_value(value, reject):
    formatted = format_p_value(value)
    return rf"\underline{{{formatted}}}" if formatted and bool(reject) else formatted


def format_holm_p_value(value, reject):
    formatted = format_p_value(value)
    return rf"\textbf{{{formatted}}}" if formatted and bool(reject) else formatted


def row_value(row, *names):
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return np.nan


def latex_output_path(output_csv, scope):
    return output_csv.with_name(f"{output_csv.stem}_{scope}.tex")


def latex_sort_key(row):
    metric_rank = {metric: i for i, metric in enumerate(LATEX_METRIC_ORDER)}
    scenario_rank = {scenario: i for i, scenario in enumerate(LATEX_SCENARIO_ORDER)}
    method_rank = {method: i for i, method in enumerate(LATEX_METHOD_ORDER)}
    return (
        metric_rank.get(row.metric, len(metric_rank)),
        scenario_rank.get(row.noise_scenario, len(scenario_rank)),
        method_rank.get(row.method, len(method_rank)),
    )


def build_latex_table(result, scope):
    scoped = result[result["scope"] == scope].copy()
    rows = sorted(scoped.itertuples(index=False), key=latex_sort_key)
    caption = LATEX_SCOPES[scope]
    use_longtable = scope == "per_metric_per_scenario"

    header = [
        r"\toprule",
        r"Metric & Scenario & Method & $n$ & $\Delta_{mean}$ & $\Delta_{median}$ & $p$ & $p_{Holm}$ \\",
        r"\midrule",
    ]

    if use_longtable:
        lines = [
            r"\begin{longtable}{lllrrrrr}",
            rf"\caption{{{caption}}}\label{{tab:fnll_wilcoxon_{scope}}}\\",
            *header,
            r"\endfirsthead",
            rf"\caption[]{{{caption} (continued)}}\\",
            *header,
            r"\endhead",
            r"\midrule",
            r"\multicolumn{8}{r}{Continued on next page} \\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    else:
        lines = [
            r"\begin{table}",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:fnll_wilcoxon_{scope}}}",
            r"\begin{tabular}{lllrrrrr}",
            *header,
        ]

    if rows:
        metric_counts = {}
        scenario_counts = {}
        for row in rows:
            metric_counts[row.metric] = metric_counts.get(row.metric, 0) + 1
            scenario_key = (row.metric, row.noise_scenario)
            scenario_counts[scenario_key] = scenario_counts.get(scenario_key, 0) + 1

        seen_metrics = set()
        seen_scenarios = set()
        previous_metric = None
        previous_scenario_key = None
        for row in rows:
            scenario_key = (row.metric, row.noise_scenario)
            if previous_metric is not None and row.metric != previous_metric:
                lines.append(r"\midrule")
            elif previous_scenario_key is not None and scenario_key != previous_scenario_key:
                lines.append(r"\cmidrule(lr){2-8}")

            metric_label = LATEX_METRIC_LABELS.get(row.metric, row.metric)
            metric_cell = ""
            if row.metric not in seen_metrics:
                metric_cell = rf"\multirow{{{metric_counts[row.metric]}}}{{*}}{{{metric_label}}}"
                seen_metrics.add(row.metric)

            scenario_cell = ""
            if scenario_key not in seen_scenarios:
                scenario_cell = rf"\multirow{{{scenario_counts[scenario_key]}}}{{*}}{{{row.noise_scenario}}}"
                seen_scenarios.add(scenario_key)

            p_value = format_uncorrected_p_value(row.p_value, row.reject_uncorrected_alpha)
            p_holm = format_holm_p_value(row.p_value_holm, row.reject_holm_alpha)
            delta_mean = row_value(row, "delta_mean", "mean_improvement")
            lines.append(
                f"{metric_cell} & {scenario_cell} & {row.method} & "
                f"{int(row.n_pairs)} & {format_delta(delta_mean)} & "
                f"{format_delta(row.median_improvement)} & "
                f"{p_value} & {p_holm} \\\\"
            )
            previous_metric = row.metric
            previous_scenario_key = scenario_key

    if use_longtable:
        lines.extend([r"\end{longtable}", ""])
    else:
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_latex_tables(result, output_csv):
    paths = []
    for scope in LATEX_SCOPES:
        path = latex_output_path(output_csv, scope)
        latex = build_latex_table(result, scope)
        path.write_text(latex)
        paths.append(path)
    return paths


def result_row(scope, metric, scenario, method, paired, alpha):
    improvements = paired.improvement.to_numpy(dtype=float) if not paired.empty else np.asarray([])
    stat, p, note = wilcoxon_greater(improvements)
    datasets = sorted(paired.dataset.dropna().unique().tolist()) if "dataset" in paired else []
    return {
        "scope": scope,
        "metric": metric,
        "noise_scenario": scenario,
        "comparison": f"{method} vs {BASELINE}",
        "method": method,
        "baseline": BASELINE,
        "alternative": "greater",
        "n_pairs": len(paired),
        "n_datasets": len(datasets),
        "datasets": " ".join(datasets),
        "pairing_unit": "case",
        "method_mean": paired.method_score.mean() if not paired.empty else np.nan,
        "baseline_mean": paired.baseline_score.mean() if not paired.empty else np.nan,
        "delta_mean": paired.improvement.mean() if not paired.empty else np.nan,
        "median_improvement": paired.improvement.median() if not paired.empty else np.nan,
        "improvement_unit": "metric_native_directional",
        "wilcoxon_statistic": stat,
        "p_value": p,
        "reject_uncorrected_alpha": bool(np.isfinite(p) and p <= alpha),
        "note": note,
    }


def run_tests(scores, metric, datasets, scenarios, alpha):
    out = []
    for scenario in scenarios:
        group = []
        for method in METHODS:
            paired = pd.concat([paired_cases(scores, metric, method, d, scenario) for d in datasets], ignore_index=True)
            group.append(result_row("per_metric_per_scenario", metric, scenario, method, paired, alpha))
        out.extend(holm(group, alpha))
    pooled = []
    for method in METHODS:
        pieces = [paired_cases(scores, metric, method, d, s) for d in datasets for s in scenarios]
        paired = pd.concat(pieces, ignore_index=True)
        pooled.append(result_row("pooled_dataset_x_scenario_per_metric", metric, "ALL", method, paired, alpha))
    return pd.DataFrame(out + holm(pooled, alpha))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=SUPPORTED_METRICS)
    parser.add_argument("--datasets", nargs="+", default=list(target_datasets))
    parser.add_argument("--datasets-for-metric", action="append")
    parser.add_argument("--noise-scenarios", nargs="+", default=NOISE_ORDER, choices=NOISE_ORDER)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nnunet-results-root", type=Path)
    args = parser.parse_args()

    default_datasets = parse_selected_datasets(args.datasets)
    overrides = parse_metric_dataset_overrides(args.datasets_for_metric)
    roots = resolve_nnunet_results_roots(args.nnunet_results_root)
    exp_paths = build_experiment_path_index_for_roots(roots)
    sheet_df = filter_records_to_included_folds(load_and_preprocess_results())

    tables = []
    for metric in args.metrics:
        datasets = overrides.get(metric, default_datasets)
        print(f"\nRunning case-paired Wilcoxon tests for {metric}: {', '.join(datasets)}")
        scores = collect_case_scores(sheet_df[sheet_df.Dataset_norm.isin(datasets)], exp_paths, metric)
        tables.append(run_tests(scores, metric, datasets, args.noise_scenarios, args.alpha))

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"\nSaved Wilcoxon/Holm results to {args.output_csv.resolve()}")
    latex_paths = write_latex_tables(result, args.output_csv)
    print("Saved LaTeX tables:")
    for path in latex_paths:
        print(f"  - {path.resolve()}")
    if not result.empty:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
