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
DEFAULT_METRICS = ["Dice", "HD95", "FgBgInstanceF1", "ClassConfusion"]
DEFAULT_OUTPUT = OUTPUT_DIR / "fnll_vs_fedavg_wilcoxon_holm.csv"
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
        "mean_improvement": paired.improvement.mean() if not paired.empty else np.nan,
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
    if not result.empty:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
