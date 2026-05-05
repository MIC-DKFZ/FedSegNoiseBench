import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    from .ranking import OUTPUT_DIR, target_algos, target_datasets
    from .visualize_ranking import (
        LOWER_IS_BETTER_METRICS,
        NOISE_ORDER,
        SUPPORTED_METRICS,
        build_cell_bootstrap_vectors,
        build_experiment_path_index_for_roots,
        collect_client_bootstrap_vectors,
        filter_records_to_included_folds,
        load_and_preprocess_results,
        parse_selected_datasets,
        resolve_nnunet_results_roots,
    )
except ImportError:
    from ranking import OUTPUT_DIR, target_algos, target_datasets
    from visualize_ranking import (
        LOWER_IS_BETTER_METRICS,
        NOISE_ORDER,
        SUPPORTED_METRICS,
        build_cell_bootstrap_vectors,
        build_experiment_path_index_for_roots,
        collect_client_bootstrap_vectors,
        filter_records_to_included_folds,
        load_and_preprocess_results,
        parse_selected_datasets,
        resolve_nnunet_results_roots,
    )


BASELINE_ALGORITHM = "FedAvg"
COMPARISON_ALGORITHMS = [
    "FedSelect",
    "IOP-FL",
    "FedCorr",
    "FedA3I",
]
DEFAULT_METRICS = ["Dice", "HD95", "FgBgInstanceF1", "ClassConfusion"]
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "fnll_vs_fedavg_wilcoxon_holm.csv"


def parse_metric_dataset_overrides(
    override_args: Optional[Sequence[str]],
) -> Dict[str, List[str]]:
    """
    Parse entries like:
      Dice=LIDC,RIGA,MouseTumor
      HD95:LIDC RIGA
    """
    overrides: Dict[str, List[str]] = {}
    if not override_args:
        return overrides

    for raw_entry in override_args:
        entry = str(raw_entry).strip()
        if not entry:
            continue
        if "=" in entry:
            metric, raw_datasets = entry.split("=", 1)
        elif ":" in entry:
            metric, raw_datasets = entry.split(":", 1)
        else:
            raise ValueError(
                "Dataset override must look like Metric=DatasetA,DatasetB "
                f"or Metric:DatasetA DatasetB. Got: {entry}"
            )

        metric = metric.strip()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric in --datasets-for-metric: {metric}. "
                f"Allowed metrics: {SUPPORTED_METRICS}"
            )

        datasets = parse_selected_datasets([raw_datasets])
        overrides[metric] = datasets

    return overrides


def holm_bonferroni(
    p_values: List[float],
    alpha: float,
) -> Tuple[List[float], List[bool]]:
    m = len(p_values)
    adjusted = [np.nan] * m
    rejected = [False] * m

    finite_items = [(idx, float(p)) for idx, p in enumerate(p_values) if np.isfinite(p)]
    finite_items.sort(key=lambda item: item[1])

    running_max = 0.0
    stop_rejecting = False
    finite_m = len(finite_items)

    for rank, (idx, p_value) in enumerate(finite_items, start=1):
        correction_factor = finite_m - rank + 1
        adj = min(1.0, correction_factor * p_value)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

        threshold = alpha / correction_factor
        if not stop_rejecting and p_value <= threshold:
            rejected[idx] = True
        else:
            stop_rejecting = True

    return adjusted, rejected


def final_vector_mean(vector: np.ndarray) -> float:
    arr = np.asarray(vector, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def build_score_table(cell_vectors_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if cell_vectors_df.empty:
        return pd.DataFrame(
            columns=["metric", "algorithm", "dataset", "noise_scenario", "score"]
        )

    rows = []
    for row in cell_vectors_df.itertuples(index=False):
        rows.append(
            {
                "metric": metric_name,
                "algorithm": row.algorithm,
                "dataset": row.dataset,
                "noise_scenario": row.noise_scenario,
                "score": final_vector_mean(row.bootstrap_vector),
            }
        )
    return pd.DataFrame(rows)


def _lookup_score(
    score_df: pd.DataFrame,
    metric_name: str,
    algorithm: str,
    dataset: str,
    noise_scenario: str,
) -> Optional[float]:
    scores = score_df[
        (score_df["metric"] == metric_name)
        & (score_df["algorithm"] == algorithm)
        & (score_df["dataset"] == dataset)
        & (score_df["noise_scenario"] == noise_scenario)
    ]["score"].dropna()
    if scores.empty:
        return None
    return float(scores.iloc[0])


def _directional_improvement(
    metric_name: str,
    method_score: float,
    baseline_score: float,
) -> float:
    if metric_name in LOWER_IS_BETTER_METRICS:
        return baseline_score - method_score
    return method_score - baseline_score


def _paired_test_from_improvements(
    improvements: Sequence[float],
    alpha: float,
) -> Tuple[float, float, str]:
    improvement_arr = np.asarray(improvements, dtype=float)
    improvement_arr = improvement_arr[np.isfinite(improvement_arr)]
    if improvement_arr.size < 2:
        return np.nan, np.nan, "too few paired observations"
    if np.allclose(improvement_arr, 0.0, equal_nan=False):
        return 0.0, 1.0, "all paired improvements are zero"

    try:
        try:
            result = wilcoxon(
                improvement_arr,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            )
        except TypeError:
            result = wilcoxon(
                improvement_arr,
                alternative="greater",
                zero_method="wilcox",
            )
        return float(result.statistic), float(result.pvalue), ""
    except ValueError as exc:
        return np.nan, np.nan, str(exc)


def _add_holm_results(rows: List[Dict], alpha: float) -> List[Dict]:
    p_values = [row["p_value"] for row in rows]
    p_adj, rejected = holm_bonferroni(p_values, alpha)
    for row, adj, reject in zip(rows, p_adj, rejected):
        row["correction"] = "Holm-Bonferroni"
        row["p_value_holm"] = adj
        row["reject_holm_alpha"] = reject
        row["alpha"] = alpha
    return rows


def _base_result_row(
    scope: str,
    metric_name: str,
    noise_scenario: str,
    method: str,
    n_pairs: int,
    pair_ids: Sequence[str],
    method_scores: Sequence[float],
    baseline_scores: Sequence[float],
    improvements: Sequence[float],
    statistic: float,
    p_value: float,
    note: str,
    alpha: float,
    improvement_unit: str,
) -> Dict:
    method_arr = np.asarray(method_scores, dtype=float)
    baseline_arr = np.asarray(baseline_scores, dtype=float)
    improvement_arr = np.asarray(improvements, dtype=float)
    return {
        "scope": scope,
        "metric": metric_name,
        "noise_scenario": noise_scenario,
        "comparison": f"{method} vs {BASELINE_ALGORITHM}",
        "method": method,
        "baseline": BASELINE_ALGORITHM,
        "alternative": "greater",
        "n_pairs": n_pairs,
        "n_datasets": np.nan,
        "datasets": "",
        "pairing_unit": " ".join(pair_ids),
        "method_mean": float(np.mean(method_arr)) if method_arr.size else np.nan,
        "baseline_mean": (
            float(np.mean(baseline_arr)) if baseline_arr.size else np.nan
        ),
        "mean_improvement": (
            float(np.mean(improvement_arr)) if improvement_arr.size else np.nan
        ),
        "median_improvement": (
            float(np.median(improvement_arr)) if improvement_arr.size else np.nan
        ),
        "improvement_unit": improvement_unit,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "reject_uncorrected_alpha": bool(np.isfinite(p_value) and p_value <= alpha),
        "note": note,
    }


def run_wilcoxon_for_metric(
    score_df: pd.DataFrame,
    metric_name: str,
    datasets: Sequence[str],
    noise_scenarios: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    rows = []

    for noise_scenario in noise_scenarios:
        group_rows = []
        for method in COMPARISON_ALGORITHMS:
            paired_method = []
            paired_baseline = []
            paired_datasets = []
            improvements = []

            for dataset in datasets:
                baseline_score = _lookup_score(
                    score_df,
                    metric_name,
                    BASELINE_ALGORITHM,
                    dataset,
                    noise_scenario,
                )
                method_score = _lookup_score(
                    score_df,
                    metric_name,
                    method,
                    dataset,
                    noise_scenario,
                )

                if baseline_score is None or method_score is None:
                    continue

                paired_baseline.append(baseline_score)
                paired_method.append(method_score)
                paired_datasets.append(dataset)
                improvements.append(
                    _directional_improvement(
                        metric_name,
                        method_score,
                        baseline_score,
                    )
                )

            statistic, p_value, note = _paired_test_from_improvements(
                improvements,
                alpha,
            )

            group_rows.append(
                _base_result_row(
                    scope="per_metric_per_scenario",
                    metric_name=metric_name,
                    noise_scenario=noise_scenario,
                    method=method,
                    n_pairs=len(paired_datasets),
                    pair_ids=paired_datasets,
                    method_scores=paired_method,
                    baseline_scores=paired_baseline,
                    improvements=improvements,
                    statistic=statistic,
                    p_value=p_value,
                    note=note,
                    alpha=alpha,
                    improvement_unit="metric_native_directional",
                )
            )
            group_rows[-1]["n_datasets"] = len(paired_datasets)
            group_rows[-1]["datasets"] = " ".join(paired_datasets)

        rows.extend(_add_holm_results(group_rows, alpha))

    return pd.DataFrame(rows)


def run_wilcoxon_across_dataset_scenarios(
    score_df: pd.DataFrame,
    metric_name: str,
    datasets: Sequence[str],
    noise_scenarios: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    group_rows = []
    for method in COMPARISON_ALGORITHMS:
        paired_method = []
        paired_baseline = []
        pair_ids = []
        improvements = []

        for dataset in datasets:
            for noise_scenario in noise_scenarios:
                baseline_score = _lookup_score(
                    score_df,
                    metric_name,
                    BASELINE_ALGORITHM,
                    dataset,
                    noise_scenario,
                )
                method_score = _lookup_score(
                    score_df,
                    metric_name,
                    method,
                    dataset,
                    noise_scenario,
                )
                if baseline_score is None or method_score is None:
                    continue

                paired_baseline.append(baseline_score)
                paired_method.append(method_score)
                pair_ids.append(f"{dataset}:{noise_scenario}")
                improvements.append(
                    _directional_improvement(
                        metric_name,
                        method_score,
                        baseline_score,
                    )
                )

        statistic, p_value, note = _paired_test_from_improvements(
            improvements,
            alpha,
        )
        datasets_present = sorted({pair_id.split(":", 1)[0] for pair_id in pair_ids})
        group_rows.append(
            _base_result_row(
                scope="pooled_dataset_x_scenario_per_metric",
                metric_name=metric_name,
                noise_scenario="ALL",
                method=method,
                n_pairs=len(pair_ids),
                pair_ids=pair_ids,
                method_scores=paired_method,
                baseline_scores=paired_baseline,
                improvements=improvements,
                statistic=statistic,
                p_value=p_value,
                note=note,
                alpha=alpha,
                improvement_unit="metric_native_directional",
            )
        )
        group_rows[-1]["n_datasets"] = len(datasets_present)
        group_rows[-1]["datasets"] = " ".join(datasets_present)

    return pd.DataFrame(_add_holm_results(group_rows, alpha))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Wilcoxon signed-rank tests comparing FNLL methods "
            "against FedAvg across datasets, per metric and client-noise scenario, "
            "plus pooled dataset×scenario pairs per metric. "
            "Scores are derived from the same ranking-style final bootstrap vectors "
            "used for result tables and ranking stability."
        )
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=SUPPORTED_METRICS,
        help=f"Metrics to test (default: {DEFAULT_METRICS}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(target_datasets),
        help=(
            "Default dataset list for all metrics. Can be overridden per metric "
            "with --datasets-for-metric."
        ),
    )
    parser.add_argument(
        "--datasets-for-metric",
        action="append",
        default=None,
        help=(
            "Metric-specific dataset selection. Repeatable. Examples: "
            "--datasets-for-metric Dice=LIDC,RIGA,MouseTumor "
            "--datasets-for-metric HD95=RIGA,MouseTumor"
        ),
    )
    parser.add_argument(
        "--noise-scenarios",
        nargs="+",
        default=NOISE_ORDER,
        choices=NOISE_ORDER,
        help=f"Noise scenarios to test (default: {NOISE_ORDER}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise alpha for Holm-Bonferroni correction per reported scope/group.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--nnunet-results-root",
        type=Path,
        default=None,
        help=(
            "Root directory of nnUNet results. If not set, uses $nnUNet_results "
            "or the default roots from visualize_ranking.py."
        ),
    )
    args = parser.parse_args()

    default_datasets = parse_selected_datasets(args.datasets)
    metric_dataset_overrides = parse_metric_dataset_overrides(args.datasets_for_metric)
    datasets_by_metric = {
        metric_name: metric_dataset_overrides.get(metric_name, default_datasets)
        for metric_name in args.metrics
    }

    nnunet_results_roots = resolve_nnunet_results_roots(args.nnunet_results_root)
    all_exp_paths = build_experiment_path_index_for_roots(nnunet_results_roots)

    df = load_and_preprocess_results()
    df = filter_records_to_included_folds(df)

    all_results = []
    for metric_name in args.metrics:
        selected_datasets = datasets_by_metric[metric_name]
        metric_df = df[df["Dataset_norm"].isin(selected_datasets)].copy()
        print(
            f"\nRunning statistical tests for {metric_name} on datasets: "
            + ", ".join(selected_datasets)
        )

        client_vectors_df = collect_client_bootstrap_vectors(
            metric_df,
            all_exp_paths,
            metric_name,
        )
        cell_vectors_df = build_cell_bootstrap_vectors(client_vectors_df)
        score_df = build_score_table(cell_vectors_df, metric_name)
        metric_results = run_wilcoxon_for_metric(
            score_df,
            metric_name,
            selected_datasets,
            args.noise_scenarios,
            args.alpha,
        )
        all_results.append(metric_results)
        all_results.append(
            run_wilcoxon_across_dataset_scenarios(
                score_df,
                metric_name,
                selected_datasets,
                args.noise_scenarios,
                args.alpha,
            )
        )

    results_df = (
        pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved Wilcoxon/Holm results to {args.output_csv.resolve()}")

    if not results_df.empty:
        print()
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
