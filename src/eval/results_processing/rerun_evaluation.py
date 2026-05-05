import argparse
import itertools
import os
import re
import subprocess
from pathlib import Path

from tqdm import tqdm

try:
    from .ranking import exp_id_col, included_folds, load_and_preprocess_results
    from .visualize_ranking import filter_records_to_included_folds
except ImportError:
    from ranking import exp_id_col, included_folds, load_and_preprocess_results
    from visualize_ranking import filter_records_to_included_folds


RESULTS_ROOTS = [
    Path("/home/m391k/cluster-data/checkpoints/nnUNet_results"),
    Path("/home/m391k/juwels/checkpoints/nnUNet_results"),
]


def find_nnunet_config_dir(pred_dir: Path) -> Path:
    for parent in pred_dir.parents:
        if (parent / "dataset.json").is_file() and (parent / "plans.json").is_file():
            return parent
    raise FileNotFoundError(f"Could not find dataset.json/plans.json above {pred_dir}")


def dataset_id_from_path(pred_dir: Path) -> str:
    dataset_dir = next(p for p in pred_dir.parents if p.name.startswith("Dataset"))
    match = re.match(r"Dataset(\d+)_", dataset_dir.name)
    if match is None:
        raise ValueError(f"Could not infer dataset id from {dataset_dir}")
    return match.group(1)


def find_preprocessed_dataset(preproc_root: Path, dataset_id: str) -> Path:
    matches = sorted(preproc_root.glob(f"Dataset{dataset_id}_*"))
    if not matches:
        raise FileNotFoundError(f"No preprocessed Dataset{dataset_id}_* in {preproc_root}")
    return matches[0]


def build_command(
    pred_dir: Path,
    preproc_root: Path,
    num_processes: int,
    output_name: str,
) -> list:
    dataset_id = dataset_id_from_path(pred_dir)
    config_dir = find_nnunet_config_dir(pred_dir)
    gt_dir = find_preprocessed_dataset(preproc_root, dataset_id) / "gt_segmentations"
    return [
        "nnUNetv2_evaluate_folder",
        str(gt_dir),
        str(pred_dir),
        "-djfile",
        str(config_dir / "dataset.json"),
        "-pfile",
        str(config_dir / "plans.json"),
        "-o",
        str(pred_dir / output_name),
        "-np",
        str(num_processes),
        "--chill",
        "-instance_iou",
        "0.1",
    ]


def iter_validation_dirs(results_roots):
    for root in results_roots:
        yield from root.glob("Dataset*/**/fold_*/**/validation")


def load_paper_experiment_ids():
    df = load_and_preprocess_results()
    df = filter_records_to_included_folds(df)
    exp_ids = {
        str(exp_id).strip()
        for exp_id in df[exp_id_col].dropna().unique()
        if str(exp_id).strip() and str(exp_id).strip().lower() != "nan"
    }
    print(
        f"Loaded {len(exp_ids)} paper experiment IDs from Google Sheets "
        f"(folds {included_folds})."
    )
    return exp_ids


def path_matches_any_experiment_id(path: Path, exp_ids: set) -> bool:
    path_str = str(path)
    return any(exp_id in path_str for exp_id in exp_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, action="append", default=None)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path(os.environ["nnUNet_preprocessed"]),
    )
    parser.add_argument("-np", "--num-processes", type=int, default=32)
    parser.add_argument("--output-name", default="summary_reran.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--all-found",
        action="store_true",
        help="Ignore the paper experiment list and rerun every validation folder found.",
    )
    args = parser.parse_args()

    roots = args.results_root or RESULTS_ROOTS
    exp_ids = None if args.all_found else load_paper_experiment_ids()
    pred_dirs = (p for p in iter_validation_dirs(roots) if p.is_dir())
    if exp_ids is not None:
        pred_dirs = (p for p in pred_dirs if path_matches_any_experiment_id(p, exp_ids))
    if args.limit is not None:
        pred_dirs = itertools.islice(pred_dirs, args.limit)

    pred_dirs_list = list(pred_dirs)
    for pred_dir in tqdm(pred_dirs_list, desc="Rerunning validation evaluation", total=len(pred_dirs_list)):
        try:
            cmd = build_command(
                pred_dir,
                args.preprocessed_root,
                args.num_processes,
                args.output_name,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {pred_dir}: {exc}")
            continue

        print("Running:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
