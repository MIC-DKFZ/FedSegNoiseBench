import subprocess
import pandas as pd
from tqdm import tqdm
import argparse
import re


def run_bootstrap_eval(experiment_id, force=False, num_workers=None):
    """Run bootstrap_nnunet_eval.py for a given experiment_id."""
    cmd = [
        "python3",
        "./src/eval/results_processing/bootstrap_nnunet_eval.py",
        "--exp_id",
        experiment_id,
    ]
    if force:
        cmd.append("--force")
    if num_workers is not None:
        cmd.extend(["--num-workers", str(num_workers)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error for experiment_id {experiment_id}: {result.stderr}")
    else:
        print(f"Completed experiment_id {experiment_id}")


def extract_fold_from_experiment_id(experiment_id):
    """Extract fold number from experiment ID like ..._fold2_..."""
    match = re.search(r"fold(\d+)", str(experiment_id))
    return int(match.group(1)) if match else None


def main(df, force=False, selected_folds=None, num_workers=None):
    """Find experiment_ids and run bootstrap for each experiment."""
    # get experiment_ids of df for rows with ID set
    experiment_ids = df[df["ID"].notna()]["Experiment ID"].dropna().unique().tolist()

    if selected_folds:
        filtered_experiment_ids = []
        skipped_without_fold = 0
        for exp_id in experiment_ids:
            fold = extract_fold_from_experiment_id(exp_id)
            if fold is None:
                skipped_without_fold += 1
                continue
            if fold in selected_folds:
                filtered_experiment_ids.append(exp_id)

        experiment_ids = filtered_experiment_ids
        print(
            f"Restricted to folds {sorted(selected_folds)}. "
            f"Found {len(experiment_ids)} matching experiments."
        )
        if skipped_without_fold:
            print(
                f"Skipped {skipped_without_fold} experiment IDs because no fold could be inferred."
            )

    print(f"Found {len(experiment_ids)} experiments")
    for exp_id in tqdm(experiment_ids, desc="Running bootstrap evaluations"):
        run_bootstrap_eval(exp_id, force=force, num_workers=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full re-evaluation even if bootstrap results already exist. Without this flag, only missing metrics are added.",
        default=False,
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of folds to run, e.g. --folds 0 1 2",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of threads to use per bootstrap_nnunet_eval.py process. "
            "If omitted, the child script chooses its default."
        ),
    )
    args = parser.parse_args()

    # google sheet details
    sheet_id = "1AP_KH1cVSDwgpI1n7qK_VZU0Vi19Wh8vKo4jYWkuIXg"
    gid = "332656109"  # use appropriate gid for the sheet tab (0 is usually the first)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    )
    gsheet_df = pd.read_csv(csv_url)

    main(
        gsheet_df,
        force=args.force,
        selected_folds=args.folds,
        num_workers=args.num_workers,
    )
