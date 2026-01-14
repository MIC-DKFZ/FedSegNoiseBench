import subprocess
import pandas as pd


def run_bootstrap_eval(experiment_id):
    """Run bootstrap_nnunet_eval.py for a given experiment_id."""
    cmd = [
        "python3",
        "./src/eval/results_processing/bootstrap_nnunet_eval.py",
        "--exp_id",
        experiment_id,
        "--force",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error for experiment_id {experiment_id}: {result.stderr}")
    else:
        print(f"Completed experiment_id {experiment_id}")


def main(df):
    """Find experiment_ids and run bootstrap for each experiment."""
    # get experiment_ids of df for rows with ID set
    experiment_ids = df[df["ID"].notna()]["Experiment ID"].dropna().unique().tolist()

    print(f"Found {len(experiment_ids)} experiments")
    for exp_id in experiment_ids:
        run_bootstrap_eval(exp_id)


if __name__ == "__main__":
    # google sheet details
    sheet_id = "1AP_KH1cVSDwgpI1n7qK_VZU0Vi19Wh8vKo4jYWkuIXg"
    gid = "332656109"  # use appropriate gid for the sheet tab (0 is usually the first)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    )
    gsheet_df = pd.read_csv(csv_url)

    main(gsheet_df)
