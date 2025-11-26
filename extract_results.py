import os
import re
import argparse


def extract_info(job_id, directory):
    err_file = os.path.join(directory, f"{job_id}.err")
    log_file = os.path.join(directory, f"{job_id}.log")

    # Extract experiment_id from .err file
    experiment_id = None
    with open(err_file, "r") as f:
        for line in f:
            match = re.search(r"Experiment ID: (.+)", line)
            if match:
                experiment_id = match.group(1).strip()
                break

    # Extract all Mean Validation Dice values from .log file
    mean_dice = []
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r"Mean Validation Dice:\s*([\d\.Ee+-]+)", line)
            if match:
                mean_dice.append(match.group(1).replace(".", ","))

    return experiment_id, mean_dice


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract experiment info and mean validation dice values."
    )
    parser.add_argument(
        "--job-id", type=str, help="The job ID corresponding to the log and err files."
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="The directory where the log and err files are located.",
        default="/home/m391k/cluster-data/logs",
    )
    args = parser.parse_args()

    job_id = args.job_id
    directory = args.directory

    experiment_id, mean_dice = extract_info(job_id, directory)
    print("Experiment ID:", experiment_id)
    print("Mean Validation Dice values:", mean_dice)
