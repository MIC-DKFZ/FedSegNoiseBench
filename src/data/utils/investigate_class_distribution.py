import json
import os
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
from PIL import Image


def get_persample_class_distribution(gt_seg_folder, gt_file_ending=".tif"):
    """
    Get the class distribution for each sample in segmentation masks.

    Args:
        gt_seg_folder: Path to folder containing ground truth segmentation masks
        gt_file_ending: File extension of segmentation mask files (default: ".tif")

    Returns:
        dict: Dictionary mapping sample IDs to sets of class indices present in each sample
    """
    # find all gt files of current dir
    gt_files = sorted(glob(os.path.join(gt_seg_folder, f"*{gt_file_ending}")))

    persample_class_distribution = {
        gt_files[i].split(os.sep)[-1].replace(gt_file_ending, ""): set()
        for i in range(len(gt_files))
    }

    for gt_file in tqdm(gt_files, desc="Processing gt segmentation masks"):
        sample_id = gt_file.split(os.sep)[-1].replace(gt_file_ending, "")
        # load segmentation mask
        mask = np.array(Image.open(gt_file))
        # get unique classes in this mask
        unique_classes = np.unique(mask)
        persample_class_distribution[sample_id] = set(int(c) for c in unique_classes)

    return persample_class_distribution

def get_perfold_class_distribution(persample_class_distribution, fold_id, client_dirs):
    print(f"\nClass distribution for fold {fold_id}:")
    for client_dir in client_dirs:
        dir = Path(client_dir)
        splits_file = dir / "splits_final.json"
        with open(splits_file, "r") as f:
            splits = json.load(f)

        train_samples = splits[fold_id]["train"]
        val_samples = splits[fold_id]["val"]

        train_class_counts = defaultdict(int)
        val_class_counts = defaultdict(int)

        for sample_id in train_samples:
            classes = persample_class_distribution.get(sample_id, set())
            for c in classes:
                train_class_counts[c] += 1

        for sample_id in val_samples:
            classes = persample_class_distribution.get(sample_id, set())
            for c in classes:
                val_class_counts[c] += 1

        print(f"\nClient: {dir.name}, Fold: {fold_id}")
        print("Train class distribution:")
        for c, count in sorted(train_class_counts.items()):
            print(f"  Class {c}: {count} samples")

        print("Validation class distribution:")
        for c, count in sorted(val_class_counts.items()):
            print(f"  Class {c}: {count} samples")

def get_perclient_class_distribution_count(perclient_persample_class_distribution):
    perclient_class_distribution_count = {}
    for c_id, persample_class_distribution in perclient_persample_class_distribution.items():
        class_count = defaultdict(int)
        for sample_id, classes in persample_class_distribution.items():
            for c in classes:
                class_count[c] += 1
        perclient_class_distribution_count[c_id] = dict(class_count)
    return perclient_class_distribution_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Investigate class distribution in segmentation masks."
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing splits_final.json and gt_segmentations folder",
    )
    args = parser.parse_args()

    # for all dirs create a dict with sample_id and class distribution
    persample_class_distribution = {}
    perclient_persample_class_distribution = {c_id : {} for c_id in range(len(args.dirs))}
    for c_id, dir_path in enumerate(args.dirs):
        dir = Path(dir_path)
        gt_seg_folder = dir / "gt_segmentations"

        persample_class_distribution.update(get_persample_class_distribution(gt_seg_folder=gt_seg_folder))
        perclient_persample_class_distribution[c_id] = persample_class_distribution
    print("\nClass distribution per sample across clients:")
    print(persample_class_distribution)
    perclient_class_distribution_count = get_perclient_class_distribution_count(perclient_persample_class_distribution)
    print("\nClass distribution per client:")
    print(perclient_class_distribution_count)

    # get per fold and per client the class distribution
    num_folds = 5
    for fold_id in range(num_folds):
        get_perfold_class_distribution(persample_class_distribution, fold_id, args.dirs)


    # num_folds = 5
    # num_clients = len(args.dirs)
    # class_dist = {
    #     f: {Path(client_dir).name: {} for client_dir in args.dirs}
    #     for f in range(num_folds)
    # }
    # for fold_id in range(num_folds):
    #     print(f"Processing fold {fold_id}...")
    #     for dir_path in args.dirs:
    #         dir = Path(dir_path)
    #         splits_file = dir / "splits_final.json"
    #         gt_seg_folder = dir / "gt_segmentations"

    #         if fold_id == 0:
    #             # get per-sample fg classes only once
    #             get

    #         class_dist = investigate_class_distribution(
    #             splits_file, fold_id, gt_seg_folder
    #         )

    # # Print results
    # for sample_id, info in sorted(class_dist.items()):
    #     print(f"\nSample {sample_id}:")
    #     for fold_split, classes in info.items():
    #         print(f"  {fold_split}: Classes {sorted(classes)}")
