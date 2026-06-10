import json
import os
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
from PIL import Image
import nibabel as nib


def get_persample_class_distribution(gt_seg_folder, gt_file_ending=".nii.gz"):
    """
    Get the class distribution for each sample in segmentation masks.

    Args:
        gt_seg_folder: Path to folder containing ground truth segmentation masks
        gt_file_ending: File extension of segmentation mask files (default: ".nii.gz")

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
        if gt_file_ending == ".nii.gz":
            mask = np.array(nib.load(gt_file).get_fdata())
        elif gt_file_ending in [".tif", ".png"]:
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

        print("\n")
        print(f"Client: {dir.name}, Fold: {fold_id}")
        print("Train class distribution:")
        for c, count in sorted(train_class_counts.items()):
            print(f"  Class {c}: {count} samples")

        print("Validation class distribution:")
        for c, count in sorted(val_class_counts.items()):
            print(f"  Class {c}: {count} samples")


def get_perclient_class_distribution_count(perclient_persample_class_distribution):
    perclient_class_distribution_count = {}
    for (
        c_id,
        persample_class_distribution,
    ) in perclient_persample_class_distribution.items():
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
        help="Directories containing splits_final.json and gt_segmentations folder. Should be sth like: \
            --dirs /home/m391k/cluster-data/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real/Dataset600_MMIA-DUKE_expert_flclient0 \
            /home/m391k/cluster-data/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real/Dataset601_MMIA-ISPY1_expert_flclient1 \
            /home/m391k/cluster-data/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real/Dataset602_MMIA-ISPY2_expert_flclient2 \
            /home/m391k/cluster-data/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real/Dataset603_MMIA-NACT_expert_flclient3",
    )
    parser.add_argument(
        "--gt_file_ending",
        type=str,
        default=".tif",
        help="File extension of segmentation mask files. Options: .nii.gz (default), .tif, .png",
    )
    args = parser.parse_args()

    # for all dirs create a dict with sample_id and class distribution
    persample_class_distribution = {}
    perclient_persample_class_distribution = {
        c_id: {} for c_id in range(len(args.dirs))
    }
    for c_id, dir_path in enumerate(args.dirs):
        dir = Path(dir_path)
        gt_seg_folder = dir / "gt_segmentations"

        perclient_persample_class_distribution[c_id] = get_persample_class_distribution(
            gt_seg_folder=gt_seg_folder, gt_file_ending=args.gt_file_ending
        )
        persample_class_distribution.update(
            perclient_persample_class_distribution[c_id]
        )

    print("\n")
    print("#" * 100)
    print("Class distribution per sample ACROSS clients:")
    print(persample_class_distribution)
    print(f"Total number of samples: {len(persample_class_distribution)}")
    perclient_class_distribution_count = get_perclient_class_distribution_count(
        perclient_persample_class_distribution
    )

    print("\n")
    print("#" * 100)
    print("Number of samples per client:")
    for c_id, samples in perclient_persample_class_distribution.items():
        print(f"Client {c_id}: {len(samples)} samples")

    print("\n")
    print("#" * 100)
    print("Class distribution per client:")
    for c_id, class_distribution in perclient_class_distribution_count.items():
        print(f"Client {c_id}: {class_distribution}")

    # get per fold and per client the class distribution
    print("\n")
    print("#" * 100)
    num_folds = 5
    for fold_id in range(num_folds):
        print("=" * 50)
        get_perfold_class_distribution(persample_class_distribution, fold_id, args.dirs)
