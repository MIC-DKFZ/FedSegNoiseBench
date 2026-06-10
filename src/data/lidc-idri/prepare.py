import argparse
import logging
import os
import shutil
import sys

_lidc_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _lidc_dir)
sys.path.insert(0, os.path.join(_lidc_dir, "fed"))

from lidc_annotations_to_nifti import process_lidc_dataset as extract_nifti
from create_empty_masks import create_missing_seg_masks
from create_single_seg_dataset import process_lidc_dataset as fuse_masks
from lidc_2_nnunet_dataset_format import lidc_2_nnunet_dataset_format
from verify_dataset_integrity import process_files as verify_integrity
from fed_data_splitting import main as split_to_clients

_FLAMBY_METADATA = os.path.join(_lidc_dir, "flamby_lidc_federated_split_metadata.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LIDC-IDRI dataset for federated noisy label learning.")
    parser.add_argument(
        "--raw_data_path",
        type=str,
        required=True,
        help="Directory for all intermediate outputs (NIfTI, fused masks, combined nnUNet dataset).",
    )
    parser.add_argument(
        "--single_seg_mode",
        type=str,
        required=True,
        choices=["annotator_majority", "random"],
        help="Label mode: 'annotator_majority' for clean clients, 'random' for noisy clients.",
    )
    parser.add_argument(
        "--dataset_ids",
        type=str,
        required=True,
        help="Four space-separated dataset IDs for the 4 FL clients (e.g., '041 042 043 044').",
    )
    parser.add_argument(
        "--lidc_manifest",
        type=str,
        required=True,
        help="Path to the LIDC TCIA download manifest CSV (contains 'Series UID' and 'Subject ID' columns).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    nnunet_raw = os.getenv("nnUNet_raw")
    assert nnunet_raw, "Environment variable $nnUNet_raw is not set."

    # auto-derived intermediate paths
    nifti_dir = os.path.join(args.raw_data_path, "nifti")
    single_seg_dir = os.path.join(args.raw_data_path, f"single_seg_{args.single_seg_mode}")
    log_csv = os.path.join(single_seg_dir, "single_seg_per_ct.csv")
    combined_nnunet_dir = os.path.join(nnunet_raw, f"DatasetTMP_LIDC_Cropped_{args.single_seg_mode}")

    # Step 1: Extract raw LIDC-IDRI annotations to cropped NIfTI (64x64x64)
    logging.info("Step 1/6: Extracting LIDC annotations to NIfTI (cropped 64x64x64)...")
    extract_nifti(nifti_dir, cropping=True)

    # Step 2: Fill in missing rater masks with zeros, then merge into nifti_dir
    logging.info("Step 2/6: Creating missing segmentation masks...")
    create_missing_seg_masks(nifti_dir, cropping=True)
    empty_masks_dir = os.path.join(nifti_dir, "empty_masks")
    if os.path.exists(empty_masks_dir):
        for fname in os.listdir(empty_masks_dir):
            shutil.move(
                os.path.join(empty_masks_dir, fname),
                os.path.join(nifti_dir, fname),
            )
        os.rmdir(empty_masks_dir)

    # Step 3: Fuse per-rater masks into a single mask per CT
    logging.info(f"Step 3/6: Fusing rater masks (mode={args.single_seg_mode})...")
    fuse_masks(nifti_dir, single_seg_dir, log_csv, mode=args.single_seg_mode, cropping=True)

    # Step 4: Convert to nnUNet raw dataset format
    logging.info("Step 4/6: Converting to nnUNet raw format...")
    lidc_2_nnunet_dataset_format(nifti_dir, single_seg_dir, combined_nnunet_dir, cropping=True)

    # Step 5: Fix any spacing or shape mismatches between images and labels
    logging.info("Step 5/6: Verifying dataset integrity...")
    verify_integrity(
        os.path.join(combined_nnunet_dir, "imagesTr"),
        os.path.join(combined_nnunet_dir, "labelsTr"),
    )

    # Step 6: Split combined dataset into 4 FL clients by CT scanner manufacturer
    logging.info("Step 6/6: Splitting into FL clients by manufacturer...")
    split_to_clients(
        combined_nnunet_dir,
        _FLAMBY_METADATA,
        args.lidc_manifest,
        args.dataset_ids.split(),
    )

    logging.info("Done. FL client datasets written to $nnUNet_raw.")
