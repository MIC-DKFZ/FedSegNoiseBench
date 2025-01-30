import os
import shutil
import json
from tqdm import tqdm
import argparse

def lidc_2_nnunet_dataset_format(
    base_ct_dir, base_seg_dir, output_dir, multi_class=False, cropping=False
):
    # Output directories
    images_tr_dir = os.path.join(output_dir, "imagesTr")
    labels_tr_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)

    # Initialize dataset.json metadata
    dataset_json = {
        "channel_names": {"0": "CT"},
        "description": "Lung Nodule Segmentation on CT (lidc)",
        "file_ending": ".nii.gz",
        "labels": (
            {"background": 0, "begnin": 1, "malignant": 2}
            if multi_class
            else {"background": 0, "lesion": 1}
        ),
        "licence": "CC-BY-SA 4.0",
        "name": "Lung LIDC",
        "numTraining": 0,
        "reference": "TCIA",
    }

    # Process files
    ct_files = sorted([f for f in os.listdir(base_ct_dir) if f.endswith("_CT.nii.gz")])
    seg_files = sorted(
        [f for f in os.listdir(base_seg_dir) if f.endswith("_SEG.nii.gz")]
    )

    paired_files = []

    # Match CT and SEG files based on their identifiers
    for ct_file in tqdm(
        ct_files, desc="Processing CT and SEG files", unit="CT&SEG files"
    ):
        ct_id = ct_file.split("_CT")[0]  # Extract ID, e.g., "LIDC-IDRI-0001" or "LIDC-IDRI-0001_0" for cropping=True
        patient_id = ct_id.split("-")[-1]  # Extract patient ID, e.g., "0001" or "0001_0" for cropping=True
        matching_seg = next((s for s in seg_files if patient_id in s), None)

        # adapt ct_id for cropping=True
        ct_id = ct_id if not cropping else ct_id.replace("_", "-")

        if matching_seg:
            paired_files.append((ct_file, matching_seg))

            # Copy CT file to imagesTr with the appropriate name
            ct_target_name = f"{ct_id.replace('LIDC-IDRI-', 'LIDC_')}_0000.nii.gz"
            shutil.copy(
                os.path.join(base_ct_dir, ct_file),
                os.path.join(images_tr_dir, ct_target_name),
            )

            # Copy SEG file to labelsTr with the appropriate name
            seg_target_name = f"{ct_id.replace('LIDC-IDRI-', 'LIDC_')}.nii.gz"
            shutil.copy(
                os.path.join(base_seg_dir, matching_seg),
                os.path.join(labels_tr_dir, seg_target_name),
            )

    # Update numTraining in dataset.json
    dataset_json["numTraining"] = len(paired_files)

    # Write dataset.json to output directory
    with open(os.path.join(output_dir, "dataset.json"), "w") as json_file:
        json.dump(dataset_json, json_file, indent=4)

    print(f"Processed {len(paired_files)} CT-SEG pairs.")
    print(f"Dataset organized in nnUNet format at: {output_dir}")


if __name__ == "__main__":
    # # CLI argument parsing
    # parser = argparse.ArgumentParser(description="Convert data into nnUNet data format.")
    # parser.add_argument("--base_ct_dir", required=True, help="Path to the CT images directory (usually: LIDC_seg-per-nodule-and-rater_nifti).")
    # parser.add_argument("--base_seg_dir", required=True, help="Path to the labels directory (e.g.: LIDC-single_seg_nifti/malignancy_annotator_majority).")
    # parser.add_argument("--output_dir", required=True, help="Datset path of nnUNet_raw data (e.g.: nnUNet_raw/Dataset028_LIDC-Malignancy-AnnotatorMajority).")
    # parser.add_argument("--multi_class", action="store_true", help="Differentiate between benign and malignant nodules.")
    # parser.add_argument("--cropping", action="store_true", help="CTs and SEGs are cropped.")

    # args = parser.parse_args()

    # lidc_2_nnunet_dataset_format(args.base_ct_dir, args.base_seg_dir, args.output_dir, args.multi_class, args.cropping)

    # MANUAL USAGE
    # Paths
    base_ct_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti-cropped"
    base_seg_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-single_seg_nifti/cropped_annotator_majority"
    output_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/nnUNet_raw/Dataset032_LIDC-Cropped-AnnotatorMajority"
    lidc_2_nnunet_dataset_format(base_ct_dir, base_seg_dir, output_dir, multi_class=False, cropping=True)