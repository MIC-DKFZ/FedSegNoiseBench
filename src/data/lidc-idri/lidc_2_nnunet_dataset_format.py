import os
import shutil
import json
from tqdm import tqdm

# Paths
base_ct_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti"
base_seg_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-single_seg_nifti/random-multi_rater"
output_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/nnUNet_raw/Dataset023_LIDC-RandomMultiRater"

images_tr_dir = os.path.join(output_dir, "imagesTr")
labels_tr_dir = os.path.join(output_dir, "labelsTr")

# Ensure output directories exist
os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)

# Initialize dataset.json metadata
dataset_json = {
    "channel_names": {"0": "CT"},
    "description": "Lung Nodule Segmentation on CT (lidc)",
    "file_ending": ".nii.gz",
    "labels": {"background": 0, "lesion": 1},
    "licence": "CC-BY-SA 4.0",
    "name": "Lung LIDC",
    "numTraining": 0,  # Will be updated dynamically
    "reference": "TCIA"
}

# Process files
ct_files = sorted([f for f in os.listdir(base_ct_dir) if f.endswith("_CT.nii.gz")])
seg_files = sorted([f for f in os.listdir(base_seg_dir) if f.endswith("_SEG.nii.gz")])

paired_files = []

# Match CT and SEG files based on their identifiers
for ct_file in tqdm(ct_files, desc="Processing CT and SEG files", unit="CT&SEG files"):
    ct_id = ct_file.split("_CT")[0]  # Extract ID, e.g., "LIDC-IDRI-0001"
    patient_id = ct_id.split("-")[-1]  # Extract patient ID, e.g., "0001"
    matching_seg = next((s for s in seg_files if patient_id in s), None)

    if matching_seg:
        paired_files.append((ct_file, matching_seg))

        # Copy CT file to imagesTr with the appropriate name
        ct_target_name = f"{ct_id.replace('LIDC-IDRI-', 'LIDC_')}_0000.nii.gz"
        shutil.copy(os.path.join(base_ct_dir, ct_file), os.path.join(images_tr_dir, ct_target_name))

        # Copy SEG file to labelsTr with the appropriate name
        seg_target_name = f"{ct_id.replace('LIDC-IDRI-', 'LIDC_')}.nii.gz"
        shutil.copy(os.path.join(base_seg_dir, matching_seg), os.path.join(labels_tr_dir, seg_target_name))

# Update numTraining in dataset.json
dataset_json["numTraining"] = len(paired_files)

# Write dataset.json to output directory
with open(os.path.join(output_dir, "dataset.json"), "w") as json_file:
    json.dump(dataset_json, json_file, indent=4)

print(f"Processed {len(paired_files)} CT-SEG pairs.")
print(f"Dataset organized in nnUNet format at: {output_dir}")
