import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def find_and_fix_nifti_files(directory):
    """
    For each NIfTI file matching 'LIDC-IDRI-<4-digit-number>_*_0_SEG.nii.gz',
    check for corresponding '_1_SEG.nii.gz', '_2_SEG.nii.gz', '_3_SEG.nii.gz' files.
    If missing, create zero-filled NIfTI files matching the reference file's dimensions.

    Args:
        directory (str): Path to the directory containing NIfTI files.
    """
    # Gather all files starting with 'LIDC-IDRI-' and ending with '_0_SEG.nii.gz'
    ct_files = [
        f for f in os.listdir(directory) 
        if f.startswith("LIDC-IDRI-") and f.endswith("_CT.nii.gz")
    ]

    monitor_already_existing_files = 0
    monitor_created_files = 0
    monitor_nonodule_created_files = 0
    for ct_file in tqdm(ct_files, desc="Checking CT files", unit="file"):
        # Extract the common identifier: LIDC-IDRI-<4-digit-number>
        patient_id = ct_file.split("_")[0]  # e.g., "LIDC-IDRI-0001"

        seg_nodule_files = [
            f for f in os.listdir(directory) 
            if f.startswith(patient_id) and f.endswith("_SEG.nii.gz")
        ]

        if len(seg_nodule_files) > 0:
            nodule_ids = []
            for seg_nodule_file in seg_nodule_files:
                # Extract the nodule ids
                nodule_ids.append(seg_nodule_file.split("_")[1])
            nodule_ids = list(set(nodule_ids))  # Remove duplicates

            patient_nodule_ids = [patient_id + "_" + nodule_id for nodule_id in nodule_ids]

            for patient_nodule_id in patient_nodule_ids:
                # patient_nodule_files = [f for f in seg_nodule_files if f.startswith(patient_nodule_id)]

                # Build paths for '_1_SEG.nii.gz', '_2_SEG.nii.gz', '_3_SEG.nii.gz'
                expected_files = [
                    f"{patient_nodule_id}_0_SEG.nii.gz",
                    f"{patient_nodule_id}_1_SEG.nii.gz",
                    f"{patient_nodule_id}_2_SEG.nii.gz",
                    f"{patient_nodule_id}_3_SEG.nii.gz"
                ]

                ref_path = None
                ref_img = None
                ref_shape = None
                ref_affine = None
                # Check for each expected file and create if missing
                for expected_file in expected_files:
                    expected_path = os.path.join(directory, expected_file)
                    if os.path.exists(expected_path):
                        print(f"File already exists!")
                        monitor_already_existing_files += 1
                    else:
                        print(f"Creating missing file: {expected_path}")
                        if not ref_path and not ref_img and not ref_shape and not ref_affine:
                            # Load the reference file to extract shape and affine
                            ref_path = os.path.join(directory, ct_file)
                            ref_img = nib.load(ref_path)
                            ref_shape = ref_img.shape
                            ref_affine = ref_img.affine

                        # Create a zero-filled volume
                        zero_volume = np.zeros(ref_shape, dtype=np.uint8)
                        new_img = nib.Nifti1Image(zero_volume, affine=ref_affine)
                        nib.save(new_img, os.path.join(directory, "empty_masks", expected_file))
                        monitor_created_files += 1
        else:
            print(f"No SEG files found for {ct_file}. Let's create 4 empty SEG files for an imaginary nodule 0.")
            # Load the reference file to extract shape and affine
            ref_path = os.path.join(directory, ct_file)
            ref_img = nib.load(ref_path)
            ref_shape = ref_img.shape
            ref_affine = ref_img.affine
            for i in range(4):
                # Create a zero-filled volume
                zero_volume = np.zeros(ref_shape, dtype=np.uint8)
                new_img = nib.Nifti1Image(zero_volume, affine=ref_affine)
                nib.save(new_img, os.path.join(directory, "empty_masks", f"{patient_id}_0_{i}_SEG.nii.gz"))
                monitor_nonodule_created_files += 1


    print(f"Created {monitor_created_files} files and found {monitor_already_existing_files} already existing files.")
    print(f"Created {monitor_nonodule_created_files} files for non-existing nodules.")

# Set the directory containing the NIfTI files
nifti_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti"  # Replace with your directory path
find_and_fix_nifti_files(nifti_directory)
