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
    reference_files = [
        f for f in os.listdir(directory) 
        if f.startswith("LIDC-IDRI-") and f.endswith("_0_SEG.nii.gz")
    ]

    monitor_already_existing_files = 0
    monitor_created_files = 0
    for ref_file in tqdm(reference_files, desc="Checking 0th reference seg mask file", unit="file"):
        # Extract the common identifier: LIDC-IDRI-<4-digit-number>
        base_identifier = ref_file.split("_")[0] + "_" + ref_file.split("_")[1]  # e.g., "LIDC-IDRI-0001_1"
        
        # Build paths for '_1_SEG.nii.gz', '_2_SEG.nii.gz', '_3_SEG.nii.gz'
        expected_files = [
            f"{base_identifier}_1_SEG.nii.gz",
            f"{base_identifier}_2_SEG.nii.gz",
            f"{base_identifier}_3_SEG.nii.gz"
        ]

        ref_path = None
        ref_img = None
        ref_shape = None
        ref_affine = None
        # Check for each expected file and create if missing
        for expected_file in expected_files:
            expected_path = os.path.join(directory, expected_file)
            if not os.path.exists(expected_path):
                print(f"Creating missing file: {expected_path}")
                if not ref_path and not ref_img and not ref_shape and not ref_affine:
                    # Load the reference file to extract shape and affine
                    ref_path = os.path.join(directory, ref_file)
                    ref_img = nib.load(ref_path)
                    ref_shape = ref_img.shape
                    ref_affine = ref_img.affine

                # Create a zero-filled volume
                zero_volume = np.zeros(ref_shape, dtype=np.uint8)
                new_img = nib.Nifti1Image(zero_volume, affine=ref_affine)
                nib.save(new_img, os.path.join(directory, "empty_masks", expected_file))
                monitor_created_files += 1
            else:
                print(f"File already exists!")
                monitor_already_existing_files += 1

    print(f"Created {monitor_created_files} files and found {monitor_already_existing_files} already existing files.")

# Set the directory containing the NIfTI files
nifti_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-nifti"  # Replace with your directory path
find_and_fix_nifti_files(nifti_directory)
