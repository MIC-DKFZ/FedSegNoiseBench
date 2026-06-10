import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def create_missing_seg_masks(directory, cropping=False):
    """
    Scans a directory for missing segmentation (SEG) mask files and generates zero-filled 
    NIfTI files to ensure a complete set of segmentation masks.

    This function processes all computed tomography (CT) files in the specified directory 
    that match the naming convention 'LIDC-IDRI-<ID>_CT.nii.gz'. For each CT file, it looks 
    for associated segmentation files ('_SEG.nii.gz'). If some or all expected segmentation 
    files are missing, it creates zero-filled NIfTI segmentation masks.

    Parameters:
    ----------
    directory : str
        The path to the directory containing the CT and SEG files.

    Process:
    -------
    1. Collects all CT files in the directory starting with 'LIDC-IDRI-' and ending 
       with '_CT.nii.gz'.
    2. Extracts patient IDs from the CT file names and searches for corresponding SEG files.
    3. For each patient, identifies the expected segmentation files:
        - '_0_SEG.nii.gz', '_1_SEG.nii.gz', '_2_SEG.nii.gz', and '_3_SEG.nii.gz'.
    4. If segmentation files are missing:
        - For existing nodules, creates zero-filled NIfTI files using the reference CT file.
        - For patients without nodules, creates four zero-filled segmentation masks 
          for an imaginary nodule with ID '0'.
    5. Saves the generated segmentation files to an 'empty_masks' folder within the directory.

    Metrics:
    -------
    - Reports the number of existing segmentation files detected.
    - Reports the number of missing segmentation files created.
    - Reports the number of segmentation files created for non-existent nodules.

    Notes:
    ------
    - The function assumes the NIfTI format for CT and segmentation files.
    - Missing files are created with the same shape and affine transformation as the 
      reference CT file.

    Example:
    -------
    >>> create_missing_seg_masks("/path/to/dataset")
    Checking CT files: 100%|█████████████████████████████| 50/50 [00:30<00:00,  1.63file/s]
    Created 120 files and found 80 already existing files.
    Created 40 files for non-existing nodules.

    Dependencies:
    ------------
    - os
    - nibabel (for NIfTI file handling)
    - numpy
    - tqdm (for progress bar visualization)
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
        patient_id = ct_file.split("_")[0]  # e.g., "LIDC-IDRI-0001"
        patient_nodule_id = ct_file.split("_")[0] + "_" + ct_file.split("_")[1]  # e.g., "LIDC-IDRI-0001_0"

        # for cropping=True, get SEG files nodule-specific
        if cropping:
            seg_nodule_files = [
                f for f in os.listdir(directory) 
                if f.startswith(patient_nodule_id) and f.endswith("_SEG.nii.gz")
            ]
        # for cropping=False, get all SEG files of patient
        else:
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

                # Build paths for '_0_SEG.nii.gz', '_1_SEG.nii.gz', '_2_SEG.nii.gz', '_3_SEG.nii.gz'
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

if __name__ == "__main__":
    nifti_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti-cropped"
    create_missing_seg_masks(nifti_directory, cropping=True)
