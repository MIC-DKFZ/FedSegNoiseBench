import os
import nibabel as nib
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

def check_non_zero_volumes(directory, output_log="log.txt"):
    """
    Checks all NIfTI files ending with SEG.nii.gz in the specified directory 
    and logs filenames without non-zero values into a log file.

    Args:
        directory (str): Path to the directory containing the NIfTI files.
        output_log (str): Path to the log file for saving filenames.
    """
    # Gather all files ending with SEG.nii.gz
    nifti_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("SEG.nii.gz"):
                nifti_files.append(os.path.join(root, file))
    
    # Open the log file in write mode
    with open(output_log, "w") as log_file:
        # Wrap file list with tqdm for the progress bar
        for file_path in tqdm(nifti_files, desc="Checking NIfTI files", unit="file"):
            try:
                # Load the NIfTI file
                nifti_img = nib.load(file_path)
                volume = nifti_img.get_fdata()

                # Check if the volume contains any non-zero values
                if not np.any(volume):
                    log_file.write(f"{file_path}\n")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Specify the directory containing the NIfTI files
nifti_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-nifti/"  # Replace with your path
output_log_file = "log.txt"

# Run the function
check_non_zero_volumes(nifti_directory, output_log_file)
