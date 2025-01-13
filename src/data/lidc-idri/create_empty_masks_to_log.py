import os

def write_filenames_to_txt(directory, output_file):
    """
    Write filenames of all NIfTI files in a directory to a text file.

    Args:
        directory (str): Path to the directory containing NIfTI files.
        output_file (str): Path to the output text file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Gather all .nii.gz files
    nifti_files = [
        f for f in os.listdir(directory) 
        if f.endswith(".nii.gz")
    ]
    
    # Write filenames to the output text file
    with open(output_file, "w") as f:
        for file in nifti_files:
            f.write(f"{file}\n")
    
    print(f"Successfully wrote {len(nifti_files)} filenames to {output_file}")

# Set the directory path and output file path
nifti_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-nifti/empty_masks"
output_txt_file = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-nifti/empty_created_seg_masks.txt"

# Run the function
write_filenames_to_txt(nifti_directory, output_txt_file)
