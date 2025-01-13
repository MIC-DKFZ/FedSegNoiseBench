import os
from tqdm import tqdm

# Define the directory containing your files
base_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-nifti"
# base_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti"
# base_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-single_seg_nifti/random-multi_rater"
log_file_path = "no_seg_ct_files.log"

# Initialize a dictionary to group CT and SEG files
ct_files = {}
seg_files = set()

# Traverse the folder structure
for root, _, files in os.walk(base_dir):
    for file in tqdm(files, desc="Checking files", unit="file"):
        if file.endswith("_CT.nii.gz"):  # CT file pattern
            base_name = file.replace("_CT.nii.gz", "")
            ct_files[base_name] = os.path.join(root, file)
        elif "_SEG" in file and file.endswith(".nii.gz"):  # SEG file pattern
            base_name = file.split("_")[0]  # Extract base CT name (e.g., LIDC-IDRI-0001)
            seg_files.add(base_name)

# Filter CT files without any SEG files
no_seg_ct_files = [path for base_name, path in ct_files.items() if base_name not in seg_files]
print(f"Found {len(ct_files)} CT files and {len(seg_files)} SEG files.")
print(f"Found {len(no_seg_ct_files)} CT files without SEG files.")

# Write unmatched CT files to the log file
with open(log_file_path, "w") as log_file:
    log_file.write(f"{no_seg_ct_files}\n")