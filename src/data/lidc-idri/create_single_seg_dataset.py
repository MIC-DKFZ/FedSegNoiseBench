import os
import random
import nibabel as nib
import numpy as np
import csv
from tqdm import tqdm
import nibabel.processing
from nibabel.processing import resample_to_output
from scipy.ndimage import zoom



# def resample_mask_to_shape(mask_data, target_shape, original_affine):
#     """
#     Resamples a 3D mask to the target shape using NIfTI image metadata.

#     Args:
#         mask_data (numpy.ndarray): Input mask data.
#         target_shape (tuple): Desired output shape (height, width, depth).
#         original_affine (numpy.ndarray): Affine transformation matrix of the original mask.

#     Returns:
#         numpy.ndarray: Resampled mask.
#     """
#     # Create a temporary NIfTI image from the mask data
#     mask_img = nib.Nifti1Image(mask_data, original_affine)
    
#     # Calculate voxel size changes based on the target shape
#     original_shape = mask_data.shape
#     voxel_sizes = [
#         original_shape[i] / target_shape[i] for i in range(len(target_shape))
#     ]
    
#     # Resample the mask
#     resampled_img = resample_to_output(mask_img, voxel_sizes)
    
#     # Return the resampled data as a NumPy array
#     return resampled_img.get_fdata().astype(np.uint8)

def resample_mask_to_shape(mask_data, target_shape, original_affine):
    """
    Resamples a 3D mask to the target shape using NIfTI image metadata.

    Args:
        mask_data (numpy.ndarray): Input mask data.
        target_shape (tuple): Desired output shape (height, width, depth).
        original_affine (numpy.ndarray): Affine transformation matrix of the original mask.

    Returns:
        numpy.ndarray: Resampled mask.
    """
    # Create a temporary NIfTI image from the mask data
    mask_img = nib.Nifti1Image(mask_data, original_affine)
    
    # Resample the mask to match the target shape
    resampled_img = resample_to_output(mask_img, target_shape)
    
    # Return the resampled data as a NumPy array
    return resampled_img.get_fdata().astype(np.uint8)

def process_lidc_dataset(input_dir, output_mask_dir, log_csv, mode):
    """
    For each CT scan, randomly select one segmentation mask per nodule, 
    fuse the selected masks into one, and log the selection details.

    Args:
        input_dir (str): Directory containing the LIDC-IDRI CT and segmentation files.
        output_mask_dir (str): Directory to save the fused segmentation masks.
        log_csv (str): Path to the CSV log file tracking selected rater masks.
    """
    # Ensure output directory exists
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Prepare CSV log file
    with open(log_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["CT_File", "Nodule_ID", "Selected Rater_ID", "Selected_Mask_File"])
        
        # Group files by patient ID
        all_files = os.listdir(input_dir)
        ct_files = [f for f in all_files if f.endswith("_CT.nii.gz")]
        
        for ct_file in tqdm(ct_files, desc="Creating/Identifying Segmentation Mask for CT scan", unit="scan"):
            patient_id = ct_file.split("_")[0].split("-")[2]  # Extract patient ID

            # check if file already exists
            if os.path.exists(os.path.join(output_mask_dir, f"{patient_id}_fused_SEG.nii.gz")):
                print(f"Segmentation mask for patient {patient_id} already exists, skipping.")
                continue
            
            # Find all segmentation masks for this patient
            mask_files = [
                f for f in all_files 
                if f.startswith(f"LIDC-IDRI-{patient_id}") and f.endswith("_SEG.nii.gz")
            ]
            
            # Group masks by nodule ID
            nodule_masks = {}
            for mask_file in mask_files:
                _, nodule_id, rater_id, _ = mask_file.split("_")
                pat_nod_key = f"{patient_id}_{nodule_id}"
                nodule_masks.setdefault(pat_nod_key, []).append(mask_file)
            
            # Fuse masks for the current CT
            fused_mask = None

            if len(nodule_masks) == 0:
                # throw error
                raise ValueError(f"No masks found for patient {patient_id}")
            
            for pat_nod_key, masks in nodule_masks.items():
                # if len(masks) != 4:
                #     print(f"Warning: Nodule {pat_nod_key} does not have exactly 4 masks.")
                #     continue
                
                if mode == "random":
                    # Randomly select one mask
                    selected_mask = random.choice(masks)
                else:
                    # throw error for invalid mode
                    raise ValueError(f"Invalid mode: {mode}. Use 'random' instead.")
                
                selected_mask_path = os.path.join(input_dir, selected_mask)
                print(f"Selected mask for {pat_nod_key}: {selected_mask}")

                # Log selection to CSV
                csv_writer.writerow([ct_file, selected_mask.split("_")[1], selected_mask.split("_")[2], selected_mask])
                
                # Load selected mask
                mask_nifti = nib.load(selected_mask_path)
                mask_data = mask_nifti.get_fdata()
                
                # Initialize fused mask array if not already
                if fused_mask is None:
                    target_shape = mask_data.shape  # Set the first mask's shape as the target
                    fused_mask = np.zeros(target_shape, dtype=np.uint8)
                
                # Resample the current mask to the target shape
                if mask_data.shape != target_shape:
                    print(f"Resampling mask {selected_mask_path} from {mask_data.shape} to {target_shape}")
                    # mask_data = resample_mask_to_shape(mask_data, target_shape, mask_nifti.affine)
                    given_shape = mask_data.shape
                    zoom_factors = [target_shape[i] / given_shape[i] for i in range(3)]
                    mask_data = zoom(mask_data, zoom_factors, order=3)

                # Fuse the resampled mask
                fused_mask = np.logical_or(fused_mask, mask_data).astype(np.uint8)
            
            # Save fused mask to output directory
            if fused_mask is not None:
                fused_mask_nifti = nib.Nifti1Image(fused_mask, affine=mask_nifti.affine)
                output_file = os.path.join(output_mask_dir, f"{patient_id}_fused_SEG.nii.gz")
                nib.save(fused_mask_nifti, output_file)
                print(f"Saved fused mask for patient {patient_id} to {output_file}")

# Define paths
input_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti"  # Replace with your input path
output_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-single_seg_nifti/random-multi_rater_v2"  # Replace with desired output directory
log_file = os.path.join(output_directory, "selected_masks_log.csv")  # Path to log file

# Run the function
process_lidc_dataset(input_directory, output_directory, log_file, mode="random")
