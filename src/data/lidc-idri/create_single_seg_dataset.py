import os
import random
import nibabel as nib
import numpy as np
import csv
from tqdm import tqdm
from scipy.ndimage import zoom
import argparse

MAX_ANNOTATORS = 4

def resample_mask(mask_data, target_shape):
    """
    Resample to target shape using scipy.ndimage.zoom.

    Args:
        mask_data (np.ndarray): Segmentation mask data.
        target_shape (tuple): Target shape for resampling.
    """
    print(f"Resampling mask from {mask_data.shape} to {target_shape}")
    given_shape = mask_data.shape
    zoom_factors = [target_shape[i] / given_shape[i] for i in range(3)]
    mask_data = zoom(mask_data, zoom_factors, order=3)
    return mask_data

def process_lidc_dataset(input_dir, output_mask_dir, log_csv, mode):
    """
    For each CT scan, randomly select one segmentation mask per nodule, 
    fuse the selected masks into one, and log the selection details.

    Args:
        input_dir (str): Directory containing the LIDC-IDRI CT and segmentation files.
        output_mask_dir (str): Directory to save the fused segmentation masks.
        log_csv (str): Path to the CSV log file tracking selected rater masks.
        mode (str): Mode for selecting the segmentation mask(s) per nodule. 
            Options: "random", "union", "lesion_majority", "annotator_majority"
            `union`: at least one annotator marked the voxel
            `lesion_majority`: majority voting per lesion (only annotators for
                the respective lesion are counted)
            `annotator_majority`: majority voting per voxel (all annotators are
                counted)

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
                nodule_masks.setdefault(pat_nod_key, []).append(os.path.join(input_dir, mask_file))
            if len(nodule_masks) == 0:
                # throw error
                raise ValueError(f"No masks found for patient {patient_id}")
            
            final_nodule_masks = []
            # iterate over N_j nodule masks
            for pat_nod_key, masks in nodule_masks.items():
                final_nodule_mask = None

                if mode == "random":
                    # Randomly select one mask
                    selected_mask = random.choice(masks)
                    selected_mask_path = os.path.join(input_dir, selected_mask)
                    print(f"Selected mask for {pat_nod_key}: {selected_mask}")
                    csv_writer.writerow([ct_file, selected_mask.split("_")[1], selected_mask.split("_")[2], selected_mask])
                    final_nodule_mask = nib.load(selected_mask_path).get_fdata()
                    affine = nib.load(selected_mask_path).affine

                elif mode == "union" or mode == "lesion_majority" or mode == "annotator_majority":
                    tmp_nodule_mask = None
                    count_masks_w_nodule_annotated = 0
                    # add annotator i's masks up for this nodule j
                    for mask in masks:
                        # Load annotator i's mask of nodule j
                        mask_data = nib.load(mask).get_fdata()

                        # Initialize fused mask array if not already
                        if tmp_nodule_mask is None:
                            target_shape = mask_data.shape  # Set the first mask's shape as the target
                            tmp_nodule_mask = np.zeros(target_shape, dtype=np.uint8)
                            affine = nib.load(mask).affine

                        # Resample the current mask to the target shape
                        if mask_data.shape != target_shape:
                            mask_data = resample_mask(mask_data, target_shape)

                        # Fuse the resampled mask for current nodule
                        tmp_nodule_mask += mask_data.astype(np.uint8)

                        # check if mask actually contains nodule or is empty
                        if np.max(mask_data) != 0:
                            count_masks_w_nodule_annotated += 1

                    # mode "union", "lesion_majority", "annotator_majority" logic
                    if mode == "union":
                        final_nodule_mask = (tmp_nodule_mask > 0).astype(np.uint8)
                    elif mode == "lesion_majority":
                        final_nodule_mask = (tmp_nodule_mask >= count_masks_w_nodule_annotated // 2).astype(np.uint8)
                    elif mode == "annotator_majority":
                        final_nodule_mask = (tmp_nodule_mask >= MAX_ANNOTATORS // 2).astype(np.uint8)
                    else:
                        # throw error for invalid mode
                        raise ValueError(f"Invalid mode: {mode}. Use 'random' instead.")

                else:
                    # throw error for invalid mode
                    raise ValueError(f"Invalid mode: {mode}. Use 'random' instead.")
                
                # add the final nodule mask to the list
                final_nodule_masks.append(final_nodule_mask)

            # Fuse the final nodule masks
            final_fused_mask = None
            for final_nodule_mask in final_nodule_masks:
                # initialize final_fused_mask and define target shape
                if final_fused_mask is None:
                    final_shape = final_nodule_mask.shape
                    final_fused_mask = np.zeros(final_shape, dtype=np.uint8)

                # for mode="random", ensure all masks have the same shape
                if final_nodule_mask.shape != final_shape:
                    final_nodule_mask = resample_mask(final_nodule_mask, final_shape)
                # just sum them up
                final_fused_mask += final_nodule_mask

            # for binary mask, set all values > 0 to 1
            tmp_final_fused_mask = (final_fused_mask > 0).astype(np.uint8)
            final_fused_mask = tmp_final_fused_mask
            
            # Save fused mask to output directory
            if final_fused_mask is not None:
                final_fused_mask_nifti = nib.Nifti1Image(final_fused_mask, affine=affine)
                output_file = os.path.join(output_mask_dir, f"{patient_id}_fused_SEG.nii.gz")
                nib.save(final_fused_mask_nifti, output_file)
                print(f"Saved fused mask for patient {patient_id} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LIDC dataset for segmentation masks.")
    parser.add_argument("input_directory", type=str, help="Path to the input directory containing CT and mask files.")
    parser.add_argument("output_directory", type=str, help="Path to the output directory for fused masks.")
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    parser.add_argument("--mode", type=str, default="lesion_majority", choices=["random", "union", "lesion_majority", "annotator_majority"],
                        help="Mode for selecting segmentation masks (default: lesion_majority).")

    args = parser.parse_args()

    process_lidc_dataset(args.input_directory, args.output_directory, args.log_file, args.mode)

# # Define paths
# input_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti"
# output_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-single_seg_nifti/lesion_majority"
# log_file = os.path.join(output_directory, "selected_masks_log.csv")
# # Run the function
# process_lidc_dataset(input_directory, output_directory, log_file, mode="lesion_majority") # mode="random", "union", "lesion_majority", "annotater_majority"