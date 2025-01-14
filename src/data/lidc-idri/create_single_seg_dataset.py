import os
import random
import nibabel as nib
import numpy as np
import csv
from tqdm import tqdm
from scipy.ndimage import zoom
import argparse
import pandas as pd
from collections import Counter

MAX_ANNOTATORS = 4
MALIGNANCY_THRESHOLD = 3


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


def process_lidc_dataset(input_dir, output_mask_dir, log_csv, mode, multi_class):
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

    # load metadata from csv file
    metadata_df = pd.read_csv(
        os.path.join(input_dir, "interrater_meta_data_w_values.csv")
    )

    # Prepare CSV log file
    with open(log_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["CT_File", "Nodule_ID", "Selected Rater_ID", "Selected_Mask_File", "Maligancy"]
        )

        # Group files by patient ID
        all_files = os.listdir(input_dir)
        ct_files = [f for f in all_files if f.endswith("_CT.nii.gz")]

        for ct_file in tqdm(
            ct_files,
            desc="Creating/Identifying Segmentation Mask for CT scan",
            unit="scan",
        ):
            patient_id = ct_file.split("_")[0].split("-")[2]  # Extract patient ID
            lidc_id = ct_file.split("_")[0]  # Extract LIDC-IDRI ID

            # check if file already exists
            if os.path.exists(
                os.path.join(output_mask_dir, f"{patient_id}_fused_{mode}_SEG.nii.gz")
            ):
                print(
                    f"Segmentation mask for patient {patient_id} already exists, skipping."
                )
                continue

            # Find all segmentation masks for this patient
            mask_files = [
                f
                for f in all_files
                if f.startswith(f"LIDC-IDRI-{patient_id}") and f.endswith("_SEG.nii.gz")
            ]

            # Group masks by nodule ID
            nodule_masks = {}
            for mask_file in mask_files:
                _, nodule_id, rater_id, _ = mask_file.split("_")
                pat_nod_key = f"{patient_id}_{nodule_id}"
                nodule_masks.setdefault(pat_nod_key, []).append(
                    os.path.join(input_dir, mask_file)
                )
            if len(nodule_masks) == 0:
                # throw error
                raise ValueError(f"No masks found for patient {patient_id}")

            final_nodule_masks = []
            final_nodule_malignancies = []
            # iterate over N_j nodule masks
            for pat_nod_key, masks in nodule_masks.items():
                final_nodule_mask = None

                if mode == "random":
                    # Randomly select one mask
                    selected_mask = random.choice(masks)
                    selected_mask_path = os.path.join(input_dir, selected_mask)
                    nodule_id = pat_nod_key.split("_")[1]
                    selected_rater_id = os.path.basename(selected_mask_path).split("_")[
                        2
                    ]
                    
                    # Load the selected mask and it's malignancy
                    final_nodule_mask = nib.load(selected_mask_path).get_fdata()
                    affine = nib.load(selected_mask_path).affine
                    malignancy_df_entry = metadata_df[
                        (metadata_df["patient_id"] == lidc_id)
                        & (metadata_df["nod_idx"] == int(nodule_id))
                        & (metadata_df["rater_idx"] == int(selected_rater_id))
                    ]
                    if malignancy_df_entry.empty:
                        final_nodule_malignancy = 0
                    else:
                        selected_malignancy = malignancy_df_entry[
                            "malignancy"
                        ].values[0]
                        final_nodule_malignancy = 2 if selected_malignancy >= MALIGNANCY_THRESHOLD else 1

                    print(f"Selected mask for {pat_nod_key}: {selected_mask} --> to logging .csv!")
                    csv_writer.writerow(
                        [ct_file, nodule_id, selected_rater_id, selected_mask, final_nodule_malignancy]
                    )

                elif (
                    mode == "union"
                    or mode == "lesion_majority"
                    or mode == "annotator_majority"
                ):
                    tmp_nodule_mask = None
                    count_masks_w_nodule_annotated = 0
                    nodule_raters_malignancies = []
                    # add annotator i's masks up for this nodule j
                    for mask in masks:
                        # Load annotator i's mask of nodule j
                        mask_data = nib.load(mask).get_fdata()
                        nodule_id = pat_nod_key.split("_")[1]
                        rater_id = os.path.basename(mask).split("_")[2]

                        # Initialize fused mask array if not already
                        if tmp_nodule_mask is None:
                            target_shape = mask_data.shape
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

                        # get malignancy of current rater
                        malignancy_df_entry = metadata_df[
                            (metadata_df["patient_id"] == lidc_id)
                            & (metadata_df["nod_idx"] == int(nodule_id))
                            & (metadata_df["rater_idx"] == int(rater_id))
                        ]
                        if malignancy_df_entry.empty:
                            nodule_raters_malignancies.append(None)
                        else:
                            nodule_raters_malignancies.append(
                                malignancy_df_entry["malignancy"]
                                .values[0]
                                .astype(np.uint8)
                            )

                    # Compose the nodule's mask based on the mode
                    if mode == "union":
                        final_nodule_mask = (tmp_nodule_mask > 0).astype(np.uint8)
                    elif mode == "lesion_majority":
                        final_nodule_mask = (
                            tmp_nodule_mask >= count_masks_w_nodule_annotated // 2
                        ).astype(np.uint8)
                    elif mode == "annotator_majority":
                        final_nodule_mask = (
                            tmp_nodule_mask >= MAX_ANNOTATORS // 2
                        ).astype(np.uint8)
                    else:
                        # throw error for invalid mode
                        raise ValueError(f"Invalid mode: {mode}. Use 'random', 'union', 'lesion_majority', 'annotator_majority' instead.")

                    # Determine malignancy based on mode
                    if mode == "union":
                        # mode="union" expects the worst case w.r.t. the annotation masks as annotation mask is maximized
                        # consequently, we also set the malignancy to the worst value, i.e. the maximum value
                        max_malignancy = np.max(nodule_raters_malignancies)
                        final_nodule_malignancy = 2 if max_malignancy >= MALIGNANCY_THRESHOLD else 1
                    elif mode == "lesion_majority":
                        # WARNING: not exactly the same as the reference implementation (https://git.dkfz.de/mic/internal/nndetection/-/blob/main/tasks/Task044_LIDC_pylidc/scripts/prepare.py?ref_type=heads#L146)
                        # remove None as they indicate no lesion found, and don't count in mode "lesion_majority"
                        nodule_raters_malignancies = [x for x in nodule_raters_malignancies if x is not None]
                        counter = Counter(nodule_raters_malignancies)
                        # get majority malignancy
                        most_common_malignancy = counter.most_common(1)[0][0]
                        # set final nodule malignancy=2 (for class label "malignant") if mean malignancy is above threshold else 1 (for class label "benign")
                        final_nodule_malignancy = 2 if most_common_malignancy >= MALIGNANCY_THRESHOLD else 1
                    elif mode == "annotator_majority":
                        # replace None with 0 as they indicate no lesion found, but do count in mode "annotator_majority"
                        nodule_raters_malignancies = [0 if x is None else x for x in nodule_raters_malignancies]
                         # compute mean
                        mean_malignancy = int(np.mean(nodule_raters_malignancies))
                        # set final nodule malignancy=2 (for class label "malignant") if mean malignancy is above threshold else 1 (for class label "benign")
                        final_nodule_malignancy = 2 if mean_malignancy >= MALIGNANCY_THRESHOLD else 1
                    else:
                        # throw error for invalid mode
                        raise ValueError(f"Invalid mode: {mode}. Use 'union', 'lesion_majority', 'annotator_majority' instead.")

                    # log to file
                    print(f"Fused mask for {pat_nod_key} --> to logging .csv!")
                    csv_writer.writerow(
                        [ct_file, nodule_id, f"fused via {mode}", f"fused via {mode}", final_nodule_malignancy]
                    )

                else:
                    # throw error for invalid mode
                    raise ValueError(f"Invalid mode: {mode}. Use 'random', 'union', 'lesion_majority', 'annotator_majority' instead.")
                
                # if differentiating between benign and malignant nodules else just background and nodule
                if multi_class:
                    # set forground values in final_nodule_mask to final_nodule_malignancy
                    final_nodule_mask[final_nodule_mask > 0] = final_nodule_malignancy

                # add the final nodule mask to the list
                final_nodule_masks.append(final_nodule_mask)
                final_nodule_malignancies.append(final_nodule_malignancy)

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
                final_fused_mask += final_nodule_mask.astype(np.uint8)

            if not multi_class:
                # for binary mask, set all values > 0 to 1
                tmp_final_fused_mask = (final_fused_mask > 0).astype(np.uint8)
                final_fused_mask = tmp_final_fused_mask
            else:
                assert np.all(final_fused_mask <= 2) 

            # Save fused mask to output directory
            if final_fused_mask is not None:
                final_fused_mask_nifti = nib.Nifti1Image(
                    final_fused_mask, affine=affine
                )
                output_file = os.path.join(
                    output_mask_dir, f"{patient_id}_fused_{mode}_SEG.nii.gz"
                )
                nib.save(final_fused_mask_nifti, output_file)
                print(f"Saved fused mask for patient {patient_id} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LIDC dataset for segmentation masks.")
    parser.add_argument("input_directory", type=str, help="Path to the input directory containing CT and mask files.")
    parser.add_argument("output_directory", type=str, help="Path to the output directory for fused masks.")
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    parser.add_argument("--mode", type=str, default="lesion_majority", choices=["random", "union", "lesion_majority", "annotator_majority"],
                        help="Mode for selecting segmentation masks (default: lesion_majority).")
    parser.add_argument("--multi_class", action="store_true", help="Differentiate between benign and malignant nodules.")

    args = parser.parse_args()

    process_lidc_dataset(args.input_directory, args.output_directory, args.log_file, args.mode, args.multi_class)

# # Define paths
# input_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti"
# output_directory = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-single_seg_nifti/malignancy_annotator_majority"
# log_file = os.path.join(output_directory, "single_seg_per_ct.csv")
# # Run the function
# process_lidc_dataset(
#     input_directory, output_directory, log_file, mode="annotator_majority", multi_class=True
# )  # mode="random", "union", "lesion_majority", "annotator_majority"
