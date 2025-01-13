import os
import csv
import numpy as np
import pylidc as pl
import nibabel as nib
from skimage.draw import polygon
from tqdm import tqdm
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def save_nifti(volume, affine, output_path):
    """Saves a 3D volume as a NIfTI file."""
    nifti_image = nib.Nifti1Image(volume, affine)
    nib.save(nifti_image, output_path)


def visualize_scan_with_mask(scan, mask):
    # Ensure scan and mask have the same dimensions
    assert scan.shape == mask.shape, "Scan and mask dimensions must match."

    # Find the slice index with the most non-zero values in the mask
    non_zero_counts = mask.sum(axis=(1, 2))  # Sum over rows and cols for each slice
    if np.all(non_zero_counts == 0):
        print("No non-zero mask slices found.")
        return

    slice_index = np.argmax(
        non_zero_counts
    )  # Index of slice with the most non-zero values
    scan_slice = scan[slice_index]
    mask_slice = mask[slice_index]

    # Normalize the scan slice for better visualization (scale to 0-1)
    scan_slice_normalized = (scan_slice - np.min(scan_slice)) / (
        np.max(scan_slice) - np.min(scan_slice)
    )

    # Create the figure
    plt.figure(figsize=(10, 10))

    # Plot the scan with the mask overlay
    plt.imshow(scan_slice_normalized, cmap="gray")
    plt.imshow(mask_slice, cmap="jet", alpha=0.5)  # Overlay mask with transparency

    plt.title(f"CT Scan with Mask Overlay (Slice {slice_index})")
    plt.axis("off")
    plt.colorbar(label="Mask Intensity")
    plt.show()


def create_segmentation_mask(scan, annotations, output_dir):
    """Creates segmentation masks for each annotation set and saves them as NIfTI files."""
    # Prepare the 3D array for the mask with the same dimensions as the CT scan
    mask = np.zeros(scan.to_volume().shape, dtype=np.uint8)

    z_positions = scan.slice_zvals.tolist()  # Convert zvals to a list for indexing

    for annotation in annotations:
        for nodule in annotation.tied_annotations:
            for roi in nodule.rois:
                if roi.image_z_position not in z_positions:
                    continue

                # Find the corresponding slice index
                z_index = z_positions.index(roi.image_z_position)

                # Create a binary mask for this slice
                rr, cc = polygon(roi.y, roi.x, mask.shape[1:])
                mask[z_index, rr, cc] = 1

    return mask


def get_affine(scan):
    """Computes the affine transformation matrix for the scan."""
    # Use slice_zvals instead of slice_z_positions
    slice_spacing = np.mean(np.diff(scan.slice_zvals))  # Spacing between slices
    pixel_spacing = scan.pixel_spacing  # In-plane resolution
    affine = np.array(
        [
            [pixel_spacing, 0, 0, 0],
            [0, pixel_spacing, 0, 0],
            [0, 0, slice_spacing, 0],
            [0, 0, 0, 1],
        ]
    )
    return affine


def process_lidc_dataset(output_dir):
    """Processes the LIDC-IDRI dataset to extract multi-rater annotations and save as NIfTI."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create .csv file for meta data
    meta_data_path = os.path.join(output_dir, "interrater_meta_data.csv")
    metadata_rows = []
    column_names = [
        "patient_id", "nod_idx", "rater_idx",
        "Calcification", "calcification",
        "InternalStructure",
        "Lobulation", "lobulation",
        "Malignancy", "malignancy",
        "Margin", "margin",
        "Sphericity", "sphericity",
        "Spiculation", "spiculation",
        "Subtlety", "subtlety",
        "Texture", "texture"
    ]

    # Query all scans
    scans = pl.query(pl.Scan).all()

    for scan in tqdm(scans, desc="Processing scans"):
        patient_id = scan.patient_id
        volume_path = os.path.join(output_dir, f"{patient_id}_CT.nii.gz")

        if not os.path.exists(volume_path):
            # Convert the scan's voxel data to a NIfTI file and save
            volume = scan.to_volume()
            affine = get_affine(scan)
            save_nifti(volume, affine, volume_path)
        else:
            print("CT already exists, skip saving of CT volume!")

        # get nodules of current scan
        nodules = scan.cluster_annotations()

        # iterate over nodules
        for nod_idx, nodule in enumerate(nodules):
            for rater_idx, rater_annotation in enumerate(nodule):
                seg_output_fname = os.path.join(
                    output_dir, f"{patient_id}_{nod_idx}_{rater_idx}_SEG.nii.gz"
                )

                try:
                    # collect data to be added to the .csv file
                    row = [
                        patient_id,nod_idx,rater_idx,
                        rater_annotation.Calcification,
                        rater_annotation.calcification,
                        rater_annotation.InternalStructure,
                        rater_annotation.Lobulation,
                        rater_annotation.lobulation,
                        rater_annotation.Malignancy,
                        rater_annotation.malignancy,
                        rater_annotation.Margin,
                        rater_annotation.margin,
                        rater_annotation.Sphericity,
                        rater_annotation.sphericity,
                        rater_annotation.Spiculation,
                        rater_annotation.spiculation,
                        rater_annotation.Subtlety,
                        rater_annotation.subtlety,
                        rater_annotation.Texture,
                        rater_annotation.texture
                    ]
                    metadata_rows.append(row)
                except AssertionError as e:
                    print(
                        f"Skipping problematic annotation for Patient ID: {patient_id}, Nodule Index: {nod_idx}, Rater Index: {rater_idx} due to error: {e}"
                    )
                    # Optionally add an error row or log it elsewhere
                    metadata_rows.append([patient_id, nod_idx, rater_idx] + ["Error"] * (len(column_names) - 3))
                    # Optionally log the error or record problematic cases elsewhere
                    # continue

                if os.path.exists(seg_output_fname):
                    print("Segmentation already exists, skip saving SEG mask!")
                else:
                    # create empty mask for each nodule and rater
                    if "volume" not in locals():
                        volume = scan.to_volume()
                        affine = get_affine(scan)
                    mask = np.zeros(volume.shape, dtype=np.uint8)
                    mask[rater_annotation.bbox()][rater_annotation.boolean_mask()] += 1
                    save_nifti(mask, affine, seg_output_fname)
        
        # Write the collected metadata to a CSV file
        if metadata_rows:
            with open(meta_data_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(column_names)
                # Write rows
                writer.writerows(metadata_rows)


if __name__ == "__main__":
    output_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-nifti"
    process_lidc_dataset(output_dir)
