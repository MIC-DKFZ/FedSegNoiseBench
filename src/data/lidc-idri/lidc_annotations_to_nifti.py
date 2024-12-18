import os
import numpy as np
import pylidc as pl
import nibabel as nib
from skimage.draw import polygon
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def save_nifti(volume, affine, output_path):
    """Saves a 3D volume as a NIfTI file."""
    nifti_image = nib.Nifti1Image(volume, affine)
    nib.save(nifti_image, output_path)

# def visualize_scan_with_mask(scan, mask):
#     # Ensure scan and mask have the same dimensions
#     assert scan.shape == mask.shape, "Scan and mask dimensions must match."

#     # Find the slice index where the mask has non-zero values
#     slice_indices = np.where(mask.sum(axis=(1, 2)) > 0)[0]  # Sum over rows and cols for each slice
#     if len(slice_indices) == 0:
#         print("No non-zero mask slices found.")
#         return

#     # Choose the first slice with non-zero mask values
#     slice_index = slice_indices[0]
#     scan_slice = scan[slice_index]
#     mask_slice = mask[slice_index]

#     # Normalize the scan slice for better visualization (scale to 0-1)
#     scan_slice_normalized = (scan_slice - np.min(scan_slice)) / (np.max(scan_slice) - np.min(scan_slice))

#     # Create the figure
#     plt.figure(figsize=(10, 10))

#     # Plot the scan with the mask overlay
#     plt.imshow(scan_slice_normalized, cmap="gray")
#     plt.imshow(mask_slice, cmap="jet", alpha=0.5)  # Overlay mask with transparency

#     plt.title(f"CT Scan with Mask Overlay (Slice {slice_index})")
#     plt.axis("off")
#     plt.colorbar(label="Mask Intensity")
#     plt.show()
def visualize_scan_with_mask(scan, mask):
    # Ensure scan and mask have the same dimensions
    assert scan.shape == mask.shape, "Scan and mask dimensions must match."

    # Find the slice index with the most non-zero values in the mask
    non_zero_counts = mask.sum(axis=(1, 2))  # Sum over rows and cols for each slice
    if np.all(non_zero_counts == 0):
        print("No non-zero mask slices found.")
        return

    slice_index = np.argmax(non_zero_counts)  # Index of slice with the most non-zero values
    scan_slice = scan[slice_index]
    mask_slice = mask[slice_index]

    # Normalize the scan slice for better visualization (scale to 0-1)
    scan_slice_normalized = (scan_slice - np.min(scan_slice)) / (np.max(scan_slice) - np.min(scan_slice))

    # Create the figure
    plt.figure(figsize=(10, 10))

    # Plot the scan with the mask overlay
    plt.imshow(scan_slice_normalized, cmap="gray")
    plt.imshow(mask_slice, cmap="jet", alpha=0.5)  # Overlay mask with transparency

    plt.title(f"CT Scan with Mask Overlay (Slice {slice_index})")
    plt.axis("off")
    plt.colorbar(label="Mask Intensity")
    plt.show()




# def create_segmentation_mask(scan, annotations, output_dir):
#     """Creates segmentation masks for each annotation set and saves them as NIfTI files."""
#     # Prepare the 3D array for the mask with the same dimensions as the CT scan
#     mask = np.zeros(scan.to_volume().shape, dtype=np.uint8)

#     for annotation in annotations:
#         for nodule in annotation.tied_annotations:
#             for roi in nodule.rois:
#                 if roi.image_z_position not in scan.slice_z_positions:
#                     continue

#                 # Find the corresponding slice index
#                 z_index = scan.slice_z_positions.index(roi.image_z_position)

#                 # Create a binary mask for this slice
#                 rr, cc = polygon(roi.y, roi.x, mask.shape[1:])
#                 mask[z_index, rr, cc] = 1


#     return mask
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


# def get_affine(scan):
#     """Computes the affine transformation matrix for the scan."""
#     slice_spacing = np.mean(np.diff(scan.slice_z_positions))  # Spacing between slices
#     pixel_spacing = scan.pixel_spacing  # In-plane resolution
#     affine = np.array([
#         [pixel_spacing[0], 0, 0, 0],
#         [0, pixel_spacing[1], 0, 0],
#         [0, 0, slice_spacing, 0],
#         [0, 0, 0, 1],
#     ])
#     return affine
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
                seg_output_fname = os.path.join(output_dir, f"{patient_id}_{nod_idx}_{rater_idx}_SEG.nii.gz")
                if os.path.exists(seg_output_fname):
                    print("Segmentation already exists, skipping")
                    continue

                # Write metadata to .csv file
                with open(meta_data_path, "a") as f:
                    try:
                        # Retrieve attributes safely
                        calcification = rater_annotation.Calcification
                        internal_structure = rater_annotation.InternalStructure
                        lobulation = rater_annotation.Lobulation
                        malignancy = rater_annotation.Malignancy
                        margin = rater_annotation.Margin
                        sphericity = rater_annotation.Sphericity
                        spiculation = rater_annotation.Spiculation
                        subtlety = rater_annotation.Subtlety
                        texture = rater_annotation.Texture

                        # Write to the CSV file
                        f.write(
                            f"{patient_id},{nod_idx},{rater_idx},{rater_annotation.calcification=},{calcification=},{internal_structure=},{lobulation=},{malignancy=},{margin=},{sphericity=},{spiculation=},{subtlety=},{texture=}\n"
                        )
                    except AssertionError as e:
                        # Write to the CSV file
                        f.write(
                            f"{patient_id},{nod_idx},{rater_idx}, Error in metadata\n"
                        )
                        print(f"Skipping problematic annotation for Patient ID: {patient_id}, Nodule Index: {nod_idx}, Rater Index: {rater_idx} due to error: {e}")
                        # Optionally log the error or record problematic cases elsewhere
                        continue

                # create empty mask for each nodule and rater
                if "volume" not in locals():
                    volume = scan.to_volume()
                    affine = get_affine(scan)
                mask = np.zeros(volume.shape, dtype=np.uint8)
                mask[rater_annotation.bbox()][rater_annotation.boolean_mask()] +=1

                save_nifti(mask, affine, seg_output_fname)


if __name__ == "__main__":
    output_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-nifti"
    process_lidc_dataset(output_dir)
