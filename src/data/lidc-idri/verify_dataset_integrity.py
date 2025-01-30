import os
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def resample_image_to_match(reference_image, target_image, is_label=False):
    """
    Resamples the target image to match the reference image in spacing and shape.

    Parameters:
        reference_image (sitk.Image): The reference image (CT).
        target_image (sitk.Image): The target image (segmentation or CT).
        is_label (bool): Whether the target image is a label (segmentation).

    Returns:
        sitk.Image: Resampled image.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    )
    resample.SetOutputSpacing(reference_image.GetSpacing())
    resample.SetSize(reference_image.GetSize())
    resample.SetOutputDirection(reference_image.GetDirection())
    resample.SetOutputOrigin(reference_image.GetOrigin())

    return resample.Execute(target_image)


def process_files(images_dir, labels_dir, multi_class=False):
    """
    Processes and fixes mismatches between images and labels.

    Parameters:
        images_dir (str): Directory containing CT images.
        labels_dir (str): Directory containing segmentations.
    """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])

    for image_file, label_file in tqdm(zip(image_files, label_files), desc="Processing files", total=len(image_files)):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        print(f"Processing {image_file} and {label_file}...")

        # Read the image and label
        ct_image = sitk.ReadImage(image_path)
        seg_image = sitk.ReadImage(label_path)
        seg_array = sitk.GetArrayFromImage(seg_image)
        unique_seg_values = np.unique(seg_array)

        write_seg_image = False
        # Fix spacing mismatch
        if not np.allclose(ct_image.GetSpacing(), seg_image.GetSpacing(), atol=1e-5):
            print(f"Fixing spacing for {label_file} to match {image_file}...")
            seg_image = resample_image_to_match(ct_image, seg_image, is_label=True)
            write_seg_image = True

        # Fix shape mismatch
        if ct_image.GetSize() != seg_image.GetSize():
            print(f"Fixing shape for {label_file} to match {image_file}...")
            seg_image = resample_image_to_match(ct_image, seg_image, is_label=True)
            write_seg_image = True

        if not multi_class:
            # Fix binary mask values
            if (unique_seg_values > 1).any():
                print(f"Fixing binary mask values for {label_file}...")
                seg_array[seg_array > 0] = 1
                tmp_seg_image = sitk.GetImageFromArray(seg_array)
                tmp_seg_image.CopyInformation(seg_image)
                seg_image = tmp_seg_image
                write_seg_image = True

        if write_seg_image:
            # Save the resampled images
            sitk.WriteImage(seg_image, os.path.join(labels_dir, label_file))

    print("Processing complete!")


if __name__ == "__main__":

    # CLI argument parsing
    parser = argparse.ArgumentParser(description="Fix mismatches between nnUNet images and labels.")
    parser.add_argument("--images_dir", required=True, help="Path to the CT images directory (imagesTr).")
    parser.add_argument("--labels_dir", required=True, help="Path to the labels directory (labelsTr).")
    parser.add_argument("--multi_class", action="store_true", help="Differentiate between benign and malignant nodules.")

    args = parser.parse_args()

    process_files(args.images_dir, args.labels_dir, args.multi_class)

    # # for debugging
    # process_files(
    #     images_dir="/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/nnUNet_raw/Dataset027_LIDC-Malignancy-RandomMultiRater/imagesTr", 
    #     labels_dir="/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/nnUNet_raw/Dataset027_LIDC-Malignancy-RandomMultiRater/labelsTr",
    #     multi_class=True
    # )