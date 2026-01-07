import argparse
import glob
from PIL import Image
import nibabel as nib
import numpy as np
import json
from tqdm import tqdm
from scipy import ndimage


def compare_occurring_classes(clean_mask, noisy_mask):
    """
    Compare the classes occurring in clean and noisy segmentation masks.

    Args:
        clean_mask: Numpy array of the clean segmentation mask
        noisy_mask: Numpy array of the noisy segmentation mask
    Returns:
        only_in_clean: Set of class indices only in the clean mask
        only_in_noisy: Set of class indices only in the noisy mask
        in_both: Set of class indices present in both masks
    """
    clean_classes = set(np.unique(clean_mask))
    noisy_classes = set(np.unique(noisy_mask))

    only_in_clean = clean_classes - noisy_classes
    only_in_noisy = noisy_classes - clean_classes
    in_both = clean_classes & noisy_classes

    return only_in_clean, only_in_noisy, in_both

def compare_contours(clean_mask, noisy_mask):
    """
    Compare the contours of clean and noisy segmentation masks.

    Args:
        clean_mask: Numpy array of the clean segmentation mask
        noisy_mask: Numpy array of the noisy segmentation mask
    Returns:
        same_contours: Boolean indicating if contours are the same. True if same, False otherwise.
    """
    contour_differs = not np.array_equal(clean_mask, noisy_mask)
    return contour_differs

def compare_for_missed_labels_pixellevel(clean_mask, noisy_mask):
    """
    Compare clean and noisy segmentation masks for missed labels on pixel level.

    Args:
        clean_mask: Numpy array of the clean segmentation mask
        noisy_mask: Numpy array of the noisy segmentation mask
    Returns:
        missed_labels: Boolean indicating if there are missed labels. True if there are missed labels, False otherwise.
    """
    # Missed labels occur when a pixel is foreground label in one mask corresponds to background in the other
    missed_in_noisy = np.any((clean_mask > 0) & (noisy_mask == 0))
    missed_in_clean = np.any((noisy_mask > 0) & (clean_mask == 0))
    missed_labels = missed_in_noisy or missed_in_clean
    return bool(missed_labels)

def compare_for_swapped_labels_pixellevel(clean_mask, noisy_mask):
    """
    Compare clean and noisy segmentation masks for swapped labels on pixel level.

    Args:
        clean_mask: Numpy array of the clean segmentation mask
        noisy_mask: Numpy array of the noisy segmentation mask
    Returns:
        swapped_labels: Boolean indicating if there are swapped labels. True if there are swapped labels, False otherwise.
    """
    swapped_labels = False
    clean_foreground = clean_mask > 0
    noisy_foreground = noisy_mask > 0

    if np.any(clean_foreground) and np.any(noisy_foreground):
        clean_labels = set(np.unique(clean_mask[clean_foreground]))
        noisy_labels = set(np.unique(noisy_mask[noisy_foreground]))

        if clean_labels != noisy_labels:
            swapped_labels = True

    return swapped_labels

def compare_for_missed_labels_regionlevel(clean_mask, noisy_mask, miss_iou_thresh=0.2, min_region_size=10):
    """
    Compare clean and noisy segmentation masks for missed labels on region level.

    Args:
        clean_mask: Numpy array of the clean segmentation mask
        noisy_mask: Numpy array of the noisy segmentation mask
    Returns:
        missed_labels: Boolean indicating if there are missed labels. True if there are missed labels, False otherwise.
    """
    missed_regions = []
    for c in np.unique(clean_mask):
        if c == 0:
            continue
        clean_bin = (clean_mask == c)
        noisy_bin = (noisy_mask == c)

        labeled, num = ndimage.label(clean_bin)
        for rid in range(1, num+1):
            region = (labeled == rid)
            inter = np.logical_and(region, noisy_bin).sum()
            union = np.logical_or(region, noisy_bin).sum()
            iou = inter / union if union > 0 else 0.0

            # Optional: ignore very small regions
            if region.sum() < min_region_size:
                continue

            if iou < miss_iou_thresh:
                missed_regions.append((c, rid))
    return missed_regions

def compare_for_swapped_labels_regionlevel(clean_mask, noisy_mask, majority_thresh=0.7, min_region_size=10):
    """
    Compare clean and noisy segmentation masks for swapped labels.

    Args:
        clean_mask: Numpy array of the clean segmentation mask
        noisy_mask: Numpy array of the noisy segmentation mask
    Returns:
        swapped_labels: Boolean indicating if there are swapped labels. True if there are swapped labels, False otherwise.
    """
    swapped_regions = []
    labeled, num = ndimage.label(clean_mask > 0)
    for rid in range(1, num+1):
        region = (labeled == rid)
        if region.sum() < min_region_size:
            continue

        clean_labels = clean_mask[region]
        # assume single class per region: majority in clean
        c = np.bincount(clean_labels.astype(int)).argmax()

        noisy_labels = noisy_mask[region]
        counts = np.bincount(noisy_labels.astype(int))
        noisy_major = counts.argmax()
        noisy_major_prop = counts[noisy_major] / counts.sum()

        if noisy_major_prop > majority_thresh and noisy_major not in (0, c):
            swapped_regions.append((c, noisy_major, rid))
    return swapped_regions

def compare_clean_noisy_labels(clean_label_file, noisy_label_file, gt_file_ending):
    """
    Compare clean and noisy segmentation masks for a given sample w.r.t.:
    - Occurring classes
    - Contour differences
    - Missed labels
    - Swapped labels

    Args:
        clean_label_file: Path to the clean segmentation mask file
        noisy_label_file: Path to the noisy segmentation mask file
        gt_file_ending: File extension of segmentation mask files (e.g., ".nii.gz", ".tif", ".png")

    Returns:
        sample_id: ID of the sample being compared
        res_dict: Dictionary containing comparison results
    """
    # ensure matching sample IDs
    clean_sample_id = clean_label_file.split("/")[-1].replace(gt_file_ending, "")
    noisy_sample_id = noisy_label_file.split("/")[-1].replace(gt_file_ending, "")
    assert (
        clean_sample_id == noisy_sample_id
    ), f"Sample IDs do not match: {clean_sample_id} vs {noisy_sample_id}"

    # DEBUG
    print(f"Comparing sample ID: {clean_sample_id}")
    
    # Load segmentation masks
    if gt_file_ending == ".nii.gz":
        clean_mask = np.array(nib.load(clean_label_file).get_fdata())
        noisy_mask = np.array(nib.load(noisy_label_file).get_fdata())
    elif gt_file_ending == ".tif":
        clean_mask = np.array(Image.open(clean_label_file))
        noisy_mask = np.array(Image.open(noisy_label_file))

    # Compare masks w.r.t.:
    # 1. Occurring classes
    only_in_clean, only_in_noisy, in_both = compare_occurring_classes(clean_mask, noisy_mask)
    # 2. Contour i.e. voxel-wise differences
    contour_differs = compare_contours(clean_mask, noisy_mask)
    # 3. Missed labels, i.e. foreground in one mask but background in the other
    missed_labels_pixellevel = compare_for_missed_labels_pixellevel(clean_mask, noisy_mask)
    missed_labels_regionlevel = compare_for_missed_labels_regionlevel(clean_mask, noisy_mask)
    # 4. Swapped labels. i.e. foreground label A in one mask but foreground label B in the other
    swapped_labels_pixellevel = compare_for_swapped_labels_pixellevel(clean_mask, noisy_mask)
    swapped_labels_regionlevel = compare_for_swapped_labels_regionlevel(clean_mask, noisy_mask)

    # build result dict
    res_dict = {
        "occurring_classes": {
            "only_in_clean": [int(x) for x in only_in_clean],
            "only_in_noisy": [int(x) for x in only_in_noisy],
            "in_both": [int(x) for x in in_both],
        },
        "contour_differs": bool(contour_differs),
        "missed_labels_pixellevel": bool(missed_labels_pixellevel),
        "missed_labels_regionlevel": [
            (int(c), int(rid)) for c, rid in missed_labels_regionlevel
        ],
        "swapped_labels_pixellevel": bool(swapped_labels_pixellevel),
        "swapped_labels_regionlevel": [
            (int(c), int(noisy_major), int(rid)) for c, noisy_major, rid in swapped_labels_regionlevel
        ],
    }
    
    return clean_sample_id, res_dict


def main(data_dir, clean_dataset_ids, noisy_dataset_ids, gt_file_ending, output_dir):
    sample_noise_analysis = {}
    for client_idx, (clean_id, noisy_id) in enumerate(
        zip(clean_dataset_ids, noisy_dataset_ids)
    ):
        # find dataset dirs
        clean_dataset_dir_cand = f"{data_dir}/Dataset{clean_id}_*"
        noisy_dataset_dir_cand = f"{data_dir}/Dataset{noisy_id}_*"
        clean_dataset_dir = glob.glob(clean_dataset_dir_cand)[0]
        noisy_dataset_dir = glob.glob(noisy_dataset_dir_cand)[0]
        clean_dataset_gt_dir = f"{clean_dataset_dir}/gt_segmentations"
        noisy_dataset_gt_dir = f"{noisy_dataset_dir}/gt_segmentations"
        print(
            f"\nInvestigating client {client_idx} with clean dataset {clean_dataset_gt_dir} and noisy dataset {noisy_dataset_gt_dir}"
        )

        # iterate over clean and noisy label files
        clean_label_files = sorted(
            glob.glob(f"{clean_dataset_gt_dir}/*{gt_file_ending}")
        )
        noisy_label_files = sorted(
            glob.glob(f"{noisy_dataset_gt_dir}/*{gt_file_ending}")
        )
        for clean_label_file, noisy_label_file in tqdm(zip(clean_label_files, noisy_label_files), desc=f"Processing client {client_idx}", total=len(clean_label_files)):
            sample_id, res_dict = compare_clean_noisy_labels(clean_label_file, noisy_label_file, gt_file_ending)
            sample_noise_analysis[sample_id] = res_dict
        
    # save results
    output_file = f"{output_dir}/noise_analysis_results_clean{'-'.join(clean_dataset_ids)}_noisy{'-'.join(noisy_dataset_ids)}.json"
    with open(output_file, "w") as f:
        json.dump(sample_noise_analysis, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Investigate noise types by comparing clean against noisy segmentation masks."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing clean and noisy nnUNet datasets. Dataset<clean_dataset_id>_<sth.> and Dataset<noisy_dataset_id>_<sth.> are subfolders in this folder.",
    )
    parser.add_argument(
        "--clean_dataset_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of clean dataset IDs corresponding to each client, e.g. 041 042 043 044.",
    )
    parser.add_argument(
        "--noisy_dataset_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of noisy dataset IDs corresponding to each client, e.g. 045 046 047 048.",
    )
    parser.add_argument(
        "--gt_file_ending",
        type=str,
        default=".nii.gz",
        help="File ending of the ground truth segmentation masks, e.g. .nii.gz (default), .tif.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./results",
        help="Directory to save the noise analysis results as JSON files.",
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        clean_dataset_ids=args.clean_dataset_ids[0].split(),
        noisy_dataset_ids=args.noisy_dataset_ids[0].split(),
        gt_file_ending=args.gt_file_ending,
        output_dir=args.output_dir,
    )