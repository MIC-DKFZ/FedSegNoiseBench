import os
import numpy as np
import tifffile as tiff
import cv2
from tqdm import tqdm
from collections import defaultdict
import glob


def find_count_unique_labels(gleason_dir):
    print(f"Finding unique labels in {gleason_dir}")
    # Dictionary to store label counts
    label_counts = defaultdict(int)

    filenames = glob.glob(os.path.join(gleason_dir, "**", "*.tif"), recursive=True)

    # Loop through all .tif files in the directory
    for filename in tqdm(filenames, desc="Counting unique labels"):
        if filename.endswith(".tif"):  #  and "_classimg_nonconvex" in filename:
            filepath = os.path.join(gleason_dir, filename)

            # Load the segmentation mask
            mask = tiff.imread(filepath)
            # mask = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            # Get unique labels in the mask
            unique_labels = np.unique(mask)

            # Increment the count for each label
            for label in unique_labels:
                label_counts[
                    str(label)
                ] += 1  # Convert label to string for dictionary keys

    # Convert defaultdict to regular dict
    label_counts = dict(label_counts)

    # Print results
    print(label_counts)


def kaggle_random_consensus_masks(raw_dir, single_seg_dir):
    """
    Get random consensus masks from Kaggle dataset.
    """
    if not os.path.exists(single_seg_dir):
        os.makedirs(single_seg_dir)

    # get fnames
    mask_fnames = glob.glob(
        os.path.join(raw_dir, "**/**/*.png"),
        recursive=True,
    )
    # extract all unique slideXXX_coreXXX names
    unique_slide_core_ids = {
        os.path.basename(x).replace("_classimg_nonconvex.png", "") for x in mask_fnames
    }

    for slide_core_id in unique_slide_core_ids:
        # get mask_fnames of current slide_core_id
        curr_mask_fnames = [x for x in mask_fnames if slide_core_id in x]
        # select random mask
        random_mask_fname = curr_mask_fnames[
            np.random.randint(0, len(curr_mask_fnames))
        ]
        # load random mask and save as .tif with new fnam
        mask = cv2.imread(random_mask_fname, cv2.IMREAD_UNCHANGED)
        new_fname = os.path.join(
            single_seg_dir, os.path.basename(random_mask_fname).replace(".png", ".tif")
        )
        tiff.imwrite(new_fname, mask)
    print("Done")


if __name__ == "__main__":
    # gleason_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/single_seg_staple_grandchallenge"
    # gleason_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/single_seg_annotatormajority"
    # gleason_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/raw_kaggle"
    gleason_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/nnUNet_raw/Dataset409_Gleason2019_random_flclient1/labelsTr"

    find_count_unique_labels(gleason_dir)
    # kaggle_random_consensus_masks(gleason_dir, single_seg_random_dir)
