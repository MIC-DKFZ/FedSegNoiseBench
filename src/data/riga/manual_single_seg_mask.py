import os
import glob
import numpy as np
import cv2

from prepare import RIGA_dataset_processor

def manual_riga_prep_single_seg_masks(
    riga_data_preper: RIGA_dataset_processor,
    img_fname: str = None,
    out_mask_fname: str = None,
    mode: str = None,
):
    # load masks of img
    fnames = glob.glob(
            os.path.join(os.path.dirname(img_fname), os.path.basename(img_fname).replace("prime", "*")), recursive=True
        )
    imgs_seg_mask_fnames = [x for x in fnames if "prime" not in x]

    # load seg masks to np array (#annotator_masks, H, W)
    seg_masks = np.array(
        [riga_data_preper.load_img_mask(x, "GRAY") for x in imgs_seg_mask_fnames]
    )

    # Create binary masks for each structure
    structure1 = (seg_masks == 120).astype(np.uint8)
    structure2 = (seg_masks == 255).astype(np.uint8)

    final_mask = np.zeros_like(seg_masks[0], dtype=np.uint8)
    # load each masks (img + contour) and convert to proper segmentation mask
    if mode == "union":
        # load all masks and take union
        union_mask = final_mask
        for seg_mask in seg_masks:
            union_mask = np.maximum(union_mask, seg_mask)
        final_mask = union_mask
    elif mode == "annotator_majority":
        # Compute majority vote
        majority_threshold = int(len(seg_masks) * 0.5)
        consensus_structure1 = (
            np.sum(structure1, axis=0) >= majority_threshold
        ) * 120
        consensus_structure2 = (
            np.sum(structure2, axis=0) >= majority_threshold
        ) * 255
        # Merge structures into one mask
        final_mask = np.maximum(consensus_structure1, consensus_structure2)

    elif mode == "random":
        # load all masks and take random mask
        random_mask = seg_masks[np.random.randint(0, len(seg_masks))]
        final_mask = random_mask
    else:
        raise ValueError(
            f"mode must be 'union', 'annotator_majority', 'random' but is {mode}"
        )

    # save to self.single_seg_data_path
    riga_data_preper.save_img_mask_tif(final_mask, out_mask_fname, save_copy="save")

if __name__ == "__main__":
    # user input
    img_fname = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/RIGA/img_segmask_tif/Magrabia/MagrabiaMale/image23prime.tif"
    mode = "annotator_majority"

    out_mask_fname = img_fname.replace("img_segmask_tif", f"single_seg_{mode.replace('_','')}").replace("prime", f"mask")
    riga_data_preper = RIGA_dataset_processor()
    manual_riga_prep_single_seg_masks(riga_data_preper, img_fname, out_mask_fname, mode)
