import argparse
import logging
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mode
import tifffile as tiff


class Gleason_dataset_processor:
    """
    Class to process and prepare Gleason dataset for (federated) learning.
    Notes:
    - before processing, rename "Train Imgs" folder to "Train_imgs"
    - consensus masks via pixel-wise majority voting (D. Kamiri et al. "Deep Learning-Based Gleason Grading of Prostate Cancer From Histopathology Images—Role of Multiscale Decision Aggregation and Data Augmentation")
    - consensus masks: use STAPLE consensus masks from here (https://www.kaggle.com/datasets/danielerussica/gleason2019-grand-challenge)
    """

    def __init__(
        self,
        raw_data_path: str = None,
        single_seg_data_path: str = None,
        single_seg_mode: str = None,
    ):
        # set input args
        self.raw_data_path = raw_data_path
        self.single_seg_data_path = single_seg_data_path
        if self.single_seg_data_path:
            if not os.path.exists(self.single_seg_data_path):
                os.makedirs(self.single_seg_data_path)
        self.single_seg_mode = single_seg_mode

    def load_img(self, fname, mode: str = None):
        """
        Load image or mask from file.
        """
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Image or mask {fname} could not be loaded."
        if mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == "GRAY" and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = (img / img.max() * 255).astype(np.uint8)
        return img

    def save_img(self, img: np.ndarray, new_fname: str = None):
        """
        Save image or mask to file.
        """
        os.makedirs(os.path.dirname(new_fname), exist_ok=True)
        img = img.astype(np.uint8)
        tiff.imwrite(new_fname, img)

    def get_fnames(self, parent_dir: str = None):
        """
        Get image and mask filenames.
        """
        fnames = []
        for file_ending in ["*.png", "*.jpg"]:
            fnames.extend(
                glob.glob(
                    os.path.join(parent_dir, f"**/{file_ending}"),
                    recursive=True,
                )
            )
        # split into img and masks
        img_fnames = [fname for fname in fnames if ("Train_imgs" in fname)]
        mask_fnames = [fname for fname in fnames if "Maps" in fname]
        return img_fnames, mask_fnames

    def generate_consensus_masks(self):
        # get fnames
        img_fnames, mask_fnames = self.get_fnames(self.raw_data_path)

        # load img and masks
        for img_fname in tqdm(img_fnames, desc="Loading images and masks"):
            # get corresponding masks
            imgs_mask_fnames = [
                fname
                for fname in mask_fnames
                if os.path.basename(img_fname).replace(".jpg", "").replace(".png", "")
                in fname
            ]
            assert len(imgs_mask_fnames) > 0, f"No masks found for image {img_fname}."
            # load img
            img = self.load_img(img_fname)
            # load masks
            masks = [
                self.load_img(imgs_mask_fname) for imgs_mask_fname in imgs_mask_fnames
            ]

            # generate consensus mask
            if self.single_seg_mode == "annotatormajority":
                # pixel-wise majority voting
                masks = np.stack(masks, axis=0).astype(np.uint8)
                # Reshape to (H*W, N) to compute pixel-wise majority vote
                reshaped_masks = masks.reshape(masks.shape[0], -1)  # (N, H*W)
                # Apply bincount along each pixel location
                majority_labels = np.apply_along_axis(
                    lambda x: np.bincount(x, minlength=7).argmax(),
                    axis=0,
                    arr=reshaped_masks,
                )
                consensus_mask = majority_labels.reshape(masks.shape[1:])

            elif self.single_seg_mode == "random":
                consensus_mask = masks[np.random.randint(0, len(masks))]
            else:
                raise ValueError(f"Invalid single_seg_mode: {self.single_seg_mode}")

            # save consensus mask as .tif
            new_mask_fname = os.path.join(
                self.single_seg_data_path,
                os.path.basename(imgs_mask_fnames[0])
                .replace(".jpg", ".tif")
                .replace(".png", ".tif"),
            )
            self.save_img(consensus_mask, new_mask_fname)
            # save img as .tif
            new_img_fname = os.path.join(
                self.single_seg_data_path,
                os.path.basename(img_fname)
                .replace(".jpg", ".tif")
                .replace(".png", ".tif"),
            )
            self.save_img(img, new_img_fname)


if __name__ == "__main__":
    # set cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="",
        help="Path to raw data and masks RIGA dataset.",
    )
    parser.add_argument(
        "--single_seg_data_path",
        type=str,
        default="",
        help="Path to single, dense segmentation mask per histopathology image.",
    )
    parser.add_argument(
        "--single_seg_mode",
        type=str,
        default="",
        help="Mode to generate single segmentation mask per histopathology image."
        "Options: 'union', 'annotator_majority', 'random'.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    # parser.add_argument(
    #     "--dataset_ids",
    #     type=str,
    #     default="",
    #     help="Raw nnUNet Dataset ID to generate from raw data.",
    # )
    # parser.add_argument(
    #     "--nnUNet_raw_data_path",
    #     type=str,
    #     default="",
    #     help="Path to nnUNet raw dataset.",
    # )

    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    gleason_ds_processor = Gleason_dataset_processor(
        args.raw_data_path, args.single_seg_data_path, args.single_seg_mode
    )

    # Step1: Multi-rater masks -> consensus masks
    gleason_ds_processor.generate_consensus_masks()
