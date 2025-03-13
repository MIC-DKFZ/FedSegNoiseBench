import os
import logging
import argparse
import glob
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import shutil

class RIGA_dataset_processor():
    """
    Class to process and prepare RIGA dataset for (federated) learning.
    Notes:
    - Magrabia/image48: Two times mask 1 but no non-annotated image => image removed
    """
    def __init__(self,
                 raw_data_path: str=None,
                 img_segmask_tif_data_path: str=None
                ):
        self.raw_data_path = raw_data_path
        self.img_segmask_tif_data_path = img_segmask_tif_data_path
        if not os.path.exists(self.img_segmask_tif_data_path):
            os.makedirs(self.img_segmask_tif_data_path)
        self.raw_img_fnames, self.raw_mask_fnames = [], []

    def get_img_mask_fnames(self):
        """
        Get image and mask filenames from raw data path.
        """
        # get filenames of images and masks from raw data path recursively with file extension .jpg and .tif
        raw_fnames = []
        for file_extension in ["**/*.jpg", "**/*.tif"]:
            raw_fnames.extend(glob.glob(os.path.join(self.raw_data_path, file_extension), recursive=True))
        
        # delete confusion BinRushed1 data and keep BinRushed1-Corrected data
        raw_fnames = [fname for fname in raw_fnames if "BinRushed1" not in fname or "BinRushed1-Corrected" in fname]
        # remove MagrabiaFemale/image48
        raw_fnames = [fname for fname in raw_fnames if "MagrabiFemale/image48" not in fname]
        
        # split raw_fnames into image and mask filenames
        for fname in raw_fnames:
            if "prime" in fname:
                self.raw_img_fnames.append(fname)
            else:
                self.raw_mask_fnames.append(fname)

    def load_img_mask(self, fname: str=None):
        """
        Load image or mask.
        """
        img = cv2.imread(fname)
        assert img is not None, f"Image or mask {fname} could not be loaded."
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def retrieve_masks(self, mask: np.ndarray):
        """
        Retrieve optical disk and optical cup masks from contours on RGB image.
        """
        # Convert mask_ to grayscale
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to smooth the edges
        blurred = cv2.GaussianBlur(gray_mask, (15, 15), 0)

        # Detect edges using Canny
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Dilate to connect broken edges
        dilation_kernel = np.ones((9,9), np.uint8)
        dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)

        # Erosion to thin the edges again
        erosion_kernel = np.ones((11,11), np.uint8)
        eroted_edges = cv2.erode(dilated_edges, erosion_kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(eroted_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (assuming the largest is the outer circle)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Create an empty mask
        one_hot_masks = [np.zeros_like(gray_mask, dtype=np.uint8) for _ in contours]
        fg_val_count = []
        # Ensure 2 contours or 4 contours exist
        if len(contours) == 2 or len(contours) == 4 or 0==0:
            for i, contour in enumerate(contours):
                cv2.drawContours(one_hot_masks[i], [contour], -1, 255, thickness=cv2.FILLED)
                fg_val_count.append(np.count_nonzero(one_hot_masks[i]))
        else:
            cv2.imshow('Mask', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            cv2.imshow("Dilated Edges", dilated_edges)
            cv2.imshow("Eroted Edges", eroted_edges)
            cv2.waitKey(0)  # Wait for a key press to close the windows
            cv2.destroyAllWindows()
            raise ValueError(f"Mask has {len(contours)} contours.")
        
        # get biggest and smallest mask
        biggest_mask = one_hot_masks[fg_val_count.index(max(fg_val_count))]
        smallest_mask = one_hot_masks[fg_val_count.index(min(fg_val_count))]

        # set biggest and smallest mask to 1 (optical disc) and 2 (optical cup)
        biggest_mask[biggest_mask == 255] = 1
        smallest_mask[smallest_mask == 255] = 2

        final_seg_mask = np.maximum(biggest_mask, smallest_mask)
        return final_seg_mask
    
    def save_img_mask_tif(self, img: np.ndarray, old_fname: str=None, save_copy: str=None):
        """
        Save image or mask as tif.
        """
        relative_path = os.path.relpath(old_fname, start=self.raw_data_path)
        new_fname = os.path.join(self.img_segmask_tif_data_path, os.path.splitext(relative_path)[0] + ".tif")
        os.makedirs(os.path.dirname(new_fname), exist_ok=True)

        if save_copy=="save":
            tiff.imwrite(new_fname, img)
        elif save_copy=="copy":
            shutil.copy(old_fname, new_fname)
        else:
            raise ValueError(f"save_copy must be 'save' or 'copy' but is {save_copy}")
        

    def mask_contours_to_seg_masks(self):
        """
        Convert masks to one-hot encoding.
        """
        # get image and mask filenames to self.raw_img_fnames and self.raw_mask_fnames
        self.get_img_mask_fnames()
        
        # per image, load mask, substract image from mask and save resultt as one-hot encoded mask
        for img_fname in tqdm(self.raw_img_fnames, desc="Processing images", unit="image"):
            masks_fnames = [mask for mask in self.raw_mask_fnames if img_fname.replace("prime.jpg", "").replace("prime.tif", "") in mask]
            assert len(masks_fnames) > 0, f"Image {img_fname} has {len(masks_fnames)} masks."

            # load each masks (img + contour) and convert to proper segmentation mask
            for mask_fname in masks_fnames:
                mask = self.load_img_mask(mask_fname)

                seg_mask = self.retrieve_masks(mask)

                # save one-hot encoded masks
                self.save_img_mask_tif(seg_mask, mask_fname, save_copy="save")

            if ".tif" not in img_fname:
                # load the image and convert to RGB
                img = self.load_img_mask(img_fname)
                # save image to tif
                self.save_img_mask_tif(img, img_fname, save_copy="save")
            else:
                # copy image to output path
                self.save_img_mask_tif(img, img_fname, save_copy="copy")


if __name__=="__main__":
    # set cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="",
        help="Path to raw data and masks RIGA dataset.",
    )
    parser.add_argument(
        "--img_segmask_tif_data_path",
        type=str,
        default="",
        help="Path to one-hot encoded masks of RIGA dataset.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    riga_ds_processor = RIGA_dataset_processor(args.raw_data_path, args.img_segmask_tif_data_path)
    riga_ds_processor.mask_contours_to_seg_masks()
