import os
import logging
import argparse
import glob
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import shutil
import hashlib
from PIL import Image, ImageChops


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
        # if ".tif" in fname:
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Image or mask {fname} could not be loaded."
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # elif ".jpg" in fname:
        #     img = Image.open(fname).convert("RGB")
        return img
    
    def retrieve_masks(self, img: np.ndarray, mask: np.ndarray, mask_fname: str=None):
        """
        Retrieve optical disk and optical cup masks from contours on RGB image.
        """
        # if ".tif" in mask_fname:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        diff_ = cv2.absdiff(gray_mask, gray_img)
        _, diff = cv2.threshold(diff_, 20, 255, cv2.THRESH_BINARY)
        # elif ".jpg" in mask_fname:
        #     diff = ImageChops.difference(mask, img)
        # save diff to file with a unique name including a hash
        # cv2.imwrite(f"diff_{hashlib.md5(diff).hexdigest()}.png", diff)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
        # Define a minimum size threshold (e.g., 50 pixels)
        min_size = 20
        # Create a mask for large components
        filtered_diff = np.zeros_like(diff)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_diff[labels == i] = 255

        contours = []
        i=0
        closed = filtered_diff
        while len(contours) != 2 and len(contours) != 4:
            # if too many closing iterations necessary, skip image and write to log file
            if i>10:
                logging.error(f"Too many closing iterations necessary for {mask_fname}.")
                with open("log.txt", "a") as file:
                    file.write(f"{mask_fname}\n")
                cv2.imshow("diff_", diff_)
                cv2.imshow("diff", diff)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return np.array([])
            kernel = np.ones((int(3+(i/5)),int(3+(i/5))), np.uint8)  # Small kernel to minimize edge distortion
            # Apply Closing (Dilation + Erosion)
            # closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
            closed = cv2.dilate(closed, kernel + np.ones((1,1), np.uint8), iterations=2) # np.ones((5,5)
            closed = cv2.erode(closed, kernel, iterations=1) # np.ones((3,3)
            # Find contours from the binary mask
            contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            i+=1
        
        contours = sorted(contours, key=cv2.contourArea)
        # segmentation_mask = np.zeros_like(mask, dtype=np.uint8)
        # # Fill the inner circle (smallest contour) with pixel value 2
        # if len(contours) > 0:
        #     cv2.drawContours(segmentation_mask, [contours[0]], -1, 255, thickness=cv2.FILLED)
        # # Fill the outer circle (largest contour) with pixel value 1, but keeping the inner circle intact
        # if len(contours) > 1:
        #     cv2.drawContours(segmentation_mask, [contours[-1]], -1, 180, thickness=cv2.FILLED)

        # cv2.imshow("diff", diff)
        # cv2.imshow("segmentation_mask", segmentation_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Create an empty mask
        one_hot_masks = [np.zeros_like(mask, dtype=np.uint8) for _ in contours]
        fg_val_count = []
        # Ensure 2 contours or 4 contours exist
        if len(contours) == 2 or len(contours) == 4 or True:
            for i, contour in enumerate(contours):
                cv2.drawContours(one_hot_masks[i], [contour], -1, 255, thickness=cv2.FILLED)
                fg_val_count.append(np.count_nonzero(one_hot_masks[i]))
        else:
            cv2.imshow('Mask', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            cv2.imshow("diff", diff)
            # cv2.imshow("Restored Edges", restored_edges)
            # cv2.imshow("Eroted Edges", eroted_edges)
            cv2.waitKey(0)  # Wait for a key press to close the windows
            cv2.destroyAllWindows()
            raise ValueError(f"Mask has {len(contours)} contours.")
        
        # get biggest and smallest mask
        biggest_mask = one_hot_masks[fg_val_count.index(max(fg_val_count))]
        sec_biggest_mask = one_hot_masks[fg_val_count.index(max(fg_val_count))-2]
        # smallest_mask = one_hot_masks[fg_val_count.index(min(fg_val_count))]

        # set biggest and smallest mask to 1 (optical disc) and 2 (optical cup)
        biggest_mask[biggest_mask == 255] = 120
        sec_biggest_mask[sec_biggest_mask == 255] = 255

        final_seg_mask = np.maximum(biggest_mask, sec_biggest_mask)
        return final_seg_mask

        cv2.imshow("Mask", diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
        # Define HSV range for blue color (adjust as needed)
        lower_blue = np.array([90, 50, 50])  # Lower bound of blue; [90, 80, 80]
        upper_blue = np.array([130, 255, 255])  # Adjust upper limit for the specific dataset; [130, 255, 255]
        # Create a binary mask where blue colors are detected
        mask_bin = cv2.inRange(hsv, lower_blue, upper_blue)

        # cv2.imshow("Colored Mask", cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        # cv2.imshow("HSV", hsv)
        # cv2.imshow("Mask", mask_bin)
        # cv2.waitKey(0)  # Wait for a key press to close the windows
        # cv2.destroyAllWindows()

        contours = []
        i=0
        closed = mask_bin
        while len(contours) != 2 and len(contours) != 4:
            # if i>0:
            #     cv2.imshow("Closed", closed)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            kernel = np.ones((3+i,3+i), np.uint8)  # Small kernel to minimize edge distortion
            # Apply Closing (Dilation + Erosion)
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
            # Find contours from the binary mask
            contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            i+=1
        # # Create an empty mask for filled segmentation
        # segmentation_mask = np.zeros_like(mask)
        # # Fill the largest detected contour
        # if contours:
        #     largest_contour = max(contours, key=cv2.contourArea)
        #     cv2.drawContours(segmentation_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        # cv2.imshow("Colored Mask", cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        # # cv2.imshow("HSV", hsv)
        # cv2.imshow("Mask", mask_bin)
        # cv2.imshow("Closed", closed)
        # cv2.waitKey(0)  # Wait for a key press to close the windows
        # cv2.destroyAllWindows()

        # # Convert img and mask to grayscale
        # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        # mask_sub_img = cv2.subtract(gray_mask, gray_img)
        # mask_sub_img[mask_sub_img >0] = 255

        # # Apply Gaussian blur to smooth the edges
        # blurred = cv2.GaussianBlur(mask_sub_img, (21,21), 0)
        # # Detect edges using Canny
        # edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # # Define kernel
        # kernel = np.ones((3,3), np.uint8)  # Small kernel to minimize edge distortion
        # # Apply Closing (Dilation + Erosion)
        # closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # # Apply Erosion, but only where the original edges existed (masking)
        # mask = (edges > 0).astype(np.uint8)  # Create mask from original edges
        # restored_edges = cv2.bitwise_and(closed, closed, mask=mask)  # Keep only the original edges
        # cv2.imshow("Mask - Image", mask_sub_img)
        # cv2.imshow("Blurred", blurred)
        # cv2.imshow("Edges", edges)
        # cv2.imshow("Closed", closed)
        # cv2.imshow("Restored Edges", restored_edges)
        # cv2.waitKey(0)  # Wait for a key press to close the windows
        # cv2.destroyAllWindows()

        # # Apply Gaussian blur to smooth the edges
        # blurred = cv2.GaussianBlur(gray_mask, (15, 15), 0)
        # # Detect edges using Canny
        # edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        # # Dilate to connect broken edges
        # dilation_kernel = np.ones((9,9), np.uint8)
        # dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)
        # # Erosion to thin the edges again
        # erosion_kernel = np.ones((11,11), np.uint8)
        # eroted_edges = cv2.erode(dilated_edges, erosion_kernel, iterations=1)


        # # Find contours
        # contours, _ = cv2.findContours(restored_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # # Sort contours by area (assuming the largest is the outer circle)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)


        # Create an empty mask
        one_hot_masks = [np.zeros_like(mask, dtype=np.uint8) for _ in contours]
        fg_val_count = []
        # Ensure 2 contours or 4 contours exist
        if len(contours) == 2 or len(contours) == 4:
            for i, contour in enumerate(contours):
                cv2.drawContours(one_hot_masks[i], [contour], -1, 255, thickness=cv2.FILLED)
                fg_val_count.append(np.count_nonzero(one_hot_masks[i]))
        else:
            cv2.imshow('Mask', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            cv2.imshow("Closed", closed)
            # cv2.imshow("Restored Edges", restored_edges)
            # cv2.imshow("Eroted Edges", eroted_edges)
            cv2.waitKey(0)  # Wait for a key press to close the windows
            cv2.destroyAllWindows()
            raise ValueError(f"Mask has {len(contours)} contours.")
        
        # get biggest and smallest mask
        biggest_mask = one_hot_masks[fg_val_count.index(max(fg_val_count))]
        smallest_mask = one_hot_masks[fg_val_count.index(min(fg_val_count))]

        # set biggest and smallest mask to 1 (optical disc) and 2 (optical cup)
        biggest_mask[biggest_mask == 255] = 125
        smallest_mask[smallest_mask == 255] = 255

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
            print(f"Processing image {img_fname}...")
            masks_fnames = [mask for mask in self.raw_mask_fnames if img_fname.replace("prime.jpg", "-").replace("prime.tif", "-") in mask]
            assert len(masks_fnames) > 0, f"Image {img_fname} has {len(masks_fnames)} masks."

            # load the image and convert to RGB
            img = self.load_img_mask(img_fname)

            if ".tif" not in img_fname:
                # save image to tif
                self.save_img_mask_tif(img, img_fname, save_copy="save")
            else:
                # copy image to output path
                self.save_img_mask_tif(img, img_fname, save_copy="copy")

            # load each masks (img + contour) and convert to proper segmentation mask
            for mask_fname in masks_fnames:
                mask = self.load_img_mask(mask_fname)

                seg_mask = self.retrieve_masks(img, mask, mask_fname)
                
                if seg_mask.any():
                    # save one-hot encoded masks
                    self.save_img_mask_tif(seg_mask, mask_fname, save_copy="save")
                else:
                    continue


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
