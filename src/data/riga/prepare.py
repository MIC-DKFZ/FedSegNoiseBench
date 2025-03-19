import os
import logging
import argparse
import glob
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import shutil

# import hashlib
# from PIL import Image, ImageChops
import json


class RIGA_dataset_processor:
    """
    Class to process and prepare RIGA dataset for (federated) learning.
    Notes:
    - MagrabiaFemale/image48: Two times mask 1 but no non-annotated image => image removed
    - MagrabiaMale: first mask always called "ImageXY-1" instead of "imageXY-1" => manually renamed
    - BinRushed1-Corrected/image44-4.jpg: inner annotation touches outer annotation => mask removed
    - BinRushed2/image20-3.jpg: inner annotation touches outer annotation => mask removed
    - MagrabiFemale/image26-3.tif: contrast too low => mask removed
    """

    def __init__(
        self,
        raw_data_path: str = None,
        img_segmask_tif_data_path: str = None,
        single_seg_mode: str = None,
        single_seg_data_path: str = None,
        dataset_ids: str = None,
        nnUNet_raw_data_path: str = None,
    ):
        # set input args
        self.raw_data_path = raw_data_path
        self.img_segmask_tif_data_path = img_segmask_tif_data_path
        if self.img_segmask_tif_data_path:
            if not os.path.exists(self.img_segmask_tif_data_path):
                os.makedirs(self.img_segmask_tif_data_path)
        self.single_seg_mode = single_seg_mode
        self.single_seg_data_path = single_seg_data_path
        if self.single_seg_data_path:
            if not os.path.exists(self.single_seg_data_path):
                os.makedirs(self.single_seg_data_path)
        self.dataset_ids = dataset_ids
        self.nnUNet_raw_data_path = nnUNet_raw_data_path
        if self.nnUNet_raw_data_path:
            if not os.path.exists(self.nnUNet_raw_data_path):
                os.makedirs(self.nnUNet_raw_data_path)

        self.raw_img_fnames, self.raw_mask_fnames = [], []

    def get_img_mask_fnames(self):
        """
        Get image and mask filenames from raw data path.
        """
        # get filenames of images and masks from raw data path recursively with file extension .jpg and .tif
        raw_fnames = []
        for file_extension in ["**/*.jpg", "**/*.tif"]:
            raw_fnames.extend(
                glob.glob(
                    os.path.join(self.raw_data_path, file_extension), recursive=True
                )
            )

        # handle special cases, see doc-string of class
        # delete confusion BinRushed1 data and keep BinRushed1-Corrected data
        raw_fnames = [
            fname
            for fname in raw_fnames
            if "BinRushed1" not in fname or "BinRushed1-Corrected" in fname
        ]
        # remove MagrabiaFemale/image48
        raw_fnames = [
            fname for fname in raw_fnames if "MagrabiFemale/image48" not in fname
        ]

        # split raw_fnames into image and mask filenames
        for fname in raw_fnames:
            if "prime" in fname:
                self.raw_img_fnames.append(fname)
            else:
                self.raw_mask_fnames.append(fname)

    def load_img_mask(self, fname: str = None, mode: str = None):
        """
        Load image or mask.
        """
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Image or mask {fname} could not be loaded."
        if mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == "GRAY" and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img / img.max() * 255).astype(np.uint8)  # Normalize to 0-255
        return img

    def retrieve_dense_masks(
        self, img: np.ndarray, mask: np.ndarray, mask_fname: str = None
    ):
        """
        Retrieve dense optical disc and optical cup masks from contours on RGB image.
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        diff_ = cv2.absdiff(gray_mask, gray_img)
        _, diff = cv2.threshold(diff_, 20, 255, cv2.THRESH_BINARY)

        # CCA to filter out small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            diff, connectivity=8
        )
        min_size = 20
        filtered_diff = np.zeros_like(diff)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_diff[labels == i] = 255

        contours = []
        i = 0
        closed = filtered_diff
        while len(contours) != 2 and len(contours) != 4:
            # if too many closing iterations necessary, skip image and write to log file
            if i > 10:
                logging.error(
                    f"Too many closing iterations necessary for {mask_fname}."
                )
                with open("log.txt", "a") as file:
                    file.write(f"{mask_fname}\n")
                return np.array([])
            kernel = np.ones((int(3 + (i / 5)), int(3 + (i / 5))), np.uint8)
            # Apply Closing (Dilation + Erosion) with adapted kernel size
            closed = cv2.dilate(
                closed, kernel + np.ones((1, 1), np.uint8), iterations=2
            )
            closed = cv2.erode(closed, kernel, iterations=1)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            # remove contour that includes background of retina image
            contours = [
                contour for contour in contours if np.array([0, 0]) not in contour
            ]
            i += 1
        contours = sorted(contours, key=cv2.contourArea)

        # Create an empty mask
        dense_masks = [np.zeros_like(mask, dtype=np.uint8) for _ in contours]
        fg_val_count = []
        # Ensure 2 contours or 4 contours exist
        for i, contour in enumerate(contours):
            cv2.drawContours(dense_masks[i], [contour], -1, 255, thickness=cv2.FILLED)
            fg_val_count.append(np.count_nonzero(dense_masks[i]))

        # get biggest and second biggest mask
        # Note: cv2.RETR_CCOMP in cv2.findContours() retrieves inner and outer edge of each contour
        #       so the biggest and second biggest mask are at index 0 and 2
        biggest_mask = dense_masks[fg_val_count.index(max(fg_val_count))]
        sec_biggest_mask = dense_masks[fg_val_count.index(max(fg_val_count)) - 2]

        # set biggest and smallest mask to 1 (optical disc) and 2 (optical cup)
        biggest_mask[biggest_mask == 255] = 120
        sec_biggest_mask[sec_biggest_mask == 255] = 255

        final_seg_mask = np.maximum(biggest_mask, sec_biggest_mask)
        return final_seg_mask

    def save_img_mask_tif(
        self,
        img: np.ndarray = None,
        new_fname: str = None,
        save_copy: str = None,
        old_fname: str = None,
    ):
        """
        Save image or mask as tif.
        """
        # ensure some stuff
        os.makedirs(os.path.dirname(new_fname), exist_ok=True)

        # save or copy image
        if save_copy == "save":
            img = img.astype(np.uint8)
            tiff.imwrite(new_fname, img)
        elif save_copy == "copy":
            shutil.copy(old_fname, new_fname)
        else:
            raise ValueError(f"save_copy must be 'save' or 'copy' but is {save_copy}")

    def mask_contours_to_seg_masks(self):
        """
        Convert masks to dense segmentation masks.
        """
        # get image and mask filenames to self.raw_img_fnames and self.raw_mask_fnames
        self.get_img_mask_fnames()

        for img_fname in tqdm(
            self.raw_img_fnames, desc="Processing images", unit="image"
        ):
            print(f"Processing image {img_fname}...")
            masks_fnames = [
                mask
                for mask in self.raw_mask_fnames
                if img_fname.replace("prime.jpg", "-").replace("prime.tif", "-") in mask
            ]
            assert (
                len(masks_fnames) > 0
            ), f"Image {img_fname} has {len(masks_fnames)} masks."

            # load the image
            img = self.load_img_mask(img_fname, "RGB")
            # compose new filename
            relative_path = os.path.relpath(img_fname, start=self.raw_data_path)
            new_fname = os.path.join(
                self.img_segmask_tif_data_path,
                os.path.splitext(relative_path)[0] + ".tif",
            )
            if ".tif" not in img_fname:
                # save image to tif
                self.save_img_mask_tif(img, new_fname, save_copy="save")
            else:
                # copy image to output path, if image is already in tif format
                self.save_img_mask_tif(
                    img, new_fname, save_copy="copy", old_fname=img_fname
                )

            # load each masks (img + contour) and convert to proper segmentation mask
            for mask_fname in masks_fnames:
                mask = self.load_img_mask(mask_fname, "RGB")

                seg_mask = self.retrieve_dense_masks(img, mask, mask_fname)

                # compose new filename
                relative_path = os.path.relpath(mask_fname, start=self.raw_data_path)
                new_fname = os.path.join(
                    self.img_segmask_tif_data_path,
                    os.path.splitext(relative_path)[0] + ".tif",
                )
                if seg_mask.any():
                    # save dense masks
                    self.save_img_mask_tif(seg_mask, new_fname, save_copy="save")
                else:
                    continue

    def generate_consensus_random_rater_masks(
        self, annotator_majority_threshold: float = 0.5
    ):
        """
        Generate consensus and random rater masks.
        """
        # load segmentation mask fnames from self.img_segmask_tif_data_path
        fnames = glob.glob(
            os.path.join(self.img_segmask_tif_data_path, "**/*.tif"), recursive=True
        )
        seg_mask_fnames, img_fnames = [], []
        for fname in fnames:
            (
                img_fnames.append(fname)
                if "prime" in fname
                else seg_mask_fnames.append(fname)
            )

        # iterate over imgs and generate mask according to self.single_seg_mode
        for img_fname in tqdm(
            img_fnames, desc="Generate seg mask for image", unit="image"
        ):
            # get seg_masks for current img
            imgs_seg_mask_fnames = [
                seg_mask
                for seg_mask in seg_mask_fnames
                if img_fname.replace("prime.tif", "-") in seg_mask
            ]
            assert (
                len(imgs_seg_mask_fnames) > 0
            ), f"Image {img_fname} has {len(imgs_seg_mask_fnames)} masks."

            # load seg masks to np array (#annotator_masks, H, W)
            seg_masks = np.array(
                [self.load_img_mask(x, "GRAY") for x in imgs_seg_mask_fnames]
            )

            # Create binary masks for each structure
            structure1 = (seg_masks == 120).astype(np.uint8)
            structure2 = (seg_masks == 255).astype(np.uint8)

            final_mask = np.zeros_like(seg_masks[0], dtype=np.uint8)
            # load each masks (img + contour) and convert to proper segmentation mask
            if self.single_seg_mode == "union":
                # load all masks and take union
                union_mask = final_mask
                for seg_mask in seg_masks:
                    union_mask = np.maximum(union_mask, seg_mask)
                final_mask = union_mask
            elif self.single_seg_mode == "annotator_majority":
                # Compute majority vote
                majority_threshold = int(len(seg_masks) * annotator_majority_threshold)
                consensus_structure1 = (
                    np.sum(structure1, axis=0) >= majority_threshold
                ) * 120
                consensus_structure2 = (
                    np.sum(structure2, axis=0) >= majority_threshold
                ) * 255
                # Merge structures into one mask
                final_mask = np.maximum(consensus_structure1, consensus_structure2)

            elif self.single_seg_mode == "random":
                # load all masks and take random mask
                random_mask = seg_masks[np.random.randint(0, len(seg_masks))]
                final_mask = random_mask
            else:
                raise ValueError(
                    f"self.single_seg_mode must be 'union', 'annotator_majority', 'random' but is {self.single_seg_mode}"
                )

            # save to self.single_seg_data_path
            # compose new filename
            relative_path = os.path.relpath(
                imgs_seg_mask_fnames[0], start=self.img_segmask_tif_data_path
            )
            new_fname = os.path.join(
                self.single_seg_data_path,
                os.path.splitext(relative_path)[0].rsplit("-", 1)[0] + "mask.tif",
            )
            self.save_img_mask_tif(final_mask, new_fname, save_copy="save")

    def get_dataset_id_name(self, dataset_to_id, img_fname):
        """
        Retrieve dataset_id based on available folder depth in dataset_to_id
        """
        base1 = os.path.basename(os.path.dirname(img_fname))  # One level up
        base2 = os.path.basename(
            os.path.dirname(os.path.dirname(img_fname))
        )  # Two levels up

        if base2 in dataset_to_id:  # Prefer deeper structure if available
            return dataset_to_id[base2], base2
        elif base1 in dataset_to_id:
            return dataset_to_id[base1], base1
        else:
            raise KeyError(f"No matching dataset_id found for {img_fname}")

    def to_nnUNet_raw_dataset(self, consecutive_label_order: bool = False):
        """
        Convert RIGA dataset to nnUNet raw dataset format.
        """
        # get images and masks fnames
        img_fnames = glob.glob(
            os.path.join(self.img_segmask_tif_data_path, "**/*prime.tif"),
            recursive=True,
        )
        seg_mask_fnames = glob.glob(
            os.path.join(self.single_seg_data_path, "**/*mask.tif"), recursive=True
        )

        # define dataset to dataset_id mapping
        dataset_to_id = {
            "BinRushed": self.dataset_ids.split()[0],
            "Magrabia": self.dataset_ids.split()[1],
            "MESSIDOR": self.dataset_ids.split()[2],
        }

        dataset_num_samples = {
            "BinRushed": 0,
            "Magrabia": 0,
            "MESSIDOR": 0,
        }
        # split img into separate R, G, B channels and save to nnUNet raw dataset format
        for idx, img_fname in tqdm(
            enumerate(img_fnames),
            total=len(img_fnames),
            desc="Save images to nnUNet raw dataset",
            unit="image",
        ):
            # load image
            img = self.load_img_mask(img_fname, "RGB")
            # split image into R, G, B channels
            r, g, b = cv2.split(img)
            # save channel-separated image
            for channel_img, channel_name in zip([r, g, b], ["0000", "0001", "0002"]):
                # compose new filename
                dataset_id, dataset_name = self.get_dataset_id_name(
                    dataset_to_id, img_fname
                )
                new_fname = os.path.join(
                    self.nnUNet_raw_data_path,
                    f"Dataset{dataset_id}_RIGA-{dataset_name}_{self.single_seg_mode}",
                    "imagesTr",
                    f"RIGA{dataset_name}_{idx:04d}_{channel_name}.tif",
                )
                self.save_img_mask_tif(channel_img, new_fname, save_copy="save")
                dataset_num_samples[dataset_name] += 1

            # corresponding seg mask
            seg_mask_fname = img_fname.replace(
                    "img_segmask_tif", f"single_seg_{self.single_seg_mode.replace('_','')}"
                ).replace("prime.tif", "mask.tif")
            new_seg_mask_fname = new_fname.replace("imagesTr", "labelsTr").replace(
                    f"_{channel_name}.tif", ".tif"
                )
            if consecutive_label_order:
                # copy img's corresponding seg mask to nnUNet raw dataset format
                self.save_img_mask_tif(
                    old_fname=seg_mask_fname, new_fname=new_seg_mask_fname, save_copy="copy"
                )
            else:
                mask = self.load_img_mask(seg_mask_fname, "GRAY")
                mask[(mask > 0) & (mask < 255)] = 1
                mask[mask == 255] = 2
                self.save_img_mask_tif(mask, new_seg_mask_fname, save_copy="save")

        # generate dataset.json file per dataset
        for dataset_name, dataset_id in dataset_to_id.items():
            # compose dataset.json
            dataset_json = {
                "channel_names": {"0": "R", "1": "G", "2": "B"},
                "description": "RIGA dataset",
                "file_ending": ".tif",
                "labels": ({"background": 0, "disc": 1, "cup": 2}),
                "name": f"RIGA-{dataset_name}",
                "numTraining": int(dataset_num_samples[dataset_name] / 3),
                "reference": "RIGA",
            }
            # save dataset.json
            dataset_json_fname = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{dataset_id}_RIGA-{dataset_name}_{self.single_seg_mode}",
                "dataset.json",
            )
            with open(dataset_json_fname, "w") as json_file:
                json.dump(dataset_json, json_file, indent=4)


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
        "--img_segmask_tif_data_path",
        type=str,
        default="",
        help="Path to dense segmentation masks of RIGA dataset.",
    )
    parser.add_argument(
        "--single_seg_mode",
        type=str,
        default="",
        help="Mode to generate single segmentation mask per retina image."
        "Options: 'union', 'annotator_majority', 'random'.",
    )
    parser.add_argument(
        "--single_seg_data_path",
        type=str,
        default="",
        help="Path to single, dense segmentation mask per retina image.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--dataset_ids",
        type=str,
        default="",
        help="Raw nnUNet Dataset ID to generate from raw data.",
    )
    parser.add_argument(
        "--nnUNet_raw_data_path",
        type=str,
        default="",
        help="Path to nnUNet raw dataset.",
    )

    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    riga_ds_processor = RIGA_dataset_processor(
        args.raw_data_path,
        args.img_segmask_tif_data_path,
        args.single_seg_mode,
        args.single_seg_data_path,
        args.dataset_ids,
        args.nnUNet_raw_data_path,
    )

    # Step 1: Convert masks to dense segmentation masks
    # riga_ds_processor.mask_contours_to_seg_masks()

    # Step 2: Generate consensus and random-rater masks
    # riga_ds_processor.generate_consensus_random_rater_masks()

    # Step 3: To nnUNet_raw dataset format
    riga_ds_processor.to_nnUNet_raw_dataset(consecutive_label_order=False)
