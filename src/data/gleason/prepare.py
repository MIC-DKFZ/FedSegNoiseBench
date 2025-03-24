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
import shutil
import json


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
        staple_raw_data_path: str = None,
        dataset_ids: str = None,
        pathoimgslides_per_flclient: str = None,
        nnUNet_raw_data_path: str = None,
    ):
        # set input args
        self.raw_data_path = raw_data_path
        self.single_seg_data_path = single_seg_data_path
        if self.single_seg_data_path:
            if not os.path.exists(self.single_seg_data_path):
                os.makedirs(self.single_seg_data_path)
        self.single_seg_mode = single_seg_mode
        self.staple_raw_data_path = staple_raw_data_path
        self.dataset_ids = dataset_ids
        self.pathoimgslides_per_flclient = pathoimgslides_per_flclient
        self.nnUNet_raw_data_path = nnUNet_raw_data_path
        if self.nnUNet_raw_data_path:
            if not os.path.exists(self.nnUNet_raw_data_path):
                os.makedirs(self.nnUNet_raw_data_path)

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

    def get_fnames(
        self,
        parent_dir: str = None,
        filter_img_masks: bool = True,
        sub_dirs: str = "**/",
    ):
        """
        Get image and mask filenames.
        """
        fnames = []
        for file_ending in ["*.png", "*.jpg", "*.tif"]:
            fnames.extend(
                glob.glob(
                    os.path.join(parent_dir, f"**/{file_ending}"),
                    recursive=True,
                )
            )
        if filter_img_masks:
            # split into img and masks
            img_fnames = [fname for fname in fnames if ("Train_imgs" in fname)]
            mask_fnames = [fname for fname in fnames if "Maps" in fname]
            return img_fnames, mask_fnames
        return fnames

    def generate_consensus_masks(self):
        # get fnames
        img_fnames, mask_fnames = self.get_fnames(
            self.raw_data_path, filter_img_masks=True
        )

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

    def staple_to_consensus_masks(self):
        """
        Convert STAPLE masks to consensus masks.
        """
        # get fnames
        _, mask_fnames = self.get_fnames(
            self.staple_raw_data_path, filter_img_masks=True
        )

        for mask_fname in tqdm(mask_fnames, desc="Loading STAPLE masks"):
            # load mask
            mask = self.load_img(mask_fname)
            # save mask as .tif
            new_mask_fname = os.path.join(
                self.single_seg_data_path,
                os.path.basename(mask_fname).replace(".png", ".tif"),
            )
            self.save_img(mask, new_mask_fname)

    def to_nnUNet_raw_dataset(self):
        """
        Generate nnUNet raw dataset with FL splits.
        """
        # define FL client's dataset_id to histopatho image slide mapping
        dataset_ids = self.dataset_ids.split(" ")
        _pathoimgslides_per_flclient = [
            int(x) for x in self.pathoimgslides_per_flclient.split(",")
        ]
        num_slides_per_flclient = len(_pathoimgslides_per_flclient) // len(dataset_ids)
        pathoimgslides_per_flclient = [
            _pathoimgslides_per_flclient[
                i * num_slides_per_flclient : (i + 1) * num_slides_per_flclient
            ]
            for i in range(len(dataset_ids))
        ]
        fl_client_data = {
            ds_id: slides_flc
            for ds_id, slides_flc in zip(dataset_ids, pathoimgslides_per_flclient)
        }

        # get fnames
        if self.single_seg_mode == "staple":
            mask_fnames = self.get_fnames(
                self.single_seg_data_path, filter_img_masks=False
            )
            fnames = self.get_fnames(
                self.single_seg_data_path.replace("staple", "random"),
                filter_img_masks=False,
            )
            img_fnames = [
                fname for fname in fnames if "classimg_nonconvex" not in fname
            ]
        else:
            fnames = self.get_fnames(
                self.single_seg_data_path, sub_dirs="", filter_img_masks=False
            )
            img_fnames = [
                fname for fname in fnames if "classimg_nonconvex" not in fname
            ]
            mask_fnames = [fname for fname in fnames if fname not in img_fnames]

        dataset_num_samples = {x: 0 for x in fl_client_data.keys()}
        # split data
        for idx, (ds_id, slides_flc) in enumerate(fl_client_data.items()):
            # create output_folder for ds_id
            output_folder = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{ds_id}_Gleason2019_{self.single_seg_mode}_flclient{idx}",
            )
            os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)

            # get img_slides of current FL client
            img_slide_fnames, mask_slides_fnames = [], []
            for slide_idx_flc in slides_flc:
                img_slide_fnames.extend(
                    [
                        x
                        for x in img_fnames
                        if (
                            (f"slide{slide_idx_flc:03d}" in x) and ("classimg" not in x)
                        )
                    ]
                )
                mask_slides_fnames.extend(
                    [
                        x
                        for x in mask_fnames
                        if ((f"slide{slide_idx_flc:03d}" in x) and ("classimg" in x))
                    ]
                )
                if self.single_seg_mode == "staple":  # and len(mask_slides_fnames) ==
                    mask_slides_fnames.extend(
                        [x for x in mask_fnames if f"s{slide_idx_flc:03d}" in x]
                    )

            for img_idx, img_slide_fname in tqdm(
                enumerate(img_slide_fnames), desc=f"Processing images of FL cient {idx}"
            ):
                print(f"Processing img {img_slide_fname}")
                # image
                img = self.load_img(img_slide_fname, "RGB")
                # split image into R, G, B channels
                r, g, b = cv2.split(img)
                # save channel-separated image
                for channel_img, channel_name in zip(
                    [r, g, b], ["0000", "0001", "0002"]
                ):
                    # compose new filename
                    new_fname = os.path.join(
                        output_folder,
                        "imagesTr",
                        f"Gleason-{os.path.basename(img_slide_fname).replace('_','').replace('.tif','')}_{img_idx:04d}_{channel_name}.tif",
                    )
                    self.save_img(channel_img, new_fname)
                dataset_num_samples[ds_id] += 1

                # corresponding seg mask
                try:
                    current_segmask_fname = [
                        x
                        for x in mask_slides_fnames
                        if (
                            (
                                os.path.basename(img_slide_fname)
                                .replace("slide", "s")
                                .replace("core", "c")
                                .replace(".tif", "")
                                in x
                            )
                            or (
                                os.path.basename(img_slide_fname).replace(".tif", "")
                                in x
                            )
                        )
                    ][0]
                except IndexError:
                    raise ValueError(f"No seg mask found for {img_slide_fname}")
                new_seg_mask_fname = new_fname.replace("imagesTr", "labelsTr").replace(
                    f"_{channel_name}.tif", ".tif"
                )
                # copy seg mask
                shutil.copy(current_segmask_fname, new_seg_mask_fname)

        # generate dataset.json file per dataset
        for idx, dataset_id in enumerate(fl_client_data.keys()):
            # compose dataset.json
            dataset_json = {
                "channel_names": {"0": "R", "1": "G", "2": "B"},
                "description": "Gleason2019 dataset",
                "file_ending": ".tif",
                "labels": (
                    {
                        "background": 0,
                        "gleason_label1": 1,
                        "gleason_label2": 2,
                        "gleason_label3": 3,
                        "gleason_label4": 4,
                        "gleason_label5": 5,
                        "gleason_label6": 6,
                    }
                ),
                "name": f"Gleason2019_flcient{idx}",
                "numTraining": dataset_num_samples[dataset_id],
                "reference": "Gleason2019",
            }
            # save dataset.json
            dataset_json_fname = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{dataset_id}_Gleason2019_{self.single_seg_mode}_flclient{idx}",
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
        "--staple_raw_data_path",
        type=str,
        default="",
        help="Path to STAPLE consensus masks.",
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
        "--pathoimgslides_per_flclient",
        type=str,
        default="",
        help="Number of histopathology image slides per federated learning client.",
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

    gleason_ds_processor = Gleason_dataset_processor(
        args.raw_data_path,
        args.single_seg_data_path,
        args.single_seg_mode,
        args.staple_raw_data_path,
        args.dataset_ids,
        args.pathoimgslides_per_flclient,
        args.nnUNet_raw_data_path,
    )

    # Step1: Multi-rater masks -> consensus masks
    # gleason_ds_processor.generate_consensus_masks()

    # Step1.1: Given STAPLE consensus masks to consensus masks
    # gleason_ds_processor.staple_to_consensus_masks()

    # Step2: Generate nnUNet datasets with FL splits
    gleason_ds_processor.to_nnUNet_raw_dataset()
