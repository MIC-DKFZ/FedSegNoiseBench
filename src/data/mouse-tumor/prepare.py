import argparse
import logging
import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
import shutil
import json


class MouseTumor_dataset_processor:
    """
    Class to process and prepare MouseTumor dataset for (federated) learning.
    Notes:
    """

    def __init__(
        self,
        raw_data_path: str = None,
        single_seg_mode: str = None,
        dataset_ids: str = None,
        nnUNet_raw_data_path: str = None,
    ):
        # set input args
        self.raw_data_path = raw_data_path
        self.single_seg_mode = single_seg_mode
        self.dataset_ids = dataset_ids
        self.nnUNet_raw_data_path = nnUNet_raw_data_path
        if self.nnUNet_raw_data_path:
            if not os.path.exists(self.nnUNet_raw_data_path):
                os.makedirs(self.nnUNet_raw_data_path)

    def load_img(self, fname, mode: str = None):
        """
        Load image or mask from file.
        """
        pass

    def save_img(self, img: np.ndarray, new_fname: str = None):
        """
        Save image or mask to file.
        """
        os.makedirs(os.path.dirname(new_fname), exist_ok=True)
        pass

    def get_fnames(
        self,
        parent_dir: str = None,
        file_endings: list = ["*.png", "*.jpg", "*.tif", "*.nii.gz"],
    ):
        """
        Get image and mask filenames.
        """
        fnames = []
        for file_ending in file_endings:
            fnames.extend(
                glob.glob(
                    os.path.join(parent_dir, f"**/{file_ending}"),
                    recursive=True,
                )
            )
        return fnames

    def get_singleseg_mask(self, fnames):
        """
        Get mask filenames according to single_seg_mode.
        """
        if self.single_seg_mode == "staple":
            mask_fname = [fname for fname in fnames if "STAPLE" in fname][0]
        elif self.single_seg_mode == "random":
            mask_fnames = [fname for fname in fnames if "Annotator" in fname]
            # select random mask
            mask_fname = mask_fnames[np.random.randint(0, len(mask_fnames))]
        else:
            raise ValueError(f"Unknown single_seg_mode: {self.single_seg_mode}")
        return mask_fname

    def to_nnUNet_raw_dataset(self):
        """
        Generate nnUNet raw dataset with FL splits.
        """
        fnames = self.get_fnames(self.raw_data_path, file_endings=["*.nii.gz"])

        # get sorted unique mouse IDs
        unique_mouse_ids = set()
        for fname in fnames:
            mouse_id = [
                x for x in os.path.basename(fname).split("_") if x.startswith("M")
            ][0]
            unique_mouse_ids.add(mouse_id)
        sorted_unique_mouse_ids = sorted(unique_mouse_ids, key=lambda x: int(x[1:]))

        # define FL client's dataset_id to mouse ID mapping
        dataset_ids = self.dataset_ids.split(" ")
        num_mice_per_flclient = len(sorted_unique_mouse_ids) // len(dataset_ids)
        mice_per_flclient = [
            sorted_unique_mouse_ids[
                i * num_mice_per_flclient : (i + 1) * num_mice_per_flclient
            ]
            for i in range(len(dataset_ids))
        ]
        fl_client_data = {
            ds_id: mice for ds_id, mice in zip(dataset_ids, mice_per_flclient)
        }

        dataset_num_samples = {x: 0 for x in fl_client_data.keys()}
        counter = 0
        # create per FL client dataset
        for idx, (ds_id, mice) in enumerate(fl_client_data.items()):
            # create output_folder for ds_id
            output_folder_img = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{ds_id}_MouseTumor_{self.single_seg_mode}_flclient{idx}",
                "imagesTr",
            )
            output_folder_labels = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{ds_id}_MouseTumor_{self.single_seg_mode}_flclient{idx}",
                "labelsTr",
            )
            os.makedirs(output_folder_img, exist_ok=True)
            os.makedirs(output_folder_labels, exist_ok=True)

            for mouse_idx, mouse_id in tqdm(
                enumerate(mice), desc=f"Processing mice of FL cient {idx}"
            ):
                print(f"Processing mouse {mouse_id}")
                # get all fnames
                mouse_fnames = [
                    fname for fname in fnames if mouse_id in os.path.basename(fname)
                ]
                img_fnames = [x for x in mouse_fnames if "CT" in x]

                # iterate over the given 10 datasets
                for dataset_index in range(10):
                    dataset_img_fnames = [
                        fname
                        for fname in img_fnames
                        if f"Dataset {dataset_index}" in fname
                    ]
                    for img_fname in dataset_img_fnames:
                        # extract mouse_time_id
                        mouse_time_id = (
                            os.path.basename(img_fname)
                            .replace(".nii.gz", "")
                            .replace("CT_", "")
                        )
                        # get mask filenames of current mouse img
                        img_masks_fnames = [
                            fname
                            for fname in mouse_fnames
                            if (
                                ("CT" not in fname)
                                and (mouse_time_id in os.path.basename(fname))
                                and (f"Dataset {dataset_index}" in fname)
                            )
                        ]
                        assert (
                            len(img_masks_fnames) == 4
                        ), f"Expected 4 masks (3 annotator, 1 STAPLE mask), got {len(img_masks_fnames)}"
                        # get single seg mask according to single_seg_mode
                        mask_fname = self.get_singleseg_mask(img_masks_fnames)

                        # put mouse data into nnUNet raw dataset
                        shutil.copy(
                            img_fname,
                            os.path.join(
                                output_folder_img,
                                f"ds{dataset_index}{mouse_time_id.replace('_','')}_0000.nii.gz",
                            ),
                        )
                        shutil.copy(
                            mask_fname,
                            os.path.join(
                                output_folder_labels,
                                f"ds{dataset_index}{mouse_time_id.replace('_','')}.nii.gz",
                            ),
                        )

                        dataset_num_samples[ds_id] += 1

        # generate dataset.json file per dataset
        for idx, dataset_id in enumerate(fl_client_data.keys()):
            # compose dataset.json
            dataset_json = {
                "channel_names": {"0": "CT"},
                "description": "Mouse Tumor dataset",
                "file_ending": ".nii.gz",
                "labels": (
                    {
                        "background": 0,
                        "tumor": 1,
                    }
                ),
                "name": f"MouseTumor_flcient{idx}",
                "numTraining": dataset_num_samples[dataset_id],
                "reference": "Mouse Tumor dataset",
            }
            # save dataset.json
            dataset_json_fname = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{dataset_id}_MouseTumor_{self.single_seg_mode}_flclient{idx}",
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
        "--single_seg_mode",
        type=str,
        default="",
        help="Single segmentation mode (staple, random).",
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
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    nnunet_raw = os.getenv("nnUNet_raw")
    assert nnunet_raw, "Environment variable $nnUNet_raw is not set."

    mousetumor_ds_processor = MouseTumor_dataset_processor(
        args.raw_data_path,
        args.single_seg_mode,
        args.dataset_ids,
        nnunet_raw,
    )

    mousetumor_ds_processor.to_nnUNet_raw_dataset()
