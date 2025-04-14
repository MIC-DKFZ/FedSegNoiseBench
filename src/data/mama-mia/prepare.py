import argparse
import logging
import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
import shutil
import json


class MamaMia_dataset_processor:
    """
    Class to process and prepare MAMA-MIA dataset for (federated) learning.
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

    def to_nnUNet_raw_dataset(self):
        """
        Convert raw data to nnUNet raw dataset format.
        """
        # get image and mask filenames
        img_fnames = self.get_fnames(
            os.path.join(self.raw_data_path, "MMIA-images"), file_endings=["*.nii.gz"]
        )
        mask_fnames = self.get_fnames(
            os.path.join(self.raw_data_path, "MMIA-segs", self.single_seg_mode),
            file_endings=["*.nii.gz"],
        )
        # only keep images of channels 0000, 0001, 0002 as these are the onyl channels that are present for all images
        img_fnames_kept = [
            x for x in img_fnames if "0000" in x or "0001" in x or "0002" in x
        ]

        # define dataset_id to datasource mapping
        dataset_ids = self.dataset_ids.split()
        datasources = ["DUKE", "ISPY1", "ISPY2", "NACT"]
        dataset_id_to_datasource = {x: y for x, y in zip(dataset_ids, datasources)}

        # create dataset folder
        for i, dataset_id in enumerate(dataset_ids):
            print(
                f"Processing dataset {dataset_id} ({dataset_id_to_datasource[dataset_id]})..."
            )

            # create dataset folder
            dataset_folder = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{dataset_id}_MMIA-{dataset_id_to_datasource[dataset_id]}_{self.single_seg_mode}_flclient{i}",
            )
            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)
            for subfolder in ["imagesTr", "labelsTr"]:
                os.makedirs(os.path.join(dataset_folder, subfolder), exist_ok=True)

            # get images and masks of current dataset/datasource
            curr_img_fnames = [
                x for x in img_fnames_kept if dataset_id_to_datasource[dataset_id] in x
            ]
            curr_mask_fnames = [
                x for x in mask_fnames if dataset_id_to_datasource[dataset_id] in x
            ]

            # copy images
            for img_fname in tqdm(curr_img_fnames, desc="Copying images"):
                new_img_fname = os.path.join(
                    dataset_folder, "imagesTr", os.path.basename(img_fname)
                )
                shutil.copyfile(img_fname, new_img_fname)

            # copy masks
            for mask_fname in tqdm(curr_mask_fnames, desc="Copying masks"):
                new_mask_fname = os.path.join(
                    dataset_folder, "labelsTr", os.path.basename(mask_fname)
                )
                shutil.copyfile(mask_fname, new_mask_fname)

            # create dataset.json file
            dataset_json = {
                "name": "MAMA-MIA",
                "description": f"MAMA-MIA {dataset_id_to_datasource[dataset_id]} dataset",
                "file_ending": ".nii.gz",
                "channel_names": {
                    "0": "MRI baseline",
                    "1": "MRI after CE",
                    "2": "MRI after CE 2",
                },
                "labels": (
                    {
                        "0": "background",
                        "1": "lesion",
                    },
                ),
                "numTraining": len(img_fnames_kept),
            }
            dataset_json_fname = os.path.join(dataset_folder, "dataset.json")
            with open(dataset_json_fname, "w") as f:
                json.dump(dataset_json, f, indent=4)


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
        help="Single segmentation mode (expert, automatic).",
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

    mamamia_ds_processor = MamaMia_dataset_processor(
        args.raw_data_path,
        args.single_seg_mode,
        args.dataset_ids,
        args.nnUNet_raw_data_path,
    )

    mamamia_ds_processor.to_nnUNet_raw_dataset()
