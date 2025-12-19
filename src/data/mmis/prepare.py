import argparse
import logging
import os
import glob
import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
import shutil
import json


class MMIS_dataset_processor:
    """
    Class to process and prepare MMIS dataset for (federated) learning.
    Notes:
    - MMIS (Multi-Modal brain MRI Segmentation) dataset contains multi-annotator labels
    - Input data format: .h5 files containing t1, t1c, t2 MRI modalities and multiple annotator labels
    - Modalities: ['t1', 't1c', 't2']
    - Labels: ['label_a1', 'label_a2', 'label_a3', 'label_a4'] (4 annotators)
    """

    def __init__(
        self,
        raw_data_path: str = None,
        nifti_output_path: str = None,
        single_seg_mode: str = None,
        dataset_ids: str = None,
        nnUNet_raw_data_path: str = None,
    ):
        # set input args
        self.raw_data_path = raw_data_path
        self.nifti_output_path = nifti_output_path
        if self.nifti_output_path:
            if not os.path.exists(self.nifti_output_path):
                os.makedirs(self.nifti_output_path)
        self.single_seg_mode = single_seg_mode
        self.dataset_ids = dataset_ids
        self.nnUNet_raw_data_path = nnUNet_raw_data_path
        if self.nnUNet_raw_data_path:
            if not os.path.exists(self.nnUNet_raw_data_path):
                os.makedirs(self.nnUNet_raw_data_path)

        # MMIS dataset specific configuration
        self.modalities = ["t1", "t1c", "t2"]
        self.label_keys = ["label_a1", "label_a2", "label_a3", "label_a4"]

    def get_h5_fnames(self):
        """
        Get all .h5 filenames from raw data path.
        """
        h5_fnames = glob.glob(
            os.path.join(self.raw_data_path, "**/*.h5"), recursive=True
        )
        return sorted(h5_fnames)

    def h5_to_nifti(self, h5_path: str, output_dir: str):
        """
        Convert a single .h5 file to NIfTI format.
        Saves MRI modalities (t1, t1c, t2) and annotator labels (label_a1-a4) as separate .nii.gz files.

        Args:
            h5_path: Path to input .h5 file
            output_dir: Directory to save NIfTI files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Read voxel spacing and construct affine matrix
        with h5py.File(h5_path, "r") as f:
            spacing = f["voxel_spacing"][:]  # [z, y, x] spacing

            # NIfTI affine matrix (identity + spacing)
            affine = np.eye(4)
            affine[0, 0] = spacing[2]  # x
            affine[1, 1] = spacing[1]  # y
            affine[2, 2] = spacing[0]  # z

            # Save MRI modalities
            for mod in self.modalities:
                if mod in f:
                    data = f[mod][:]  # Already in [z, y, x] NIfTI order
                    nii_img = nib.Nifti1Image(data.astype(np.float32), affine)
                    nib.save(nii_img, os.path.join(output_dir, f"{mod}.nii.gz"))

            # Save labels
            for lbl in self.label_keys:
                if lbl in f:
                    data = f[lbl][:]  # uint8 labels, already [z, y, x]
                    nii_img = nib.Nifti1Image(data.astype(np.uint8), affine)
                    nib.save(nii_img, os.path.join(output_dir, f"{lbl}.nii.gz"))

    def convert_all_h5_to_nifti(self):
        """
        Convert all .h5 files in raw_data_path to NIfTI format.
        Creates one subdirectory per .h5 file containing all modalities and labels.
        """
        h5_fnames = self.get_h5_fnames()

        if len(h5_fnames) == 0:
            logging.warning(f"No .h5 files found in {self.raw_data_path}")
            return

        logging.info(f"Found {len(h5_fnames)} .h5 files to convert")

        for h5_fname in tqdm(h5_fnames, desc="Converting .h5 to NIfTI"):
            # Create output directory for this sample
            sample_name = os.path.splitext(os.path.basename(h5_fname))[0]
            output_dir = os.path.join(self.nifti_output_path, sample_name)

            # Convert h5 to nifti
            self.h5_to_nifti(h5_fname, output_dir)

        logging.info(f"All NIfTI files saved to: {self.nifti_output_path}")

    def get_nifti_fnames(self, parent_dir: str = None):
        """
        Get all NIfTI filenames from a directory.
        """
        nifti_fnames = glob.glob(
            os.path.join(parent_dir, "**/*.nii.gz"), recursive=True
        )
        return sorted(nifti_fnames)

    def generate_all_singlerater_masks(self):
        """
        Generate consensus masks from multiple annotator labels.
        Supports different consensus strategies: 'majority', 'random', 'annotator1', etc.
        """
        nifti_dirs = sorted(
            [
                d
                for d in glob.glob(os.path.join(self.nifti_output_path, "*"))
                if os.path.isdir(d)
            ]
        )

        if len(nifti_dirs) == 0:
            logging.warning(f"No directories found in {self.nifti_output_path}")
            return

        logging.info(f"Found {len(nifti_dirs)} sample directories")

        for sample_dir in tqdm(nifti_dirs, desc="Generating single-rater masks"):
            self.generate_singlerater_mask(sample_dir)

    def generate_singlerater_mask(self, sample_dir: str):
        """
        Generate single-rater mask for a single sample directory.

        Args:
            sample_dir: Directory containing label NIfTI files from multiple annotators
        """
        # Load all annotator labels
        labels = {}
        for lbl in self.label_keys:
            lbl_path = os.path.join(sample_dir, f"{lbl}.nii.gz")
            if os.path.exists(lbl_path):
                labels[lbl] = nib.load(lbl_path).get_fdata().astype(np.uint8)

        if not labels:
            logging.warning(f"No labels found in {sample_dir}")
            return

        # Stack labels: shape (n_annotators, z, y, x)
        label_stack = np.stack(list(labels.values()), axis=0)

        # Generate single-rater mask based on mode
        if self.single_seg_mode == "majority":
            singlerater_mask = np.round(np.mean(label_stack, axis=0)).astype(np.uint8)
            fname_suffix = self.single_seg_mode
        elif self.single_seg_mode == "random":
            idx = np.random.randint(0, len(labels))
            singlerater_mask = list(labels.values())[idx]
            fname_suffix = self.single_seg_mode
        elif self.single_seg_mode == "rater":
            # randomly select one rater with equal probability
            idx = np.random.randint(0, len(labels))
            singlerater_mask = list(labels.values())[idx]
            fname_suffix = self.single_seg_mode + str(idx + 1)
        else:
            logging.warning(f"Unknown single-rater mode: {self.single_seg_mode}")
            return

        # Save single-rater mask
        affine = nib.load(
            os.path.join(sample_dir, f"{self.label_keys[0]}.nii.gz")
        ).affine
        nii_img = nib.Nifti1Image(singlerater_mask, affine)
        nib.save(
            nii_img,
            os.path.join(sample_dir, f"singlerater_label_{fname_suffix}.nii.gz"),
        )

    def to_clients_nnUNet_raw_dataset(self):
        """
        Convert processed NIfTI data to nnUNet raw dataset format.
        Creates separate datasets for federated learning clients if needed.
        """
        # identify client-sample assignment based on selected rater for label singlerater_label_rater<rater-id>.nii.gz per sample
        all_datasamples = sorted(
            [
                d
                for d in glob.glob(os.path.join(self.nifti_output_path, "*"))
                if os.path.isdir(d)
            ]
        )
        client_2_sample_assignment = {}
        for sample_dir in all_datasamples:
            sample_name = os.path.basename(sample_dir)
            # determine which rater was used for this sample
            label_files = glob.glob(
                os.path.join(sample_dir, "singlerater_label_rater*.nii.gz")
            )
            if not label_files:
                logging.warning(
                    f"No single-rater label found for sample {sample_name}, skipping."
                )
                continue
            label_file = label_files[0]
            rater_id = label_file.split("singlerater_label_rater")[-1].split(".nii.gz")[
                0
            ]
            client_id = int(rater_id) - 1  # client_id starts from 0

            if client_id not in client_2_sample_assignment:
                client_2_sample_assignment[client_id] = []
            client_2_sample_assignment[client_id].append(sample_dir)

        # create per dataset_id nnUNet directories
        dataset_id_list = self.dataset_ids.split()
        for client_id, dataset_id in enumerate(dataset_id_list):
            client_raw_data_dir = os.path.join(
                self.nnUNet_raw_data_path,
                f"Dataset{dataset_id}_MMIS_{self.single_seg_mode}_client{client_id}",
            )
            imagesTr_dir = os.path.join(client_raw_data_dir, "imagesTr")
            labelsTr_dir = os.path.join(client_raw_data_dir, "labelsTr")
            os.makedirs(imagesTr_dir, exist_ok=True)
            os.makedirs(labelsTr_dir, exist_ok=True)

            # copy NIfTI files to nnUNet format
            nifti_dirs = sorted(
                [
                    d
                    for d in glob.glob(os.path.join(self.nifti_output_path, "*"))
                    if os.path.isdir(d)
                ]
            )
            nifti_dirs_of_current_client = [
                d
                for d in nifti_dirs
                if client_id in client_2_sample_assignment
                and d in client_2_sample_assignment[client_id]
            ]
            for sample_dir in tqdm(
                nifti_dirs_of_current_client,
                desc=f"Preparing client {client_id} nnUNet dataset",
            ):
                sample_name = os.path.basename(sample_dir)

                # Copy modalities
                for mod in self.modalities:
                    src_path = os.path.join(sample_dir, f"{mod}.nii.gz")
                    if os.path.exists(src_path):
                        dst_fname = (
                            f"{sample_name}_0000.nii.gz"
                            if mod == "t1"
                            else f"{sample_name}_000{self.modalities.index(mod)}.nii.gz"
                        )
                        dst_path = os.path.join(imagesTr_dir, dst_fname)
                        shutil.copy(src_path, dst_path)

                # Copy single-rater label
                if (
                    client_id in client_2_sample_assignment
                    and sample_dir in client_2_sample_assignment[client_id]
                ):
                    singlerater_label_fname = (
                        f"singlerater_label_{self.single_seg_mode}.nii.gz"
                        if self.single_seg_mode == "majority"
                        else f"singlerater_label_{self.single_seg_mode}{client_id + 1}.nii.gz"
                    )
                    label_files = glob.glob(
                        os.path.join(sample_dir, singlerater_label_fname)
                    )
                    if label_files:
                        src_path = label_files[0]
                        dst_path = os.path.join(labelsTr_dir, f"{sample_name}.nii.gz")
                        shutil.copy(src_path, dst_path)

            # create dataset.json file
            dataset_json = {
                "name": f"MMIS_{self.single_seg_mode}_client{client_id}",
                "description": f"MMIS dataset for client {client_id} using {self.single_seg_mode} single-rater labels",
                "file_ending": ".nii.gz",
                "channel_names": {"0": "t1", "1": "t1c", "2": "t2"},
                "labels": {"background": "0", "gtv_nasopharyngeal_carcinoma": "1"},
                "numTraining": len(nifti_dirs_of_current_client),
            }

            with open(os.path.join(client_raw_data_dir, "dataset.json"), "w") as f:
                json.dump(dataset_json, f, indent=4)


if __name__ == "__main__":
    # set cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="",
        help="Path to raw MMIS dataset containing .h5 files.",
    )
    parser.add_argument(
        "--nifti_output_path",
        type=str,
        default="",
        help="Path to save converted NIfTI files.",
    )
    parser.add_argument(
        "--single_seg_mode",
        type=str,
        default="",
        help="Single segmentation mode: majority, random, rater.",
    )
    parser.add_argument(
        "--dataset_ids",
        type=str,
        default="",
        help="Dataset IDs for nnUNet (space-separated, e.g., '401 402 403').",
    )
    parser.add_argument(
        "--nnUNet_raw_data_path",
        type=str,
        default="",
        help="Path to nnUNet raw data directory.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["h5_to_nifti", "generate_single_rater_masks", "to_clients_nnunet_ds"],
        default="h5_to_nifti",
        help="Action to perform: h5_to_nifti, generate_single_rater_masks, or to_clients_nnunet_ds.",
    )
    args = parser.parse_args()

    # set logging
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # initialize dataset processor
    processor = MMIS_dataset_processor(
        raw_data_path=args.raw_data_path,
        nifti_output_path=args.nifti_output_path,
        single_seg_mode=args.single_seg_mode,
        dataset_ids=args.dataset_ids,
        nnUNet_raw_data_path=args.nnUNet_raw_data_path,
    )

    # execute action
    if args.action == "h5_to_nifti":
        logging.info("Converting .h5 files to NIfTI format...")
        processor.convert_all_h5_to_nifti()
    elif args.action == "generate_single_rater_masks":
        logging.info("Generating single-rater masks...")
        processor.generate_all_singlerater_masks()
    elif args.action == "to_clients_nnunet_ds":
        logging.info("Converting to nnUNet raw dataset format...")
        processor.to_clients_nnUNet_raw_dataset()
    else:
        logging.error(f"Unknown action: {args.action}")
