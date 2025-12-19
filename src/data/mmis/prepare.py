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
        self.modalities = ['t1', 't1c', 't2']
        self.label_keys = ['label_a1', 'label_a2', 'label_a3', 'label_a4']

    def get_h5_fnames(self):
        """
        Get all .h5 filenames from raw data path.
        """
        h5_fnames = glob.glob(
            os.path.join(self.raw_data_path, "**/*.h5"),
            recursive=True
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
        with h5py.File(h5_path, 'r') as f:
            spacing = f['voxel_spacing'][:]  # [z, y, x] spacing
            
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
            os.path.join(parent_dir, "**/*.nii.gz"),
            recursive=True
        )
        return sorted(nifti_fnames)

    def generate_all_singlerater_masks(self):
        """
        Generate consensus masks from multiple annotator labels.
        Supports different consensus strategies: 'majority', 'random', 'annotator1', etc.
        """
        nifti_dirs = sorted([d for d in glob.glob(os.path.join(self.nifti_output_path, "*")) if os.path.isdir(d)])
        
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
            fname_suffix =  self.single_seg_mode + str(idx + 1)
        else:
            logging.warning(f"Unknown single-rater mode: {self.single_seg_mode}")
            return
        
        # Save single-rater mask
        affine = nib.load(os.path.join(sample_dir, f"{self.label_keys[0]}.nii.gz")).affine
        nii_img = nib.Nifti1Image(singlerater_mask, affine)
        nib.save(nii_img, os.path.join(sample_dir, f"singlerater_label_{fname_suffix}.nii.gz"))
        

    def to_nnUNet_raw_dataset(self):
        """
        Convert processed NIfTI data to nnUNet raw dataset format.
        Creates separate datasets for federated learning clients if needed.
        """
        # TODO: Implement nnUNet dataset creation
        # Similar structure to MamaMia dataset's to_nnUNet_raw_dataset method
        raise NotImplementedError("nnUNet dataset conversion not yet implemented")


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
        choices=["h5_to_nifti", "generate_single_rater_masks", "to_nnunet"],
        default="h5_to_nifti",
        help="Action to perform: h5_to_nifti, generate_single_rater_masks, or to_nnunet.",
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
    elif args.action == "to_nnunet":
        logging.info("Converting to nnUNet raw dataset format...")
        processor.to_nnUNet_raw_dataset()
    else:
        logging.error(f"Unknown action: {args.action}")
