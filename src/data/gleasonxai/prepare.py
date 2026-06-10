import argparse
import json
import os
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import SimpleITK as sitk


class GleasonXAIDataPreparer:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = Path(raw_data_dir)

        self.explanation2gleasongrade_mapping = (
            self._load_gleason_explanation_mappings()
        )

        self.gleasongrade2seglabel_mapping = {
            "0": 0,
            "3": 1,
            "4": 2,
            "5": 3,
        }

        # get list of all raw image files
        # TissueArray
        self.tissue_array_raw_imgs = glob(
            str(
                self.raw_data_dir
                / "27301845"
                / "tissuearray_com_data"
                / "tissuemicroarray_com_data"
                / "*.jpg"
            )
        )
        # Harvard Dataverse
        self.harvard_dataverse_raw_imgs = glob(
            str(self.raw_data_dir / "dataverse_files" / "*" / "*.jpg")
        )
        self.harvard_dataverse_raw_imgs = [
            img for img in self.harvard_dataverse_raw_imgs if "mask" not in img
        ]
        # Gleason2019
        self.gleason2019_raw_imgs = glob(
            str(
                self.raw_data_dir
                / ".."
                / "Gleason2019"
                / "raw_grandchallenge"
                / "Train_imgs"
                / "*.jpg"
            )
        )
        self.gleason2019_raw_imgs += glob(
            str(
                self.raw_data_dir
                / ".."
                / "Gleason2019"
                / "raw_grandchallenge"
                / "Test_imgs"
                / "*.jpg"
            )
        )

    def _load_gleason_explanation_mappings(self):
        gleasongrade2explanation_mapping = json.load(
            open(self.raw_data_dir / "27301845" / "label_remapping.json")
        )["hierarchy"]
        explanation2gleasongrade_mapping = {}
        for grade, categories in gleasongrade2explanation_mapping.items():
            for category, explanations in categories.items():
                # Map category name to grade
                explanation2gleasongrade_mapping[category] = int(grade)
                # Map each detailed explanation to grade
                for expl in explanations:
                    explanation2gleasongrade_mapping[expl] = int(grade)
        return explanation2gleasongrade_mapping

    def init_generate_labels_task(self, label_mode, output_dir):
        self.label_mode = "all_raters" if label_mode == "all" else label_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # also create subdirectories for each sub dataset
        for datasource in ["tissue_array", "harvard_dataverse", "gleason2019"]:
            (self.output_dir / datasource / self.label_mode).mkdir(
                parents=True, exist_ok=True
            )

    def init_convert_images_task(self, output_dir, convert_images_mode="jpg_to_png", labels_dir=None, label_mode=None):
        self.convert_img_mode = convert_images_mode
        self.convert_output_dir = Path(output_dir)
        self.convert_output_dir.mkdir(parents=True, exist_ok=True)
        self.convert_labels_dir = Path(labels_dir) if labels_dir else None
        self.convert_label_mode = label_mode

    def init_to_nnunet_raw_dataset_task(
        self,
        dataset_ids,
        label_mode,
        labels_input_dir,
        images_input_dir,
        output_dir,
        split_mode="separate_sources",
        single_source=None,
    ):
        """Initialize task to create nnUNet raw datasets.

        Args:
            dataset_ids: Space-separated string of 3 dataset IDs for the 3 sources
            label_mode: Label mode (e.g., 'consensus_staple', 'random_rater')
            labels_input_dir: Directory containing generated labels (output from generate_labels)
            images_input_dir: Directory containing converted images (output from convert_images)
            output_dir: Output directory for nnUNet raw datasets
        """
        self.nnunet_dataset_ids = dataset_ids.split()
        if len(self.nnunet_dataset_ids) != 3:
            raise ValueError(
                f"Expected 3 dataset IDs, got {len(self.nnunet_dataset_ids)}"
            )
        self.nnunet_label_mode = label_mode
        self.nnunet_labels_input_dir = Path(labels_input_dir)
        self.nnunet_images_input_dir = Path(images_input_dir)
        self.nnunet_output_dir = Path(output_dir)
        self.nnunet_output_dir.mkdir(parents=True, exist_ok=True)
        self.nnunet_split_mode = split_mode
        self.nnunet_single_source = single_source

    def _load_image(self, tma_identifier):
        # search for the image in all raw image lists
        for img_list in [
            self.tissue_array_raw_imgs,
            self.harvard_dataverse_raw_imgs,
            self.gleason2019_raw_imgs,
        ]:
            for img_path in img_list:
                if tma_identifier in img_path:
                    datasource = ""
                    if "tissuemicroarray_com_data" in img_path:
                        datasource = "tissue_array"
                    elif "dataverse_files" in img_path:
                        datasource = "harvard_dataverse"
                    elif "Gleason2019" in img_path:
                        datasource = "gleason2019"
                    # load and return image
                    return Image.open(img_path), datasource

        raise FileNotFoundError(
            f"Image with TMA_identifier {tma_identifier} not found."
        )

    def _save_label_mask(
        self,
        label_mask,
        tma_identifier,
        datasource,
        expected_size=None,
        output_filename=None,
    ):
        """Save label mask with optional size validation.

        Args:
            label_mask: The label mask array
            tma_identifier: TMA identifier for the image
            datasource: Source dataset name
            expected_size: Optional (H, W) tuple to validate mask size
            output_filename: Optional explicit filename (defaults to
                "{tma_identifier}_{label_mode}_mask.png")
        """
        # Validate mask size if expected size is provided
        if expected_size is not None:
            actual_size = label_mask.shape[:2]  # (H, W)
            if actual_size != expected_size:
                raise ValueError(
                    f"Mask size mismatch for {tma_identifier}: "
                    f"expected {expected_size}, got {actual_size}. "
                    f"Ensure image and mask are generated with same dimensions."
                )

        # save label mask
        if output_filename is None:
            output_filename = f"{tma_identifier}_{self.label_mode}_mask.png"
        output_path = self.output_dir / datasource / self.label_mode / output_filename
        Image.fromarray(label_mask).save(output_path)
        print(f"Saved label mask to {output_path}")

    def _build_label_mask_from_df(self, annotator_df, img_size):
        """Create one segmentation mask from a dataframe slice of one rater."""
        label_mask = np.zeros(img_size, dtype=np.uint8)

        for _, grade_annotator_frame in annotator_df.groupby(
            "grade", sort=True, observed=True
        ):
            for exp, coords in zip(
                grade_annotator_frame["explanations"],
                grade_annotator_frame["coords"],
            ):
                coords = np.array(eval(coords))
                new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :])
                label_slice = np.zeros(list(img_size), dtype=np.int8)

                cv2.fillPoly(label_slice, [new_coords.T], color=1)

                gleason_grade = self.explanation2gleasongrade_mapping[exp]
                seg_label = self.gleasongrade2seglabel_mapping[str(gleason_grade)]
                label_mask[label_slice > 0] = seg_label

        return label_mask

    def _generate_random_rater_labels(self, tma_df, tma_raters_pairs):
        # randomly select a tma-rater pair
        selected_pair = tma_raters_pairs.sample(n=1).iloc[0]
        selected_df = tma_df[
            (tma_df["TMA_identifier"] == selected_pair["TMA_identifier"])
            & (tma_df["annotator"] == selected_pair["annotator"])
        ]
        print(
            f"Given raters for TMA {selected_pair['TMA_identifier']}: {tma_raters_pairs['annotator'].tolist()} ; selected rater: {selected_pair['annotator']}"
        )

        # load image for reference
        img, datasource = self._load_image(selected_df.iloc[0]["TMA_identifier"])
        img_size = np.array(img.size)[::-1]  # (H, W)

        label_mask = self._build_label_mask_from_df(selected_df, img_size)

        # save label mask with size validation
        self._save_label_mask(
            label_mask,
            selected_df.iloc[0]["TMA_identifier"],
            datasource,
            expected_size=tuple(img_size),
        )

    def _generate_consensus_staple_labels(self, tma_df, tma_raters_pairs):
        # load image for reference
        img, datasource = self._load_image(tma_df.iloc[0]["TMA_identifier"])
        img_size = np.array(img.size)[::-1]

        # create label masks for each rater
        rater_label_masks = []
        for _, row in tma_raters_pairs.iterrows():
            rater = row["annotator"]
            rater_df = tma_df[
                (tma_df["TMA_identifier"] == row["TMA_identifier"])
                & (tma_df["annotator"] == rater)
            ]

            label_mask = self._build_label_mask_from_df(rater_df, img_size)

            rater_label_masks.append(label_mask)

        staple_mask = sitk.MultiLabelSTAPLE(
            [sitk.GetImageFromArray(mask) for mask in rater_label_masks],
        )
        staple_mask_array = sitk.GetArrayFromImage(staple_mask)

        # STAPLE produces floating-point values; clip to valid label range first, then round
        staple_mask_array = np.clip(staple_mask_array, 0, 3)
        staple_mask_array = np.round(staple_mask_array).astype(np.uint8)
        # check that staple_mask_array has only valid labels 0,1,2,3
        unique_labels = np.unique(staple_mask_array)
        assert all(
            label in [0, 1, 2, 3] for label in unique_labels
        ), f"Invalid labels found in STAPLE mask: {unique_labels}"

        # save label mask with size validation
        self._save_label_mask(
            staple_mask_array,
            tma_df.iloc[0]["TMA_identifier"],
            datasource,
            expected_size=tuple(img_size),
        )

    def _generate_all_rater_labels(self, tma_df, tma_raters_pairs):
        """Generate one label mask per rater for one TMA."""
        img, datasource = self._load_image(tma_df.iloc[0]["TMA_identifier"])
        img_size = np.array(img.size)[::-1]
        tma_identifier = tma_df.iloc[0]["TMA_identifier"]

        for _, row in tma_raters_pairs.iterrows():
            rater = row["annotator"]
            rater_df = tma_df[
                (tma_df["TMA_identifier"] == row["TMA_identifier"])
                & (tma_df["annotator"] == rater)
            ]

            label_mask = self._build_label_mask_from_df(rater_df, img_size)

            safe_rater = "".join(
                char if str(char).isalnum() else "_" for char in str(rater)
            )
            self._save_label_mask(
                label_mask,
                tma_identifier,
                datasource,
                expected_size=tuple(img_size),
                output_filename=(
                    f"{tma_identifier}_{self.label_mode}_annotator_{safe_rater}_mask.png"
                ),
            )

    def generate_labels(self):
        print(f"Generating labels in mode: {self.label_mode}")

        # load final_filtered_explanations_df.csv from raw_data_dir
        df = pd.read_csv(
            f"{self.raw_data_dir}/27301845/final_filtered_explanations_df.csv"
        )

        # get unique images ("TMA_identifier") and image-rater pairs
        all_tmas = df["TMA_identifier"].unique()
        all_tmas_raters_pairs = df[["TMA_identifier", "annotator"]].drop_duplicates()

        for tma in tqdm(all_tmas, desc="Processing TMAs"):
            # get all tma_rater_pair for this tma
            tma_raters_pairs = all_tmas_raters_pairs[
                all_tmas_raters_pairs["TMA_identifier"] == tma
            ]

            # get df for this tma
            tma_df = df[df["TMA_identifier"] == tma]

            if self.label_mode == "consensus_staple":
                self._generate_consensus_staple_labels(tma_df, tma_raters_pairs)
            elif self.label_mode == "random_rater":
                self._generate_random_rater_labels(tma_df, tma_raters_pairs)
            elif self.label_mode in ["all_raters", "all"]:
                self._generate_all_rater_labels(tma_df, tma_raters_pairs)
            else:
                raise ValueError(
                    f"Unknown generate_labels mode: {self.label_mode}. "
                    "Supported modes: consensus_staple, random_rater, all_raters"
                )

    def convert_images(self):
        """Convert all images to new format given convert_images_mode."""
        print(f"Converting images from {self.convert_img_mode} format...")

        # Collect all image lists
        all_img_lists = [
            (self.tissue_array_raw_imgs, "tissue_array"),
            (self.harvard_dataverse_raw_imgs, "harvard_dataverse"),
            (self.gleason2019_raw_imgs, "gleason2019"),
        ]

        # load annotation table to only convert images that have annotations
        df = pd.read_csv(
            f"{self.raw_data_dir}/27301845/final_filtered_explanations_df.csv"
        )

        total_images = sum(len(img_list) for img_list, _ in all_img_lists)

        for img_list, datasource in all_img_lists:
            # Create output subdirectory for this datasource
            output_subdir = self.convert_output_dir / datasource
            output_subdir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(img_list, desc=f"Converting {datasource} images"):
                # check if image has annotations
                tma_identifier = Path(img_path).stem
                if tma_identifier not in df["TMA_identifier"].values:
                    continue

                # Load image
                img = Image.open(img_path)
                img_size = img.size  # (W, H)

                # load corresponding mask and check size match
                mask = Image.open(
                    self.convert_labels_dir / datasource / self.convert_label_mode / f"{tma_identifier}_{self.convert_label_mode}_mask.png"
                )
                mask_size = mask.size  # (W, H)
                if img_size != mask_size:
                    # resize image to mask size
                    print(
                        f"Resizing image {tma_identifier} from size {img_size} to mask size {mask_size}"
                    )
                    img = img.resize(mask_size, Image.LANCZOS)
                    img_size = img.size  # update size after resize
                assert (
                    img_size == mask_size
                ), f"Size mismatch for {tma_identifier}: image {img_size} vs mask {mask_size}"

                # Get filename without extension
                img_filename = Path(img_path).stem

                if self.convert_img_mode == "jpg_to_png":
                    # Save as PNG
                    output_path = output_subdir / f"{img_filename}.png"
                    img.save(output_path, "PNG")
                else:
                    raise ValueError(
                        f"Unknown convert_images_mode: {self.convert_img_mode}"
                    )

        print(f"Converted {total_images} images to PNG format.")

    def to_nnunet_raw_dataset(self):
        """Create nnUNet raw datasets from labels and images.

        Maps 3 source datasets (tissue_array, harvard_dataverse, gleason2019) to 3 FL clients.
        For each client:
        - Copies labels from respective subdirectory to labelsTr
        - Loads images, separates RGB channels (_0000, _0001, _0002), saves to imagesTr
        - Creates dataset.json
        """
        print(f"Creating nnUNet raw datasets with label_mode={self.nnunet_label_mode}")

        # Map datasources to dataset IDs and FL clients
        datasources = ["tissue_array", "harvard_dataverse", "gleason2019"]

        for client_idx, (datasource, dataset_id) in enumerate(
            zip(datasources, self.nnunet_dataset_ids)
        ):
            print(
                f"Processing client {client_idx} (dataset_id={dataset_id}, datasource={datasource})"
            )

            # Create nnUNet dataset directory
            nnunet_dataset_name = f"Dataset{dataset_id}_GleasonXAI_{self.nnunet_label_mode}_client{client_idx}"
            nnunet_dataset_dir = self.nnunet_output_dir / nnunet_dataset_name
            imagesTr_dir = nnunet_dataset_dir / "imagesTr"
            labelsTr_dir = nnunet_dataset_dir / "labelsTr"
            imagesTr_dir.mkdir(parents=True, exist_ok=True)
            labelsTr_dir.mkdir(parents=True, exist_ok=True)

            # Get labels for this datasource
            labels_subdir = (
                self.nnunet_labels_input_dir / datasource / self.nnunet_label_mode
            )
            if not labels_subdir.exists():
                print(f"Warning: Labels directory not found: {labels_subdir}")
                label_files = []
            else:
                label_files = list(labels_subdir.glob("*_mask.png"))

            # Get images for this datasource
            images_subdir = self.nnunet_images_input_dir / datasource
            if not images_subdir.exists():
                print(f"Warning: Images directory not found: {images_subdir}")
                image_files = []
            else:
                image_files = list(images_subdir.glob("*.png"))

            sample_count = 0

            # Process each image-label pair
            for img_file in tqdm(image_files, desc=f"Processing {datasource} images"):
                tma_identifier = img_file.stem

                # Find corresponding label
                label_file = (
                    labels_subdir
                    / f"{tma_identifier}_{self.nnunet_label_mode}_mask.png"
                )
                if not label_file.exists():
                    print(
                        f"Warning: Label not found for image {tma_identifier}, skipping"
                    )
                    continue

                # Load image and convert to RGB if needed
                img = Image.open(img_file).convert("RGB")
                img_array = np.array(img)  # (H, W, 3) with RGB channels

                # Load label
                label_img = Image.open(label_file)
                label_array = np.array(label_img)  # (H, W)

                # Verify sizes match (should be guaranteed from generation)
                if img_array.shape[:2] != label_array.shape[:2]:
                    raise ValueError(
                        f"Size mismatch for {tma_identifier}: "
                        f"image {img_array.shape[:2]} vs label {label_array.shape[:2]}. "
                        f"Check mask generation."
                    )

                # Split into R, G, B channels
                r_channel = img_array[:, :, 0]
                g_channel = img_array[:, :, 1]
                b_channel = img_array[:, :, 2]

                # Save channels with nnUNet naming convention
                # Channel 0 (R), 1 (G), 2 (B)
                for channel_idx, channel_data in enumerate(
                    [r_channel, g_channel, b_channel]
                ):
                    channel_img = Image.fromarray(channel_data, mode="L")
                    channel_filename = f"{tma_identifier}_{channel_idx:04d}.png"
                    channel_img.save(imagesTr_dir / channel_filename)

                # Save label to labelsTr with nnUNet naming
                label_img.save(labelsTr_dir / f"{tma_identifier}.png")

                sample_count += 1

            # Create dataset.json
            dataset_json = {
                "name": f"GleasonXAI_{self.nnunet_label_mode}_client{client_idx}",
                "description": f"GleasonXAI dataset from {datasource} using {self.nnunet_label_mode} labels",
                "reference": "GleasonXAI",
                "license": "CC-BY-4.0",
                "release": "1.0",
                "file_ending": ".png",
                "channel_names": {"0": "R", "1": "G", "2": "B"},
                "labels": {
                    "background": 0,
                    "gleason_3": 1,
                    "gleason_4": 2,
                    "gleason_5": 3,
                },
                "numTraining": sample_count,
                "numTest": 0,
                "training": [],
                "test": [],
            }

            # Add file references
            for img_file in image_files[:sample_count]:  # Use processed count
                tma_identifier = img_file.stem
                if (labelsTr_dir / f"{tma_identifier}.png").exists():
                    dataset_json["training"].append(
                        {
                            "image": f"./imagesTr/{tma_identifier}_0000.png",
                            "label": f"./labelsTr/{tma_identifier}.png",
                        }
                    )

            # Save dataset.json
            with open(nnunet_dataset_dir / "dataset.json", "w") as f:
                json.dump(dataset_json, f, indent=4)

            print(f"Processed {sample_count} samples for client {client_idx}")

        print(f"nnUNet raw datasets created in {self.nnunet_output_dir}")

    def _compute_single_source_splits(
        self, datasource, single_source_splitting_strategy="least_freq_label_uniform"
    ):
        """
        Computes splitting of a single source dataset into 3 FL clients.
        Splitting is done either randomly or using least frequent label uniform strategy.
        Before splitting is computed, check whether existing splits are available to load.

        Args:
            datasource: The source dataset to split (e.g., 'tissue_array', 'harvard_dataverse', 'gleason2019')
            single_source_splitting_strategy: Strategy for splitting ('random', 'least_freq_label_uniform')

        Returns:
            Dictionary mapping client indices to lists of TMA_identifiers.
        """
        # check whether existing splits are available to load
        splits_file = (
            self.nnunet_output_dir
            / f"{datasource}_splits_{single_source_splitting_strategy}.json"
        )
        if splits_file.exists():
            print(f"Loading existing splits from {splits_file}")
            with open(splits_file, "r") as f:
                fl_client_splits = json.load(f)
            return fl_client_splits

        if single_source_splitting_strategy == "random":
            # implement random splitting
            raise NotImplementedError("Random splitting not yet implemented.")
        elif single_source_splitting_strategy == "least_freq_label_uniform":
            # load all labels for the given datasource and label mode
            labels_subdir = (
                self.nnunet_labels_input_dir / datasource / self.nnunet_label_mode
            )
            label_files = list(labels_subdir.glob("*_mask.png"))
            # create dict with TMA_identifier as key and occuring labels as values
            tma_labels_dict = {}
            for label_file in tqdm(
                label_files, desc=f"Loading labels for {datasource}"
            ):
                tma_identifier = label_file.stem.replace(
                    f"_{self.nnunet_label_mode}_mask", ""
                )
                label_img = Image.open(label_file)
                label_array = np.array(label_img)
                unique_labels = np.unique(label_array)
                tma_labels_dict[tma_identifier] = unique_labels.tolist()

            # identify least frequent label across all TMAs
            label_counts = {}
            for labels in tma_labels_dict.values():
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
            least_frequent_label = min(label_counts, key=label_counts.get)
            print(
                f"Least frequent label in {datasource} is {least_frequent_label} with count {label_counts[least_frequent_label]}"
            )

            # assign TMAs to clients to balance least frequent label
            fl_client_splits = {0: [], 1: [], 2: []}
            client_least_freq_counts = {0: 0, 1: 0, 2: 0}
            for tma_identifier, labels in tma_labels_dict.items():
                if least_frequent_label in labels:
                    # assign to client with currently least count of least frequent label
                    target_client = min(
                        client_least_freq_counts, key=client_least_freq_counts.get
                    )
                    fl_client_splits[target_client].append(tma_identifier)
                    client_least_freq_counts[target_client] += 1
                else:
                    # assign to client with least total samples
                    target_client = min(
                        fl_client_splits, key=lambda k: len(fl_client_splits[k])
                    )
                    fl_client_splits[target_client].append(tma_identifier)

            # save splits to file for future use
            with open(splits_file, "w") as f:
                json.dump(fl_client_splits, f, indent=4)
            print(f"Saved computed splits to {splits_file}")

            return fl_client_splits
        else:
            raise ValueError(
                f"Unknown single_source_splitting_strategy: {single_source_splitting_strategy}"
            )

    def to_nnunet_raw_dataset_single_source(self):
        """
        Create nnUNet raw dataset from a single source dataset.

        Splits self.nnunet_single_source with given self.nnunet_label_mode to 3 FL clients.
        Various splitting strategies are supported: 'random', 'least_freq_label_uniform'
        """
        print(
            f"Creating nnUNet raw dataset from single source: {self.nnunet_single_source} with label_mode={self.nnunet_label_mode} to dataset IDs={self.nnunet_dataset_ids}"
        )

        datasource = self.nnunet_single_source

        # define splitting
        fl_client_splits = self._compute_single_source_splits(datasource)

        for client_idx, dataset_id in enumerate(self.nnunet_dataset_ids):
            print(
                f"Processing client {client_idx} (dataset_id={dataset_id}, datasource={datasource})"
            )

            # Create nnUNet dataset directory
            nnunet_dataset_name = f"Dataset{dataset_id}_GleasonXAI_{self.nnunet_label_mode}_client{client_idx}"
            nnunet_dataset_dir = self.nnunet_output_dir / nnunet_dataset_name
            imagesTr_dir = nnunet_dataset_dir / "imagesTr"
            labelsTr_dir = nnunet_dataset_dir / "labelsTr"
            imagesTr_dir.mkdir(parents=True, exist_ok=True)
            labelsTr_dir.mkdir(parents=True, exist_ok=True)

            tma_identifiers = fl_client_splits[str(client_idx)]

            sample_count = 0

            for tma_identifier in tqdm(
                tma_identifiers,
                desc=f"Processing {datasource} TMAs for client {client_idx}",
            ):
                # Load image, split it in its channels and save it, and same with labels
                img_file = (
                    self.nnunet_images_input_dir / datasource / f"{tma_identifier}.png"
                )
                label_file = (
                    self.nnunet_labels_input_dir
                    / datasource
                    / self.nnunet_label_mode
                    / f"{tma_identifier}_{self.nnunet_label_mode}_mask.png"
                )
                if not img_file.exists() or not label_file.exists():
                    print(
                        f"Warning: Image or label not found for TMA {tma_identifier}, skipping"
                    )
                    continue

                # Load image and convert to RGB if needed
                img = Image.open(img_file).convert("RGB")
                img_array = np.array(img)  # (H, W, 3) with RGB channels

                # Load label
                label_img = Image.open(label_file)
                label_array = np.array(label_img)  # (H, W)

                # Verify sizes match (should be guaranteed from generation)
                if img_array.shape[:2] != label_array.shape[:2]:
                    raise ValueError(
                        f"Size mismatch for {tma_identifier}: "
                        f"image {img_array.shape[:2]} vs label {label_array.shape[:2]}. "
                        f"Check mask generation."
                    )

                # Split into R, G, B channels
                r_channel = img_array[:, :, 0]
                g_channel = img_array[:, :, 1]
                b_channel = img_array[:, :, 2]

                # Save channels with nnUNet naming convention
                # Channel 0 (R), 1 (G), 2 (B)
                for channel_idx, channel_data in enumerate(
                    [r_channel, g_channel, b_channel]
                ):
                    channel_img = Image.fromarray(channel_data, mode="L")
                    channel_filename = f"{tma_identifier}_{channel_idx:04d}.png"
                    channel_img.save(imagesTr_dir / channel_filename)

                # Save label to labelsTr with nnUNet naming
                label_img.save(labelsTr_dir / f"{tma_identifier}.png")
                sample_count += 1

            # Create dataset.json as before
            dataset_json = {
                "name": f"GleasonXAI_{self.nnunet_label_mode}_client{client_idx}",
                "description": f"GleasonXAI dataset from {datasource} using {self.nnunet_label_mode} labels",
                "reference": "GleasonXAI",
                "license": "CC-BY-4.0",
                "release": "1.0",
                "file_ending": ".png",
                "channel_names": {"0": "R", "1": "G", "2": "B"},
                "labels": {
                    "background": 0,
                    "gleason_3": 1,
                    "gleason_4": 2,
                    "gleason_5": 3,
                },
                "numTraining": sample_count,
                "numTest": 0,
                "training": [],
                "test": [],
            }
            # Add file references
            for tma_identifier in tma_identifiers:
                if (labelsTr_dir / f"{tma_identifier}.png").exists():
                    dataset_json["training"].append(
                        {
                            "image": f"./imagesTr/{tma_identifier}_0000.png",
                            "label": f"./labelsTr/{tma_identifier}.png",
                        }
                    )

            # Save dataset.json
            with open(nnunet_dataset_dir / "dataset.json", "w") as f:
                json.dump(dataset_json, f, indent=4)

            print(f"Processed {sample_count} samples for client {client_idx}")

        print(f"nnUNet raw datasets created in {self.nnunet_output_dir}")


def main(args):
    preparer = GleasonXAIDataPreparer(raw_data_dir=args.raw_data_dir)

    if args.task == "generate_labels":
        preparer.init_generate_labels_task(
            label_mode=args.generate_labels_mode,
            output_dir=args.generate_labels_output_dir,
        )
        preparer.generate_labels()
    elif args.task == "convert_images":
        preparer.init_convert_images_task(
            convert_images_mode=args.convert_images_mode,
            output_dir=args.convert_images_output_dir,
        )
        preparer.convert_images()
    elif args.task == "to_nnunet_raw_dataset":
        preparer.init_to_nnunet_raw_dataset_task(
            dataset_ids=args.nnunet_dataset_ids,
            label_mode=args.nnunet_label_mode,
            labels_input_dir=args.nnunet_labels_input_dir,
            images_input_dir=args.nnunet_images_input_dir,
            output_dir=args.nnunet_output_dir,
            split_mode=args.nnunet_split_mode,
            single_source=args.nnunet_single_source,
        )
        if preparer.nnunet_split_mode == "single_source":
            preparer.to_nnunet_raw_dataset_single_source()
        elif preparer.nnunet_split_mode == "separate_sources":
            preparer.to_nnunet_raw_dataset()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Gleason XAI dataset")
    parser.add_argument(
        "--raw_data_dir", type=str, required=True, help="Path to the raw data directory"
    )
    parser.add_argument(
        "--single_seg_mode",
        type=str,
        required=True,
        choices=["consensus_staple", "random_rater"],
        help="Label mode: 'consensus_staple' for clean clients, 'random_rater' for noisy clients.",
    )
    parser.add_argument(
        "--dataset_ids",
        type=str,
        required=True,
        help="Three space-separated dataset IDs for the 3 FL clients (e.g., '430 431 432')",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    nnunet_raw = os.getenv("nnUNet_raw")
    assert nnunet_raw, "Environment variable $nnUNet_raw is not set."

    # auto-derive intermediate paths from raw_data_dir
    generated_labels_dir = Path(args.raw_data_dir) / "generated_labels"
    converted_images_dir = Path(args.raw_data_dir) / "converted_images"

    preparer = GleasonXAIDataPreparer(raw_data_dir=args.raw_data_dir)

    # Step 1: Generate labels
    logging.info(f"Generating {args.single_seg_mode} labels...")
    preparer.init_generate_labels_task(
        label_mode=args.single_seg_mode,
        output_dir=generated_labels_dir,
    )
    preparer.generate_labels()

    # Step 2: Convert images to PNG (uses generated labels for size reference)
    logging.info("Converting images to PNG...")
    preparer.init_convert_images_task(
        output_dir=converted_images_dir,
        labels_dir=generated_labels_dir,
        label_mode=args.single_seg_mode,
    )
    preparer.convert_images()

    # Step 3: To nnUNet raw dataset format
    logging.info("Creating nnUNet raw datasets...")
    preparer.init_to_nnunet_raw_dataset_task(
        dataset_ids=args.dataset_ids,
        label_mode=args.single_seg_mode,
        labels_input_dir=generated_labels_dir,
        images_input_dir=converted_images_dir,
        output_dir=nnunet_raw,
        split_mode="single_source",
        single_source="harvard_dataverse",
    )
    preparer.to_nnunet_raw_dataset_single_source()
