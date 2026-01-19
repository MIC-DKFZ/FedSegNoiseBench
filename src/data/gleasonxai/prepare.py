import argparse
import json
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
        self.label_mode = label_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # also create subdirectories for each sub dataset
        for datasource in ["tissue_array", "harvard_dataverse", "gleason2019"]:
            (self.output_dir / datasource / self.label_mode).mkdir(
                parents=True, exist_ok=True
            )

    def init_convert_images_task(self, output_dir, convert_images_mode="jpg_to_png"):
        self.convert_img_mode = convert_images_mode
        self.convert_output_dir = Path(output_dir)
        self.convert_output_dir.mkdir(parents=True, exist_ok=True)

    def init_to_nnunet_raw_dataset_task(
        self, dataset_ids, label_mode, labels_input_dir, images_input_dir, output_dir
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
        self, label_mask, tma_identifier, datasource, expected_size=None
    ):
        """Save label mask with optional size validation.

        Args:
            label_mask: The label mask array
            tma_identifier: TMA identifier for the image
            datasource: Source dataset name
            expected_size: Optional (H, W) tuple to validate mask size
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
        output_path = (
            self.output_dir
            / datasource
            / self.label_mode
            / f"{tma_identifier}_{self.label_mode}_mask.png"
        )
        Image.fromarray(label_mask).save(output_path)
        print(f"Saved label mask to {output_path}")

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

        # create label mask with grade_frame_order like the reference function
        label_mask = np.zeros(img_size, dtype=np.uint8)

        # Process by grade order (sort=True) to match grade_frame_order behavior
        for grade_image, grade_annotator_frame in selected_df.groupby(
            "grade", sort=True, observed=True
        ):
            for exp, coords in zip(
                grade_annotator_frame["explanations"],
                grade_annotator_frame["coords"],
            ):
                coords = np.array(eval(coords))
                new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :])
                label_slice = np.zeros(list(img_size), dtype=np.int8)

                # cv2.fillPoly expects (W,H) coordinates
                cv2.fillPoly(label_slice, [new_coords.T], color=1)

                gleason_grade = self.explanation2gleasongrade_mapping[exp]
                seg_label = self.gleasongrade2seglabel_mapping[str(gleason_grade)]
                label_mask[label_slice > 0] = seg_label

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

            label_mask = np.zeros(img_size, dtype=np.uint8)
            for exp, coords in zip(
                rater_df["explanations"],
                rater_df["coords"],
            ):
                coords = np.array(eval(coords))
                new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :])
                label_slice = np.zeros(list(img_size), dtype=np.int8)
                cv2.fillPoly(label_slice, [new_coords.T], color=1)
                gleason_grade = self.explanation2gleasongrade_mapping[exp]
                seg_label = self.gleasongrade2seglabel_mapping[str(gleason_grade)]
                label_mask[label_slice > 0] = seg_label

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
                    f"{self.raw_data_dir}/generated_labels/{datasource}/random_rater/{tma_identifier}_random_rater_mask.png"
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
        )
        preparer.to_nnunet_raw_dataset()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Gleason XAI dataset")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to perform: 'generate_labels', 'convert_images', 'to_nnunet_raw_dataset'",
    )

    parser.add_argument(
        "--raw_data_dir", type=str, required=True, help="Path to the raw data directory"
    )

    # args for 'generate_labels' task
    parser.add_argument(
        "--generate_labels_mode",
        type=str,
        required=False,
        help="Mode for label generation: 'consensus_staple', 'random_rater', 'all', ...",
    )
    parser.add_argument(
        "--generate_labels_output_dir",
        type=str,
        required=False,
        help="Output directory for generated labels",
    )

    # args for 'convert_images' task
    parser.add_argument(
        "--convert_images_mode",
        type=str,
        required=False,
        help="Mode for image conversion: 'jpg_to_png', ...",
    )
    parser.add_argument(
        "--convert_images_output_dir",
        type=str,
        required=False,
        help="Output directory for converted images.",
    )

    # args for 'to_nnunet_raw_dataset' task
    parser.add_argument(
        "--nnunet_dataset_ids",
        type=str,
        required=False,
        help="Three space-separated dataset IDs for the 3 FL clients (e.g., '430 431 432')",
    )
    parser.add_argument(
        "--nnunet_label_mode",
        type=str,
        required=False,
        help="Label mode to use (e.g., 'consensus_staple', 'random_rater')",
    )
    parser.add_argument(
        "--nnunet_labels_input_dir",
        type=str,
        required=False,
        help="Input directory containing generated labels (output from generate_labels task)",
    )
    parser.add_argument(
        "--nnunet_images_input_dir",
        type=str,
        required=False,
        help="Input directory containing converted images (output from convert_images task)",
    )
    parser.add_argument(
        "--nnunet_output_dir",
        type=str,
        required=False,
        help="Output directory for nnUNet raw datasets",
    )

    args = parser.parse_args()

    main(args)
