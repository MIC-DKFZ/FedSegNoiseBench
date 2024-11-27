import os
import json

def update_dataset_json(dataset_json_path, images_dir, labels_dir, output_json_path):
    """
    Update the dataset.json file to include paths to images and labels.

    Args:
        dataset_json_path (str): Path to the input dataset.json file.
        images_dir (str): Path to the directory containing the images.
        labels_dir (str): Path to the directory containing the labels.
        output_json_path (str): Path to save the updated dataset.json.
    """
    # Load the existing dataset.json
    with open(dataset_json_path, 'r') as file:
        dataset_json = json.load(file)
    
    # Prepare dataset structure
    dataset = {}

    # Find all image files in the images directory
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".nii.gz")]

    # Create the dataset entries
    for image_file in image_files:
        # Derive the case identifier (filename without extension)
        case_id = image_file.rsplit("_", 1)[0]

        # Construct paths for the image and corresponding label
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

        # Check if the label exists before adding the entry
        if os.path.exists(label_path):
            dataset[case_id] = {
                "images": [image_path],
                "label": label_path
            }
        else:
            print(f"Warning: No label found for {case_id}")

    # Update the dataset.json structure
    dataset_json["dataset"] = dataset

    # Save the updated dataset.json to the specified output path
    with open(output_json_path, 'w') as file:
        json.dump(dataset_json, file, indent=4)

    print(f"Updated dataset.json has been saved to {output_json_path}")

# Example usage
if __name__ == "__main__":
    dataset_json_path = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/KiTS2023/clean/Dataset220_KiTS2023/dataset.json"
    images_dir = "/home/m391k/E132-Rohdaten/nnUNetv2/Dataset220_KiTS2023/imagesTr"
    labels_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/KiTS2023/clean/Dataset220_KiTS2023/labelsTr"
    output_json_path = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/KiTS2023/clean/Dataset220_KiTS2023/dataset_imgs_lbls_sep.json"

    update_dataset_json(dataset_json_path, images_dir, labels_dir, output_json_path)
