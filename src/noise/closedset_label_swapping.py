import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def swap_labels(label_array, class_mapping):
    """
    Swap the class labels in the label array according to the class_mapping.
    """
    swapped_array = np.copy(label_array)
    for original, new in class_mapping.items():
        swapped_array[label_array == original] = new
    return swapped_array

def generate_class_mapping(unique_classes):
    """
    Generate a random permutation mapping for the unique class labels,
    ensuring that no class is mapped to itself.
    """
    permuted_classes = unique_classes.copy()
    while True:
        np.random.shuffle(permuted_classes)
        # Check that no class maps to itself
        if np.all(permuted_classes != unique_classes):
            break
    return dict(zip(unique_classes, permuted_classes))

def create_closedset_label_swapped_labels(label_dir, output_dir, seed=42):
    """
    Create and save noisy closed-set label-swapped labels.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for label_file in tqdm(os.listdir(label_dir), desc="Processing labels"):
        if not label_file.endswith(".nii.gz"):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_dir, label_file)
        
        # Load the label file
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata().astype(np.int32)
        
        # Find unique classes in the label and rm background class (=0)
        unique_classes = np.unique(label_data)
        unique_classes = unique_classes[unique_classes != 0]
        
        # Generate a random class mapping
        class_mapping = generate_class_mapping(unique_classes)
        
        # Apply the label swapping
        swapped_data = swap_labels(label_data, class_mapping)
        
        # Save the modified label
        swapped_nii = nib.Nifti1Image(swapped_data, label_nii.affine, label_nii.header)
        nib.save(swapped_nii, output_path)

if __name__ == "__main__":
    # Paths
    label_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/KiTS2023/clean/Dataset220_KiTS2023/labelsTr"
    output_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/KiTS2023/closedset_label_swapping/Dataset220_KiTS2023/labelsTr"

    # Create swapped labels
    create_closedset_label_swapped_labels(label_dir, output_dir, seed=42)
