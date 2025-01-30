import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from tqdm import tqdm

def apply_morphological_operation_on_class_3d(label_array, class_value, kernel_size=3, iterations=1):
    """
    Apply dilation or erosion to a specific class (foreground) in a 3D label mask.
    """
    # Create a 3D structuring element (spherical kernel)
    structuring_element = generate_binary_structure(rank=3, connectivity=1)
    structuring_element = np.pad(structuring_element, (kernel_size // 2,))
    
    # Create a binary mask for the specific class
    class_mask = (label_array == class_value).astype(np.uint8)

    # Randomly choose whether to apply dilation or erosion
    if np.random.rand() > 0.5:
        # Apply 3D dilation to the class mask
        modified_mask = binary_dilation(class_mask, structure=structuring_element, iterations=iterations)
    else:
        # Apply 3D erosion to the class mask
        modified_mask = binary_erosion(class_mask, structure=structuring_element, iterations=iterations)
    
    # Update the label array with the modified mask
    label_array[class_mask != 0] = 0  # Clear the original class
    label_array[modified_mask] = class_value  # Insert the modified class
    return label_array

def apply_morphological_noise_to_labels(label_dir, output_dir, kernel_size=3, iterations=1, seed=42):
    """
    Apply random 3D dilation or erosion to each foreground class in the label masks.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for label_file in tqdm(os.listdir(label_dir), desc="Processing labels"):
        print(f"Processing {label_file}")
        if not label_file.endswith(".nii.gz"):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_dir, label_file)
        
        # Load the label file
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata().astype(np.int32)
        
        # Find unique classes in the label (excluding background class 0)
        unique_classes = np.unique(label_data)
        unique_classes = unique_classes[unique_classes != 0]  # Exclude background class (0)
        
        # Apply random 3D morphological operation (dilation or erosion) per class
        for class_value in unique_classes:
            label_data = apply_morphological_operation_on_class_3d(label_data, class_value, kernel_size, iterations)
        
        # Save the modified label
        noisy_nii = nib.Nifti1Image(label_data, label_nii.affine, label_nii.header)
        nib.save(noisy_nii, output_path)

if __name__ == "__main__":
    # Paths
    label_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/KiTS2023/clean/Dataset220_KiTS2023/labelsTr"
    output_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/KiTS2023/label_erosion_dilation/Dataset220_KiTS2023/labelsTr"

    # Apply morphological noise (dilation or erosion)
    apply_morphological_noise_to_labels(label_dir, output_dir, kernel_size=3, iterations=1)
