# import h5py

# h5_path = "/home/m391k/Documents/my_documents/Publications/fed_noisy_label_benchmark/data/MMIS2024TASK1/training/Sample_0.h5"

# def print_structure(name, obj):
#     print(name)
#     if isinstance(obj, h5py.Dataset):
#         print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")

# with h5py.File(h5_path, 'r') as f:
#     f.visititems(print_structure)

import h5py
import nibabel as nib
import numpy as np
import os

h5_path = "/home/m391k/Documents/my_documents/Publications/fed_noisy_label_benchmark/data/MMIS2024TASK1/training/Sample_0.h5"
output_dir = "/home/m391k/Documents/my_documents/Publications/fed_noisy_label_benchmark/data/MMIS2024TASK1/training/Sample_0_nifti"
os.makedirs(output_dir, exist_ok=True)

# Get voxel spacing and shape
with h5py.File(h5_path, 'r') as f:
    spacing = f['voxel_spacing'][:]  # [z, y, x] spacing
    shape = f['t1'].shape  # (22, 145, 179) -> [z, y, x]
    
print(f"Shape: {shape}, Spacing: {spacing}")

# NIfTI affine matrix (identity + spacing)
affine = np.eye(4)
affine[0, 0] = spacing[2]  # x
affine[1, 1] = spacing[1]  # y  
affine[2, 2] = spacing[0]  # z

modalities = ['t1', 't1c', 't2']
labels = ['label_a1', 'label_a2', 'label_a3', 'label_a4']

# Save MRI modalities
print("Saving MRI modalities...")
with h5py.File(h5_path, 'r') as f:
    for mod in modalities:
        data = f[mod][:]  # Already in [z, y, x] NIfTI order
        nii_img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(nii_img, os.path.join(output_dir, f"{mod}.nii.gz"))
        print(f"Saved {mod}.nii.gz")

# Save labels
print("Saving labels...")
with h5py.File(h5_path, 'r') as f:
    for lbl in labels:
        data = f[lbl][:]  # uint8 labels, already [z, y, x]
        nii_img = nib.Nifti1Image(data.astype(np.uint8), affine)
        nib.save(nii_img, os.path.join(output_dir, f"{lbl}.nii.gz"))
        print(f"Saved {lbl}.nii.gz")

print(f"All files saved to: {output_dir}")