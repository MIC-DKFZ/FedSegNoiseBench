import os

def count_unique_ids(input_dir):
    # Initialize a set to store unique IDs
    unique_ids = set()

    # Iterate over files in the directory
    for filename in os.listdir(input_dir):
        # Split the filename and extract the first part as the ID
        if filename.endswith(".nii.gz"):
            unique_id = filename.split("_")[0]
            unique_ids.add(unique_id)

    # Print the count of unique IDs
    print(f"Number of unique IDs: {len(unique_ids)}")
    # print("Unique IDs:", unique_ids)

if __name__ == "__main__":
    # Specify the directory containing the files
    # input_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC-single_seg_nifti/malignancy_cropped_random-multi_rater"
    input_dir = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/LIDC_seg-per-nodule-and-rater_nifti-cropped"
    count_unique_ids(input_dir)