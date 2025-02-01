import os
import shutil
import pandas as pd
import glob
from tqdm import tqdm
import json


def main(
    datafolder: str = None,
    fed_split_metadata_fname: str = None,
    lidc_metadata_fname: str = None,
    client_ds_ids: list = None,
):
    # load csv to df
    fed_split_metadata_df = pd.read_csv(fed_split_metadata_fname)
    lidc_metadata_df = pd.read_csv(lidc_metadata_fname)

    # merge both df
    lidc_metadata_df.rename(columns={"Series UID": "SeriesInstanceUID"}, inplace=True)
    merged_df = pd.merge(
        fed_split_metadata_df,
        lidc_metadata_df[["SeriesInstanceUID", "Subject ID"]],
        on="SeriesInstanceUID",
        how="left",
    )

    # load all nifti file paths from datafolder
    nifti_files = glob.glob(f"{datafolder}/**/*.nii.gz", recursive=True)

    # create outputfolder for each client's train and test image and label data
    src_ds_id = os.path.basename(datafolder).split("_")[0]
    clientid_dsname_dict = {}
    for client_id, client_ds_id in enumerate(client_ds_ids):
        clientid_dsname_dict[client_id] = (
            data_folder.replace(src_ds_id, f"Dataset{client_ds_id}")
            + f"_fedclient{client_id}"
        )
        any(
            os.makedirs(os.path.join(clientid_dsname_dict[client_id], x), exist_ok=True)
            for x in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]
        )

    for nifti_file in tqdm(
        nifti_files, desc="Splitting .nii.gz files to client datasets", unit="file"
    ):
        # get lidc_id from nifti file in matching format to occurence in metadata
        lidc_id = (
            "LIDC-IDRI-" + os.path.basename(nifti_file).split("_")[1].split("-")[0]
        )
        curr_client_id = merged_df.loc[merged_df["Subject ID"] == lidc_id][
            "Manufacturer"
        ].values[0]
        curr_train_test = merged_df.loc[merged_df["Subject ID"] == lidc_id][
            "Split"
        ].values[0]

        # copy nifti file to client's train or test image or label output folder
        _nifti_file = nifti_file
        if curr_train_test == "test":
            _nifti_file = _nifti_file.replace("imagesTr", "imagesTs")
            _nifti_file = _nifti_file.replace("labelsTr", "labelsTs")
        new_fname = _nifti_file.replace(
            os.path.dirname(os.path.dirname(_nifti_file)),
            clientid_dsname_dict[curr_client_id],
        )
        shutil.copy(nifti_file, new_fname)

    # create new dataset.json
    for client_id, client_ds_name in clientid_dsname_dict.items():
        shutil.copy(
            os.path.join(datafolder, "dataset.json"),
            os.path.join(client_ds_name, "dataset.json"),
        )
        with open(os.path.join(client_ds_name, "dataset.json"), "r+") as f:
            data = json.load(f)
            data["numTraining"] = len(glob.glob(f"{client_ds_name}/imagesTr/*.nii.gz"))
            data["numTest"] = len(glob.glob(f"{client_ds_name}/imagesTs/*.nii.gz"))
            f.seek(0)  # Move cursor to beginning of file
            json.dump(data, f, indent=4)  # Write updated JSON
            f.truncate()  # Remove any leftover content from previous version


if __name__ == "__main__":
    # define in and out dirs
    data_folder = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/nnUNet_raw/Dataset029_LIDC-Malignancy-Cropped-RandomMultiRater"
    fed_split_metadata_fname = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/flamby_fed_lidc_metadata.csv"
    lidc_metadata_fname = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/LIDC-IDRI_raw/manifest-1600709154662/metadata.csv"

    client_ds_ids = ["037", "038", "039", "040"]

    main(data_folder, fed_split_metadata_fname, lidc_metadata_fname, client_ds_ids)
