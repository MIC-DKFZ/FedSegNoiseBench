from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_seg(seg_path:Path):
    seg_array = np.array(Image.open(seg_path))

    print(f"Seg shape: {seg_array.shape}, unique classes: {np.unique(seg_array)}")

    plt.figure(figsize=(12, 6))
    plt.imshow(seg_array, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_segs(seg_paths:dict):
    n_segs = len(seg_paths)
    plt.figure(figsize=(5*n_segs, 5))
    for i, (label, seg_path) in enumerate(seg_paths.items()):
        seg_array = np.array(Image.open(seg_path))
        print(f"{label} - Seg shape: {seg_array.shape}, unique classes: {np.unique(seg_array)}")

        plt.subplot(1, n_segs, i+1)
        plt.imshow(seg_array, cmap="gray")
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    # seg_file = Path("/home/m391k/cluster-data/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real/Dataset412_Gleason2019_staple_flclient2/gt_segmentations/" \
    # "Gleason-slide007core044_0058.tif")
    # plot_seg(seg_file)

    seg_files = {
        # "rater1": Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/raw_grandchallenge/Maps1_T/slide001_core003_classimg_nonconvex.png"),
        # "rater2": Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/raw_grandchallenge/Maps3_T/slide001_core003_classimg_nonconvex.png"),
        # "rater3": Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/raw_grandchallenge/Maps4_T/slide001_core003_classimg_nonconvex.png"),
        # "rater4": Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/raw_grandchallenge/Maps5_T/slide001_core003_classimg_nonconvex.png"),
        "random": Path("/home/m391k/cluster-data/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real/Dataset419_Gleason2019_random_flclient0/gt_segmentations/Gleason-slide001core005_0024.tif"),
        # "annotator_majority": Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/Gleason2019/single_seg_annotatormajority_grandchallenge/slide001_core145_classimg_nonconvex.tif"),
        "staple": Path("/home/m391k/cluster-data/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real/Dataset416_Gleason2019_staple_flclient0/gt_segmentations/Gleason-slide001core005_0024.tif"),
    }
    plot_segs(seg_files)

    # very intense difference: Gleason-slide007core002_0073.tif; Gleason-slide007core008_0062.tif
    # massive contour difference: Gleason-slide007core013_0048.tif