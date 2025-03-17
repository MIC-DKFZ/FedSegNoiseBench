import cv2
import tifffile as tiff
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

from prepare import RIGA_dataset_processor


def manual_riga_prep(
    riga_data_preper: RIGA_dataset_processor,
    img_fname: str = None,
    mask_fname: str = None,
    out_fname: str = None,
):
    """
    Manuelly retrieve dense segmentation mask from sampled failed in riga/prepare.py's RIGA_dataset_processor class.
    """
    # load mask
    img = riga_data_preper.load_img_mask(img_fname)
    mask = riga_data_preper.load_img_mask(mask_fname)

    # manually retrieve dense masks by adapting hyperparameters
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    diff_ = cv2.absdiff(gray_mask, gray_img)
    _, diff = cv2.threshold(diff_, 20, 255, cv2.THRESH_BINARY)

    # CCA to filter out small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff, connectivity=8
    )
    min_size = 50
    filtered_diff = np.zeros_like(diff)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_diff[labels == i] = 255

    skeleton = skeletonize(filtered_diff)

    contours = []
    i = 0
    closed = skeleton.copy()
    # closed = filtered_diff
    closed = (closed * 255).astype(np.uint8)
    while len(contours) != 2 and len(contours) != 4:
        kernel = np.ones((int(3 + (i / 5)), int(3 + (i / 5))), np.uint8)
        # Apply Closing (Dilation + Erosion) with adapted kernel size
        closed = cv2.dilate(
            closed, kernel + np.ones((1, 1), np.uint8), iterations=1 #  + np.ones((1, 1), np.uint8)
        )
        closed = cv2.erode(closed, kernel, iterations=1)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        # remove contour that includes background of retina image
        contours = [contour for contour in contours if np.array([0,0]) not in contour]
        i += 1
    print(i)
    contours = sorted(contours, key=cv2.contourArea)

    # Create an empty mask
    one_hot_masks = [np.zeros_like(mask, dtype=np.uint8) for _ in contours]
    fg_val_count = []
    # Ensure 2 contours or 4 contours exist
    for i, contour in enumerate(contours):
        cv2.drawContours(one_hot_masks[i], [contour], -1, 255, thickness=cv2.FILLED)
        fg_val_count.append(np.count_nonzero(one_hot_masks[i]))

    # get biggest and second biggest mask
    # Note: cv2.RETR_CCOMP in cv2.findContours() retrieves inner and outer edge of each contour
    #       so the biggest and second biggest mask are at index 0 and 2
    biggest_mask = one_hot_masks[fg_val_count.index(max(fg_val_count))]
    sec_biggest_mask = one_hot_masks[fg_val_count.index(max(fg_val_count)) - 2]

    # set biggest and smallest mask to 1 (optical disc) and 2 (optical cup)
    biggest_mask[biggest_mask == 255] = 120
    sec_biggest_mask[sec_biggest_mask == 255] = 255

    final_seg_mask = np.maximum(biggest_mask, sec_biggest_mask)

    # # visualize
    images = [img, mask, diff_, diff, filtered_diff, closed, final_seg_mask]
    titles = ["img", "mask", "diff_", "diff", "filtered_diff", "closed", "final_seg_mask"]
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(images), figsize=(25, 5))  # Adjust figsize as needed
    # Loop over images and display them
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray")  # Use cmap="gray" for single-channel images
        ax.set_title(title)
        ax.axis("off")  # Hide axis labels
        print(title)
    # Show all images
    plt.show()

    # save dense mask
    tiff.imwrite(out_fname, final_seg_mask)

if __name__ == "__main__":
    # user input
    mask_fname = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/RIGA/raw/BinRushedcorrected/BinRushed/BinRushed2/image20-3.jpg"
    img_fname = "/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/RIGA/raw/BinRushedcorrected/BinRushed/BinRushed2/image20prime.jpg"

    riga_data_preper = RIGA_dataset_processor()
    out_fname = mask_fname.replace("raw", "img_segmask_tif").replace(".jpg", ".tif")
    manual_riga_prep(riga_data_preper, img_fname, mask_fname, out_fname)
