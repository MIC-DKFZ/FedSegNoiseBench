import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple

def load_multichannel_image(image_paths: Union[str, List[str]]) -> np.ndarray:
    """
    Load a potentially multi-channel image.
    
    Args:
        image_paths: Single file path or list of file paths (one per channel)
    
    Returns:
        Image array of shape (H, W, C) or (H, W) for single channel
    """
    if isinstance(image_paths, (str, Path)):
        img = cv2.imread(str(image_paths), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image file: {image_paths}")
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        channels = []
        expected_shape = None
        for path in image_paths:
            channel = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if channel is None:
                raise FileNotFoundError(f"Could not read image channel file: {path}")
            if expected_shape is None:
                expected_shape = channel.shape
            elif channel.shape != expected_shape:
                raise ValueError(
                    "All image channels must have the same shape, but got "
                    f"{channel.shape} for {path} and {expected_shape} for the first channel."
                )
            channels.append(channel.astype(np.float32))

        stacked = np.stack(channels, axis=-1)
        # For qualitative overlays we want a grayscale background, not a false-color
        # rendering of the individual nnUNet channels.
        grayscale = np.mean(stacked, axis=-1).astype(np.uint8)
        return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def load_mask(mask_path: str) -> np.ndarray:
    """Load a segmentation mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask file: {mask_path}")
    return mask


def overlay_contours(
    image: np.ndarray,
    consensus_mask: np.ndarray,
    noisy_mask: np.ndarray,
    consensus_color: Tuple[int, int, int] = (252, 233, 79),
    noisy_color: Tuple[int, int, int] = (5, 214, 254),
    thickness: int = 2
) -> np.ndarray:
    """
    Overlay mask contours on image.
    
    Args:
        image: RGB image array
        consensus_mask: Binary consensus segmentation mask
        noisy_mask: Binary noisy segmentation mask
        consensus_color: BGR color for consensus contours (yellow)
        noisy_color: BGR color for noisy contours (cyan)
        thickness: Contour line thickness
    
    Returns:
        Image with overlaid contours
    """
    result = image.copy()

    def draw_mask_contours(
        mask: np.ndarray,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw contours for every non-background label in the mask."""
        labels = sorted(int(v) for v in np.unique(mask) if int(v) > 0)
        if not labels:
            return

        for label in labels:
            binary_mask = (mask == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                result,
                contours,
                -1,
                color,
                thickness,
            )

    # Draw noisy first so the consensus contour remains visible in front.
    
    draw_mask_contours(consensus_mask, consensus_color)
    draw_mask_contours(noisy_mask, noisy_color)

    return result


def visualize_segmentations(
    image_paths: Union[str, List[str]],
    consensus_mask_path: str,
    noisy_mask_path: str,
    output_path: str = None,
    line_width: int = 2,
) -> np.ndarray:
    """
    Load image and masks, then overlay contours.
    
    Args:
        image_paths: Path(s) to image file(s)
        consensus_mask_path: Path to consensus segmentation mask
        noisy_mask_path: Path to noisy segmentation mask
        output_path: Optional path to save result
        line_width: Contour line width in pixels
    
    Returns:
        Visualization image
    """
    image = load_multichannel_image(image_paths)
    consensus_mask = load_mask(consensus_mask_path)
    noisy_mask = load_mask(noisy_mask_path)
    
    result = overlay_contours(
        image,
        consensus_mask,
        noisy_mask,
        thickness=line_width,
    )
    
    if output_path:
        cv2.imwrite(str(output_path), result)
    
    return result


def default_output_path(
    image_paths: Union[str, List[str]],
    consensus_mask_path: str,
    noisy_mask_path: str,
) -> Path:
    """Derive a default output path next to the masks/images."""
    if isinstance(image_paths, (list, tuple)) and len(image_paths) > 0:
        base_source = Path(image_paths[0])
    elif image_paths:
        base_source = Path(image_paths)
    else:
        base_source = Path(consensus_mask_path)

    return base_source.parent / (
        f"{base_source.stem}_overlay_"
        f"{Path(consensus_mask_path).stem}_vs_{Path(noisy_mask_path).stem}.png"
    )


def parse_image_paths(raw_paths: List[str]) -> Union[str, List[str]]:
    """Use a string for single-channel input, list for multi-channel input."""
    if len(raw_paths) == 1:
        return raw_paths[0]
    return raw_paths


def main(args):
    """Command-line entry point for qualitative segmentation overlays."""
    image_paths = parse_image_paths(args.image_paths)
    output_path = Path(args.output) if args.output else default_output_path(
        image_paths,
        args.consensus_mask,
        args.noisy_mask,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = visualize_segmentations(
        image_paths=image_paths,
        consensus_mask_path=args.consensus_mask,
        noisy_mask_path=args.noisy_mask,
        output_path=str(output_path),
        line_width=args.line_width,
    )

    print(f"Saved qualitative overlay to: {output_path}")
    print(f"Result shape: {result.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay consensus and noisy segmentation contours on an image."
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One image path for grayscale/RGB input, or multiple image paths "
            "for multi-channel input."
        ),
    )
    parser.add_argument(
        "--consensus_mask",
        type=str,
        required=True,
        help="Path to the consensus segmentation mask.",
    )
    parser.add_argument(
        "--noisy_mask",
        type=str,
        required=True,
        help="Path to the noisy segmentation mask.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the overlay image.",
    )
    parser.add_argument(
        "--line_width",
        type=int,
        default=2,
        help="Contour line width in pixels (default: 2).",
    )

    main(parser.parse_args())
