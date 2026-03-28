"""
SIFT-based feature matching for automated alignment refinement.

Compares a microscope snapshot against a WSI region to find the spatial
offset between predicted and actual stage position. Handles different
pixel sizes between the microscope and WSI by rescaling before matching.
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def match_sift(
    microscope_image: np.ndarray,
    wsi_region: np.ndarray,
    microscope_pixel_size_um: float,
    wsi_pixel_size_um: float,
    flip_x: bool = False,
    flip_y: bool = False,
    min_match_count: int = 10,
    ratio_threshold: float = 0.7,
) -> Optional[Tuple[float, float, int, float]]:
    """
    Match a microscope snapshot to a WSI region using SIFT features.

    Args:
        microscope_image: Image from the microscope camera (BGR or grayscale)
        wsi_region: Region extracted from the WSI around the predicted tile
                    (should be larger than the microscope FOV to allow offset detection)
        microscope_pixel_size_um: Pixel size of microscope image in um/pixel
        wsi_pixel_size_um: Pixel size of the WSI region in um/pixel
        flip_x: Whether the WSI is flipped in X relative to the microscope
        flip_y: Whether the WSI is flipped in Y relative to the microscope
        min_match_count: Minimum SIFT matches required for a valid result
        ratio_threshold: Lowe's ratio test threshold (lower = stricter)

    Returns:
        Tuple of (offset_x_um, offset_y_um, n_inliers, confidence) or None if matching failed.
        Offset is the correction to apply to the stage position:
        stage should move by (offset_x, offset_y) to center on the matching region.
    """
    # Convert to grayscale
    gray_micro = _to_gray(microscope_image)
    gray_wsi = _to_gray(wsi_region)

    # Apply flips to WSI region to match microscope orientation
    if flip_x and flip_y:
        gray_wsi = cv2.flip(gray_wsi, -1)  # Both axes
    elif flip_x:
        gray_wsi = cv2.flip(gray_wsi, 1)  # Horizontal
    elif flip_y:
        gray_wsi = cv2.flip(gray_wsi, 0)  # Vertical

    # Rescale to match pixel sizes
    # If microscope has 0.17 um/px and WSI has 0.25 um/px, WSI pixels are larger
    # Scale WSI to match microscope pixel size
    scale_factor = wsi_pixel_size_um / microscope_pixel_size_um
    if abs(scale_factor - 1.0) > 0.01:
        new_w = int(gray_wsi.shape[1] * scale_factor)
        new_h = int(gray_wsi.shape[0] * scale_factor)
        gray_wsi_scaled = cv2.resize(gray_wsi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        logger.info(
            f"Rescaled WSI region from {gray_wsi.shape[1]}x{gray_wsi.shape[0]} "
            f"to {new_w}x{new_h} (scale={scale_factor:.3f}, "
            f"micro={microscope_pixel_size_um:.4f} um/px, wsi={wsi_pixel_size_um:.4f} um/px)"
        )
    else:
        gray_wsi_scaled = gray_wsi
        logger.info("Pixel sizes match, no rescaling needed")

    logger.info(
        f"SIFT matching: microscope {gray_micro.shape[1]}x{gray_micro.shape[0]} "
        f"vs WSI {gray_wsi_scaled.shape[1]}x{gray_wsi_scaled.shape[0]}"
    )

    # Run SIFT
    sift = cv2.SIFT_create()
    kp_micro, des_micro = sift.detectAndCompute(gray_micro, None)
    kp_wsi, des_wsi = sift.detectAndCompute(gray_wsi_scaled, None)

    logger.info(f"SIFT keypoints: microscope={len(kp_micro)}, WSI={len(kp_wsi)}")

    if des_micro is None or des_wsi is None or len(kp_micro) < 2 or len(kp_wsi) < 2:
        logger.warning("Insufficient keypoints for matching")
        return None

    # Match features using FLANN
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_micro, des_wsi, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    logger.info(f"Good matches after ratio test: {len(good_matches)} (threshold={ratio_threshold})")

    if len(good_matches) < min_match_count:
        logger.warning(
            f"Too few matches ({len(good_matches)}) for reliable alignment "
            f"(minimum {min_match_count})"
        )
        return None

    # Extract matched point coordinates
    src_pts = np.float32([kp_micro[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_wsi[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography with RANSAC to filter outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        logger.warning("Could not find homography")
        return None

    n_inliers = int(mask.sum())
    logger.info(f"Homography found: {n_inliers} inliers out of {len(good_matches)} matches")

    if n_inliers < min_match_count:
        logger.warning(f"Too few inliers ({n_inliers}) for reliable alignment")
        return None

    # Calculate offset: where does the center of the microscope image land in the WSI?
    micro_center = np.float32([[[
        gray_micro.shape[1] / 2.0,
        gray_micro.shape[0] / 2.0
    ]]])
    wsi_center_in_scaled = cv2.perspectiveTransform(micro_center, H)[0][0]

    # The WSI region center (in scaled pixels) is where the stage currently points
    wsi_scaled_center_x = gray_wsi_scaled.shape[1] / 2.0
    wsi_scaled_center_y = gray_wsi_scaled.shape[0] / 2.0

    # Offset in scaled WSI pixels (= microscope pixels since we matched scale)
    offset_px_x = wsi_center_in_scaled[0] - wsi_scaled_center_x
    offset_px_y = wsi_center_in_scaled[1] - wsi_scaled_center_y

    # Convert to microns using microscope pixel size
    offset_um_x = offset_px_x * microscope_pixel_size_um
    offset_um_y = offset_px_y * microscope_pixel_size_um

    # Confidence: inlier ratio
    confidence = n_inliers / len(good_matches) if len(good_matches) > 0 else 0

    logger.info(
        f"SIFT result: offset=({offset_um_x:.1f}, {offset_um_y:.1f}) um, "
        f"({offset_px_x:.1f}, {offset_px_y:.1f}) px, "
        f"inliers={n_inliers}, confidence={confidence:.2f}"
    )

    return (offset_um_x, offset_um_y, n_inliers, confidence)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert image to 8-bit grayscale."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    # Convert to 8-bit if needed
    if gray.dtype == np.uint16:
        gray = (gray / 256).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    return gray
