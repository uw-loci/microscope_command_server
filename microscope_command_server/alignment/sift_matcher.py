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
    min_pixel_size_um: float = 1.0,
    contrast_threshold: float = 0.04,
    nfeatures: int = 0,
    mono_normalization: str = "PERCENTILE",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    clahe_enabled: bool = True,
    clahe_clip_limit: float = 2.0,
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
        min_pixel_size_um: Minimum target pixel size in um. Both images are
            downsampled to at least this resolution. This speeds up matching
            and suppresses JPEG compression artifacts (8x8 block boundaries),
            sensor noise, and other high-frequency artifacts that create
            spurious keypoints. Default 1.0 um/px is sufficient for
            tissue-level structural features.
        mono_normalization: How to convert >8-bit grayscale input to 8-bit.
            "PERCENTILE" (default) stretches [percentile_low, percentile_high]
            to [0, 255] -- best when the camera doesn't use the full bit range
            (typical 12-14 bit cameras). "MIN_MAX" stretches the full data
            range. "BIT_SHIFT" preserves the legacy /256 behaviour for
            cameras that already produce data spanning the full 16-bit range.
        percentile_low / percentile_high: Percentile clip points used when
            mono_normalization == "PERCENTILE". Defaults 2/98 are robust
            against a few saturated pixels.
        clahe_enabled: Apply Contrast-Limited Adaptive Histogram
            Equalisation to both grayscale images before SIFT. This is
            the standard cross-modality robustness trick (e.g. matching
            8-bit H&E vs. monochrome brightfield) and dramatically
            improves keypoint compatibility. Default on.
        clahe_clip_limit: CLAHE clipLimit. Higher = more aggressive
            equalisation. 2.0 is a safe default; raise to 4.0 if matches
            are still scarce.

    Returns:
        Tuple of (offset_x_um, offset_y_um, n_inliers, confidence) or None if matching failed.
        Offset is the correction to apply to the stage position:
        stage should move by (offset_x, offset_y) to center on the matching region.
    """
    # Convert to 8-bit grayscale, normalising as configured.
    gray_micro = _to_gray(
        microscope_image,
        mono_normalization=mono_normalization,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    gray_wsi = _to_gray(
        wsi_region,
        mono_normalization=mono_normalization,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )

    # Cross-modality contrast normalisation. Applied AFTER the per-image
    # to-8-bit conversion so we equalise on the same scale for both.
    if clahe_enabled:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))
        gray_micro = clahe.apply(gray_micro)
        gray_wsi = clahe.apply(gray_wsi)
        logger.info(
            f"Applied CLAHE (clipLimit={clahe_clip_limit}) to both images"
        )

    # Apply flips to WSI region to match microscope orientation
    if flip_x and flip_y:
        gray_wsi = cv2.flip(gray_wsi, -1)  # Both axes
    elif flip_x:
        gray_wsi = cv2.flip(gray_wsi, 1)  # Horizontal
    elif flip_y:
        gray_wsi = cv2.flip(gray_wsi, 0)  # Vertical

    # Downsample both images to a common resolution.
    # Target = max of (lower resolution image, min_pixel_size_um).
    # This ensures:
    #   1. No upscaling (never invent fake detail)
    #   2. Always at least min_pixel_size_um to suppress JPEG block artifacts,
    #      sensor noise, and speed up matching
    target_pixel_size = max(microscope_pixel_size_um, wsi_pixel_size_um, min_pixel_size_um)

    micro_scale = microscope_pixel_size_um / target_pixel_size
    wsi_scale = wsi_pixel_size_um / target_pixel_size

    if micro_scale < 0.99:
        new_w = max(1, int(gray_micro.shape[1] * micro_scale))
        new_h = max(1, int(gray_micro.shape[0] * micro_scale))
        gray_micro = cv2.resize(gray_micro, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(
            f"Downscaled microscope image to {new_w}x{new_h} "
            f"(scale={micro_scale:.3f}, {microscope_pixel_size_um:.4f} -> {target_pixel_size:.4f} um/px)"
        )

    if wsi_scale < 0.99:
        new_w = max(1, int(gray_wsi.shape[1] * wsi_scale))
        new_h = max(1, int(gray_wsi.shape[0] * wsi_scale))
        gray_wsi = cv2.resize(gray_wsi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(
            f"Downscaled WSI region to {new_w}x{new_h} "
            f"(scale={wsi_scale:.3f}, {wsi_pixel_size_um:.4f} -> {target_pixel_size:.4f} um/px)"
        )

    if micro_scale >= 0.99 and wsi_scale >= 0.99:
        logger.info(f"Both images already at or below target resolution ({target_pixel_size:.4f} um/px)")

    # Both images now at target_pixel_size um/px
    gray_wsi_scaled = gray_wsi

    logger.info(
        f"SIFT matching: microscope {gray_micro.shape[1]}x{gray_micro.shape[0]} "
        f"vs WSI {gray_wsi_scaled.shape[1]}x{gray_wsi_scaled.shape[0]} "
        f"(both at {target_pixel_size:.4f} um/px)"
    )

    # Run SIFT with configurable parameters
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
    )
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

    # Offset in matched-resolution pixels (both images at target_pixel_size)
    offset_px_x = wsi_center_in_scaled[0] - wsi_scaled_center_x
    offset_px_y = wsi_center_in_scaled[1] - wsi_scaled_center_y

    # Convert to microns using the common target pixel size
    offset_um_x = offset_px_x * target_pixel_size
    offset_um_y = offset_px_y * target_pixel_size

    # Confidence: inlier ratio
    confidence = n_inliers / len(good_matches) if len(good_matches) > 0 else 0

    logger.info(
        f"SIFT result: offset=({offset_um_x:.1f}, {offset_um_y:.1f}) um, "
        f"({offset_px_x:.1f}, {offset_px_y:.1f}) px, "
        f"inliers={n_inliers}, confidence={confidence:.2f}"
    )

    return (offset_um_x, offset_um_y, n_inliers, confidence)


def _to_gray(
    image: np.ndarray,
    mono_normalization: str = "PERCENTILE",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> np.ndarray:
    """Convert image to 8-bit grayscale with configurable normalization.

    For multi-channel input (typical 8-bit RGB H&E), this collapses to
    luminance via OpenCV. For single-channel input above 8-bit (typical
    12-14 bit camera packed in 16-bit container), this rescales to 8-bit
    using the requested mode:

    - PERCENTILE (default): clip to [percentile_low, percentile_high] of
      the actual data, then linearly stretch to [0, 255]. Robust to a
      few saturated pixels and to cameras that don't use the full bit
      range (the common case). This is the right default for SIFT.
    - MIN_MAX: linear stretch from min to max of the image.
    - BIT_SHIFT: legacy /256 behaviour. Use only when the camera is
      known to span the full 16-bit range.
    """
    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    if gray.dtype == np.uint8:
        return gray

    # Need to compress to 8-bit. The naive /256 collapses dynamic range
    # whenever the source doesn't use the full 16-bit range (very common
    # on cameras that output 12 or 14 effective bits packed into uint16).
    mode = (mono_normalization or "PERCENTILE").upper()

    if mode == "BIT_SHIFT":
        if gray.dtype == np.uint16:
            out = (gray / 256).astype(np.uint8)
        else:
            out = gray.astype(np.uint8)
        logger.info(
            f"_to_gray: BIT_SHIFT applied (input dtype={gray.dtype}, "
            f"min={int(gray.min())}, max={int(gray.max())})"
        )
        return out

    if mode == "MIN_MAX":
        lo = float(gray.min())
        hi = float(gray.max())
    else:  # PERCENTILE (default)
        lo = float(np.percentile(gray, max(0.0, min(percentile_low, 100.0))))
        hi = float(np.percentile(gray, max(0.0, min(percentile_high, 100.0))))

    if hi <= lo:
        # Degenerate (flat) image -- avoid divide-by-zero, return zeros.
        logger.warning(
            f"_to_gray: degenerate range lo={lo}, hi={hi} (input dtype={gray.dtype}); "
            f"returning zero image"
        )
        return np.zeros(gray.shape, dtype=np.uint8)

    scaled = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    out = np.clip(scaled, 0, 255).astype(np.uint8)
    logger.info(
        f"_to_gray: mode={mode}, input dtype={gray.dtype} range=[{int(gray.min())},{int(gray.max())}], "
        f"clip=[{lo:.0f},{hi:.0f}], output 8-bit range=[{int(out.min())},{int(out.max())}]"
    )
    return out
