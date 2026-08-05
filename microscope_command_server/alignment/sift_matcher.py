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
    coarse_pixel_size_um: float = 0.0,
    coarse_to_fine_enabled: bool = False,
    rgb_conversion: str = "GREEN",
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
        coarse_pixel_size_um: Target resolution (um/px) for the coarse pass
            of a coarse-to-fine search. Only used when
            coarse_to_fine_enabled is True and this value is coarser than
            the fine target (max of microscope/WSI/min_pixel_size). Typical
            4.0 (vs. a 1.0 fine target).
        coarse_to_fine_enabled: When True, run a heavily-downsampled SIFT
            pass over the whole WSI region first to find a rough offset,
            then crop both images to a small window around that offset and
            re-run SIFT at full (fine) resolution for precision. This lets
            the caller enlarge the search area (a bigger WSI region) without
            paying the cost of running full-resolution SIFT over the whole
            region. Falls back to the single full-resolution pass if the
            coarse pass finds no match.
        rgb_conversion: How to collapse a colour (RGB, e.g. H&E) input to a
            single channel before matching. Only affects 3/4-channel images --
            the monochrome microscope snapshot is untouched.
            "GREEN" (default) takes the green channel: for H&E both
            hematoxylin (nuclei) and eosin (cytoplasm) absorb green strongly,
            so green carries the most tissue structure AND keeps the same
            intensity polarity (tissue dark on a bright transmitted-light
            background) as a brightfield monochrome camera, which is exactly
            what makes the two commensurate for SIFT. "LUMINANCE" is the
            legacy cv2.COLOR_BGR2GRAY weighting (0.299R+0.587G+0.114B); its
            red term brightens eosin and washes out cytoplasm contrast, so it
            matches an H&E scan against a mono brightfield camera poorly.
            NOTE: do NOT use an absorbance/optical-density conversion here --
            it inverts polarity (tissue bright) and its gradients run opposite
            to the intensity microscope image, which breaks descriptor
            matching. Absorbance would only be valid if BOTH images were
            converted, which this path does not do.

    Returns:
        Tuple of (offset_x_um, offset_y_um, n_inliers, confidence) or None if matching failed.
        Offset is the correction to apply to the stage position:
        stage should move by (offset_x, offset_y) to center on the matching region.
    """
    # Reject a non-positive / non-finite microscope pixel size before it can
    # drive the resolution scaling. Micro-Manager returns 0.0 um/px when no
    # pixel-size calibration is bound to the active objective; with a 0 (or NaN)
    # here the scale ratio below collapses the microscope image toward 1x1 px,
    # producing zero keypoints and a misleading "insufficient features" result.
    # Fail loudly instead so the caller reports the real cause. (The Java client
    # also guards this, but the server must not silently degrade.)
    if not np.isfinite(microscope_pixel_size_um) or microscope_pixel_size_um <= 0:
        logger.error(
            "SIFT: invalid microscope_pixel_size_um=%r (Micro-Manager pixel-size "
            "calibration missing for the active objective); cannot scale images",
            microscope_pixel_size_um,
        )
        return None
    if not np.isfinite(wsi_pixel_size_um) or wsi_pixel_size_um <= 0:
        logger.error("SIFT: invalid wsi_pixel_size_um=%r; cannot scale images", wsi_pixel_size_um)
        return None

    # Convert to 8-bit grayscale, normalising as configured.
    gray_micro = _to_gray(
        microscope_image,
        mono_normalization=mono_normalization,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        rgb_conversion=rgb_conversion,
    )
    gray_wsi = _to_gray(
        wsi_region,
        mono_normalization=mono_normalization,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        rgb_conversion=rgb_conversion,
    )

    # Cross-modality contrast normalisation. Applied AFTER the per-image
    # to-8-bit conversion so we equalise on the same scale for both.
    if clahe_enabled:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))
        gray_micro = clahe.apply(gray_micro)
        gray_wsi = clahe.apply(gray_wsi)
        logger.info(f"Applied CLAHE (clipLimit={clahe_clip_limit}) to both images")

    # Apply flips to WSI region to match microscope orientation
    if flip_x and flip_y:
        gray_wsi = cv2.flip(gray_wsi, -1)  # Both axes
    elif flip_x:
        gray_wsi = cv2.flip(gray_wsi, 1)  # Horizontal
    elif flip_y:
        gray_wsi = cv2.flip(gray_wsi, 0)  # Vertical

    # gray_micro / gray_wsi are now 8-bit, CLAHE-equalised, and the WSI is
    # flipped to the microscope orientation. All downstream matching operates
    # in this common (flipped) full-resolution space, so offsets compose
    # cleanly across a coarse-to-fine search.
    fine_target = max(microscope_pixel_size_um, wsi_pixel_size_um, min_pixel_size_um)

    if coarse_to_fine_enabled and coarse_pixel_size_um > fine_target:
        result = _match_coarse_to_fine(
            gray_micro,
            gray_wsi,
            microscope_pixel_size_um,
            wsi_pixel_size_um,
            fine_target,
            coarse_pixel_size_um,
            min_match_count=min_match_count,
            ratio_threshold=ratio_threshold,
            contrast_threshold=contrast_threshold,
            nfeatures=nfeatures,
        )
        if result is not None:
            return result
        logger.info("Coarse-to-fine found no match; falling back to single full-resolution pass")

    return _match_at_resolution(
        gray_micro,
        gray_wsi,
        microscope_pixel_size_um,
        wsi_pixel_size_um,
        fine_target,
        min_match_count=min_match_count,
        ratio_threshold=ratio_threshold,
        contrast_threshold=contrast_threshold,
        nfeatures=nfeatures,
    )


def _match_at_resolution(
    gray_micro: np.ndarray,
    gray_wsi: np.ndarray,
    microscope_pixel_size_um: float,
    wsi_pixel_size_um: float,
    target_pixel_size: float,
    min_match_count: int,
    ratio_threshold: float,
    contrast_threshold: float,
    nfeatures: int,
) -> Optional[Tuple[float, float, int, float]]:
    """Downsample both (already gray/CLAHE/flipped) images to target_pixel_size
    and run a single SIFT match.

    Returns (offset_um_x, offset_um_y, n_inliers, confidence) measured at the
    centre of gray_wsi, or None if matching failed. Inputs are not mutated.
    """
    micro_scale = microscope_pixel_size_um / target_pixel_size
    wsi_scale = wsi_pixel_size_um / target_pixel_size

    if micro_scale < 0.99:
        # Floor each dimension at a SIFT-meaningful minimum (16 px) so a
        # degenerate scale can never collapse the image to a keypoint-free
        # sliver. A no-op for normal downscales; the invalid-pixel-size guard in
        # match_sift already rejects the 0 um/px case that produced 1x1 before.
        new_w = max(16, int(gray_micro.shape[1] * micro_scale))
        new_h = max(16, int(gray_micro.shape[0] * micro_scale))
        gray_micro = cv2.resize(gray_micro, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(
            f"Downscaled microscope image to {new_w}x{new_h} "
            f"(scale={micro_scale:.3f}, {microscope_pixel_size_um:.4f} -> {target_pixel_size:.4f} um/px)"
        )

    if wsi_scale < 0.99:
        new_w = max(16, int(gray_wsi.shape[1] * wsi_scale))
        new_h = max(16, int(gray_wsi.shape[0] * wsi_scale))
        gray_wsi = cv2.resize(gray_wsi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(
            f"Downscaled WSI region to {new_w}x{new_h} "
            f"(scale={wsi_scale:.3f}, {wsi_pixel_size_um:.4f} -> {target_pixel_size:.4f} um/px)"
        )

    if micro_scale >= 0.99 and wsi_scale >= 0.99:
        logger.info(
            f"Both images already at or below target resolution ({target_pixel_size:.4f} um/px)"
        )

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
    index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
    search_params = {"checks": 50}
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
    micro_center = np.float32([[[gray_micro.shape[1] / 2.0, gray_micro.shape[0] / 2.0]]])
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


def _match_coarse_to_fine(
    gray_micro: np.ndarray,
    gray_wsi: np.ndarray,
    microscope_pixel_size_um: float,
    wsi_pixel_size_um: float,
    fine_target: float,
    coarse_pixel_size_um: float,
    min_match_count: int,
    ratio_threshold: float,
    contrast_threshold: float,
    nfeatures: int,
) -> Optional[Tuple[float, float, int, float]]:
    """Two-stage SIFT: a coarse pass over the whole region to localise the
    microscope FOV, then a full-resolution pass over a small crop for
    precision. Inputs are the common gray/CLAHE/flipped full-resolution
    images. Returns the composed offset (microns, relative to the centre of
    gray_wsi) or None if the coarse pass fails.
    """
    coarse_target = max(microscope_pixel_size_um, wsi_pixel_size_um, coarse_pixel_size_um)
    logger.info(
        f"Coarse-to-fine: coarse pass at {coarse_target:.3f} um/px over full region, "
        f"fine pass at {fine_target:.3f} um/px"
    )

    coarse = _match_at_resolution(
        gray_micro,
        gray_wsi,
        microscope_pixel_size_um,
        wsi_pixel_size_um,
        coarse_target,
        min_match_count=min_match_count,
        ratio_threshold=ratio_threshold,
        contrast_threshold=contrast_threshold,
        nfeatures=nfeatures,
    )
    if coarse is None:
        return None

    coarse_off_x_um, coarse_off_y_um, _, _ = coarse

    h_wsi, w_wsi = gray_wsi.shape[:2]
    wsi_cx = w_wsi / 2.0
    wsi_cy = h_wsi / 2.0

    # Where the coarse pass says the microscope centre lands, in full-res WSI px.
    p_x = wsi_cx + coarse_off_x_um / wsi_pixel_size_um
    p_y = wsi_cy + coarse_off_y_um / wsi_pixel_size_um

    # Fine crop: microscope FOV plus a half-FOV of slack on each side, to
    # absorb coarse-pass error while staying small relative to the full region.
    micro_fov_w_um = gray_micro.shape[1] * microscope_pixel_size_um
    micro_fov_h_um = gray_micro.shape[0] * microscope_pixel_size_um
    slack_um = max(micro_fov_w_um, micro_fov_h_um) * 0.5
    half_w_px = (micro_fov_w_um / 2.0 + slack_um) / wsi_pixel_size_um
    half_h_px = (micro_fov_h_um / 2.0 + slack_um) / wsi_pixel_size_um

    x0 = int(max(0, round(p_x - half_w_px)))
    y0 = int(max(0, round(p_y - half_h_px)))
    x1 = int(min(w_wsi, round(p_x + half_w_px)))
    y1 = int(min(h_wsi, round(p_y + half_h_px)))

    if x1 - x0 < 4 or y1 - y0 < 4:
        logger.info("Coarse-to-fine: degenerate fine crop, using coarse result")
        return coarse

    crop = gray_wsi[y0:y1, x0:x1]
    crop_cx = (x0 + x1) / 2.0
    crop_cy = (y0 + y1) / 2.0
    logger.info(
        f"Coarse-to-fine: coarse offset=({coarse_off_x_um:.1f}, {coarse_off_y_um:.1f}) um; "
        f"fine crop {crop.shape[1]}x{crop.shape[0]} px at WSI ({x0},{y0})-({x1},{y1})"
    )

    fine = _match_at_resolution(
        gray_micro,
        crop,
        microscope_pixel_size_um,
        wsi_pixel_size_um,
        fine_target,
        min_match_count=min_match_count,
        ratio_threshold=ratio_threshold,
        contrast_threshold=contrast_threshold,
        nfeatures=nfeatures,
    )
    if fine is None:
        logger.info("Coarse-to-fine: fine pass failed, using coarse result")
        return coarse

    fine_off_x_um, fine_off_y_um, n_inliers, confidence = fine

    # Compose: the fine offset is relative to the crop centre; shift it back to
    # the full-region centre. (Handles a clamped crop where crop centre != p.)
    final_off_x_um = (crop_cx - wsi_cx) * wsi_pixel_size_um + fine_off_x_um
    final_off_y_um = (crop_cy - wsi_cy) * wsi_pixel_size_um + fine_off_y_um

    logger.info(
        f"Coarse-to-fine result: offset=({final_off_x_um:.1f}, {final_off_y_um:.1f}) um, "
        f"inliers={n_inliers}, confidence={confidence:.2f}"
    )

    return (final_off_x_um, final_off_y_um, n_inliers, confidence)


def _to_gray(
    image: np.ndarray,
    mono_normalization: str = "PERCENTILE",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    rgb_conversion: str = "GREEN",
) -> np.ndarray:
    """Convert image to 8-bit grayscale with configurable normalization.

    For multi-channel input (typical 8-bit RGB H&E), this collapses to a
    single channel per ``rgb_conversion`` ("GREEN" default, or "LUMINANCE"
    for the legacy BGR2GRAY weighting). For single-channel input above 8-bit
    (typical 12-14 bit camera packed in 16-bit container), this rescales to
    8-bit using the requested mode:

    - PERCENTILE (default): clip to [percentile_low, percentile_high] of
      the actual data, then linearly stretch to [0, 255]. Robust to a
      few saturated pixels and to cameras that don't use the full bit
      range (the common case). This is the right default for SIFT.
    - MIN_MAX: linear stretch from min to max of the image.
    - BIT_SHIFT: legacy /256 behaviour. Use only when the camera is
      known to span the full 16-bit range.
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        conv = (rgb_conversion or "GREEN").upper()
        if conv == "GREEN":
            # OpenCV loads colour as BGR; index 1 is the green channel. Green
            # maximises H&E tissue structure and keeps intensity polarity
            # (tissue dark) commensurate with a brightfield mono camera.
            gray = image[:, :, 1]
            logger.info("_to_gray: RGB->green-channel conversion")
        else:  # LUMINANCE (legacy)
            code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, code)
            logger.info("_to_gray: RGB->luminance (BGR2GRAY) conversion")
    elif image.ndim == 3:
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
        out = (gray / 256).astype(np.uint8) if gray.dtype == np.uint16 else gray.astype(np.uint8)
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
