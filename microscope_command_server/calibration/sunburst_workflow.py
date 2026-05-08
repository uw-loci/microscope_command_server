"""
Sunburst Calibration Workflow.

This module provides the server-side workflow for PPM reference slide calibration,
which creates a hue-to-angle mapping using radial spoke sampling.

The workflow:
1. Acquires a single image using the current camera settings
2. Runs ppm_library.RadialCalibrator on the acquired image
3. Saves calibration results (NPZ file and plot PNG)
4. Returns results as JSON for the client

The user is expected to configure camera exposure and white balance via the
Camera Control dialog before running calibration. This workflow does NOT
modify any camera settings -- it acquires with whatever is currently set.

All calibration logic (plotting, quality checking, mask generation) lives in
ppm_library. This module is a thin hardware orchestrator.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def run_sunburst_calibration(
    hardware,
    config_manager,
    output_folder: str,
    modality: str,
    expected_spokes: int = 16,
    saturation_threshold: float = 0.1,
    value_threshold: float = 0.1,
    calibration_name: Optional[str] = None,
    radius_inner: int = 30,
    radius_outer: int = 150,
    logger: Optional[logging.Logger] = None,
    existing_image_path: Optional[str] = None,
    center: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Run sunburst calibration workflow.

    Acquires an image of the calibration slide using the current camera
    settings and runs RadialCalibrator to create a hue-to-angle mapping.

    Camera exposure and white balance should be configured by the user
    via the Camera Control dialog before calling this function. No camera
    settings are modified by this workflow.

    When existing_image_path is provided, skips image acquisition and
    loads the specified image for re-analysis. When center is provided,
    passes it to RadialCalibrator to bypass auto-detection.

    Args:
        hardware: Hardware interface for camera control
        config_manager: MicroscopeConfigManager instance for accessing modality settings
        output_folder: Directory to save calibration results (files saved directly here)
        modality: Modality name (e.g., "ppm_20x") for logging
        expected_spokes: Number of spokes in the sunburst pattern (default 16)
        saturation_threshold: Minimum saturation for foreground detection (default 0.1)
        value_threshold: Minimum brightness for foreground detection (default 0.1)
        calibration_name: Optional name for calibration files (auto-generated if None)
        radius_inner: Inner sampling radius in pixels from center (default 30)
        radius_outer: Outer sampling radius in pixels from center (default 150)
        logger: Logger instance (creates one if None)
        existing_image_path: Optional path to existing image (skips acquisition)
        center: Optional (y, x) center coordinates for manual override

    Returns:
        Dict with results:
            - success: bool
            - r_squared: float (0-1)
            - spokes_detected: int
            - center: list [y, x] center coordinates
            - plot_path: str (path to calibration plot PNG)
            - calibration_path: str (path to calibration NPZ)
            - image_path: str (path to acquired calibration image)
            - warnings: list of warning strings
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    warnings_list = []
    image_path = None
    mask_path = None

    # Generate calibration name if not provided
    if calibration_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_retry" if existing_image_path is not None else ""
        calibration_name = f"sunburst_cal_{timestamp}{suffix}"

    # Use output folder directly (no modality subfolder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting radial calibration for modality: {modality}")
    logger.info(f"Output folder: {output_path}")
    logger.info(f"Number of spokes: {expected_spokes}")
    logger.info(f"Radial sampling: inner={radius_inner}px, outer={radius_outer}px")

    try:
        if existing_image_path is not None:
            # Reuse existing image - skip acquisition
            logger.info(f"Reusing existing image: {existing_image_path}")
            if center is not None:
                logger.info(f"Using manually specified center: y={center[0]}, x={center[1]}")

            try:
                import tifffile

                image = tifffile.imread(str(existing_image_path))
            except ImportError:
                from PIL import Image as PILImage

                image = np.array(PILImage.open(str(existing_image_path)))

            image_path = Path(existing_image_path)
            extra_info = None
        else:
            # Read current camera settings for reporting (do NOT modify them)
            extra_info = _read_current_camera_settings(hardware, logger)

            # Acquire with current camera settings as-is
            logger.info("Acquiring calibration image with current camera settings...")
            image, metadata = hardware.snap_image()

            # Save the raw calibration image
            image_filename = f"{calibration_name}_image.tif"
            image_path = output_path / image_filename
            _save_calibration_image(image, image_path, logger)
            logger.info(f"Saved calibration image: {image_path}")

        # === Library calls: all calibration logic ===
        logger.info("Running RadialCalibrator...")
        try:
            from ppm_library.calibration import RadialCalibrator
        except ImportError as e:
            logger.error(f"Failed to import ppm_library: {e}")
            return {
                "success": False,
                "error": f"ppm_library not available: {e}",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "warnings": warnings_list,
            }

        calibrator = RadialCalibrator(
            n_spokes=expected_spokes,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
            radius_inner=radius_inner,
            radius_outer=radius_outer,
        )

        # Save detection mask for troubleshooting (before calibration attempt)
        mask_filename = f"{calibration_name}_mask.png"
        mask_path = output_path / mask_filename
        try:
            calibrator.save_detection_mask(image, mask_path)
            logger.info(f"Saved detection mask: {mask_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug mask: {e}")

        # Run calibration
        result = calibrator.calibrate(str(image_path), center=center, debug_plot=False)

        # Log results
        spokes_detected = len(result.samples)
        logger.info(f"Detected {spokes_detected} spokes")
        logger.info(f"R-squared: {result.r_squared:.4f}")
        logger.info(f"Center: y={result.center[0]}, x={result.center[1]}")
        logger.info(f"Hue offset: {result.hue_offset:.4f}")

        # Quality checking via library
        quality_warnings = result.check_quality(expected_spokes)
        for w in quality_warnings:
            warnings_list.append(w)
            logger.warning(w)

        # Include any warnings from the calibrator itself (e.g., saturation)
        if result.warnings:
            for w in result.warnings:
                warnings_list.append(w)
                logger.warning(f"Calibrator warning: {w}")

        # Save calibration file (NPZ)
        calibration_filename = f"{calibration_name}.npz"
        calibration_path = output_path / calibration_filename
        result.save(str(calibration_path))
        logger.info(f"Saved calibration: {calibration_path}")

        # Save calibration plot via library
        plot_filename = f"{calibration_name}_plot.png"
        plot_path = output_path / plot_filename
        try:
            result.save_plot(plot_path, image, calibrator, extra_info=extra_info)
            logger.info(f"Saved calibration plot: {plot_path}")
        except Exception as e:
            logger.error(f"Failed to create calibration plot: {e}")

        # Return success result
        return {
            "success": True,
            "r_squared": float(result.r_squared),
            "spokes_detected": spokes_detected,
            "center": [int(result.center[0]), int(result.center[1])],
            "hue_offset": float(result.hue_offset),
            "plot_path": str(plot_path),
            "calibration_path": str(calibration_path),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "warnings": warnings_list,
        }

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {
            "success": False,
            "error": f"File not found: {e}",
            "image_path": str(image_path) if image_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "warnings": warnings_list,
        }
    except ValueError as e:
        logger.error(f"Calibration failed: {e}")
        return {
            "success": False,
            "error": f"Calibration failed: {e}",
            "image_path": str(image_path) if image_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "warnings": warnings_list,
        }
    except Exception as e:
        logger.error(f"Unexpected error during calibration: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Unexpected error: {e}",
            "image_path": str(image_path) if image_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "warnings": warnings_list,
        }


def _read_current_camera_settings(hardware, logger) -> Dict[str, str]:
    """
    Read the current camera settings for reporting in the calibration plot.

    Does NOT modify any camera settings. Reads whatever the user has
    configured via the Camera Control dialog.

    For JAI cameras, reads per-channel exposures and gain.
    For other cameras, reads the unified exposure.

    Args:
        hardware: Hardware interface with core property access
        logger: Logger instance

    Returns:
        Dict of display-only strings for the calibration plot info panel,
        e.g. {"Exposure R": "50.0 ms", "Exposure G": "40.0 ms", ...}
    """
    extra_info = {}

    try:
        camera = None
        try:
            camera = hardware.get_camera_name()
        except Exception:
            pass

        cam = hardware.camera
        if cam.supports_per_channel_exposure():
            exposures = cam.get_channel_exposures()
            extra_info["Exposure R"] = f"{exposures['red']:.1f} ms"
            extra_info["Exposure G"] = f"{exposures['green']:.1f} ms"
            extra_info["Exposure B"] = f"{exposures['blue']:.1f} ms"

            gain = cam.get_unified_gain()
            if gain > 1.0:
                extra_info["Unified Gain"] = f"{gain:.2f}x"

            logger.info(
                f"Current camera settings: "
                f"R={exposures['red']:.1f}ms, "
                f"G={exposures['green']:.1f}ms, "
                f"B={exposures['blue']:.1f}ms, "
                f"gain={gain:.2f}x"
            )
        else:
            exposure_ms = hardware.get_exposure()
            extra_info["Exposure"] = f"{exposure_ms:.1f} ms"
            if camera:
                extra_info["Camera"] = camera
            logger.info(f"Current camera exposure: {exposure_ms:.1f} ms")

    except Exception as e:
        logger.warning(f"Could not read current camera settings: {e}")

    return extra_info


def _save_calibration_image(image: np.ndarray, path: Path, logger) -> None:
    """
    Save calibration image as TIFF.

    Args:
        image: Image array (H, W, C) or (H, W)
        path: Output path
        logger: Logger instance
    """
    try:
        import tifffile

        tifffile.imwrite(str(path), image)
    except ImportError:
        # Fall back to PIL
        try:
            from PIL import Image as PILImage

            if image.ndim == 3 and image.shape[2] == 3:
                pil_img = PILImage.fromarray(image, mode="RGB")
            else:
                pil_img = PILImage.fromarray(image)
            pil_img.save(str(path))
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise
