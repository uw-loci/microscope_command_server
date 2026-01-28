"""
Starburst Calibration Workflow.

This module provides the server-side workflow for starburst/sunburst calibration,
which creates a hue-to-angle mapping from a calibration slide with oriented rectangles.

The workflow:
1. Retrieves camera exposure from modality profile (uncrossed/90 deg settings)
2. Sets camera exposure and acquires a single image
3. Runs ppm_library.SunburstCalibrator on the acquired image
4. Saves calibration results (NPZ file and plot PNG)
5. Returns results as JSON for the client
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def run_starburst_calibration(
    hardware,
    config_manager,
    output_folder: str,
    modality: str,
    expected_rectangles: int = 16,
    saturation_threshold: float = 0.1,
    value_threshold: float = 0.1,
    calibration_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run starburst calibration workflow.

    Acquires an image of the calibration slide and runs SunburstCalibrator
    to create a hue-to-angle mapping.

    Args:
        hardware: Hardware interface for camera control
        config_manager: MicroscopeConfigManager instance for accessing modality settings
        output_folder: Directory to save calibration results
        modality: Modality name (e.g., "ppm_20x") for exposure lookup
        expected_rectangles: Number of rectangles expected on calibration slide (default 16)
        saturation_threshold: Minimum saturation for foreground detection (default 0.1)
        value_threshold: Minimum brightness for foreground detection (default 0.1)
        calibration_name: Optional name for calibration files (auto-generated if None)
        logger: Logger instance (creates one if None)

    Returns:
        Dict with results:
            - success: bool
            - r_squared: float (0-1)
            - rectangles_detected: int
            - plot_path: str (path to calibration plot PNG)
            - calibration_path: str (path to calibration NPZ)
            - image_path: str (path to acquired calibration image)
            - warnings: list of warning strings
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    warnings_list = []

    # Generate calibration name if not provided
    if calibration_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibration_name = f"starburst_cal_{timestamp}"

    # Create modality-specific output folder
    output_path = Path(output_folder) / modality
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting starburst calibration for modality: {modality}")
    logger.info(f"Output folder: {output_path}")
    logger.info(f"Expected rectangles: {expected_rectangles}")

    try:
        # Step 1: Get camera exposure from modality profile
        # Use the uncrossed (90 deg) exposure as it provides good signal
        exposure_ms = _get_calibration_exposure(config_manager, modality, logger)
        logger.info(f"Using exposure: {exposure_ms} ms")

        # Step 2: Set camera exposure and acquire image
        logger.info("Setting camera exposure...")
        hardware.set_exposure(exposure_ms)

        logger.info("Acquiring calibration image...")
        image, metadata = hardware.snap_image()

        # Step 3: Save the raw calibration image
        image_filename = f"{calibration_name}_image.tif"
        image_path = output_path / image_filename
        _save_calibration_image(image, image_path, logger)
        logger.info(f"Saved calibration image: {image_path}")

        # Step 4: Run calibration using ppm_library
        logger.info("Running SunburstCalibrator...")
        try:
            from ppm_library.calibration import SunburstCalibrator
        except ImportError as e:
            logger.error(f"Failed to import ppm_library: {e}")
            return {
                "success": False,
                "error": f"ppm_library not available: {e}",
                "warnings": warnings_list,
            }

        calibrator = SunburstCalibrator(
            n_expected_rectangles=expected_rectangles,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
        )

        # Run calibration (without debug_plot since we save it manually)
        result = calibrator.calibrate(str(image_path), debug_plot=False)

        # Check results
        rectangles_detected = len(result.rectangles)
        logger.info(f"Detected {rectangles_detected} rectangles")
        logger.info(f"R-squared: {result.r_squared:.4f}")

        if rectangles_detected != expected_rectangles:
            warning = (
                f"Expected {expected_rectangles} rectangles but found {rectangles_detected}. "
                "Consider repositioning slide or adjusting detection thresholds."
            )
            warnings_list.append(warning)
            logger.warning(warning)

        if result.r_squared < 0.95:
            warning = (
                f"R-squared ({result.r_squared:.4f}) is below 0.95. "
                "Calibration may be inaccurate. Check for slide positioning or detection issues."
            )
            warnings_list.append(warning)
            logger.warning(warning)

        # Step 5: Save calibration file (NPZ)
        calibration_filename = f"{calibration_name}.npz"
        calibration_path = output_path / calibration_filename
        result.save(str(calibration_path))
        logger.info(f"Saved calibration: {calibration_path}")

        # Step 6: Create and save calibration plot
        plot_filename = f"{calibration_name}_plot.png"
        plot_path = output_path / plot_filename
        _create_calibration_plot(image, result, calibrator, plot_path, logger)
        logger.info(f"Saved calibration plot: {plot_path}")

        # Return success result
        return {
            "success": True,
            "r_squared": float(result.r_squared),
            "rectangles_detected": rectangles_detected,
            "plot_path": str(plot_path),
            "calibration_path": str(calibration_path),
            "image_path": str(image_path),
            "warnings": warnings_list,
        }

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {
            "success": False,
            "error": f"File not found: {e}",
            "warnings": warnings_list,
        }
    except ValueError as e:
        logger.error(f"Calibration failed: {e}")
        return {
            "success": False,
            "error": f"Calibration failed: {e}",
            "warnings": warnings_list,
        }
    except Exception as e:
        logger.error(f"Unexpected error during calibration: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Unexpected error: {e}",
            "warnings": warnings_list,
        }


def _get_calibration_exposure(config_manager, modality: str, logger) -> float:
    """
    Get camera exposure for calibration from modality profile.

    Uses the uncrossed (90 deg) exposure from the modality's angle settings,
    which typically provides good signal for calibration.

    Args:
        config_manager: MicroscopeConfigManager instance
        modality: Modality name (e.g., "ppm_20x")
        logger: Logger instance

    Returns:
        Exposure time in milliseconds
    """
    try:
        # Try to get exposure from modality profile
        modality_config = config_manager.get_modality_config(modality)

        if modality_config:
            angles = modality_config.get("angles", [])
            exposures = modality_config.get("exposures_ms", [])

            # Look for 90 degree angle (uncrossed)
            if 90 in angles and len(exposures) > angles.index(90):
                exposure = exposures[angles.index(90)]
                logger.debug(f"Using 90 deg exposure from modality: {exposure} ms")
                return float(exposure)

            # Fall back to first exposure if available
            if exposures:
                exposure = exposures[0]
                logger.debug(f"Using first exposure from modality: {exposure} ms")
                return float(exposure)

        # Default exposure if modality config not found
        logger.warning(
            f"Could not find exposure for modality {modality}, using default 50ms"
        )
        return 50.0

    except Exception as e:
        logger.warning(f"Error getting exposure from config: {e}, using default 50ms")
        return 50.0


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


def _create_calibration_plot(
    image: np.ndarray, result, calibrator, output_path: Path, logger
) -> None:
    """
    Create and save calibration visualization plot.

    Creates a 2x2 plot showing:
    - Original image with detected rectangles
    - Segmentation mask colored by hue
    - Hue vs angle scatter with regression line
    - Color wheel representation

    Args:
        image: Original calibration image
        result: CalibrationResult from SunburstCalibrator
        calibrator: SunburstCalibrator instance
        output_path: Path to save plot
        logger: Logger instance
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for server
        import matplotlib.pyplot as plt
        from skimage import color

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Original image with detected rectangles
        ax1 = axes[0, 0]
        ax1.imshow(image)
        for rect in result.rectangles:
            cy, cx = rect.centroid
            ax1.plot(cx, cy, "ro", markersize=8)
            ax1.annotate(
                f"{rect.angle:.1f} deg",
                (cx, cy),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )
        ax1.set_title(f"Detected Rectangles ({len(result.rectangles)} found)")
        ax1.axis("off")

        # Plot 2: Segmentation mask
        ax2 = axes[0, 1]
        mask_combined = np.zeros(image.shape[:2], dtype=np.float32)
        for rect in result.rectangles:
            if rect.mask is not None:
                mask_combined[rect.mask] = rect.hue_mode
        ax2.imshow(mask_combined, cmap="hsv", vmin=0, vmax=1)
        ax2.set_title("Segmented Regions (colored by hue)")
        ax2.axis("off")

        # Plot 3: Scatter plot of hue vs angle with regression
        ax3 = axes[1, 0]
        ax3.scatter(
            result.hue_values, result.angles, s=100, c="blue", edgecolors="black"
        )

        # Plot regression line
        hue_range = np.linspace(0, 1, 100)
        predicted_angles = result.hue_to_angle(hue_range)
        ax3.plot(
            hue_range,
            predicted_angles,
            "r-",
            linewidth=2,
            label=f"Regression (R^2={result.r_squared:.4f})",
        )

        ax3.set_xlabel("Hue Value [0-1]")
        ax3.set_ylabel("Angle (degrees)")
        ax3.set_title("Hue to Angle Calibration")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 180)

        # Plot 4: Calibration info text
        ax4 = axes[1, 1]
        ax4.axis("off")
        info_text = (
            f"Starburst Calibration Results\n"
            f"{'=' * 35}\n\n"
            f"R-squared: {result.r_squared:.6f}\n"
            f"Rectangles detected: {len(result.rectangles)}\n\n"
            f"Regression (hue -> angle):\n"
            f"  angle = {result.inv_slope:.4f} * hue + {result.inv_intercept:.4f}\n\n"
            f"Regression (angle -> hue):\n"
            f"  hue = {result.slope:.6f} * angle + {result.intercept:.4f}\n\n"
            f"Calibration file saved for use in PPM analysis."
        )
        ax4.text(
            0.1,
            0.9,
            info_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Saved calibration plot to {output_path}")

    except Exception as e:
        logger.error(f"Failed to create calibration plot: {e}")
        # Don't raise - plot is optional
