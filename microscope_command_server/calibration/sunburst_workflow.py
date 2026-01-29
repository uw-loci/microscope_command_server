"""
Sunburst Calibration Workflow.

This module provides the server-side workflow for PPM reference slide calibration,
which creates a hue-to-angle mapping using radial spoke sampling.

The workflow:
1. Retrieves camera exposure from modality profile (uncrossed/90 deg settings)
2. Sets camera exposure and acquires a single image
3. Runs ppm_library.RadialCalibrator on the acquired image
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


def run_sunburst_calibration(
    hardware,
    config_manager,
    output_folder: str,
    modality: str,
    expected_rectangles: int = 16,
    saturation_threshold: float = 0.1,
    value_threshold: float = 0.1,
    calibration_name: Optional[str] = None,
    radius_inner: int = 30,
    radius_outer: int = 150,
    rotation_search_degrees: float = 5.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run sunburst calibration workflow.

    Acquires an image of the calibration slide and runs RadialCalibrator
    to create a hue-to-angle mapping using radial spoke sampling.

    Args:
        hardware: Hardware interface for camera control
        config_manager: MicroscopeConfigManager instance for accessing modality settings
        output_folder: Directory to save calibration results
        modality: Modality name (e.g., "ppm_20x") for exposure lookup
        expected_rectangles: Number of spokes in the sunburst pattern (default 16)
        saturation_threshold: Minimum saturation for foreground detection (default 0.1)
        value_threshold: Minimum brightness for foreground detection (default 0.1)
        calibration_name: Optional name for calibration files (auto-generated if None)
        radius_inner: Inner sampling radius in pixels from center (default 30)
        radius_outer: Outer sampling radius in pixels from center (default 150)
        rotation_search_degrees: Search range +/- degrees for spoke alignment (default 5.0)
        logger: Logger instance (creates one if None)

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
        calibration_name = f"sunburst_cal_{timestamp}"

    # Create modality-specific output folder
    output_path = Path(output_folder) / modality
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting radial calibration for modality: {modality}")
    logger.info(f"Output folder: {output_path}")
    logger.info(f"Number of spokes: {expected_rectangles}")
    logger.info(f"Radial sampling: inner={radius_inner}px, outer={radius_outer}px")
    logger.info(f"Rotation search: +/- {rotation_search_degrees} deg")

    try:
        # Step 1: Get camera exposure from hardware settings
        # Use the uncrossed (90 deg) exposure as it provides good signal
        exposure_config = _get_calibration_exposures(hardware, modality, logger)

        # Step 2: Set camera exposure based on camera type
        logger.info("Setting camera exposure...")

        if exposure_config.get('per_channel', False):
            # JAI camera with per-channel exposures
            try:
                from microscope_control.jai import JAICameraProperties
                jai_props = JAICameraProperties(hardware.core)
                jai_props.set_channel_exposures(
                    red=exposure_config['r'],
                    green=exposure_config['g'],
                    blue=exposure_config['b'],
                    auto_enable=True
                )
                logger.info(f"Set per-channel exposures: R={exposure_config['r']}, "
                           f"G={exposure_config['g']}, B={exposure_config['b']}")
            except ImportError:
                # Fall back to unified exposure if JAI module not available
                avg_exposure = (exposure_config['r'] + exposure_config['g'] + exposure_config['b']) / 3
                logger.warning(f"JAI module not available, using average exposure: {avg_exposure} ms")
                hardware.set_exposure(avg_exposure)
        else:
            # Unified exposure for non-JAI cameras
            exposure_ms = exposure_config.get('exposure_ms', 100.0)
            logger.info(f"Using unified exposure: {exposure_ms} ms")
            hardware.set_exposure(exposure_ms)

        logger.info("Acquiring calibration image...")
        image, metadata = hardware.snap_image()

        # Step 3: Save the raw calibration image
        image_filename = f"{calibration_name}_image.tif"
        image_path = output_path / image_filename
        _save_calibration_image(image, image_path, logger)
        logger.info(f"Saved calibration image: {image_path}")

        # Step 4: Save debug mask for troubleshooting (before calibration attempt)
        # This helps users understand what the thresholds are detecting
        mask_filename = f"{calibration_name}_mask.png"
        mask_path = output_path / mask_filename
        _save_debug_mask(image, saturation_threshold, value_threshold, mask_path, logger)
        logger.info(f"Saved detection mask: {mask_path}")

        # Step 5: Run calibration using ppm_library RadialCalibrator
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
            n_spokes=expected_rectangles,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            rotation_search_degrees=rotation_search_degrees,
        )

        # Run calibration (without debug_plot since we save it manually)
        result = calibrator.calibrate(str(image_path), debug_plot=False)

        # Check results
        spokes_detected = len(result.samples)
        logger.info(f"Detected {spokes_detected} spokes")
        logger.info(f"R-squared: {result.r_squared:.4f}")
        logger.info(f"Center: y={result.center[0]}, x={result.center[1]}")
        logger.info(f"Hue offset: {result.hue_offset:.4f}")

        if spokes_detected < expected_rectangles:
            warning = (
                f"Expected {expected_rectangles} spokes but found {spokes_detected}. "
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

        # Include any warnings from the calibrator itself (e.g., saturation warnings)
        if result.warnings:
            for w in result.warnings:
                warnings_list.append(w)
                logger.warning(f"Calibrator warning: {w}")

        # Step 6: Save calibration file (NPZ)
        calibration_filename = f"{calibration_name}.npz"
        calibration_path = output_path / calibration_filename
        result.save(str(calibration_path))
        logger.info(f"Saved calibration: {calibration_path}")

        # Step 7: Create and save calibration plot
        plot_filename = f"{calibration_name}_plot.png"
        plot_path = output_path / plot_filename
        _create_calibration_plot(image, result, calibrator, plot_path, logger)
        logger.info(f"Saved calibration plot: {plot_path}")

        # Return success result
        return {
            "success": True,
            "r_squared": float(result.r_squared),
            "spokes_detected": spokes_detected,
            "rectangles_detected": spokes_detected,  # backward compat for Java client
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


def _get_calibration_exposures(hardware, modality: str, logger) -> Dict[str, Any]:
    """
    Get camera exposure settings for calibration from hardware settings.

    Looks up exposure settings from hardware.settings which contains
    the loaded configuration. Uses 90-degree (uncrossed) exposure if
    available, otherwise uses a sensible default.

    For JAI cameras, returns per-channel (R/G/B) exposures.
    For other cameras, returns a single unified exposure.

    Args:
        hardware: Hardware interface with settings dictionary
        modality: Modality name (e.g., "ppm_20x", "ppm")
        logger: Logger instance

    Returns:
        Dict with keys:
            - 'per_channel': bool - True if per-channel exposures
            - 'r', 'g', 'b': floats - per-channel exposures (if per_channel=True)
            - 'exposure_ms': float - unified exposure (if per_channel=False)
    """
    default_exposure = 100.0  # Sensible default for calibration

    try:
        settings = getattr(hardware, 'settings', None)
        if settings is None:
            logger.warning("No settings available in hardware, using default exposure")
            return {'per_channel': False, 'exposure_ms': default_exposure}

        # Check camera type
        camera = None
        try:
            camera = hardware.core.get_property("Core", "Camera")
        except Exception:
            pass
        is_jai = camera == "JAICamera"

        # Extract base modality (e.g., "ppm" from "ppm_20x")
        base_modality = modality.split("_")[0] if "_" in modality else modality

        # Get current objective and detector from settings
        objective_id = settings.get("id_objective")
        detector_id = settings.get("id_detector")

        # If we have a specific detector, extract its ID
        if isinstance(detector_id, dict):
            # Use the first detector ID found
            detector_id = list(detector_id.keys())[0] if detector_id else None

        logger.info(f"Looking up exposures for modality={base_modality}, "
                   f"objective={objective_id}, detector={detector_id}, camera={camera}")

        # Check imaging_profiles for per-channel exposures (PPM format)
        imaging_profiles = settings.get("imaging_profiles", {})
        modality_profiles = imaging_profiles.get(base_modality, {})

        # Try to find profile for current objective
        profile = None
        if objective_id and objective_id in modality_profiles:
            obj_profiles = modality_profiles[objective_id]
            if detector_id and detector_id in obj_profiles:
                profile = obj_profiles[detector_id]
            elif obj_profiles:
                # Use first detector profile
                profile = list(obj_profiles.values())[0]
        elif modality_profiles:
            # Use first objective's first detector
            first_obj = list(modality_profiles.values())[0]
            if isinstance(first_obj, dict) and first_obj:
                profile = list(first_obj.values())[0]

        if profile and "exposures_ms" in profile:
            exposures_config = profile["exposures_ms"]

            # PPM has angle-based exposures: negative, crossed, positive, uncrossed
            # For calibration, use "uncrossed" (90 deg parallel polars - brightest)
            if "uncrossed" in exposures_config:
                uncrossed = exposures_config["uncrossed"]

                # Check for per-channel exposures
                if isinstance(uncrossed, dict):
                    if all(k in uncrossed for k in ['r', 'g', 'b']):
                        logger.info(f"Using per-channel uncrossed exposures: "
                                   f"R={uncrossed['r']}, G={uncrossed['g']}, B={uncrossed['b']}")
                        return {
                            'per_channel': True,
                            'r': float(uncrossed['r']),
                            'g': float(uncrossed['g']),
                            'b': float(uncrossed['b']),
                        }
                    elif 'all' in uncrossed:
                        exp_all = float(uncrossed['all'])
                        logger.info(f"Using uncrossed exposure (all channels): {exp_all} ms")
                        return {'per_channel': False, 'exposure_ms': exp_all}
                else:
                    # Single value
                    logger.info(f"Using uncrossed exposure: {uncrossed} ms")
                    return {'per_channel': False, 'exposure_ms': float(uncrossed)}

            # Fall back to checking for simple exposure format
            if "single" in exposures_config:
                single = exposures_config["single"]
                if isinstance(single, dict):
                    if all(k in single for k in ['r', 'g', 'b']):
                        logger.info(f"Using per-channel single exposures: "
                                   f"R={single['r']}, G={single['g']}, B={single['b']}")
                        return {
                            'per_channel': True,
                            'r': float(single['r']),
                            'g': float(single['g']),
                            'b': float(single['b']),
                        }
                    elif 'all' in single:
                        exp_all = float(single['all'])
                        logger.info(f"Using single exposure (all channels): {exp_all} ms")
                        return {'per_channel': False, 'exposure_ms': exp_all}
                else:
                    logger.info(f"Using single exposure: {single} ms")
                    return {'per_channel': False, 'exposure_ms': float(single)}

        # Pattern 2: modalities -> ppm -> rotation_angles (older format)
        modalities = settings.get("modalities", {})
        modality_settings = modalities.get(base_modality, {})
        rotation_angles = modality_settings.get("rotation_angles", [])

        # Look for 90-degree angle (uncrossed, good signal)
        for angle_config in rotation_angles:
            if isinstance(angle_config, dict):
                angle = angle_config.get("angle", angle_config.get("rotation_angle"))
                if angle == 90:
                    exposure = angle_config.get("exposure_ms")
                    if exposure:
                        logger.info(f"Using 90-deg exposure from modality config: {exposure} ms")
                        return {'per_channel': False, 'exposure_ms': float(exposure)}

        # No exposure found in config
        logger.warning(
            f"Could not find exposure for modality '{modality}' in settings, "
            f"using default {default_exposure}ms. "
            "Consider setting exposure in your configuration."
        )
        return {'per_channel': False, 'exposure_ms': default_exposure}

    except Exception as e:
        logger.warning(f"Error getting exposure from settings: {e}, using default {default_exposure}ms")
        return {'per_channel': False, 'exposure_ms': default_exposure}


def _save_debug_mask(
    image: np.ndarray,
    saturation_threshold: float,
    value_threshold: float,
    output_path: Path,
    logger,
) -> None:
    """
    Save a debug visualization of the foreground detection mask.

    This creates a side-by-side comparison showing:
    - Original image
    - Foreground mask (what the thresholds detect)
    - Overlay of mask on original

    Helps users troubleshoot when rectangle detection fails.

    Args:
        image: Original RGB image
        saturation_threshold: HSV saturation threshold used
        value_threshold: HSV value threshold used
        output_path: Path to save the debug image
        logger: Logger instance
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage import color, morphology
        from scipy import ndimage

        # Convert to HSV
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image

        hsv = color.rgb2hsv(img_uint8)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Create foreground mask using same logic as SunburstCalibrator
        foreground_mask = (saturation > saturation_threshold) & (value > value_threshold)

        # Clean up mask
        min_area = 100
        try:
            foreground_clean = morphology.remove_small_objects(foreground_mask, min_size=min_area)
            foreground_clean = morphology.remove_small_holes(foreground_clean, area_threshold=500)
            foreground_clean = ndimage.median_filter(foreground_clean.astype(np.uint8), size=5).astype(bool)
        except Exception:
            foreground_clean = foreground_mask

        # Count connected components
        from skimage import measure
        labels = measure.label(foreground_clean)
        n_regions = labels.max()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Original image
        ax1 = axes[0, 0]
        ax1.imshow(img_uint8)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Plot 2: Raw foreground mask
        ax2 = axes[0, 1]
        ax2.imshow(foreground_mask, cmap="gray")
        ax2.set_title(f"Foreground Mask (sat>{saturation_threshold}, val>{value_threshold})")
        ax2.axis("off")

        # Plot 3: Cleaned mask with region labels
        ax3 = axes[1, 0]
        ax3.imshow(labels, cmap="nipy_spectral")
        ax3.set_title(f"Detected Regions: {n_regions} found")
        ax3.axis("off")

        # Plot 4: Overlay on original
        ax4 = axes[1, 1]
        overlay = img_uint8.copy()
        # Highlight detected regions in green
        overlay[foreground_clean, 1] = np.minimum(255, overlay[foreground_clean, 1] + 100)
        ax4.imshow(overlay)
        ax4.set_title("Overlay (detected regions highlighted)")
        ax4.axis("off")

        # Add text with threshold info
        fig.suptitle(
            f"Detection Debug - Saturation threshold: {saturation_threshold}, "
            f"Value threshold: {value_threshold}\n"
            f"Regions found: {n_regions} (need at least 3 for calibration)",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Saved debug mask to {output_path}")

    except Exception as e:
        logger.warning(f"Failed to save debug mask: {e}")
        # Don't raise - debug output is optional


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
    - Original image with center crosshair and radial sampling lines
    - B&W foreground mask
    - Color-coded scatter plot with hue colorbar
    - Calibration info text

    Args:
        image: Original calibration image
        result: RadialCalibrationResult from RadialCalibrator
        calibrator: RadialCalibrator instance
        output_path: Path to save plot
        logger: Logger instance
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for server
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from skimage import color

        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        # Plot 1: Original image with center crosshair and radial sampling lines
        ax1 = axes[0, 0]
        ax1.imshow(image)
        cy, cx = result.center
        ax1.plot(cx, cy, 'w+', markersize=20, markeredgewidth=3)
        for sample in result.samples:
            angle_rad = np.radians(sample.angle)
            x_inner = cx + calibrator.radius_inner * np.cos(angle_rad)
            y_inner = cy - calibrator.radius_inner * np.sin(angle_rad)
            x_outer = cx + calibrator.radius_outer * np.cos(angle_rad)
            y_outer = cy - calibrator.radius_outer * np.sin(angle_rad)
            ax1.plot([x_inner, x_outer], [y_inner, y_outer], 'w-', alpha=0.5, linewidth=1)
        rot_text = f", rot={result.rotation:.1f} deg" if result.rotation != 0 else ""
        ax1.set_title(f"Radial Sampling ({len(result.samples)} spokes{rot_text})")
        ax1.axis("off")

        # Plot 2: B&W foreground mask
        ax2 = axes[0, 1]
        hsv = color.rgb2hsv(image if image.dtype == np.uint8 else
                            (image * 255).astype(np.uint8) if image.max() <= 1.0 else
                            image.astype(np.uint8))
        foreground_mask = (
            (hsv[:, :, 1] > calibrator.saturation_threshold) &
            (hsv[:, :, 2] > calibrator.value_threshold)
        )
        ax2.imshow(foreground_mask, cmap="gray")
        ax2.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
        ax2.set_title(
            f"Foreground Mask (sat>{calibrator.saturation_threshold}, "
            f"val>{calibrator.value_threshold})"
        )
        ax2.axis("off")

        # Plot 3: Color-coded scatter plot with hue colorbar
        ax3 = axes[1, 0]

        # Build color array from raw hue values
        raw_hue_values = np.array([s.hue_mean for s in result.samples])

        # Check for saturation warnings
        saturated_angles = set()
        if result.warnings:
            for warning in result.warnings:
                if "SATURATION" in warning.upper():
                    for sample in result.samples:
                        saturated_angles.add(sample.angle)

        for i, (shifted_hue, angle) in enumerate(zip(result.hue_values, result.angles)):
            raw_hue = raw_hue_values[i]
            hsv_color = np.array([[[raw_hue, 1.0, 1.0]]])
            rgb_color = mcolors.hsv_to_rgb(hsv_color)[0, 0]
            marker = 'X' if result.samples[i].angle in saturated_angles else 'o'
            edge_color = 'red' if result.samples[i].angle in saturated_angles else 'black'
            ax3.scatter(
                shifted_hue, angle,
                s=100, c=[rgb_color], edgecolors=edge_color,
                linewidths=2, marker=marker, zorder=5,
            )

        # Plot regression line
        hue_line = np.linspace(0, 1, 100)
        predicted_angles = result.inv_slope * hue_line + result.inv_intercept
        ax3.plot(hue_line, predicted_angles, 'r-', linewidth=2,
                 label=f"R^2={result.r_squared:.4f}")

        ax3.set_xlabel("Shifted Hue Value")
        ax3.set_ylabel("Angle (degrees)")
        ax3.set_title("Hue to Angle Calibration")
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 180)

        # Add shifted rainbow colorbar below the scatter plot
        _add_shifted_hue_colorbar(ax3, result.hue_offset)

        # Plot 4: Calibration info text
        ax4 = axes[1, 1]
        ax4.axis("off")
        info_text = (
            f"Radial Calibration Results\n"
            f"{'=' * 35}\n\n"
            f"R-squared: {result.r_squared:.6f}\n"
            f"Spokes detected: {len(result.samples)}\n"
            f"Center: y={result.center[0]}, x={result.center[1]}\n"
            f"Hue offset: {result.hue_offset:.4f}\n\n"
            f"Regression (hue -> angle):\n"
            f"  angle = {result.inv_slope:.4f} * hue + {result.inv_intercept:.4f}\n\n"
            f"Regression (angle -> hue):\n"
            f"  hue = {result.slope:.6f} * angle + {result.intercept:.4f}\n\n"
            f"Sampling: r_inner={calibrator.radius_inner}, "
            f"r_outer={calibrator.radius_outer}\n"
        )
        if result.warnings:
            info_text += f"\nWarnings: {len(result.warnings)}\n"
            for w in result.warnings:
                info_text += f"  - {w}\n"
        info_text += "\nCalibration file saved for use in PPM analysis."

        ax4.text(
            0.1,
            0.9,
            info_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # Room for colorbar
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Saved calibration plot to {output_path}")

    except Exception as e:
        logger.error(f"Failed to create calibration plot: {e}")
        # Don't raise - plot is optional


# ROYGBIV color positions in hue space (0-1)
_ROYGBIV_COLORS = [
    (0.000, 'R', 'red'),
    (0.083, 'O', 'orange'),
    (0.167, 'Y', 'yellow'),
    (0.333, 'G', 'green'),
    (0.583, 'B', 'blue'),
    (0.667, 'I', 'indigo'),
    (0.750, 'V', 'violet'),
]


def _add_shifted_hue_colorbar(ax, hue_offset: float) -> None:
    """
    Add a rainbow colorbar shifted by hue_offset below the given axes.

    Args:
        ax: Matplotlib axes
        hue_offset: Hue offset to shift the colorbar
    """
    import matplotlib.colors as mcolors

    pos = ax.get_position()
    cbar_ax = ax.figure.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.015])

    # Create shifted hue gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    hsv = np.zeros((1, 256, 3))
    hsv[0, :, 0] = (gradient + hue_offset) % 1.0
    hsv[0, :, 1] = 1.0
    hsv[0, :, 2] = 1.0
    rgb = mcolors.hsv_to_rgb(hsv)

    cbar_ax.imshow(rgb, aspect='auto', extent=[0, 1, 0, 1])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])

    for hue, letter, color_name in _ROYGBIV_COLORS:
        shifted_pos = (hue - hue_offset) % 1.0
        cbar_ax.text(shifted_pos, -0.5, letter, ha='center', va='top',
                     fontsize=9, fontweight='bold', color='black')
