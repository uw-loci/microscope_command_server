"""PPM (Polarized light microscopy) modality configuration and command handlers.

Registers the PPM modality config and provides handler functions for
PPM-specific server commands (PPMSENS, PPMBIREF).
"""

import logging
from typing import Any, Dict, Optional, Callable

from .config import ModalityConfig
from .registry import register

logger = logging.getLogger(__name__)


PPM_CONFIG = ModalityConfig(
    # Autofocus at 90 deg (uncrossed) -- brightest, fastest
    autofocus_angle=90.0,
    # PPM uses a rotation stage
    has_rotation=True,
    default_angle_count=4,
    # WB settings stored under settings["white_balance"]["ppm"]
    wb_settings_key="ppm",
    # Canonical PPM angle names
    angle_names={
        0.0: "crossed",
        90.0: "uncrossed",
        7.0: "positive",
        -7.0: "negative",
    },
    # Angle-specific target intensities for exposure calibration
    # Keys: (lo_abs, hi_abs) inclusive ranges on abs(angle)
    angle_intensity_targets={
        (88, 92): 245.0,  # Near-uncrossed (around 90 deg) -- brightest
        (4, 10): 160.0,  # Birefringence angles (5-7 deg neighbors)
        (0, 3): 125.0,  # Near-crossed (around 0 deg) -- dimmest
    },
    # Generic fallback for intermediate angles (10-88 deg)
    default_target_intensity=180.0,
    # Post-processing directory suffixes
    post_processing_suffixes=[".biref", ".sum"],
)


def register_ppm():
    """Register PPM modality config with the registry."""
    register("ppm", PPM_CONFIG)


# ---------------------------------------------------------------------------
# PPM-specific server command handlers
# ---------------------------------------------------------------------------


def handle_sensitivity_test(
    params: Dict[str, Any],
    port: int,
    _logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Run PPM rotation sensitivity test.

    Returns result directory path on success, None on failure.
    Raises ImportError if ppm_library is not available.
    """
    log = _logger or logger
    from ppm_library.ppm.sensitivity_test import run_ppm_sensitivity_test

    log.info(
        "Starting PPM sensitivity test (type=%s, base_angle=%s, repeats=%s)",
        params["test_type"],
        params["base_angle"],
        params["n_repeats"],
    )

    result_dir = run_ppm_sensitivity_test(
        config_yaml=params["yaml_file_path"],
        output_dir=params["output_folder_path"],
        host="127.0.0.1",
        port=port,
        test_type=params["test_type"],
        base_angle=params["base_angle"],
        n_repeats=params["n_repeats"],
        keep_images=True,
    )
    return result_dir


def handle_birefringence_test(
    params: Dict[str, Any],
    port: int,
    progress_callback: Optional[Callable] = None,
    stage_move_callback: Optional[Callable] = None,
    _logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Run PPM birefringence maximization test.

    Returns result directory path on success, None on failure.
    Raises ImportError if ppm_library is not available.
    """
    log = _logger or logger
    from ppm_library.ppm.birefringence_test import (
        run_birefringence_maximization_test,
    )

    log.info(
        "Starting PPM birefringence test (range=%s-%s, step=%s, mode=%s)",
        params["min_angle"],
        params["max_angle"],
        params["angle_step"],
        params["exposure_mode"],
    )

    result_dir = run_birefringence_maximization_test(
        config_yaml=params["yaml_file_path"],
        output_dir=params["output_folder_path"],
        host="127.0.0.1",
        port=port,
        angle_range=(params["min_angle"], params["max_angle"]),
        angle_step=params["angle_step"],
        exposure_mode=params["exposure_mode"],
        fixed_exposure_ms=params.get("fixed_exposure_ms"),
        keep_images=True,
        target_intensity=params["target_intensity"],
        progress_callback=progress_callback,
        stage_move_callback=(
            stage_move_callback if params["exposure_mode"] == "calibrate" else None
        ),
    )
    return result_dir
