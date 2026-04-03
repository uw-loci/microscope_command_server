"""Brightfield modality configuration.

Registers the brightfield modality config. Brightfield is a single-snap
modality with no rotation stage, no per-angle exposures, and no
white balance calibration. Images are monochrome or color from a
transmitted-light camera (sCMOS, CCD).
"""

import logging

from .config import ModalityConfig
from .registry import register

logger = logging.getLogger(__name__)

BRIGHTFIELD_CONFIG = ModalityConfig(
    # No rotation needed
    autofocus_angle=None,
    has_rotation=False,
    default_angle_count=1,
    # No WB settings (uniform transmitted illumination)
    wb_settings_key=None,
    # No post-processing directories
    post_processing_suffixes=[],
    # Target intensity for exposure calibration (8-bit range)
    default_target_intensity=200.0,
)


def register_brightfield():
    """Register brightfield modality config with the registry."""
    register("bf", BRIGHTFIELD_CONFIG)
    register("brightfield", BRIGHTFIELD_CONFIG)
    logger.debug("Registered brightfield modality config")
