"""SHG (Second Harmonic Generation) modality configuration.

Registers the SHG/multiphoton modality config. SHG is a single-snap
modality with no rotation stage, no per-angle exposures, and no
white balance calibration. Images are grayscale 16-bit from the PMT.
"""

import logging

from .config import ModalityConfig
from .registry import register

logger = logging.getLogger(__name__)

SHG_CONFIG = ModalityConfig(
    # No autofocus angle rotation needed (PMT sees all angles the same)
    autofocus_angle=None,
    # No rotation stage
    has_rotation=False,
    default_angle_count=1,
    # No WB settings (grayscale PMT output)
    wb_settings_key=None,
    # No post-processing directories (no biref/sum)
    post_processing_suffixes=[],
    # PMT target intensity for exposure calibration
    default_target_intensity=128.0,
)


def register_shg():
    """Register SHG modality config with the registry."""
    register("shg", SHG_CONFIG)
    logger.debug("Registered SHG modality config")
