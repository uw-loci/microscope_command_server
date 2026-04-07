"""Widefield fluorescence modality configuration.

Registers the widefield fluorescence modality config. Widefield
fluorescence is a single-snap modality with no rotation stage, no
per-angle exposures, and no white balance calibration. Images are
monochrome (sCMOS) or color depending on camera. Epi-illumination
(LED or arc lamp) with filter cube selection via acquisition profiles.
"""

import logging

from .config import ModalityConfig
from .registry import register

logger = logging.getLogger(__name__)

WIDEFIELD_CONFIG = ModalityConfig(
    # No rotation needed for widefield fluorescence
    autofocus_angle=None,
    has_rotation=False,
    default_angle_count=1,
    # No WB settings (fluorescence does not use white balance)
    wb_settings_key=None,
    # No post-processing directories
    post_processing_suffixes=[],
    # Target intensity for exposure calibration
    default_target_intensity=200.0,
)


def register_widefield():
    """Register widefield fluorescence modality config with the registry."""
    register("fl", WIDEFIELD_CONFIG)
    register("fluorescence", WIDEFIELD_CONFIG)
    register("widefield", WIDEFIELD_CONFIG)
    register("epi", WIDEFIELD_CONFIG)
    logger.debug("Registered widefield fluorescence modality config")
