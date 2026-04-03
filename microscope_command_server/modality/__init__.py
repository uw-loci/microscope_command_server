"""Modality configuration and registry.

Provides capability-based modality configuration so core acquisition
code can query modality capabilities (rotation, autofocus angle, etc.)
without hardcoding modality names.

Usage::

    from microscope_command_server.modality import get_config

    config = get_config("ppm_20x")
    if config.autofocus_angle is not None:
        hardware.set_psg_ticks(config.autofocus_angle)
"""

from .config import ModalityConfig
from .registry import get_config, register, registered_prefixes
from .ppm import register_ppm
from .shg import register_shg
from .brightfield import register_brightfield

# Register built-in modalities on import
register_ppm()
register_shg()
register_brightfield()

__all__ = [
    "ModalityConfig",
    "get_config",
    "register",
    "registered_prefixes",
]
