"""Modality registry -- prefix-based lookup of ModalityConfig instances.

Modalities register themselves with a prefix (e.g. "ppm") and callers
look up configs with the full modality string (e.g. "ppm_20x").
The first matching prefix wins, mirroring the Java ModalityRegistry pattern.
"""

import logging
from typing import Optional

from .config import ModalityConfig

logger = logging.getLogger(__name__)

_registry: dict[str, ModalityConfig] = {}

# Default config returned for unknown modalities (no rotation, generic defaults)
_default = ModalityConfig()


def register(prefix: str, config: ModalityConfig) -> None:
    """Register a modality config under a prefix (case-insensitive)."""
    key = prefix.lower()
    if key in _registry:
        logger.warning("Overwriting modality config for prefix '%s'", key)
    _registry[key] = config
    logger.debug("Registered modality config: prefix='%s'", key)


def get_config(modality: Optional[str] = None) -> ModalityConfig:
    """Look up ModalityConfig by modality string (prefix match).

    Args:
        modality: Full modality string, e.g. "ppm_20x", "brightfield".
                  None returns the default config.

    Returns:
        Matching ModalityConfig, or the default (no-capability) config.
    """
    if modality is None:
        return _default
    mod_lower = modality.lower()
    for prefix, config in _registry.items():
        if mod_lower.startswith(prefix):
            return config
    return _default


def registered_prefixes() -> list[str]:
    """Return list of registered modality prefixes (for diagnostics)."""
    return list(_registry.keys())
