"""Modality registry -- prefix-based lookup of ModalityConfig instances.

Modalities register themselves with a prefix (e.g. "ppm") and callers
look up configs with the full modality string (e.g. "ppm_20x").
The longest matching prefix wins, as in the Java ModalityRegistry, so
"bf_if_20x" resolves to the "bf_if" entry and not to "bf".
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
        The config registered under the longest prefix the modality string
        starts with, or the default (no-capability) config.
    """
    if modality is None:
        return _default
    mod_lower = modality.lower()
    # Longest prefix, not first registered: "bf" is the start of "bf_if", and
    # which of the two was registered first is an accident of import order.
    best = None
    for prefix in _registry:
        if mod_lower.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return _registry[best] if best is not None else _default


def registered_prefixes() -> list[str]:
    """Return list of registered modality prefixes (for diagnostics)."""
    return list(_registry.keys())
