"""LC-PolScope modality configuration.

Quantitative polarized light using a liquid-crystal universal compensator
(Meadowlark D5020). Five polarization states are set *electrically* per tile
via MicroManager ConfigGroup presets -- there is no rotation stage, so this is
a channel modality (State0..State4) and not an angle modality like PPM.

The reconstruction is QLIPP-style Stokes inversion (Guo et al., eLife 2020;9:
e55502), performed by ``polscope_library``. Nothing in this module does math;
it declares capabilities so core acquisition code can dispatch without
hardcoding the modality name.

Two invariants this modality does NOT enforce alone
---------------------------------------------------
1. **All five states share one exposure and gain.** The Stokes inversion treats
   the five intensities as samples of one radiometric scale, so a per-state
   difference biases retardance and orientation with *no visible symptom* --
   the images still look fine. This is held by three cooperating layers: the
   Java ``LCPolScopeModalityHandler.enforceEqualExposure``, equal ``exposure_ms``
   on every channel in ``config_LCPolScope.yml``, and the deliberate absence of
   ``channel_overrides`` on the LC-PolScope acquisition profiles. Do not add
   per-channel exposure tuning here or in the profile.

2. **State order is positional.** The states are consumed in calibration order;
   a permutation silently rotates or mirrors the orientation map rather than
   raising. If a calibration's provenance is unknown, identify it from data
   with ``polscope-scheme-check`` before trusting any orientation output.
"""

import logging

from .config import ModalityConfig
from .registry import register

logger = logging.getLogger(__name__)


# The extinction state. Excluded from autofocus: it is dark by design, so a
# focus metric computed on it is dominated by noise.
EXTINCTION_CHANNEL_ID = "State0"


LCPOLSCOPE_CONFIG = ModalityConfig(
    # No rotation stage -- polarization states are set electrically, so there
    # is no angle to move to before autofocus.
    autofocus_angle=None,
    has_rotation=False,
    # Channel modality: one "angle" per tile, five channels within it.
    default_angle_count=1,
    # Polarized transmitted light through a monochrome camera at a single
    # wavelength (549 nm). White balance is meaningless here, and applying one
    # would rescale the states unequally -- see invariant 1 above.
    wb_settings_key=None,
    # Deliberately no angle_intensity_targets: the per-state brightness spread
    # (extinction is dark by design) is signal, not something to normalize away.
    default_target_intensity=200.0,
    # Derived outputs written per tile by the reconstruction hook. Retardance
    # and orientation are separate directories so each stitches independently.
    post_processing_suffixes=[".retardance", ".orientation"],
)


def register_lcpolscope():
    """Register LC-PolScope modality config with the registry."""
    register("lcpolscope", LCPOLSCOPE_CONFIG)
    register("lcps", LCPOLSCOPE_CONFIG)
    logger.debug("Registered LC-PolScope modality config")
