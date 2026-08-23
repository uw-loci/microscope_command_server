"""LC-PolScope modality registration and its capability invariants.

These are cheap guards on decisions that fail *silently* if reversed. The
Stokes inversion has no self-check: if the five states stop sharing one
radiometric scale, or the modality starts being treated as an angle modality,
the reconstruction still produces a plausible-looking retardance and
orientation map that is simply wrong. Nothing downstream will notice.
"""

import pytest

from microscope_command_server.modality import get_config, registered_prefixes
from microscope_command_server.modality.lcpolscope import (
    EXTINCTION_CHANNEL_ID,
    LCPOLSCOPE_CONFIG,
)


@pytest.mark.parametrize("modality", ["lcpolscope", "lcpolscope_20x", "lcps", "lcps_20x"])
def test_prefixes_resolve_to_the_lcpolscope_config(modality):
    """Both spellings the Java ModalityRegistry uses must resolve here too."""
    assert get_config(modality) is LCPOLSCOPE_CONFIG


def test_prefixes_are_registered():
    prefixes = registered_prefixes()
    assert "lcpolscope" in prefixes
    assert "lcps" in prefixes


def test_lcpolscope_does_not_collide_with_other_modalities():
    """Prefix matching is first-match-wins, so a collision would silently
    hand LC-PolScope another modality's rotation and white-balance settings."""
    for other in ["ppm_20x", "brightfield", "fl_20x", "widefield", "shg"]:
        assert get_config(other) is not LCPOLSCOPE_CONFIG


def test_is_a_channel_modality_not_an_angle_modality():
    """States are set electrically by the liquid crystal -- there is no stage
    to rotate, and no angle to move to before autofocus."""
    assert LCPOLSCOPE_CONFIG.has_rotation is False
    assert LCPOLSCOPE_CONFIG.autofocus_angle is None
    assert LCPOLSCOPE_CONFIG.default_angle_count == 1


def test_no_white_balance():
    """Monochrome camera at a single wavelength. A white balance would also
    rescale the five states unequally, biasing the inversion."""
    assert LCPOLSCOPE_CONFIG.wb_settings_key is None


def test_no_per_state_intensity_targets():
    """The per-state brightness spread is signal, not something to normalize.

    State0 is dark *by design* -- it is the extinction state. Per-state
    intensity targets would drive the states to a common brightness and
    destroy exactly the modulation the reconstruction measures.
    """
    assert LCPOLSCOPE_CONFIG.angle_intensity_targets == {}
    for angle in [0.0, 7.0, 45.0, 90.0]:
        assert LCPOLSCOPE_CONFIG.get_target_intensity(angle) == 200.0


def test_derived_outputs_are_separate_directories():
    """Retardance and orientation stitch independently, so they need their own
    tile directories -- and orientation must stay separable because it is
    axial data that cannot be resampled like an ordinary scalar."""
    assert LCPOLSCOPE_CONFIG.post_processing_suffixes == [".retardance", ".orientation"]


def test_extinction_channel_id_matches_the_java_handler():
    """LCPolScopeModalityHandler.EXTINCTION_CHANNEL_ID must agree; autofocus
    skips this state on both sides."""
    assert EXTINCTION_CHANNEL_ID == "State0"
