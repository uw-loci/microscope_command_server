"""Orientation colour maps: cyclic over 180 deg, uniform lightness, PPM hues from a calibration."""

import numpy as np
import pytest

from microscope_command_server.modality.orientation_colormap import (
    UNIFORM_C,
    UNIFORM_L,
    ColormapSettings,
    colormap_settings,
    orientation_rgb,
)

# The PPM -7 deg calibration of 2026-10-16 (Negative_7_Oct2026.npz): R^2 0.998.
NEG7 = {
    "source": "Negative_7_Oct2026.npz",
    "slope": -0.005541962395299577,
    "intercept": 0.9860845130836212,
    "hue_offset": 0.28,
}

ANGLES = np.radians(np.arange(0.0, 180.0, 0.5))[None, :]
FULL = np.full(ANGLES.shape, 1.0)


def _srgb_to_oklab(rgb8):
    c = np.asarray(rgb8, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return np.cbrt(lin @ m1.T) @ m2.T


def test_uniform_map_has_one_lightness_and_one_chroma():
    rgb = orientation_rgb(FULL, ANGLES, 1.0, ColormapSettings(name="uniform"))[0]
    lab = _srgb_to_oklab(rgb)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    # 8-bit rounding is the only spread left.
    assert np.abs(lab[:, 0] - UNIFORM_L).max() < 0.005
    assert np.abs(chroma - UNIFORM_C).max() < 0.006


def test_hsv_lightness_varies_with_angle_and_uniform_does_not():
    # Why uniform is the default: at equal retardance HSV yellow is far lighter than blue.
    hsv = _srgb_to_oklab(orientation_rgb(FULL, ANGLES, 1.0, ColormapSettings(name="hsv"))[0])
    assert np.ptp(hsv[:, 0]) > 0.3


@pytest.mark.parametrize("name", ["uniform", "hsv"])
def test_maps_are_cyclic_over_180_degrees(name):
    s = ColormapSettings(name=name)
    a = orientation_rgb(np.array([[1.0]]), np.radians(np.array([[0.0]])), 1.0, s).astype(int)
    b = orientation_rgb(np.array([[1.0]]), np.radians(np.array([[179.99]])), 1.0, s).astype(int)
    assert np.abs(a - b).max() <= 2


def test_ppm_map_reproduces_the_calibration_hues():
    import colorsys

    s = colormap_settings({"display_colormap": "ppm", "display_hue_calibration": NEG7})
    deg = np.array([10.0, 45.0, 90.0, 135.0, 170.0])
    rgb = orientation_rgb(np.ones((1, 5)), np.radians(deg)[None, :], 1.0, s)[0] / 255.0
    got = np.array([colorsys.rgb_to_hsv(*px)[0] for px in rgb])
    want = np.mod(NEG7["slope"] * deg + NEG7["intercept"] + NEG7["hue_offset"], 1.0)
    diff = np.abs((got - want + 0.5) % 1.0 - 0.5)
    # One cycle per 180 deg instead of the fitted 0.998: within 0.002 of the fit, plus 8-bit rounding.
    assert diff.max() < 0.006, (got, want)


def test_registration_rotates_and_mirrors_the_angle():
    plain = ColormapSettings(name="uniform")
    mirrored = ColormapSettings(name="uniform", sign=-1, offset_deg=30.0)
    one = np.ones((1, 1))
    a = orientation_rgb(one, np.radians(np.array([[20.0]])), 1.0, mirrored)
    b = orientation_rgb(one, np.radians(np.array([[10.0]])), 1.0, plain)  # -20 + 30
    assert (a == b).all()
    assert "registered: angle x -1 +30 deg" in mirrored.describe()


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="display_colormap"):
        colormap_settings({"display_colormap": "jet"})
    with pytest.raises(ValueError, match="display_hue_calibration"):
        colormap_settings({"display_colormap": "ppm"})
    with pytest.raises(ValueError, match="hue circle"):
        colormap_settings(
            {"display_colormap": "ppm", "display_hue_calibration": {**NEG7, "slope": 0.001}}
        )
    with pytest.raises(ValueError, match="sign"):
        colormap_settings({"display_orientation_sign": 2})


def test_undefined_and_zero_pixels_are_black():
    rgb = orientation_rgb(
        np.array([[np.nan, 0.0, 1.0]]), np.radians(np.array([[10.0, 10.0, np.nan]])), 1.0
    )
    assert not rgb.any()


def test_the_dialog_choice_reaches_the_reconstruction(monkeypatch):
    # --display-colormap (the acquisition dialog's Colour map) overrides the YAML for one run.
    from types import SimpleNamespace

    from microscope_command_server.acquisition import workflow
    from microscope_command_server.modality import lcpolscope

    params = workflow.parse_acquisition_message(
        "--yaml c.yml --projects p --sample s --scan-type lcpolscope_20x --region r "
        "--angles () --exposures (300.0) --display-colormap hsv"
    )
    assert params["display_colormap"] == "hsv"

    seen = {}
    monkeypatch.setattr(
        lcpolscope, "submit_tile_reconstruction", lambda **kw: seen.update(kw["reconstruction_cfg"])
    )
    ctx = SimpleNamespace(
        modality="lcpolscope",
        params=params,
        ppm_settings={
            "modalities": {"lcpolscope": {"reconstruction": {"display_colormap": "ppm"}}}
        },
        output_path=None,
        hardware=SimpleNamespace(get_pixel_size_um=lambda: 0.27),
        write_pool=None,
        channel_background_images=None,
        lcps_background_provenance=None,
        logger=None,
    )
    workflow._maybe_reconstruct_lcpolscope_tile(ctx, [{"id": "State0"}], {}, "0.tif")
    assert seen["display_colormap"] == "hsv"
