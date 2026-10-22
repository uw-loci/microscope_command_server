"""Colour maps for axial orientation images: hue = orientation over 0..180 deg, brightness = magnitude.

Display only. The measured orientation and retardance are written separately as 16-bit
images; this module only decides what colour a pixel is shown in.

Three maps, chosen by ``reconstruction.display_colormap``:

``uniform`` (default)
    A constant-lightness, constant-chroma hue circle in OKLCH (Ottosson 2020,
    https://bottosson.github.io/posts/oklab/). Equal angle steps look like equal colour
    steps, and every orientation has the same perceived lightness, so brightness carries
    only the magnitude. L = 0.75 and C = 0.125 are the largest chroma that every hue
    reaches inside sRGB at the lightest such L (the limit is 0.1275 at L = 0.75).
``hsv``
    Hue round the HSV wheel once per 180 deg, saturation 1 (OpenPolScope's look). Its
    yellows and cyans look much brighter than its blues at the same value.
``ppm``
    The hue a PPM camera image shows for the angle, from a PPM sunburst calibration
    (ppm_library.calibration.radial: hue = slope * angle + intercept + hue_offset, mod 1),
    with the slope forced to exactly one cycle per 180 deg so the map stays cyclic, then
    HSV with saturation 1.

The angle is first registered: ``angle = sign * angle + offset_deg``, from
``display_orientation_sign`` (+1 or -1) and ``display_orientation_offset_deg``, which put
this instrument's angles in the frame of the instrument whose colours are borrowed. Both
default to the identity. Design notes and reference values for re-implementing the maps
elsewhere: claude-reports/design/2026-10-22_orientation-colormaps.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

COLORMAPS = ("uniform", "hsv", "ppm")
DEFAULT_COLORMAP = "uniform"

#: OKLCH lightness and chroma of the uniform map, and the OKLCH hue (deg) at 0 deg orientation.
UNIFORM_L = 0.75
UNIFORM_C = 0.125
UNIFORM_HUE0_DEG = 29.2  # OKLCH hue of sRGB red, so 0 deg is red as in the hsv map

#: Entries in the uniform map's lookup table over 0..180 deg (0.18 deg apart).
UNIFORM_LUT_SIZE = 1000

_M1 = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
_M2 = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)


def oklch_to_srgb(lightness, chroma, hue_deg):
    """OKLCH -> gamma-encoded sRGB in [0, 1], shape ``(..., 3)``; out-of-gamut values clipped."""
    h = np.radians(np.asarray(hue_deg, dtype=np.float64))
    a = chroma * np.cos(h)
    b = chroma * np.sin(h)
    lab = np.stack([np.full_like(a, lightness), a, b], axis=-1)
    lms = (lab @ _M1.T) ** 3
    linear = np.clip(lms @ _M2.T, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )


def _uniform_lut():
    angles = np.arange(UNIFORM_LUT_SIZE) * (180.0 / UNIFORM_LUT_SIZE)
    return oklch_to_srgb(UNIFORM_L, UNIFORM_C, UNIFORM_HUE0_DEG + 2.0 * angles)


_UNIFORM_LUT = _uniform_lut()


def _hsv_full(hue):
    """HSV with saturation and value 1 -> RGB in [0, 1]; ``hue`` in [0, 1)."""
    h6 = np.asarray(hue, dtype=np.float64) * 6.0
    sector = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    one = np.ones_like(f)
    zero = np.zeros_like(f)
    r = np.choose(sector, [one, 1 - f, zero, zero, f, one])
    g = np.choose(sector, [f, one, one, 1 - f, zero, zero])
    b = np.choose(sector, [zero, zero, f, one, one, 1 - f])
    return np.stack([r, g, b], axis=-1)


@dataclass(frozen=True)
class ColormapSettings:
    name: str = DEFAULT_COLORMAP
    offset_deg: float = 0.0
    sign: int = 1
    #: ppm only: raw PPM hue at 90 deg and the direction hue runs (+1 / -1 per 180 deg).
    ppm_hue_at_90: Optional[float] = None
    ppm_direction: int = 1
    ppm_source: Optional[str] = None

    def describe(self) -> str:
        """One line for the image metadata."""
        reg = (
            "unregistered"
            if (self.offset_deg, self.sign) == (0.0, 1)
            else f"registered: angle x {self.sign:+d} {self.offset_deg:+g} deg"
        )
        if self.name == "uniform":
            return (
                f"uniform: OKLCH L {UNIFORM_L} C {UNIFORM_C}, hue {UNIFORM_HUE0_DEG} deg at 0 deg, "
                f"one cycle per 180 deg ({reg})"
            )
        if self.name == "ppm":
            return (
                f"ppm: PPM hue from {self.ppm_source}, one cycle per 180 deg, HSV saturation 1 "
                f"({reg})"
            )
        return f"hsv: hue = angle / 180, saturation 1 ({reg})"


def colormap_settings(reconstruction_cfg) -> ColormapSettings:
    """The display colour map from ``modalities.lcpolscope.reconstruction``.

    Raises ValueError on an unknown map, a bad registration, or ``ppm`` without a usable
    ``display_hue_calibration``, so a typo is refused at acquisition start.
    """
    cfg = reconstruction_cfg or {}
    name = str(cfg.get("display_colormap") or DEFAULT_COLORMAP)
    if name not in COLORMAPS:
        raise ValueError(f"display_colormap is {name!r}; expected one of {list(COLORMAPS)}")
    offset = float(cfg.get("display_orientation_offset_deg") or 0.0)
    sign = int(cfg.get("display_orientation_sign") or 1)
    if sign not in (1, -1):
        raise ValueError(f"display_orientation_sign is {sign!r}; expected 1 or -1")
    if name != "ppm":
        return ColormapSettings(name=name, offset_deg=offset, sign=sign)

    cal = cfg.get("display_hue_calibration")
    if not isinstance(cal, dict):
        raise ValueError(
            "display_colormap 'ppm' needs display_hue_calibration: {source, slope, intercept, "
            "hue_offset} from a PPM sunburst calibration (.npz)"
        )
    try:
        slope = float(cal["slope"])
        intercept = float(cal["intercept"])
        hue_offset = float(cal["hue_offset"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"display_hue_calibration needs numeric slope, intercept and hue_offset: {exc}"
        )
    span = abs(slope) * 180.0
    if not 0.8 <= span <= 1.2:
        raise ValueError(
            f"display_hue_calibration covers {span:.2f} of the hue circle per 180 deg; a PPM "
            "calibration covers about one. Is it the right file?"
        )
    return ColormapSettings(
        name=name,
        offset_deg=offset,
        sign=sign,
        ppm_hue_at_90=float((slope * 90.0 + intercept + hue_offset) % 1.0),
        ppm_direction=1 if slope > 0 else -1,
        ppm_source=str(cal.get("source") or "unnamed calibration"),
    )


def orientation_rgb(
    magnitude, orientation_rad, ceiling, settings: Optional[ColormapSettings] = None
):
    """``(Y, X, 3)`` uint8: hue from the axial orientation, brightness from ``magnitude / ceiling``.

    Brightness scales the full-brightness colour, so at equal magnitude every orientation
    keeps the lightness its map gives it (all equal for ``uniform``). Undefined pixels
    (NaN in either input) are black.
    """
    settings = settings or ColormapSettings()
    mag = np.asarray(magnitude, dtype=np.float64)
    ori = np.asarray(orientation_rad, dtype=np.float64)
    bad = ~np.isfinite(mag) | ~np.isfinite(ori)
    deg = np.where(bad, 0.0, np.degrees(ori))
    deg = np.mod(settings.sign * deg + settings.offset_deg, 180.0)
    value = np.where(bad, 0.0, np.clip(mag / float(ceiling), 0.0, 1.0))

    if settings.name == "uniform":
        idx = np.rint(deg * (UNIFORM_LUT_SIZE / 180.0)).astype(int) % UNIFORM_LUT_SIZE
        full = _UNIFORM_LUT[idx]
    elif settings.name == "ppm":
        hue = np.mod(settings.ppm_hue_at_90 + settings.ppm_direction * (deg - 90.0) / 180.0, 1.0)
        full = _hsv_full(hue)
    else:
        full = _hsv_full(deg / 180.0)
    rgb = full * value[..., None]
    return np.rint(rgb * 255.0).astype(np.uint8)
