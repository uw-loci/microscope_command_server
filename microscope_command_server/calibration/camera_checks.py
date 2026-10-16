"""Camera settings that make polarimetric intensities meaningless.

The Stokes inversion treats the five state images as samples of one linear
radiometric scale, and the LC calibration search compares frame means with
each other. Both are silently destroyed by a camera that changes its own gain
or exposure between frames, or applies a gamma curve. The 2026-10-16 rig run
had ``Gain Auto: Continuous`` and ``Gamma Enabled: 1``: every state the
search settled on read 173 counts, the camera's own auto-gain target, and the
palette was the camera's doing, not the crystals'.

Property names are GenICam-flavoured and vary by adapter, so the match is on
the name, loosely. A camera that lacks a property is not reported for it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

_OFF_VALUES = {"off", "0", "false", "none", "", "disabled"}
_ON_VALUES = {"1", "on", "true", "enabled"}

__all__ = ["camera_setting_problems", "camera_snapshot", "describe_camera", "significant_bits"]

logger = logging.getLogger(__name__)


def camera_snapshot(hardware) -> Tuple[Dict[str, object], Dict[str, str]]:
    """``(summary, properties)``: the camera as it is set right now.

    A frame mean in raw counts means nothing without the exposure and the
    pixel format it was taken at; the first two rig calibrations were analysed
    blind for want of this. The summary is a handful of values for a reply and
    a log line; ``properties`` is every camera property as a string, for a
    metadata file and for :func:`camera_setting_problems`.

    Nothing here is fatal: a camera that cannot be asked something is recorded
    as not knowing it.
    """
    summary: Dict[str, object] = {}
    properties: Dict[str, str] = {}
    try:
        summary["exposure_ms"] = float(hardware.get_exposure())
    except Exception as exc:
        logger.debug("Could not read the exposure: %s", exc)
    core = getattr(hardware, "core", None)
    if core is None:
        return summary, properties

    camera = None
    for getter in (lambda: core.get_camera_device(), lambda: hardware.get_camera_name()):
        try:
            camera = str(getter())
            if camera:
                break
        except Exception as exc:
            logger.debug("Could not name the camera: %s", exc)
    if camera:
        summary["device"] = camera

    for key, getter in (
        ("bit_depth", lambda: int(core.get_image_bit_depth())),
        ("bytes_per_pixel", lambda: int(core.get_bytes_per_pixel())),
        ("width", lambda: int(core.get_image_width())),
        ("height", lambda: int(core.get_image_height())),
    ):
        try:
            summary[key] = getter()
        except Exception as exc:
            logger.debug("Could not read camera %s: %s", key, exc)

    if camera:
        try:
            names = core.get_device_property_names(camera)
            try:
                names = list(names)
            except TypeError:
                names = [names.get(i) for i in range(names.size())]
        except Exception as exc:
            logger.debug("Could not list %s properties: %s", camera, exc)
            names = []
        for name in names:
            try:
                properties[str(name)] = str(core.get_property(camera, str(name)))
            except Exception as exc:
                logger.debug("Could not read %s.%s: %s", camera, name, exc)
        # The handful worth putting in front of the operator. Adapters name
        # these differently, so it is a loose match on the name, and a camera
        # without one simply does not report it.
        for key, needles in (
            ("binning", ("binning",)),
            ("pixel_type", ("pixeltype", "pixelformat", "pixel type", "pixel format")),
            ("gain", ("gain",)),
        ):
            for name, value in properties.items():
                lowered = name.lower()
                if any(needle in lowered for needle in needles) and "auto" not in lowered:
                    summary[key] = value
                    break
    bits = significant_bits(summary.get("bit_depth"), summary.get("pixel_type"))
    if bits is not None:
        summary["significant_bits"] = bits
    return summary, properties


def significant_bits(bit_depth: Optional[int], pixel_type: Optional[str]) -> Optional[int]:
    """How many bits of each pixel carry data: the smaller of Micro-Manager's bit depth
    and the number in the pixel format's name (``Mono12p`` -> 12, ``Mono16`` -> 16).

    The two can disagree. An adapter may report a 12-bit packed format as 16-bit
    because it unpacks into a 16-bit buffer, while the values still run 0 to 4095;
    taking the 16 would put full scale at 65535, so the too-dark guard would refuse
    every run and a clipped state would never be seen. A name with no number
    leaves the bit depth as it is.
    """
    import re

    from_name = None
    if pixel_type:
        match = re.search(r"(\d+)", str(pixel_type))
        if match:
            value = int(match.group(1))
            if 1 <= value <= 32:
                from_name = value
    candidates = [b for b in (bit_depth, from_name) if isinstance(b, int) and b > 0]
    return min(candidates) if candidates else None


def describe_camera(camera: Dict[str, object]) -> str:
    """One line: ``Oryx, 50 ms (request), 12-bit, Mono12, 2448x2048, binning 1``."""
    parts = []
    if camera.get("device"):
        parts.append(str(camera["device"]))
    if camera.get("exposure_ms") is not None:
        source = camera.get("exposure_source")
        parts.append(f"{camera['exposure_ms']:g} ms" + (f" ({source})" if source else ""))
    bits = camera.get("significant_bits", camera.get("bit_depth"))
    if bits is not None:
        parts.append(f"{bits}-bit")
    if camera.get("pixel_type"):
        parts.append(str(camera["pixel_type"]))
    if camera.get("width") is not None and camera.get("height") is not None:
        parts.append(f"{camera['width']}x{camera['height']}")
    if camera.get("binning") is not None:
        parts.append(f"binning {camera['binning']}")
    if camera.get("gain") is not None:
        parts.append(f"gain {camera['gain']}")
    return ", ".join(parts) if parts else "unknown (the camera could not be asked)"


def camera_setting_problems(
    properties: Dict[str, str], bit_depth: Optional[int]
) -> Tuple[List[str], List[str]]:
    """``(fatal, advisory)`` problems with the camera as it is set.

    Fatal: the frames cannot be compared with each other at all (auto gain or
    auto exposure on). Advisory: the counts are usable for finding a palette
    but not for a quantitative acquisition (gamma on, 8-bit frames).
    """
    fatal: List[str] = []
    advisory: List[str] = []
    lowered = {name: str(value).strip().lower() for name, value in properties.items()}

    for name, value in lowered.items():
        key = name.lower()
        if "auto" in key and ("gain" in key or "exposure" in key) and value not in _OFF_VALUES:
            what = "gain" if "gain" in key else "exposure"
            fatal.append(
                f"{name} is {properties[name]}: the camera changes its own {what} between frames, "
                f"so no two intensities are comparable and the search settles wherever the camera "
                f"steers it. Set {name} to Off in Micro-Manager."
            )

    gamma_enabled = [n for n in lowered if "gamma" in n.lower() and "enable" in n.lower()]
    if gamma_enabled:
        for name in gamma_enabled:
            if lowered[name] in _ON_VALUES:
                advisory.append(
                    f"{name} is {properties[name]}: the camera applies a gamma curve, so counts are "
                    "not proportional to light. The Stokes inversion needs a linear camera. "
                    f"Set {name} to 0 (off) in Micro-Manager."
                )
    else:
        for name in lowered:
            if name.lower() == "gamma":
                try:
                    gamma = float(lowered[name])
                except ValueError:
                    continue
                if abs(gamma - 1.0) > 1e-3:
                    advisory.append(
                        f"{name} is {properties[name]}: the camera applies a gamma curve, so counts "
                        "are not proportional to light. The Stokes inversion needs a linear camera. "
                        "Set it to 1.0 in Micro-Manager."
                    )

    if bit_depth is not None and bit_depth <= 8:
        advisory.append(
            f"The camera is delivering {bit_depth}-bit frames. The swing states differ from "
            "extinction by about 1 percent of the open intensity, which 8 bits resolve to a "
            "couple of counts; retardance from such data is quantised. Select a 12- or 16-bit "
            "pixel format (and the deepest ADC the camera offers) in Micro-Manager, then "
            "re-measure the black level in that format."
        )
    return fatal, advisory
