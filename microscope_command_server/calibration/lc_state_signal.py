"""How bright each LC-PolScope state is at one exposure (LCSIGNAL).

The Live Viewer Camera tab sets ONE exposure for all five states. To choose it
the operator needs to see every state at that exposure against the camera's
range, not just whichever state happens to be live: the extinction state is
near black by design, and the brightest swing state is the one that clips
first. This applies each state in turn exactly as the acquisition does (the
channel's presets and device properties, then the exposure under test), snaps,
and reports per-state statistics plus the exposure that would put the brightest
swing state at a target fraction of full scale.

It only reads the camera and switches states. The state left applied at the end
is the last one measured; the caller re-applies whatever it wants live.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Where the brightest swing state's 99.9th percentile should land, as a
#: fraction of full scale, when suggesting an exposure. Below the calibration's
#: EXPOSURE_HEADROOM_FRACTION (0.9, applied there to the single brightest
#: pixel) so that a brighter part of the slide still fits.
DEFAULT_TARGET_FRACTION = 0.75
#: The percentile reported as "bright". The single brightest pixel is set by
#: one hot pixel or one speck; the 99.9th is the bright part of the field.
BRIGHT_PERCENTILE = 99.9


def frame_stats(frame: np.ndarray, full_scale: Optional[float]) -> Dict[str, Optional[float]]:
    """Mean, 99.9th percentile, max and clipped fraction of one frame."""
    data = np.asarray(frame)
    # A 5 MP frame: the percentile of a 16x subsample is the same number for a
    # few ms instead of the cost of a full sort.
    step = max(1, int(np.sqrt(data.size / 400_000)))
    sample = data[tuple(slice(None, None, step) for _ in data.shape)]
    top = full_scale
    if top is None and np.issubdtype(data.dtype, np.integer):
        top = float(np.iinfo(data.dtype).max)
    return {
        "mean": float(data.astype(np.float64).mean()),
        "p999": float(np.percentile(sample, BRIGHT_PERCENTILE)),
        "max": float(data.max()),
        "saturated_fraction": float(np.mean(data >= top)) if top is not None else None,
    }


def suggest_exposure_ms(
    exposure_ms: float,
    bright_counts: float,
    full_scale: float,
    black_level: float = 0.0,
    target_fraction: float = DEFAULT_TARGET_FRACTION,
) -> Optional[float]:
    """Exposure that moves ``bright_counts`` (measured at ``exposure_ms``) to
    ``target_fraction`` of full scale, assuming a linear camera above its black
    level. None when the measurement carries no light above the black level or
    is clipped (a clipped value says only "at least this bright")."""
    light = bright_counts - black_level
    if light <= 0 or bright_counts >= full_scale or exposure_ms <= 0:
        return None
    wanted = target_fraction * full_scale - black_level
    if wanted <= 0:
        return None
    return round(exposure_ms * wanted / light, 1)


def measure_lc_states(
    hardware,
    settings,
    profile_name: str,
    exposure_ms: Optional[float] = None,
    target_fraction: float = DEFAULT_TARGET_FRACTION,
) -> dict:
    """Apply every state of ``profile_name`` at one exposure and measure it.

    ``exposure_ms`` None measures at the camera's current exposure. Returns a
    JSON-ready dict; problems are reported in ``error`` rather than raised when
    the cause is the configuration.
    """
    from microscope_command_server.acquisition.workflow import apply_channel_hardware_state
    from microscope_command_server.calibration.camera_checks import (
        camera_setting_problems,
        camera_snapshot,
    )
    from microscope_command_server.modality.lcpolscope import EXTINCTION_CHANNEL_ID
    from microscope_command_server.server.handlers.illumination import (
        profile_channel_entry,
        resolve_profile_modality,
    )

    if not 0.0 < target_fraction < 1.0:
        return {"error": f"target fraction must be between 0 and 1, got {target_fraction}"}
    resolved = resolve_profile_modality(settings, profile_name)
    if resolved is None:
        return {"error": f"unknown acquisition profile '{profile_name}'"}
    modality_name, modality_cfg, profile_cfg = resolved
    library = [
        e for e in (modality_cfg.get("channels") or []) if isinstance(e, dict) and e.get("id")
    ]
    if not library:
        return {"error": f"modality '{modality_name}' has no channel library"}
    recon = modality_cfg.get("reconstruction") or {}
    black_level = float(recon.get("black_level") or 0.0)

    summary, properties = camera_snapshot(hardware)
    bits = summary.get("significant_bits")
    full_scale = float(2**bits - 1) if bits else None
    fatal, advisory = camera_setting_problems(properties, bits or summary.get("bit_depth"))

    if exposure_ms is None or exposure_ms <= 0:
        exposure_ms = float(hardware.get_exposure())
    exposure_ms = float(exposure_ms)

    states: List[dict] = []
    for entry in library:
        channel_id = entry["id"]
        ch_entry = profile_channel_entry(modality_cfg, profile_cfg, channel_id)
        apply_channel_hardware_state(hardware, ch_entry, logger)
        # After the channel: a channel entry may carry its own exposure, and
        # the point here is every state at the SAME exposure.
        hardware.set_exposure(exposure_ms)
        frame, _ = hardware.snap_image()
        stats = frame_stats(frame, full_scale)
        stats["id"] = channel_id
        stats["extinction"] = channel_id == EXTINCTION_CHANNEL_ID
        states.append(stats)

    swing = [s for s in states if not s["extinction"]]
    report = {
        "profile": profile_name,
        "exposure_ms": exposure_ms,
        "full_scale": full_scale,
        "significant_bits": bits,
        "black_level": black_level,
        "target_fraction": target_fraction,
        "states": states,
        "camera_fatal": fatal,
        "camera_advisory": advisory,
    }
    if swing and full_scale:
        brightest = max(swing, key=lambda s: s["p999"])
        report["brightest_state"] = brightest["id"]
        report["brightest_fraction"] = round(brightest["p999"] / full_scale, 4)
        report["clipped"] = any((s["saturated_fraction"] or 0.0) > 0.001 for s in states)
        report["suggested_exposure_ms"] = suggest_exposure_ms(
            exposure_ms, brightest["p999"], full_scale, black_level, target_fraction
        )
    logger.info(
        "LCSIGNAL %s at %.1f ms: %s",
        profile_name,
        exposure_ms,
        ", ".join(f"{s['id']} mean {s['mean']:.0f} p99.9 {s['p999']:.0f}" for s in states),
    )
    return report
