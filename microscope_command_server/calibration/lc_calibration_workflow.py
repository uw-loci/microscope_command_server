"""Running an LC-PolScope calibration on the microscope.

A thin orchestrator, in the shape ``sunburst_workflow`` established: hardware
and config are injected, artifacts are written to an output folder, and a plain
dict comes back on every path including failure. Nothing here decides how to
search -- that is the library's job.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

#: Used only when the microscope YAML declares no wavelength, and always with
#: a warning. 546 nm is what OpenPolScope persists in its own registry entry
#: (ps.acq.MeadowlarkLC.wavelength) for this instrument, and what its device
#: adapter converts waves to volts at. The lamp filter was read on 2026-10-13:
#: a Semrock FF01-549/15, so the LC-PolScope's YAML says 549 and this default
#: is only ever a placeholder for a YAML that says nothing.
DEFAULT_WAVELENGTH_NM = 546.0


def _adapter_wavelength_nm(device_properties: Dict[str, str]) -> Optional[float]:
    """The LC adapter's own wavelength setting, in nm, if it exposes one.

    Matched by name rather than hard-coded: the MeadowlarkLC adapter calls it
    "Wavelength", other adapters may not, and a missing property is simply
    "unknown", not an error.
    """
    for name, value in device_properties.items():
        if "wavelength" in name.lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


DEFAULT_SETTLE_MS = 50.0

#: The brightest frame of the extinction grid must average at least this
#: fraction of the camera's full scale, or the run is refused as too dark. A
#: tenth: the grid visits settings far from extinction, which at any usable
#: exposure are far brighter than the swing states, so a field that cannot
#: reach a tenth of full scale anywhere has nothing for the search to see.
#: The two first rig runs peaked at 16.7 counts (2026-10-12 and -13).
MIN_GRID_PEAK_FRACTION = 0.10


def _describe_camera(camera: Dict[str, Any]) -> str:
    """One line for the log and an error message: ``Oryx, 50.0 ms (request), 12-bit, 2448x2048, binning 1``."""
    parts = []
    if camera.get("device"):
        parts.append(str(camera["device"]))
    if camera.get("exposure_ms") is not None:
        source = camera.get("exposure_source")
        parts.append(f"{camera['exposure_ms']:g} ms" + (f" ({source})" if source else ""))
    if camera.get("bit_depth") is not None:
        parts.append(f"{camera['bit_depth']}-bit")
    if camera.get("pixel_type"):
        parts.append(str(camera["pixel_type"]))
    if camera.get("width") is not None and camera.get("height") is not None:
        parts.append(f"{camera['width']}x{camera['height']}")
    if camera.get("binning") is not None:
        parts.append(f"binning {camera['binning']}")
    if camera.get("gain") is not None:
        parts.append(f"gain {camera['gain']}")
    return ", ".join(parts) if parts else "unknown (the camera could not be asked)"


def run_lc_calibration(
    hardware,
    settings_yaml: Optional[Dict[str, Any]] = None,
    *,
    output_folder: str,
    modality: str = "lcpolscope",
    swing_waves: Optional[float] = None,
    scheme: Optional[str] = None,
    wavelength_nm: Optional[float] = None,
    black_level: Optional[float] = None,
    exposure_ms: Optional[float] = None,
    strategy: str = "single_pass",
    lc_control_mode: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    logger_: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Calibrate the liquid crystals and write the result.

    Every parameter left as ``None`` is taken from
    ``modalities.<modality>.reconstruction`` in the microscope YAML, so the
    calibration and the acquisition cannot drift apart. ``exposure_ms`` is the
    exception: ``None`` means "whatever Micro-Manager is set to", and either
    way the exposure used is recorded in the result, with the camera's bit
    depth and pixel format, because a frame mean in counts is unreadable
    without them.

    Returns
    -------
    dict
        Always. ``success`` says whether a palette was produced; ``warnings``
        collects anything the operator should see, including a poor extinction
        ratio -- which is reported, never treated as a failure.
    """
    log = logger_ or logger
    warnings: list = []
    out = Path(output_folder)

    def fail(message: str) -> Dict[str, Any]:
        log.error("LC calibration failed: %s", message)
        return {"success": False, "error": message, "warnings": warnings, "output_folder": str(out)}

    try:
        from polscope_library.calibration import (
            CalibrationSettings,
            InsufficientSignalError,
            calibrate,
            moves_for,
        )
    except ImportError as exc:
        return fail(f"polscope_library is not installed: {exc}")

    from .lc_instrument import LC_DEVICE, RETARDANCE_PROPERTY, MicroManagerLC

    recon = ((settings_yaml or {}).get("modalities", {}).get(modality, {}) or {}).get(
        "reconstruction", {}
    ) or {}
    swing_waves = float(swing_waves if swing_waves is not None else recon.get("swing_waves", 0.03))
    scheme = scheme or recon.get("scheme", "5-State")

    # Wavelength is a pure scale on reported retardance, so a wrong one is
    # invisible -- the maps look right and are uniformly off. It is also the
    # value we are least sure of on this instrument. So a fallback is used
    # loudly and recorded, never applied in silence: if the calibration is
    # later compared against an acquisition, the two must have agreed, and a
    # silent default is exactly how they would not.
    if wavelength_nm is not None:
        wavelength_nm = float(wavelength_nm)
        wavelength_source = "request"
    elif recon.get("wavelength_nm") is not None:
        wavelength_nm = float(recon["wavelength_nm"])
        wavelength_source = "config"
    else:
        wavelength_nm = float(DEFAULT_WAVELENGTH_NM)
        wavelength_source = "fallback"
        warnings.append(
            f"modalities.{modality}.reconstruction.wavelength_nm is not set; falling back to "
            f"{DEFAULT_WAVELENGTH_NM} nm. Retardance is reported in nanometres by scaling with "
            "this value, so if the acquisition uses a different one the two will disagree by "
            "that ratio with nothing to show for it. Set it in the microscope YAML."
        )
    # Black level: the request, else the YAML, else the library measures one or
    # assumes 0. The YAML value is the camera's own offset in counts (on the
    # LC-PolScope it is OpenPolScope's stored 103, measured at 102 in its
    # data), so it is a number; "assumed 0" is not, and it overstates the
    # extinction ratio by about 10 percent per 50 counts. The first rig run
    # (2026-10-12) reported "0.0 (assumed)" with 103 sitting in the YAML. A
    # YAML 0 means "no clamp" to the reconstruction, not a measurement, so it
    # is not used here.
    if black_level is not None:
        black_level = float(black_level)
        black_level_origin: Optional[str] = "request"
    elif recon.get("black_level"):
        black_level = float(recon["black_level"])
        black_level_origin = f"config: modalities.{modality}.reconstruction.black_level"
    else:
        black_level_origin = None

    mode = lc_control_mode or recon.get("lc_control_mode", "MM-Retardance")

    if mode == "MM-Voltage":
        # Deliberately refused rather than silently falling back: the operator
        # asked for voltage control, and quietly giving them retardance
        # control would produce a working calibration stored the wrong way.
        return fail(
            "MM-Voltage needs a retardance/voltage curve, which is not wired up yet. "
            "Set lc_control_mode: MM-Retardance to calibrate now -- the device adapter "
            "does the conversion, and the resulting palette is equally valid."
        )

    try:
        instrument = MicroManagerLC(
            hardware, mode=mode, settle_ms=recon.get("lc_settle_ms", DEFAULT_SETTLE_MS)
        )
    except Exception as exc:
        return fail(f"could not reach the liquid crystals: {exc}")

    # The adapter converts the palette's waves to volts at ITS wavelength. If
    # that is not the reconstruction wavelength, the swing the crystals apply
    # is swing * adapter / reconstruction waves, and every retardance comes
    # out scaled by the same ratio with nothing to show for it. Recorded, and
    # a disagreement is said out loud; it is not corrected here because the
    # right fix is to make the two agree once the filter is identified.
    lc_device_properties = instrument.device_snapshot()
    lc_adapter_wavelength_nm = _adapter_wavelength_nm(lc_device_properties)
    if lc_adapter_wavelength_nm is not None and abs(lc_adapter_wavelength_nm - wavelength_nm) > 0.5:
        warnings.append(
            f"The LC adapter converts waves to volts at {lc_adapter_wavelength_nm:g} nm but the "
            f"reconstruction wavelength is {wavelength_nm:g} nm. The palette is reproducible "
            "either way, but the swing the crystals apply is then "
            f"{swing_waves * lc_adapter_wavelength_nm / wavelength_nm:.5f} waves at the "
            f"reconstruction wavelength instead of {swing_waves:g}, a "
            f"{abs(lc_adapter_wavelength_nm / wavelength_nm - 1) * 100:.2f} percent retardance "
            "scale error. Set the adapter wavelength and modalities.lcpolscope.reconstruction."
            "wavelength_nm to the same value."
        )

    # Exposure: set when asked, and always recorded. The calibration measures
    # whole-frame means in raw counts; without the exposure and the pixel
    # format those numbers cannot be read later, and the first two rig runs
    # (frame means of 1 to 17 counts) could not be diagnosed for exactly that
    # reason. The previous exposure is restored on every exit path so a
    # calibration never leaves the camera where the acquisition did not set it.
    restore_exposure_ms: Optional[float] = None
    if exposure_ms is not None:
        try:
            restore_exposure_ms = float(hardware.get_exposure())
            hardware.set_exposure(float(exposure_ms))
        except Exception as exc:
            return fail(f"could not set the camera exposure to {exposure_ms:g} ms: {exc}")
        exposure_source = "request"
    else:
        exposure_source = "micro-manager"
    camera, camera_properties = instrument.camera_snapshot()
    camera["exposure_source"] = exposure_source
    full_scale: Optional[float] = None
    if isinstance(camera.get("bit_depth"), int) and camera["bit_depth"] > 0:
        full_scale = float(2 ** camera["bit_depth"] - 1)
    log.info("LC calibration camera: %s", _describe_camera(camera))
    if full_scale is None:
        warnings.append(
            "The camera did not report its bit depth, so the calibration could not check that "
            "the field is bright enough; read the frame means against the pixel format by hand."
        )

    def restore_exposure() -> None:
        if restore_exposure_ms is None:
            return
        try:
            hardware.set_exposure(restore_exposure_ms)
        except Exception as exc:
            log.error(
                "Could not restore the camera exposure to %g ms: %s", restore_exposure_ms, exc
            )

    def report(step: int, total: int, stage: str, message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(step, total, stage, message)
            except Exception as exc:  # a failing callback must not kill a run
                log.debug("progress callback raised: %s", exc)

    log.info(
        "LC calibration starting: scheme=%s swing=%s wavelength=%snm mode=%s strategy=%s",
        scheme,
        swing_waves,
        wavelength_nm,
        mode,
        strategy,
    )
    report(0, 1, "calibrate", f"{scheme} at swing {swing_waves} waves")

    started = time.time()
    try:
        result = calibrate(
            instrument,
            CalibrationSettings(
                swing_waves=swing_waves,
                scheme=scheme,
                wavelength_nm=wavelength_nm,
                black_level=black_level,
                min_grid_peak=(
                    MIN_GRID_PEAK_FRACTION * full_scale if full_scale is not None else None
                ),
            ),
            strategy=strategy,
        )
    except InsufficientSignalError as exc:
        # Not a crash: the search was stopped on purpose, with the camera state
        # it was stopped under, because that state is the thing to change.
        failure = fail(f"{exc} Camera: {_describe_camera(camera)}.")
        failure["camera"] = camera
        restore_exposure()
        return failure
    except Exception as exc:
        log.exception("LC calibration raised")
        restore_exposure()
        return fail(str(exc))
    restore_exposure()

    if should_abort is not None and should_abort():
        return {
            "success": False,
            "aborted": True,
            "error": "cancelled by the operator",
            "warnings": warnings,
            "output_folder": str(out),
        }

    warnings.extend(result.warnings)

    # What the camera saw, in one place: the counts the ratio was computed
    # from, against the camera's range. This is the line that says "dark" or
    # "fine" without anyone opening the trace.
    extinction_move = moves_for(result.scheme)[0].name
    grid_points = [p.intensity for p in result.trace if p.state == extinction_move]
    state_values = list(result.intensities.values())
    signal: Dict[str, Any] = {
        "extinction": round(state_values[0], 2) if state_values else None,
        "brightest_state": round(max(state_values), 2) if state_values else None,
        "grid_peak": round(max(grid_points), 2) if grid_points else None,
        "full_scale": full_scale,
        "frame_dtype": instrument.frame_dtype,
        "frame_max_seen": instrument.frame_max_seen,
    }
    if full_scale and signal["grid_peak"] is not None:
        signal["grid_peak_fraction"] = round(signal["grid_peak"] / full_scale, 4)
    if result.black_level is not None and state_values and result.black_level > state_values[0]:
        # 103 above frames that never exceed 17: the number is not in these
        # units. OpenPolScope recorded its 103 in its own pixel format, and a
        # camera switched to another format (or 8-bit) has a different offset.
        warnings.append(
            f"Black level {result.black_level:.1f} is above the darkest state measured "
            f"({state_values[0]:.1f} counts, camera {_describe_camera(camera)}). It cannot be "
            "this camera's offset in these units -- OpenPolScope's 103 was recorded in its own "
            "pixel format -- so the extinction ratio is unmeasurable. Measure the offset in the "
            "current pixel format (a frame with the lamp blocked) and set "
            f"modalities.{modality}.reconstruction.black_level to it."
        )

    payload = {
        "success": True,
        "scheme": result.scheme,
        "swing_waves": result.swing_waves,
        "wavelength_nm": result.wavelength_nm,
        "wavelength_source": wavelength_source,
        "lc_control_mode": mode,
        "strategy": strategy,
        "black_level": result.black_level,
        "black_level_source": black_level_origin or result.black_level_source,
        "extinction_ratio": round(result.extinction_ratio, 2),
        "assessment": result.assessment,
        # Retardance in WAVES, (LC-A, LC-B), keyed by the Micro-Manager channel
        # the state becomes. Units are in the key on purpose: this crosses the
        # socket and is written into the microscope YAML by QPSC.
        "palette_waves": {state: list(values) for state, values in result.palette_waves.items()},
        # What QPSC writes the palette to so the acquisition reproduces it.
        # The adapter converts waves to volts; no voltage curve is involved.
        "lc_device": LC_DEVICE,
        "lc_retardance_properties_waves": dict(RETARDANCE_PROPERTY),
        "lc_adapter_wavelength_nm": lc_adapter_wavelength_nm,
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state_intensities": result.intensities,
        "exposures": result.exposures,
        "elapsed_s": round(time.time() - started, 1),
        # The camera as the frames were taken, and what the frames held. A
        # frame mean in counts means nothing without the first, and the
        # second is the one-line answer to "was the field bright enough".
        "camera": camera,
        "signal": signal,
        "warnings": warnings,
        "output_folder": str(out),
    }

    try:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"lc_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
        # The trace is written but kept out of the returned payload: it is one
        # entry per exposure and the socket reply should stay small.
        on_disk = dict(payload)
        # The adapter state the palette was measured against. File only: it
        # is a dozen strings the socket reply does not need.
        on_disk["lc_device_properties"] = lc_device_properties
        on_disk["camera_properties"] = camera_properties
        on_disk["trace"] = [
            {
                "state": p.state,
                "lca_waves": p.lca_waves,
                "lcb_waves": p.lcb_waves,
                "intensity": p.intensity,
            }
            for p in result.trace
        ]
        path.write_text(json.dumps(on_disk, indent=1))
        payload["metadata_path"] = str(path)
        log.info("LC calibration written to %s", path)
    except Exception as exc:
        # The calibration itself succeeded; losing the file is worth a warning,
        # not a failure that discards a good palette.
        warnings.append(f"calibration succeeded but could not be written to {out}: {exc}")
        log.error("Could not write LC calibration metadata: %s", exc)

    log.info(
        "LC calibration done: ER %.1f (%s), %d exposures in %.1fs",
        result.extinction_ratio,
        result.assessment,
        result.exposures,
        payload["elapsed_s"],
    )
    # The palette and the black level are what the operator acts on, and the
    # dialog that shows them is not a record; the log is.
    log.info(
        "LC calibration black level %.1f (%s); palette_waves=%s",
        result.black_level,
        payload["black_level_source"],
        payload["palette_waves"],
    )
    log.info(
        "LC calibration signal: extinction %s, brightest state %s, grid peak %s of %s%s; camera %s",
        signal["extinction"],
        signal["brightest_state"],
        signal["grid_peak"],
        "unknown full scale" if full_scale is None else f"{full_scale:g}",
        (
            f" ({signal['grid_peak_fraction'] * 100:.1f}%)"
            if signal.get("grid_peak_fraction") is not None
            else ""
        ),
        _describe_camera(camera),
    )
    for message in warnings:
        log.warning("LC calibration: %s", message)
    report(
        1, 1, "calibrate", f"extinction ratio {result.extinction_ratio:.1f} ({result.assessment})"
    )
    return payload
