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
#: a warning. 549 nm is what the OpenPolScope acquisition summary records for
#: this instrument -- the same role this value plays -- but the interference
#: filter has never been identified, so treat it as a placeholder.
DEFAULT_WAVELENGTH_NM = 549.0

DEFAULT_SETTLE_MS = 50.0


def run_lc_calibration(
    hardware,
    settings_yaml: Optional[Dict[str, Any]] = None,
    *,
    output_folder: str,
    modality: str = "lcpolscope",
    swing: Optional[float] = None,
    scheme: Optional[str] = None,
    wavelength_nm: Optional[float] = None,
    black_level: Optional[float] = None,
    strategy: str = "single_pass",
    lc_control_mode: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    logger_: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Calibrate the liquid crystals and write the result.

    Every parameter left as ``None`` is taken from
    ``modalities.<modality>.reconstruction`` in the microscope YAML, so the
    calibration and the acquisition cannot drift apart.

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
        from polscope_library.calibration import CalibrationSettings, calibrate
    except ImportError as exc:
        return fail(f"polscope_library is not installed: {exc}")

    from .lc_instrument import MicroManagerLC

    recon = ((settings_yaml or {}).get("modalities", {}).get(modality, {}) or {}).get(
        "reconstruction", {}
    ) or {}
    swing = float(swing if swing is not None else recon.get("swing_waves", 0.03))
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

    def report(step: int, total: int, stage: str, message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(step, total, stage, message)
            except Exception as exc:  # a failing callback must not kill a run
                log.debug("progress callback raised: %s", exc)

    log.info(
        "LC calibration starting: scheme=%s swing=%s wavelength=%snm mode=%s strategy=%s",
        scheme,
        swing,
        wavelength_nm,
        mode,
        strategy,
    )
    report(0, 1, "calibrate", f"{scheme} at swing {swing}")

    started = time.time()
    try:
        result = calibrate(
            instrument,
            CalibrationSettings(
                swing=swing, scheme=scheme, wavelength_nm=wavelength_nm, black_level=black_level
            ),
            strategy=strategy,
        )
    except Exception as exc:
        log.exception("LC calibration raised")
        return fail(str(exc))

    if should_abort is not None and should_abort():
        return {
            "success": False,
            "aborted": True,
            "error": "cancelled by the operator",
            "warnings": warnings,
            "output_folder": str(out),
        }

    warnings.extend(result.warnings)
    payload = {
        "success": True,
        "scheme": result.scheme,
        "swing_waves": result.swing,
        "wavelength_nm": result.wavelength_nm,
        "wavelength_source": wavelength_source,
        "lc_control_mode": mode,
        "strategy": strategy,
        "black_level": result.black_level,
        "black_level_source": result.black_level_source,
        "extinction_ratio": round(result.extinction_ratio, 2),
        "assessment": result.assessment,
        # Waves, keyed by the Micro-Manager channel the state becomes.
        "palette": {state: list(values) for state, values in result.palette.items()},
        "state_intensities": result.intensities,
        "exposures": result.exposures,
        "elapsed_s": round(time.time() - started, 1),
        "warnings": warnings,
        "output_folder": str(out),
    }

    try:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"lc_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
        # The trace is written but kept out of the returned payload: it is one
        # entry per exposure and the socket reply should stay small.
        on_disk = dict(payload)
        on_disk["trace"] = [
            {"state": p.state, "lca": p.lca, "lcb": p.lcb, "intensity": p.intensity}
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
    report(
        1, 1, "calibrate", f"extinction ratio {result.extinction_ratio:.1f} ({result.assessment})"
    )
    return payload
