"""Measure each liquid crystal's own voltage-to-retardance curve (LCSWEEP).

The palette is set in the Meadowlark adapter's waves, which the adapter turns
into volts through its own table (``mmgr_dal_MeadowlarkLC.csv``). That table
is not a measurement of these crystals, and on the LC-PolScope its LC-B column
is kinked across LC-B's working range (4.1-4.6 V; 2026-10-28), which is the
leading explanation for the lopsided LC-B palette. This measures the truth.

Per crystal: park the partner where :data:`polscope_library.calibration.
SWEEP_GEOMETRY` says (LC-B at a half wave to sweep LC-A; LC-A at a QUARTER
wave to sweep LC-B, whose sweep is flat at a half wave), drive the swept
crystal's VOLTAGE directly so the adapter's table is out of the loop, snap a
clear field at every step, and hand the fringes to
:func:`polscope_library.calibration.curve_from_sweep`.

Three things are reported beside the measured curve:

- the adapter's own table, sampled by writing retardances and reading back
  the volts it chose (exactly the conversion the rig uses; no file path);
- for each palette state in the YAML, the commanded waves, the volts the
  adapter drives, and the waves the crystal REALLY has at those volts;
- the measured curve as a plain CSV (millivolts, nm at the reconstruction
  wavelength). Not in the adapter's multi-wavelength format and never
  installed: replacing the adapter's table is a human decision.

The exposure is chosen for the sweep, not reused: a sweep passes maximum
transmission, which at the calibration exposure is several times the
camera's full scale.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Brightest sweep frame, as a fraction of the camera's range above black.
TARGET_PEAK_FRACTION = 0.7
#: Points in the coarse pre-sweep that sets the exposure.
PRESWEEP_POINTS = 41
#: Retardances written to sample the adapter's table, in waves.
TABLE_SAMPLE_WAVES = np.round(np.arange(0.05, 1.70, 0.01), 4)
#: Table samples within this many volts of a voltage limit are clamped there, not real.
RAIL_MARGIN_V = 0.05
#: Half-width, in volts, around the palette over which the table is judged; the curve is flat
#: and poorly measured near the rails, which are not where the palette works.
PALETTE_WINDOW_V = 0.75
#: Saturated-pixel fraction above which a sweep frame counts as clipped.
CLIPPED_FRACTION = 0.001


def _palette_from_yaml(settings_yaml, modality: str) -> Dict[str, List[float]]:
    """The applied palette, state -> [LC-A, LC-B] waves, from the modality's channels."""
    from .lc_instrument import RETARDANCE_PROPERTY

    channels = ((settings_yaml or {}).get("modalities", {}).get(modality, {}) or {}).get(
        "channels"
    ) or []
    out: Dict[str, List[float]] = {}
    for ch in channels:
        values = {}
        for dp in ch.get("device_properties") or []:
            for axis, prop in RETARDANCE_PROPERTY.items():
                if dp.get("property") == prop:
                    with contextlib.suppress(TypeError, ValueError):
                        values[axis] = float(dp.get("value"))
        if "LCA" in values and "LCB" in values and ch.get("id"):
            out[str(ch["id"])] = [values["LCA"], values["LCB"]]
    return out


def _voltage_limits(core, device: str, prop: str) -> tuple:
    try:
        if core.has_property_limits(device, prop):
            return (
                float(core.get_property_lower_limit(device, prop)),
                float(core.get_property_upper_limit(device, prop)),
            )
    except Exception:
        pass
    return (0.0, 20.0)


def run_lc_voltage_sweep(
    hardware,
    settings_yaml: Optional[Dict[str, Any]] = None,
    *,
    output_folder: str,
    modality: str = "lcpolscope",
    axes: Sequence[str] = ("LCA", "LCB"),
    start_v: Optional[float] = None,
    stop_v: Optional[float] = None,
    step_v: float = 0.02,
    exposure_ms: Optional[float] = None,
    roi_fraction: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    logger_: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Sweep each requested crystal, write ``lc_sweep_<stamp>.json``, return a summary.

    ``start_v`` / ``stop_v`` default to the adapter's voltage limits (0-20 V on
    the D5020). The labelling needs the sweep to reach below the last fringe,
    so stopping early is refused rather than mislabelled. ``exposure_ms`` set
    skips the automatic choice. The crystals, the exposure and the camera ROI
    are restored on every exit path.

    Returns a dict with ``success`` always present.
    """
    log = logger_ or logger
    warnings: list = []
    out = Path(output_folder)

    def fail(message: str) -> Dict[str, Any]:
        log.error("LC voltage sweep failed: %s", message)
        return {"success": False, "error": message, "warnings": warnings, "output_folder": str(out)}

    try:
        from polscope_library.calibration import SWEEP_GEOMETRY, curve_from_sweep
    except ImportError as exc:
        return fail(
            f"polscope_library is missing or too old for the sweep ({exc}); pull and reinstall it."
        )

    from .lc_calibration_workflow import DEFAULT_SETTLE_MS, prepare_extinction_hardware
    from .lc_instrument import LC_DEVICE, RETARDANCE_PROPERTY, VOLTAGE_PROPERTY, MicroManagerLC

    axes = [a for a in axes if a in SWEEP_GEOMETRY]
    if not axes:
        return fail("no crystal to sweep: axes must be LCA and/or LCB")
    recon = ((settings_yaml or {}).get("modalities", {}).get(modality, {}) or {}).get(
        "reconstruction", {}
    ) or {}
    wavelength_nm = float(recon.get("wavelength_nm") or 546.0)
    black = float(recon.get("black_level") or 0.0)
    settle_s = float(recon.get("lc_settle_ms", DEFAULT_SETTLE_MS)) / 1000.0

    def report(step: int, total: int, stage: str, message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(step, total, stage, message)
            except Exception as exc:
                log.debug("progress callback raised: %s", exc)

    warnings.extend(prepare_extinction_hardware(hardware, settings_yaml, modality, log))
    try:
        instrument = MicroManagerLC(hardware, settle_ms=0.0)
        core = instrument._core()
    except Exception as exc:
        return fail(f"could not reach the liquid crystals: {exc}")

    # What to put back afterwards.
    restore_waves: Dict[str, Optional[float]] = {}
    for axis in ("LCA", "LCB"):
        try:
            restore_waves[axis] = float(core.get_property(LC_DEVICE, RETARDANCE_PROPERTY[axis]))
        except Exception:
            restore_waves[axis] = None
    try:
        restore_exposure_ms = float(hardware.get_exposure())
    except Exception:
        restore_exposure_ms = None
    camera, _ = instrument.camera_snapshot()
    bits = camera.get("significant_bits", camera.get("bit_depth"))
    full_scale = float(2**bits - 1) if isinstance(bits, int) and bits > 0 else None
    instrument.full_scale = full_scale
    roi = instrument.apply_roi(roi_fraction)

    def set_volts(axis: str, volts: float) -> None:
        core.set_property(LC_DEVICE, VOLTAGE_PROPERTY[axis], f"{volts:.4f}")
        with contextlib.suppress(Exception):
            core.wait_for_device(LC_DEVICE)
        if settle_s > 0:
            time.sleep(settle_s)

    def set_waves(axis: str, waves: float, settle: bool = True) -> None:
        core.set_property(LC_DEVICE, RETARDANCE_PROPERTY[axis], f"{waves:.4f}")
        with contextlib.suppress(Exception):
            core.wait_for_device(LC_DEVICE)
        if settle and settle_s > 0:
            time.sleep(settle_s)

    def snap() -> tuple:
        mean = instrument.measure_intensity()
        frame = instrument.frames[-1]
        clipped = (frame.get("saturated_fraction") or 0.0) > CLIPPED_FRACTION
        return mean, clipped

    stamp = time.strftime("%Y%m%d_%H%M%S")
    record: Dict[str, Any] = {
        "kind": "lc_voltage_sweep",
        "swept_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modality": modality,
        "wavelength_nm": wavelength_nm,
        "black_level": black,
        "camera": camera,
        "roi": roi,
        "settle_ms": settle_s * 1000.0,
        "lc_device_properties_before": instrument.device_snapshot(),
        "axes": {},
    }

    try:
        total_axes = len(axes)
        for n, axis in enumerate(axes):
            g = SWEEP_GEOMETRY[axis]
            lo, hi = _voltage_limits(core, LC_DEVICE, VOLTAGE_PROPERTY[axis])
            v0 = lo if start_v is None else max(lo, float(start_v))
            v1 = hi if stop_v is None else min(hi, float(stop_v))
            volts = np.round(np.arange(v0, v1 + 1e-9, float(step_v)), 4)
            entry: Dict[str, Any] = {
                "partner": g.partner,
                "partner_waves": g.partner_waves,
                "offset_waves": g.offset_waves,
                "range_v": [float(v0), float(v1)],
                "step_v": float(step_v),
            }
            record["axes"][axis] = entry
            if len(volts) < 20:
                entry["error"] = f"only {len(volts)} sweep points between {v0} and {v1} V"
                continue

            # The adapter's own table: write waves, read the volts it chose.
            report(
                n * 3,
                total_axes * 3,
                "table",
                f"{axis}: reading the adapter's waves-to-volts table",
            )
            table_w, table_v = [], []
            for w in TABLE_SAMPLE_WAVES:
                try:
                    set_waves(axis, float(w), settle=False)
                    volts_read = float(core.get_property(LC_DEVICE, VOLTAGE_PROPERTY[axis]))
                except Exception:
                    continue
                # A retardance the crystal cannot reach is clamped to a voltage rail by the
                # adapter: that pair says nothing about the table, so it is dropped.
                if volts_read <= lo + RAIL_MARGIN_V or volts_read >= hi - RAIL_MARGIN_V:
                    continue
                table_v.append(volts_read)
                table_w.append(float(w))
            entry["adapter_table"] = {"waves": table_w, "volts": table_v}

            set_waves(g.partner, g.partner_waves)

            # Exposure: the brightest point of a coarse pre-sweep at ~70% of range.
            if exposure_ms is not None:
                hardware.set_exposure(float(exposure_ms))
                entry["exposure_source"] = "request"
            else:
                report(n * 3 + 1, total_axes * 3, "exposure", f"{axis}: choosing the exposure")
                coarse = np.linspace(v0, v1, PRESWEEP_POINTS)
                exp = float(hardware.get_exposure())
                for _attempt in range(8):
                    peaks, any_clipped = [], False
                    for v in coarse:
                        set_volts(axis, float(v))
                        mean, clipped = snap()
                        peaks.append(mean)
                        any_clipped |= clipped
                    peak = max(peaks)
                    if any_clipped and exp > 0.05:
                        exp = exp / 4.0
                        hardware.set_exposure(exp)
                        continue
                    if full_scale is not None and peak - black > 0:
                        exp = exp * TARGET_PEAK_FRACTION * (full_scale - black) / (peak - black)
                        hardware.set_exposure(exp)
                    break
                entry["exposure_source"] = "auto"
            entry["exposure_ms"] = float(hardware.get_exposure())

            # The sweep, low to high voltage.
            report(n * 3 + 2, total_axes * 3, "sweep", f"{axis}: sweeping {len(volts)} points")
            intensities, clipped_points = [], 0
            for i, v in enumerate(volts):
                set_volts(axis, float(v))
                mean, clipped = snap()
                intensities.append(mean)
                clipped_points += int(clipped)
                if i % 100 == 0:
                    report(
                        n * 3 + 2,
                        total_axes * 3,
                        "sweep",
                        f"{axis}: {i}/{len(volts)} at {v:.2f} V",
                    )
            entry["volts"] = [float(v) for v in volts]
            entry["intensities"] = [float(x) for x in intensities]
            entry["clipped_points"] = clipped_points
            if clipped_points:
                entry["error"] = (
                    f"{clipped_points} sweep frames clipped; the fringe extrema cannot be trusted. "
                    "Lower the exposure or the lamp and sweep again."
                )
                continue

            try:
                curve = curve_from_sweep(volts, intensities, offset_waves=g.offset_waves)
            except ValueError as exc:
                entry["error"] = f"fringe analysis failed: {exc}"
                continue
            entry["landmarks"] = [
                {"volts": lm.voltage, "waves": lm.retardance_waves, "kind": lm.kind}
                for lm in curve.landmarks
            ]
            entry["curve"] = {
                "volts": [float(x) for x in curve.voltages],
                "waves": [float(x) for x in curve.retardances],
            }
            entry["fringe_contrast"] = float(
                (max(intensities) - min(intensities)) / max(max(intensities) - black, 1e-9)
            )
            if table_v:
                tw = np.asarray(table_w)
                tv = np.asarray(table_v)
                measured = curve.retardance_for(tv)
                inside = (tv >= curve.voltages.min()) & (tv <= curve.voltages.max())
                entry["table_vs_measured"] = {
                    "volts": [float(x) for x in tv[inside]],
                    "table_waves": [float(x) for x in tw[inside]],
                    "measured_waves": [float(x) for x in measured[inside]],
                }

        # Each palette state: commanded waves, the volts the adapter drives, the real waves.
        palette = _palette_from_yaml(settings_yaml, modality)
        rows = {}
        for state, (lca, lcb) in palette.items():
            row: Dict[str, Any] = {"lca_waves_commanded": lca, "lcb_waves_commanded": lcb}
            try:
                set_waves("LCA", lca, settle=False)
                set_waves("LCB", lcb, settle=False)
                for axis, key in (("LCA", "lca"), ("LCB", "lcb")):
                    volts_now = float(core.get_property(LC_DEVICE, VOLTAGE_PROPERTY[axis]))
                    row[f"{key}_volts"] = volts_now
                    curve_rec = record["axes"].get(axis, {}).get("curve")
                    if curve_rec:
                        row[f"{key}_waves_measured"] = float(
                            np.interp(volts_now, curve_rec["volts"], curve_rec["waves"])
                        )
            except Exception as exc:
                row["error"] = str(exc)
            rows[state] = row
        record["palette"] = rows
        _add_offsets(rows)
    except Exception as exc:
        log.exception("LC voltage sweep raised")
        return fail(str(exc))
    finally:
        for axis, waves in restore_waves.items():
            if waves is not None:
                try:
                    set_waves(axis, waves, settle=False)
                except Exception as exc:
                    log.error(
                        "Could not restore %s to %s waves after the sweep: %s", axis, waves, exc
                    )
        instrument.restore_roi()
        if restore_exposure_ms is not None:
            try:
                hardware.set_exposure(restore_exposure_ms)
            except Exception as exc:
                log.error("Could not restore the exposure after the sweep: %s", exc)

    summary = _summarise(record)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"lc_sweep_{stamp}.json"
    path.write_text(json.dumps(record, indent=1))
    csv_path = out / f"lc_sweep_{stamp}_curve_{wavelength_nm:g}nm.csv"
    _write_curve_csv(csv_path, record, wavelength_nm)
    summary.update(
        success=any("curve" in a for a in record["axes"].values()),
        metadata_path=str(path),
        curve_csv_path=str(csv_path),
        warnings=warnings,
        output_folder=str(out),
    )
    log.info("LC voltage sweep written to %s: %s", path, summary.get("headline"))
    return summary


def _add_offsets(rows: Dict[str, Dict[str, Any]]) -> None:
    """Each swing state's offset from the extinction state, commanded and measured."""
    if not rows:
        return
    ext_state = next(iter(rows))
    ext = rows[ext_state]
    for state, row in rows.items():
        for key in ("lca", "lcb"):
            for kind in ("commanded", "measured"):
                a = row.get(f"{key}_waves_{kind}")
                b = ext.get(f"{key}_waves_{kind}")
                if a is not None and b is not None:
                    row[f"{key}_offset_{kind}"] = a - b
            if row.get(f"{key}_volts") is not None and ext.get(f"{key}_volts") is not None:
                row[f"{key}_offset_volts"] = row[f"{key}_volts"] - ext[f"{key}_volts"]


def _summarise(record: Dict[str, Any]) -> Dict[str, Any]:
    """The socket reply: everything but the raw sweeps."""
    axes = {}
    for axis, entry in record["axes"].items():
        a = {
            k: entry[k] for k in ("partner", "partner_waves", "exposure_ms", "error") if k in entry
        }
        if "landmarks" in entry:
            a["landmarks"] = entry["landmarks"]
            a["fringe_contrast"] = entry.get("fringe_contrast")
        tvm = entry.get("table_vs_measured")
        if tvm and tvm["volts"]:
            v = np.asarray(tvm["volts"])
            d = np.asarray(tvm["table_waves"]) - np.asarray(tvm["measured_waves"])
            key = "lca_volts" if axis == "LCA" else "lcb_volts"
            used = [
                row[key]
                for row in (record.get("palette") or {}).values()
                if row.get(key) is not None
            ]
            if used:
                window = [min(used) - PALETTE_WINDOW_V, max(used) + PALETTE_WINDOW_V]
            else:
                window = [float(v.min()) + 1.0, float(v.max()) - 1.0]
            inside = (v >= window[0]) & (v <= window[1])
            if inside.any():
                worst = int(np.argmax(np.abs(np.where(inside, d, 0.0))))
                a["table_minus_measured_waves"] = {
                    "max_abs": float(abs(d[worst])),
                    "at_volts": float(v[worst]),
                    "window_volts": [float(window[0]), float(window[1])],
                }
        axes[axis] = a
    lines = []
    for state, row in (record.get("palette") or {}).items():
        parts = []
        for key, name in (("lca", "LC-A"), ("lcb", "LC-B")):
            oc, om = row.get(f"{key}_offset_commanded"), row.get(f"{key}_offset_measured")
            if oc is not None and abs(oc) > 1e-4:
                parts.append(
                    f"{name} {oc:+.4f} commanded"
                    + (f" = {om:+.4f} measured" if om is not None else "")
                    + (
                        f" ({row[f'{key}_offset_volts'] * 1000:+.0f} mV)"
                        if row.get(f"{key}_offset_volts") is not None
                        else ""
                    )
                )
        if parts:
            lines.append(f"{state}: " + ", ".join(parts))
    return {
        "axes": axes,
        "palette": record.get("palette", {}),
        "palette_lines": lines,
        "headline": "; ".join(lines) if lines else "no palette offsets to compare",
        "wavelength_nm": record["wavelength_nm"],
    }


def _write_curve_csv(path: Path, record: Dict[str, Any], wavelength_nm: float) -> None:
    """Measured curves as millivolts -> nm, one column per crystal, on a shared voltage grid."""
    curves = {a: e["curve"] for a, e in record["axes"].items() if "curve" in e}
    if not curves:
        return
    lo = max(min(c["volts"]) for c in curves.values())
    hi = min(max(c["volts"]) for c in curves.values())
    if hi <= lo:
        lo = min(min(c["volts"]) for c in curves.values())
        hi = max(max(c["volts"]) for c in curves.values())
    grid = np.round(np.arange(lo, hi + 1e-9, 0.02), 4)
    names = sorted(curves)
    lines = [
        f"# Measured LC voltage curves at {wavelength_nm:g} nm (LCSWEEP {record['swept_at']}).",
        "# Not the Meadowlark adapter's file format and not installed anywhere.",
        "Voltage(mv)," + ",".join(f"{wavelength_nm:g}-{n[-1]}" for n in names),
    ]
    for v in grid:
        nm = [
            float(np.interp(v, curves[n]["volts"], curves[n]["waves"])) * wavelength_nm
            for n in names
        ]
        lines.append(f"{v * 1000:.1f}," + ",".join(f"{x:.4f}" for x in nm))
    path.write_text("\n".join(lines) + "\n")
