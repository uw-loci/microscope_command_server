"""LC-PolScope drift correction: remove the retardance that builds up after the background.

The measured background removes the instrument's own retardance only as it was when the
background was taken. The light path keeps changing afterwards (most likely the liquid
crystals settling or warming), so every later tile carries a leftover that is uniform across
the tile and grows with time: on 2026-11-17 the glass of one scan read 0.36 nm at 8 min after
the background and 1.0 nm at 63 min, along one axis, and the leftover adds to fibers near that
axis and subtracts from fibers across it. See
claude-reports/2026-11-16_jove-liver-l002-roi4-reacquisition.md (2026-11-19 entries).

Two optional measurements, both in ``modalities.lcpolscope.reconstruction.drift_correction``:

A. ``tile_glass: true`` -- in every tile, find the glass (no specimen) and record its leftover
   with the time. Glass is found by brightness against the background images: the mean of the
   five states divided by the mean of the five background states is about 1.0 on glass at any
   exposure and vignetting, and below about 0.93 under tissue.
B. ``reference_interval_min: N`` -- every N minutes, go back to the spot where the background
   was taken, take the five states there and record the whole field's leftover. This tracks
   drift even while the tiles hold no glass.

C. After the last tile, the leftover is modeled against time and removed from copies of the
   retardance and orientation tiles, written to ``driftcorr_retardance``,
   ``driftcorr_orientation`` and ``driftcorr_orientation_color``. QPSC stitches every folder
   whose name ends in one of the LC-PolScope output names, so the corrected images are
   stitched beside the originals, which are never changed.

The model: with reference visits, linear interpolation between them (anchored at zero at the
moment the background was taken, where the leftover is zero by construction); with tile glass
as well, the reference curve is shifted by the median difference between the tile glass and
the curve, because the background spot and the scanned area need not read the same (they
differed by 0.2-0.3 nm on 2026-11-17). With tile glass alone, a local linear fit in time
(Gaussian weights, ``time_bandwidth_min``), held constant before the first and after the last
glass tile.

The removal is first order: the leftover vector (retardance along a doubled angle) is
subtracted in the plane of R cos 2theta, R sin 2theta. That is exact for two small retarders
in series and good to well under 1 % of the leftover for specimen retardance far below a
quarter wave (137 nm at 546 nm).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Prefix of the corrected output folders. QPSC's LC-PolScope handler stitches every folder whose
#: name ENDS in retardance / orientation / orientation_color, so the prefix keeps them stitched.
DRIFT_PREFIX = "driftcorr_"
#: Measurements, the fitted model and the first and last reference visits' state images, inside
#: the acquisition's _protected folder (kept by tile cleanup and Make Project Portable) with the
#: background and LC calibration the run used.
DRIFT_FOLDER = "drift_correction"
#: Subfolders of DRIFT_FOLDER holding the five state images of the first and the last visit.
FIRST_VISIT_FOLDER = "reference_first"
LAST_VISIT_FOLDER = "reference_last"
MEASUREMENTS_FILE = "drift_measurements.ndjson"
MODEL_FILE = "drift_model.json"


@dataclass(frozen=True)
class DriftSettings:
    """``reconstruction.drift_correction`` as read from the YAML."""

    tile_glass: bool = False
    reference_interval_min: float = 0.0
    reference_frames: int = 1
    glass_min_ratio: float = 0.94
    glass_max_ratio: float = 1.06
    glass_max_local_sd: float = 0.02
    min_glass_fraction: float = 0.02
    time_bandwidth_min: float = 5.0

    @property
    def enabled(self) -> bool:
        return self.tile_glass or self.reference_interval_min > 0

    def describe(self) -> str:
        parts = []
        if self.tile_glass:
            parts.append(
                f"tile glass (ratio {self.glass_min_ratio:g}-{self.glass_max_ratio:g}, local SD <= "
                f"{self.glass_max_local_sd:g}, >= {self.min_glass_fraction:.0%} of the tile)"
            )
        if self.reference_interval_min > 0:
            parts.append(
                f"background spot every {self.reference_interval_min:g} min "
                f"({self.reference_frames} frame(s) per state)"
            )
        return "; ".join(parts) if parts else "off"


def drift_settings(reconstruction_cfg) -> Optional[DriftSettings]:
    """``reconstruction.drift_correction`` as settings, or None when absent or switched off.

    Unknown keys and bad values raise ValueError, so a typo is reported at acquisition start
    rather than ignored.
    """
    block = (reconstruction_cfg or {}).get("drift_correction")
    if not block:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"reconstruction.drift_correction must be a mapping, got {block!r}")
    allowed = set(DriftSettings.__dataclass_fields__)
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ValueError(
            f"reconstruction.drift_correction has unknown key(s) {unknown}; expected {sorted(allowed)}"
        )
    kwargs: Dict[str, Any] = {}
    if "tile_glass" in block:
        if not isinstance(block["tile_glass"], bool):
            raise ValueError(f"tile_glass must be true or false, got {block['tile_glass']!r}")
        kwargs["tile_glass"] = block["tile_glass"]
    for key in (
        "reference_interval_min",
        "glass_min_ratio",
        "glass_max_ratio",
        "glass_max_local_sd",
        "min_glass_fraction",
        "time_bandwidth_min",
    ):
        if key in block and block[key] is not None:
            kwargs[key] = float(block[key])
    if "reference_frames" in block:
        kwargs["reference_frames"] = int(block["reference_frames"])
    s = DriftSettings(**kwargs)
    if s.reference_interval_min < 0:
        raise ValueError("reference_interval_min must be 0 (off) or a number of minutes")
    if s.reference_frames < 1:
        raise ValueError("reference_frames must be at least 1")
    if not 0 < s.glass_min_ratio < s.glass_max_ratio:
        raise ValueError("glass_min_ratio must be above 0 and below glass_max_ratio")
    if not 0 < s.min_glass_fraction <= 1:
        raise ValueError("min_glass_fraction must be in (0, 1]")
    if s.time_bandwidth_min <= 0:
        raise ValueError("time_bandwidth_min must be positive")
    return s if s.enabled else None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def glass_mask(states, background, black_level, settings: DriftSettings, exposure_ratio=1.0):
    """Pixels of glass: as bright as the background (state means) and locally uniform.

    ``exposure_ratio`` is the tile exposure over the background exposure, so a background taken
    at another exposure still reads 1.0 on glass.
    """
    from scipy import ndimage as ndi

    tile = np.mean(np.asarray(states, dtype=np.float32), axis=0) - float(black_level)
    bg = np.mean(np.asarray(background, dtype=np.float32), axis=0) - float(black_level)
    ratio = tile / np.maximum(bg, 1.0) / float(exposure_ratio)
    mu = ndi.uniform_filter(ratio, 5)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(ratio * ratio, 5) - mu * mu, 0.0))
    mask = (mu >= settings.glass_min_ratio) & (mu <= settings.glass_max_ratio)
    mask &= sd <= settings.glass_max_local_sd
    # border_value=1: glass at the edge of the camera frame is still glass.
    return ndi.binary_erosion(mask, iterations=2, border_value=1)


def leftover_vector(retardance_nm, orientation_rad, mask=None) -> Tuple[float, float, int]:
    """Median doubled-angle vector (nm along cos 2theta, sin 2theta) and the pixel count."""
    r = np.asarray(retardance_nm, dtype=np.float64)
    o = np.asarray(orientation_rad, dtype=np.float64)
    if mask is not None:
        r, o = r[mask], o[mask]
    r, o = r.ravel(), o.ravel()
    ok = np.isfinite(r) & np.isfinite(o)
    r, o = r[ok], o[ok]
    if r.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.median(r * np.cos(2 * o))), float(np.median(r * np.sin(2 * o))), int(r.size)


def vector_nm_axis(vx: float, vy: float) -> Tuple[float, float]:
    """(magnitude nm, slow axis deg in [0, 180)) of a doubled-angle vector."""
    return float(np.hypot(vx, vy)), float(np.degrees(np.arctan2(vy, vx)) / 2.0 % 180.0)


def remove_vector(retardance_nm, orientation_rad, vx: float, vy: float):
    """Subtract a uniform leftover (doubled-angle vector, nm) from a retardance/orientation pair."""
    r = np.asarray(retardance_nm, dtype=np.float64)
    o = np.asarray(orientation_rad, dtype=np.float64)
    u = r * np.cos(2 * o) - vx
    v = r * np.sin(2 * o) - vy
    return np.hypot(u, v), (np.arctan2(v, u) / 2.0) % np.pi


# ---------------------------------------------------------------------------
# Collection during the run
# ---------------------------------------------------------------------------


@dataclass
class DriftLog:
    """Everything the drift correction records during one acquisition. Thread-safe: tiles are
    added from the write pool, reference visits from the acquisition thread."""

    settings: DriftSettings
    background_time: Optional[float] = None
    reference_xyz: Optional[Tuple[float, float, float]] = None
    exposure_ratio: float = 1.0
    folder: Optional[Path] = None
    tiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    references: List[Dict[str, Any]] = field(default_factory=list)
    display: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _append(self, record: Dict[str, Any]) -> None:
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with open(self.folder / MEASUREMENTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("Could not record a drift measurement: %s", exc)

    def add_tile(self, filename: str, t: float, annotations: Dict[str, Any], glass=None) -> None:
        """One reconstructed tile: its time, its metadata, and its glass leftover when measured
        (``glass`` = (vx, vy, pixels, fraction))."""
        record: Dict[str, Any] = {"kind": "tile", "file": filename, "time": t}
        if glass is not None:
            vx, vy, n, frac = glass
            record.update(vx=vx, vy=vy, glass_pixels=n, glass_fraction=round(frac, 4))
        with self._lock:
            self.tiles[filename] = {**record, "annotations": annotations}
            self._append(record)

    def add_reference(
        self, t: float, vx: float, vy: float, pixels: int, images_folder: Optional[str] = None
    ) -> None:
        record = {"kind": "reference", "time": t, "vx": vx, "vy": vy, "pixels": pixels}
        if images_folder:
            record["images"] = images_folder
        with self._lock:
            self.references.append(record)
            self._append(record)

    def save_reference_images(self, subfolder: str, images: Dict[str, np.ndarray]) -> Optional[str]:
        """Keep one visit's five state images (uint16, one TIFF per state, named like the
        background's) in DRIFT_FOLDER/<subfolder>. Only the first and the last visit are kept:
        about 42 MB each for 2048 x 2048 states, where keeping every visit would grow a
        multi-hour run's _protected folder by gigabytes. Returns the subfolder, or None."""
        import tifffile

        if self.folder is None:
            return None
        target = self.folder / subfolder
        try:
            target.mkdir(parents=True, exist_ok=True)
            for cid, img in images.items():
                data = np.clip(np.rint(np.asarray(img, dtype=np.float64)), 0, 65535).astype(
                    np.uint16
                )
                tifffile.imwrite(str(target / f"{cid}.tif"), data)
            return subfolder
        except OSError as exc:
            logger.warning("Could not keep the reference visit images in %s: %s", target, exc)
            return None

    def reference_due(self, now: float) -> bool:
        if self.settings.reference_interval_min <= 0 or self.reference_xyz is None:
            return False
        with self._lock:
            if not self.references:
                return True
            last = max(r["time"] for r in self.references)
        return now - last >= self.settings.reference_interval_min * 60.0


def background_spot(background_folder) -> Optional[Tuple[float, float, float]]:
    """Stage X, Y, Z where the background was taken (QPSC's background_settings.yml), or None."""
    import yaml

    if not background_folder:
        return None
    path = Path(background_folder) / "background_settings.yml"
    if not path.is_file():
        return None
    try:
        ctx = (yaml.safe_load(path.read_text()) or {}).get("context") or {}
        xy = ctx.get("stage_xy_um")
        z = ctx.get("stage_z_um")
        if xy is None or z is None or len(xy) != 2:
            return None
        return float(xy[0]), float(xy[1]), float(z)
    except Exception as exc:
        logger.warning("Could not read the background position from %s: %s", path, exc)
        return None


def parse_time(stamp) -> Optional[float]:
    """An ISO time (the provenance's collected_at) as epoch seconds, or None."""
    from datetime import datetime

    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _local_linear(t, y, tq, bandwidth):
    """Gaussian-weighted local linear fit of y(t) at each tq, held constant outside [t.min, t.max].
    Two passes; the second drops points more than 4 robust SDs (and 0.05 nm) from the first."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    tq = np.clip(np.asarray(tq, float), t.min(), t.max())

    def fit(weights, at):
        out = np.empty(len(at))
        for i, q in enumerate(at):
            w = weights * np.exp(-0.5 * ((t - q) / bandwidth) ** 2)
            sw = w.sum()
            if sw <= 1e-9:
                out[i] = y[np.argmin(np.abs(t - q))]
                continue
            d = t - q
            m1 = (w * d).sum() / sw
            var = (w * (d - m1) ** 2).sum() / sw
            ym = (w * y).sum() / sw
            if var <= 1e-12:
                out[i] = ym
                continue
            slope = (w * (d - m1) * (y - ym)).sum() / sw / var
            out[i] = ym - slope * m1
        return out

    weights = np.ones_like(t)
    if len(t) >= 5:
        res = y - fit(weights, t)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        weights = (np.abs(res) <= max(4 * mad, 0.05)).astype(float)
    return fit(weights, tq)


@dataclass
class DriftModel:
    """The leftover as a function of time (epoch seconds) -> (vx, vy) nm."""

    kind: str
    predict: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    detail: Dict[str, Any]


def fit_drift_model(log: DriftLog) -> Optional[DriftModel]:
    """The model C applies, from whatever A and B recorded; None when there is too little."""
    s = log.settings
    glass = [
        r
        for r in log.tiles.values()
        if "vx" in r and r.get("glass_fraction", 0) >= s.min_glass_fraction
    ]
    refs = sorted(log.references, key=lambda r: r["time"])
    if refs:
        t_ref = [r["time"] for r in refs]
        vx_ref = [r["vx"] for r in refs]
        vy_ref = [r["vy"] for r in refs]
        anchored = log.background_time is not None and log.background_time < t_ref[0]
        if anchored:
            t_ref = [log.background_time] + t_ref
            vx_ref = [0.0] + vx_ref
            vy_ref = [0.0] + vy_ref
        t_ref, vx_ref, vy_ref = (np.asarray(a, float) for a in (t_ref, vx_ref, vy_ref))
        off_x = off_y = 0.0
        if len(glass) >= 3:
            tg = np.array([g["time"] for g in glass])
            off_x = float(
                np.median(np.array([g["vx"] for g in glass]) - np.interp(tg, t_ref, vx_ref))
            )
            off_y = float(
                np.median(np.array([g["vy"] for g in glass]) - np.interp(tg, t_ref, vy_ref))
            )

        def predict(tq):
            tq = np.asarray(tq, float)
            return np.interp(tq, t_ref, vx_ref) + off_x, np.interp(tq, t_ref, vy_ref) + off_y

        kind = "reference visits" + (" + tile glass offset" if len(glass) >= 3 else "")
        return DriftModel(
            kind,
            predict,
            {
                "references": len(refs),
                "anchored_at_background": bool(anchored),
                "glass_tiles": len(glass),
                "offset_from_tile_glass_nm": [round(off_x, 4), round(off_y, 4)],
            },
        )
    if len(glass) >= 3:
        tg = np.array([g["time"] for g in glass])
        gx = np.array([g["vx"] for g in glass])
        gy = np.array([g["vy"] for g in glass])
        bw = s.time_bandwidth_min * 60.0

        def predict(tq):
            return _local_linear(tg, gx, tq, bw), _local_linear(tg, gy, tq, bw)

        return DriftModel(
            "tile glass, local linear in time",
            predict,
            {"glass_tiles": len(glass), "bandwidth_min": s.time_bandwidth_min},
        )
    return None


# ---------------------------------------------------------------------------
# Writing the corrected copies (C)
# ---------------------------------------------------------------------------


def write_corrected_tiles(
    output_path: Path,
    log: DriftLog,
    *,
    read_tile: Callable[[Path], np.ndarray],
    write_tile: Callable[..., None],
    retardance_dir: str,
    orientation_dir: str,
    color_dir: str,
    retardance_counts_per_nm: float,
    orientation_counts_per_degree: float,
    orientation_max_counts: int,
    to_uint16: Callable,
    channel_annotations: Callable[[], Tuple[Dict[str, Dict], Dict[str, str]]],
    color_image: Callable,
    copy_tile_configuration: Callable[[Path, Path], None],
    logger_=None,
) -> Dict[str, Any]:
    """Fit the model and write corrected copies of every recorded tile. Returns the summary that
    also goes to drift_correction/drift_model.json; ``{"applied": False, ...}`` when nothing was
    written. The I/O and the LC-PolScope conventions are passed in, so this module needs neither
    the OME writer nor the modality constants."""
    log_ = logger_ or logger
    output_path = Path(output_path)
    model = fit_drift_model(log)
    summary: Dict[str, Any] = {
        "settings": log.settings.describe(),
        "tiles": len(log.tiles),
        "references": len(log.references),
        "background_time": log.background_time,
    }
    if model is None or not log.tiles:
        summary.update(
            applied=False, reason="too few measurements (need 1 reference visit or 3 glass tiles)"
        )
        _write_summary(log, summary)
        log_.warning("LC-PolScope drift correction NOT applied: %s", summary["reason"])
        return summary

    names = sorted(log.tiles)
    times = np.array([log.tiles[n]["time"] for n in names])
    px, py = model.predict(times)
    per_channel, channel_label = channel_annotations()
    out_ret = output_path / (DRIFT_PREFIX + retardance_dir)
    out_ori = output_path / (DRIFT_PREFIX + orientation_dir)
    out_col = output_path / (DRIFT_PREFIX + color_dir)
    rows = []
    t_start = time.perf_counter()
    for name, t, vx, vy in zip(names, times, px, py):
        rec = log.tiles[name]
        src_r = output_path / retardance_dir / name
        src_o = output_path / orientation_dir / name
        if not (src_r.is_file() and src_o.is_file()):
            log_.warning("Drift correction: %s has no retardance/orientation tile; skipped", name)
            continue
        ret = read_tile(src_r).astype(np.float64) / retardance_counts_per_nm
        ori = np.deg2rad(read_tile(src_o).astype(np.float64) / orientation_counts_per_degree)
        new_r, new_o = remove_vector(ret, ori, float(vx), float(vy))
        nm, axis = vector_nm_axis(float(vx), float(vy))
        minutes = (t - log.background_time) / 60.0 if log.background_time is not None else None
        drift_keys = {
            "polscope.drift_correction": f"removed {nm:.3f} nm at {axis:.1f} deg ({model.kind})",
            "polscope.drift_removed_nm": round(nm, 4),
            "polscope.drift_removed_axis_deg": round(axis, 2),
            "polscope.drift_removed_vector_nm": f"{vx:.4f},{vy:.4f}",
            "polscope.drift_model": model.kind,
            "polscope.drift_first_order": "true -- vector subtraction in the R cos2theta, R sin2theta plane",
        }
        if minutes is not None:
            drift_keys["polscope.drift_minutes_after_background"] = round(minutes, 2)
        shared = {**rec.get("annotations", {}), **drift_keys}
        r_counts, _ = to_uint16(new_r, retardance_counts_per_nm)
        o_counts, _ = to_uint16(
            np.rad2deg(new_o), orientation_counts_per_degree, orientation_max_counts
        )
        for out_dir, src_dir, data in (
            (out_ret, retardance_dir, r_counts),
            (out_ori, orientation_dir, o_counts),
        ):
            out_dir.mkdir(parents=True, exist_ok=True)
            copy_tile_configuration(output_path / retardance_dir, out_dir)
            (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
            write_tile(
                filename=str(out_dir / name),
                data=data,
                channel_name=channel_label[src_dir],
                map_annotations={**shared, **per_channel[src_dir]},
            )
        if log.display:
            out_col.mkdir(parents=True, exist_ok=True)
            copy_tile_configuration(output_path / retardance_dir, out_col)
            (out_col / name).parent.mkdir(parents=True, exist_ok=True)
            write_tile(
                filename=str(out_col / name),
                data=color_image(new_r, new_o),
                channel_name="Orientation colour (hue = orientation, brightness = retardance)",
                map_annotations={**shared, **log.display.get("annotations", {})},
            )
        rows.append(
            {
                "file": name,
                "minutes_after_background": None if minutes is None else round(minutes, 2),
                "removed_nm": round(nm, 4),
                "removed_axis_deg": round(axis, 2),
                "vx": round(float(vx), 4),
                "vy": round(float(vy), 4),
                "glass_vx": rec.get("vx"),
                "glass_vy": rec.get("vy"),
                "glass_fraction": rec.get("glass_fraction"),
            }
        )
    removed = [r["removed_nm"] for r in rows]
    summary.update(
        applied=bool(rows),
        model=model.kind,
        model_detail=model.detail,
        corrected_tiles=len(rows),
        removed_nm_min_max=[min(removed), max(removed)] if removed else None,
        folders=[p.name for p in (out_ret, out_ori, out_col) if p.is_dir()],
        per_tile=rows,
    )
    _write_summary(log, summary)
    if rows:
        log_.info(
            "LC-PolScope drift correction (%s): %d tiles, removed %.3f-%.3f nm, written to %s in %.1f s",
            model.kind,
            len(rows),
            min(removed),
            max(removed),
            ", ".join(summary["folders"]),
            time.perf_counter() - t_start,
        )
    return summary


def _write_summary(log: DriftLog, summary: Dict[str, Any]) -> None:
    if log.folder is None:
        return
    try:
        log.folder.mkdir(parents=True, exist_ok=True)
        (log.folder / MODEL_FILE).write_text(json.dumps(summary, indent=1, default=str))
    except OSError as exc:
        logger.warning("Could not write the drift model summary: %s", exc)
