"""A real microscope behind the calibration seam.

``polscope_library.calibration`` is written against a four-method protocol so
the search can be developed and tested with no hardware. This is the other
side of that seam: the same four methods, implemented against Micro-Manager.

Nothing here decides anything. The search strategy, the scheme, the bounds and
the black-level policy all live in the library; this only moves crystals and
takes pictures.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Optional

import numpy as np


def central_crop(frame: np.ndarray, fraction: float) -> np.ndarray:
    """The central ``fraction`` of ``frame`` along each of its first two axes."""
    if frame.ndim < 2 or not (0.0 < fraction < 1.0):
        return frame
    height, width = frame.shape[0], frame.shape[1]
    crop_h = max(1, int(round(height * fraction)))
    crop_w = max(1, int(round(width * fraction)))
    y, x = (height - crop_h) // 2, (width - crop_w) // 2
    return frame[y : y + crop_h, x : x + crop_w]


logger = logging.getLogger(__name__)

LC_DEVICE = "MeadowlarkLC"

#: Property names per control mode. Retardance mode asks the Micro-Manager
#: adapter for a retardance and lets it work out the voltage; voltage mode
#: computes the voltage itself and needs a curve.
#:
#: The waves properties are the writable ones. The adapter also exposes
#: "Retardance LC-A [in nm]", which is READ-ONLY -- confirmed on the rig
#: 2026-08-28 -- so it can be reported but never commanded.
RETARDANCE_PROPERTY = {"LCA": "Retardance LC-A [in waves]", "LCB": "Retardance LC-B [in waves]"}
VOLTAGE_PROPERTY = {"LCA": "Voltage (V) LC-A", "LCB": "Voltage (V) LC-B"}


class MicroManagerLC:
    """Drives the liquid crystals and the camera through Micro-Manager.

    Parameters
    ----------
    hardware
        The command server's hardware object; needs ``.core``, ``snap_image``
        and ``set_exposure``.
    mode
        ``MM-Retardance`` writes retardance in waves and lets the device
        adapter convert. ``MM-Voltage`` converts here and writes volts, which
        requires ``curve``.
    curve
        Retardance/voltage conversion, required only for ``MM-Voltage``.
    settle_ms
        Pause after each crystal move. The adapter reports "not busy" before
        the liquid crystal has finished relaxing, so waiting on the device is
        necessary but not sufficient.
    lamp
        ``(device, property, off_value, on_value)`` for darkening the field to
        measure a black level. ``None`` means :meth:`measure_dark` is
        unavailable and the library's black-level chain falls through.
    """

    def __init__(
        self,
        hardware,
        *,
        mode: str = "MM-Retardance",
        curve=None,
        settle_ms: float = 50.0,
        limits=None,
        lamp: Optional[tuple] = None,
        dark_exposure_ms: Optional[float] = None,
    ):
        # RetardanceLimits is imported inside _device_limits, which __init__
        # calls, so a missing polscope_library still fails here rather than
        # later on a crystal move.
        if mode not in ("MM-Retardance", "MM-Voltage"):
            raise ValueError(f"unknown LC control mode {mode!r}")
        if mode == "MM-Voltage" and curve is None:
            raise ValueError("MM-Voltage mode needs a retardance/voltage curve; none was supplied")

        self.hardware = hardware
        self.mode = mode
        self.curve = curve
        self.settle_ms = float(settle_ms)
        self.lamp = lamp
        self.dark_exposure_ms = dark_exposure_ms
        self.exposures = 0
        #: What the frames looked like, so a mean of 9 counts can be read as
        #: "dark" or "fine" later: the array dtype and the brightest and
        #: darkest single pixel seen over the run. ``None`` until a snap.
        self.frame_dtype: Optional[str] = None
        self.frame_shape: Optional[tuple] = None
        self.frame_max_seen: Optional[float] = None
        self.frame_min_seen: Optional[float] = None
        #: Top of the camera's range, set by the caller once it has asked the
        #: camera its bit depth. Without it the dtype's maximum stands in,
        #: which overstates a 12-bit camera delivering uint16 by 16x.
        self.full_scale: Optional[float] = None
        #: One entry per :meth:`measure_intensity`, in order: mean, brightest
        #: pixel, fraction of pixels at the top of the range, and a 256-bin
        #: histogram of a subsample. The library's trace has one point per
        #: call too, so index i here is trace point i.
        self.frames: list = []
        #: What the camera measures through, from :meth:`apply_roi`: the whole
        #: field (``None``), a camera ROI, or a software crop of every frame.
        self.roi: Optional[dict] = None
        self._camera_roi_set = False
        self._software_crop: Optional[float] = None
        self.limits = limits if limits is not None else self._device_limits()

    def _device_limits(self):
        """Ask the adapter what the crystals will actually accept.

        The library default (0.001 to 1.600 waves) is recOrder's number, and
        it does not match this rig: the D5020 adapter here reports a range of
        about 0 to 1 wave on the writable "[in waves]" properties. Asking the
        device is the only way to be right on a rig we have not seen, and it
        removes one more constant transcribed from another program's source.

        Both axes are queried and the INTERSECTION is used, because one
        RetardanceLimits governs the clamp for either crystal; a range wider
        than a crystal's own would let the clamp hand it a value it rejects.

        Falls back to the library default if the device cannot be asked --
        the calibration itself works around a quarter wave, far from either
        rail, so an unqueryable device is not a reason to refuse to run.
        """
        from polscope_library.calibration import RetardanceLimits

        default = RetardanceLimits()
        if self.mode != "MM-Retardance":
            return default
        try:
            core = self._core()
        except RuntimeError:
            return default

        lower, upper = None, None
        for prop in RETARDANCE_PROPERTY.values():
            try:
                if not core.has_property_limits(LC_DEVICE, prop):
                    continue
                low = float(core.get_property_lower_limit(LC_DEVICE, prop))
                high = float(core.get_property_upper_limit(LC_DEVICE, prop))
            except Exception as exc:
                logger.debug("Could not read limits for %s.%s: %s", LC_DEVICE, prop, exc)
                continue
            lower = low if lower is None else max(lower, low)
            upper = high if upper is None else min(upper, high)

        if lower is None or upper is None or not upper > lower:
            logger.warning(
                "Could not read retardance limits from %s; using the library default "
                "%.3f to %.3f waves. If this rig's crystals travel less than that, a "
                "commanded value near the top will be clamped to something the adapter "
                "then refuses.",
                LC_DEVICE,
                default.min_waves,
                default.max_waves,
            )
            return default

        # Never command an exact zero: it is a degenerate point for the
        # compensator model and some adapters treat it as "off".
        lower = max(lower, default.min_waves)
        logger.info(
            "Retardance limits read from %s: %.4f to %.4f waves (library default was "
            "%.3f to %.3f).",
            LC_DEVICE,
            lower,
            upper,
            default.min_waves,
            default.max_waves,
        )
        return RetardanceLimits(min_waves=lower, max_waves=upper)

    # -- LiquidCrystalInstrument ------------------------------------------

    def set_retardance(self, axis: str, waves: float) -> None:
        # Clamp rather than raise: a search that wanders near a rail must lose
        # one measurement, not the whole run. See the protocol docstring.
        value = self.limits.clamp(float(waves))
        core = self._core()
        if self.mode == "MM-Retardance":
            core.set_property(LC_DEVICE, RETARDANCE_PROPERTY[axis], str(value))
        else:
            core.set_property(
                LC_DEVICE, VOLTAGE_PROPERTY[axis], str(self.curve.voltage_for(axis, value))
            )
        try:
            core.wait_for_device(LC_DEVICE)
        except Exception as exc:  # not fatal; the settle below still applies
            logger.debug("wait_for_device(%s) failed: %s", LC_DEVICE, exc)
        if self.settle_ms > 0:
            time.sleep(self.settle_ms / 1000.0)

    def get_retardance(self, axis: str) -> float:
        """Read back what the crystal is at, in waves.

        Always read, never assume the commanded value: in voltage mode the
        command round-trips through a curve and the hardware quantises, and
        the palette we record has to be what the instrument reproduces.
        """
        core = self._core()
        if self.mode == "MM-Retardance":
            return float(core.get_property(LC_DEVICE, RETARDANCE_PROPERTY[axis]))
        volts = float(core.get_property(LC_DEVICE, VOLTAGE_PROPERTY[axis]))
        return float(self.curve.retardance_for(axis, volts))

    def get_voltage(self, axis: str) -> float:
        """The volts the controller is driving one crystal at, as the adapter reports them.

        The palette is set in the adapter's waves, which it turns into volts
        through its own table (``mmgr_dal_MeadowlarkLC.csv``); the volts are
        what the crystal actually received. Recorded on every calibration
        trace point so a table that does not fit the crystal shows up.
        """
        return float(self._core().get_property(LC_DEVICE, VOLTAGE_PROPERTY[axis]))

    def measure_intensity(self) -> float:
        self.exposures += 1
        image, _ = self.hardware.snap_image()
        frame = np.asarray(image)
        if self._software_crop is not None:
            frame = central_crop(frame, self._software_crop)
        self._note_frame(frame)
        return float(frame.astype(np.float64).mean())

    def apply_roi(self, fraction: Optional[float]) -> dict:
        """Measure through the central ``fraction`` of the field (each axis).

        Dust and other junk at the edges of the sensor would otherwise be
        averaged into every intensity, and a smaller frame reads out faster.
        The camera's own ROI is tried first, through the core; when the core
        cannot set one, every frame is cropped in software instead, which
        keeps the junk out but not the readout time. ``None`` or anything not
        strictly between 0 and 1 measures the whole field. The record returned
        (also kept as :attr:`roi`) says which happened.
        """
        self.restore_roi()
        if fraction is None or not (0.0 < float(fraction) < 1.0):
            self.roi = None
            return {"fraction": 1.0, "mode": "full"}
        fraction = float(fraction)
        record: dict = {"fraction": fraction}
        try:
            core = self._core()
            width, height = int(core.get_image_width()), int(core.get_image_height())
            roi_w = max(1, int(round(width * fraction)))
            roi_h = max(1, int(round(height * fraction)))
            x, y = (width - roi_w) // 2, (height - roi_h) // 2
            core.set_roi(x, y, roi_w, roi_h)
            self._camera_roi_set = True
            record.update(
                mode="camera",
                x=x,
                y=y,
                width=roi_w,
                height=roi_h,
                full_width=width,
                full_height=height,
            )
            logger.info(
                "LC calibration measures through a camera ROI of %dx%d at (%d, %d) of %dx%d",
                roi_w,
                roi_h,
                x,
                y,
                width,
                height,
            )
        except Exception as exc:
            self._software_crop = fraction
            record.update(mode="software", reason=str(exc))
            logger.info(
                "Camera ROI not available (%s); cropping every frame to the central %.0f%% instead",
                exc,
                fraction * 100,
            )
        self.roi = record
        return record

    def restore_roi(self) -> None:
        """Back to the whole field: the camera ROI cleared, the software crop dropped."""
        if self._camera_roi_set:
            try:
                self._core().clear_roi()
            except Exception as exc:
                logger.error("Could not clear the camera ROI after the calibration: %s", exc)
            self._camera_roi_set = False
        self._software_crop = None

    def _note_frame(self, frame: np.ndarray) -> None:
        self.frame_dtype = str(frame.dtype)
        self.frame_shape = tuple(int(n) for n in frame.shape)
        if frame.size == 0:
            return
        high, low = float(frame.max()), float(frame.min())
        self.frame_max_seen = (
            high if self.frame_max_seen is None else max(self.frame_max_seen, high)
        )
        self.frame_min_seen = low if self.frame_min_seen is None else min(self.frame_min_seen, low)

        top = self.full_scale
        if top is None and np.issubdtype(frame.dtype, np.integer):
            top = float(np.iinfo(frame.dtype).max)
        saturated = float(np.mean(frame >= top)) if top is not None else None
        # A histogram of every pixel of a 5 MP frame costs as much as the
        # exposure; one of a 16x subsample is a few ms and the same shape.
        step = max(1, int(np.sqrt(frame.size / 400_000)))
        sample = frame[tuple(slice(None, None, step) for _ in frame.shape)]
        upper = top if top is not None else max(high, 1.0)
        hist, _ = np.histogram(sample, bins=256, range=(0.0, upper))
        self.frames.append(
            {
                "mean": float(frame.astype(np.float64).mean()),
                "max": high,
                "min": low,
                "saturated_fraction": saturated,
                "histogram": hist,
                "histogram_top": upper,
            }
        )

    def measure_dark(self) -> float:
        """Mean of a frame with the field darkened.

        Raises :class:`NotImplementedError` when there is no way to darken it,
        so the library falls through to the next black-level source rather
        than blocking on a person.
        """
        if self.lamp is None:
            raise NotImplementedError("no lamp control configured; cannot measure a dark frame")
        device, prop, off_value, on_value = self.lamp
        core = self._core()
        restore_exposure = None
        try:
            if self.dark_exposure_ms is not None:
                restore_exposure = float(self.hardware.get_exposure())
                self.hardware.set_exposure(self.dark_exposure_ms)
            core.set_property(device, prop, str(off_value))
            with contextlib.suppress(Exception):
                core.wait_for_device(device)
            return self.measure_intensity()
        finally:
            # Restore unconditionally: leaving the lamp off after an exception
            # is the sort of state that makes the next acquisition mysterious.
            try:
                core.set_property(device, prop, str(on_value))
            except Exception as exc:
                logger.error("Failed to restore %s.%s after a dark frame: %s", device, prop, exc)
            if restore_exposure is not None:
                try:
                    self.hardware.set_exposure(restore_exposure)
                except Exception as exc:
                    logger.error("Failed to restore exposure after a dark frame: %s", exc)

    def device_snapshot(self) -> dict:
        """Every property of the LC device, as strings, or ``{}`` if it cannot be asked.

        Recorded with each calibration because the palette is in the ADAPTER's
        waves: the MeadowlarkLC adapter converts waves to volts at its own
        wavelength setting, so a palette is reproducible only while that
        setting (and the rest of the adapter state) is what it was. Keeping
        the snapshot is what makes a later disagreement diagnosable.
        """
        try:
            core = self._core()
            names = core.get_device_property_names(LC_DEVICE)
        except Exception as exc:
            logger.debug("Could not list %s properties: %s", LC_DEVICE, exc)
            return {}
        try:
            names = list(names)
        except TypeError:
            try:
                names = [names.get(i) for i in range(names.size())]
            except Exception:
                return {}
        snapshot = {}
        for name in names:
            try:
                snapshot[str(name)] = str(core.get_property(LC_DEVICE, str(name)))
            except Exception as exc:
                logger.debug("Could not read %s.%s: %s", LC_DEVICE, name, exc)
        return snapshot

    def camera_snapshot(self) -> tuple:
        """``(summary, properties)``: see :func:`camera_checks.camera_snapshot`.

        Lives in ``camera_checks`` so the acquisition can take the same
        snapshot without constructing an instrument.
        """
        from .camera_checks import camera_snapshot

        return camera_snapshot(self.hardware)

    # -- internals ---------------------------------------------------------

    def _core(self):
        core = getattr(self.hardware, "core", None)
        if core is None:
            raise RuntimeError("hardware has no .core; cannot reach the liquid crystals")
        return core
