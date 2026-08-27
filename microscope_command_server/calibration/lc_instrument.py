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

logger = logging.getLogger(__name__)

LC_DEVICE = "MeadowlarkLC"

#: Property names per control mode. Retardance mode asks the Micro-Manager
#: adapter for a retardance and lets it work out the voltage; voltage mode
#: computes the voltage itself and needs a curve.
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
        from polscope_library.calibration import RetardanceLimits

        if mode not in ("MM-Retardance", "MM-Voltage"):
            raise ValueError(f"unknown LC control mode {mode!r}")
        if mode == "MM-Voltage" and curve is None:
            raise ValueError("MM-Voltage mode needs a retardance/voltage curve; none was supplied")

        self.hardware = hardware
        self.mode = mode
        self.curve = curve
        self.settle_ms = float(settle_ms)
        self.limits = limits or RetardanceLimits()
        self.lamp = lamp
        self.dark_exposure_ms = dark_exposure_ms
        self.exposures = 0

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

    def measure_intensity(self) -> float:
        self.exposures += 1
        image, _ = self.hardware.snap_image()
        return float(np.asarray(image, dtype=np.float64).mean())

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

    # -- internals ---------------------------------------------------------

    def _core(self):
        core = getattr(self.hardware, "core", None)
        if core is None:
            raise RuntimeError("hardware has no .core; cannot reach the liquid crystals")
        return core
