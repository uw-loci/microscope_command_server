"""Pure geometry helpers for streaming autofocus.

Lives outside ``server.handlers`` deliberately, following ``server.probe_parsers``:
importing anything from the handlers package pulls in the hardware chain
(``microscope_control`` -> ``pycromanager``), which a unit test has no business
needing to compute an interpolation.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np


def build_z_from_poll(
    z_poll_samples: List[Tuple[float, float]],
) -> Optional[Callable[[float], float]]:
    """Map a frame's wall time to the MEASURED stage Z at that moment.

    During a streaming scan a background thread polls ``core.get_position()`` and
    records ``(wall_ms, z)``. This turns that trace into a lookup a frame can be
    dated against.

    Returns ``None`` when the trace is too short to interpolate over, which is the
    caller's signal to fall back to the modelled ``wall_ms * velocity`` Z -- the only
    thing available then.

    Why the measurement rather than the model: the model extrapolates from a
    hand-calibrated constant (``slow_speed_um_per_s``) and any error in it INTEGRATES
    over the scan. Measured on PPM 20x over a 267 um approach traverse (2026-08-26),
    configured 11.5 um/s against a real ~11.96 um/s -- 4% fast, which by the sample
    plane had accumulated into an 8.6 um labelling error. The operator's own focus was
    -237.0; the metric peak sat at z_polled = -236.65 and z_model = -228.08.

    Times outside the polled range clamp to the nearest endpoint rather than
    extrapolating, so a frame retrieved after the last poll cannot be projected past
    the end of travel.

    :param z_poll_samples: ``(wall_ms, z_um)`` pairs; need not be sorted
    :return: a callable mapping ``wall_ms`` to interpolated Z, or ``None``
    """
    if len(z_poll_samples) < 2:
        return None
    poll_t = np.asarray([t for t, _ in z_poll_samples], dtype=float)
    poll_z = np.asarray([z for _, z in z_poll_samples], dtype=float)
    order = np.argsort(poll_t)
    poll_t = poll_t[order]
    poll_z = poll_z[order]

    def _z_at(wall_ms: float) -> float:
        return float(np.interp(wall_ms, poll_t, poll_z))

    return _z_at
