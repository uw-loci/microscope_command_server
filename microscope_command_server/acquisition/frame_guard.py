"""The acquisition workflow's half of the stage frame guard.

The guard itself lives in microscope_control: the XY motion profile and the two
checks in ``hardware/xy_motion.py``, applied at the single XY choke point in
``hardware/stage.py``. This module contributes the two decisions that belong to
an acquisition rather than to a stage:

* **when the check is armed** -- only while we are the only thing moving the
  stage. Outside an acquisition the operator may move it from the joystick or
  the Live Viewer, and a disagreement with the last commanded position is then
  not evidence of anything.
* **what a suspect frame costs** -- the run, immediately.

Kept free of hardware imports so it can be tested without Micro-Manager.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def set_frame_check_armed(hardware: Any, armed: bool) -> None:
    """Arm or disarm the stage's commanded-vs-reported frame check.

    Tolerant of hardware without the hook, so mock rigs and any stage
    implementation that predates it still acquire.
    """
    setter = getattr(getattr(hardware, "stage", None), "set_frame_check_armed", None)
    if setter is None:
        return
    try:
        setter(armed)
    except Exception as e:
        logger.debug("Could not %s the stage frame check: %s", "arm" if armed else "disarm", e)


def frame_suspect_reason(hardware: Any) -> Optional[str]:
    """Why the stage's coordinate frame is untrustworthy, or None."""
    reason = getattr(getattr(hardware, "stage", None), "xy_frame_suspect", None)
    return reason or None


def raise_if_frame_suspect(hardware: Any) -> None:
    """Stop the acquisition if the stage's coordinate frame has moved.

    A Prior controller that restarts re-zeros wherever it stands, and from then
    on it goes exactly where commanded and reports exactly that -- so every
    remaining tile would be imaged at coordinates that no longer describe the
    sample. The 2026-10-02 four-slide run spent thirty hours that way and
    returned two regions of slide label. Failing here costs the rest of one
    region; not failing costs the rest of the run.

    Raised as a failure rather than a cancellation: nobody asked for it, and the
    operator has to re-establish the frame at the calibration fiducial before
    any coordinate is worth acting on again.
    """
    reason = frame_suspect_reason(hardware)
    if reason:
        raise RuntimeError(f"Stage coordinate frame is no longer trustworthy. {reason}")
