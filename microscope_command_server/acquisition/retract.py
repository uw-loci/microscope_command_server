"""Retract the objective to the declared safe Z when an acquisition is done.

**Why at the end of an acquisition.** An acquisition covers one annotation, and no
annotation spans two slides, so "finished an annotation" is a point at which the next
stage motion is unconstrained -- it may be the next region, the next slide, or an operator
driving the stage by hand. Keying the retraction to that event rather than to a move
distance means it also covers cases nobody thought to enumerate.

**What it is protecting against.** On the 2026-10-02 four-slide run, every long XY move --
all fourteen, including an 85.9 mm traverse across the whole holder -- ran at focus height,
between -306 and -459 um, while the declared safe Z was 0. That is around 400 um of
clearance given away on every traverse, including the end-of-region return move below,
which on that run was itself up to 12.7 mm. An objective that touches a slide or a holder
rib mid-traverse can shove the slide, and on an open-loop stepper it can also cost steps --
which the controller cannot report, because it counts what it was asked to do rather than
what the stage did.

**Why the direction is checked rather than trusted.** Moving to a "safe" Z that is on the
wrong side drives the objective INTO the sample. The PPM config carries the scar: an
earlier safe Z of -500 was wrong-side, inferred from "focus is near -400, so retract
further" without establishing the sign. So this refuses to move unless
``stage.focus.retract_sign`` says which way is away from the sample AND the target is on
that side of where the stage currently is. A refusal costs clearance; a wrong guess costs
the objective and the slide.

Takes plain callables for the stage so the decision can be tested without Micro-Manager.
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _stage_config(config_manager: Any) -> dict:
    """Return the ``stage`` block, or an empty dict.

    Deliberately goes through ``get_config()`` rather than a dotted-path accessor:
    ConfigManager has no such method, and a call to one raises into whatever except
    block surrounds it.
    """
    try:
        cfg = config_manager.get_config()
    except Exception as e:
        logger.debug("Could not read the microscope config: %s", e)
        return {}
    if not isinstance(cfg, dict):
        return {}
    stage = cfg.get("stage")
    return stage if isinstance(stage, dict) else {}


def resolve_safe_z(config_manager: Any) -> Optional[float]:
    """The declared retracted Z, from ``stage.safe_z_um``, or None.

    Per-insert and per-modality overrides (``stage.inserts.configurations.<id>.safe_z_um``)
    are NOT resolved here, because the acquisition message does not carry which insert is
    fitted. On a rig where an insert overrides the scope-level value, the direction check
    below is what keeps a scope-level value from being used unsafely: a target on the wrong
    side of the sample is refused rather than driven to.
    """
    raw = _stage_config(config_manager).get("safe_z_um")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("stage.safe_z_um is %r, which is not a number; not retracting.", raw)
        return None


def resolve_retract_sign(config_manager: Any) -> Optional[float]:
    """+1.0 when increasing Z retracts, -1.0 when decreasing does, else None."""
    focus = _stage_config(config_manager).get("focus")
    raw = focus.get("retract_sign") if isinstance(focus, dict) else None
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("positive", "+1", "1"):
        return 1.0
    if text in ("negative", "-1"):
        return -1.0
    logger.warning(
        "stage.focus.retract_sign is %r; expected positive or negative. Not retracting.", raw
    )
    return None


def retract_to_safe_z(
    config_manager: Any,
    read_z: Callable[[], float],
    move_z: Callable[[float], None],
    log: Any = None,
    tolerance_um: float = 1.0,
) -> bool:
    """Move the focus device to the declared safe Z. Returns True if it moved.

    Never raises: a retraction that cannot be made safely, or at all, must not turn a
    completed acquisition into a failed one. Every refusal says why, because a run that
    quietly stopped retracting would give the clearance back without anyone noticing.
    """
    out = log or logger

    safe_z = resolve_safe_z(config_manager)
    if safe_z is None:
        out.debug("No stage.safe_z_um declared; leaving Z where the acquisition left it.")
        return False

    sign = resolve_retract_sign(config_manager)
    if sign is None:
        out.warning(
            "Not retracting to safe Z %.2f um: stage.focus.retract_sign is not declared, so "
            "which direction is away from the sample is unknown. Measure it rather than "
            "guessing -- moving the wrong way drives the objective into the slide.",
            safe_z,
        )
        return False

    try:
        current_z = float(read_z())
    except Exception as e:
        out.warning("Not retracting to safe Z: could not read the current Z (%s).", e)
        return False

    delta = safe_z - current_z
    if abs(delta) <= tolerance_um:
        out.debug("Already at the safe Z %.2f um; nothing to retract.", safe_z)
        return False

    if delta * sign < 0:
        out.error(
            "NOT retracting: the safe Z (%.2f um) is on the SAMPLE side of the current Z "
            "(%.2f um), given that %s Z retracts. Moving there would drive the objective "
            "toward the sample. Check stage.safe_z_um and stage.focus.retract_sign.",
            safe_z,
            current_z,
            "increasing" if sign > 0 else "decreasing",
        )
        return False

    try:
        move_z(safe_z)
    except Exception as e:
        out.warning("Retraction to safe Z %.2f um failed: %s", safe_z, e)
        return False

    out.info(
        "Retracted to the safe Z %.2f um from %.2f um (%.1f um of clearance gained) before "
        "any further stage motion.",
        safe_z,
        current_z,
        abs(delta),
    )
    return True
