"""Single-point acquisition helper for the Z-stack + time-lapse refactor.

This module provides the contract skeleton for the unified acquisition path
described in the refactor plan. It is intentionally BEHAVIOR-NEUTRAL at this
stage: no caller in workflow.py routes through this module yet. Downstream
teams will fill in the T-outer / Z-middle / (angle|channel)-inner loop and
wire the StackWriter here in later rollout tasks.

ASCII-only per project policy: this module runs on Windows cp1252 as well as
Linux/WSL. Do not use Unicode characters (arrows, Greek letters, deg sign) in
code, logging, or comments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------
@dataclass
class ZParams:
    """Z-stack parameters for a single-point acquisition.

    enabled=False is the default and corresponds to a single Z plane at the
    current stage Z. When enabled, z_offsets is a list of deltas around the
    center Z (e.g. [-2, -1, 0, 1, 2] for a 5-plane stack with 1 um step).
    """

    enabled: bool = False
    z_offsets: List[float] = field(default_factory=lambda: [0.0])
    z_step_um: Optional[float] = None
    projection: Optional[str] = None  # "max" / "min" / "sum" / "mean" / "std"


@dataclass
class TParams:
    """Time-lapse parameters for a single-point acquisition.

    timepoints=1 (default) disables time-lapse. interval_sec governs the
    "start at t0 + n*dt" pacing enforced by the TimepointScheduler
    downstream teams will add. interval_sec=0 means "acquire as fast as
    possible" (back-to-back timepoints).
    """

    timepoints: int = 1
    interval_sec: float = 0.0


@dataclass
class SinglePointContext:
    """Inputs needed to drive a single-point (T, Z, (C|angle)) acquisition.

    This dataclass is the cross-team contract: the Z-stack team, time-lapse
    team, and main workflow team all read it. Mutations are coordinator
    sign-off only.

    Fields are typed loosely (Any) where they reference existing workflow
    types (``AcquisitionContext``, ``ModalityConfig``, hardware handles) to
    keep this module free of circular imports. The concrete types are
    resolved at call-site.
    """

    # Identity / output
    sample: str
    modality: str
    output_path: Path

    # Per-acquisition plans.
    z_params: ZParams
    t_params: TParams

    # Hardware abstraction and the larger acquisition context the caller
    # has already populated (positions, write pool, cancellation, etc.).
    hardware: Any
    acquisition_context: Any  # workflow.AcquisitionContext

    # Optional per-channel plan (resolved list from resolve_channel_plan).
    # Empty list means "this is a non-channel modality" (brightfield, PPM).
    channel_plan: List[Dict[str, Any]] = field(default_factory=list)

    # Optional background correction configuration (passthrough; values
    # duplicate fields in acquisition_context but are surfaced here so the
    # single-point helper can reason about them without digging).
    bg_config: Optional[Dict[str, Any]] = None

    # Optional StackWriter handle (wired in a later rollout task).
    writer: Any = None


@dataclass
class SinglePointResult:
    """Summary of a single-point acquisition run.

    Matches the existing per-tile return convention in workflow.py:
    the tile_worst_sat dict is the aggregated saturation percentages,
    and xy_move_pending is the trailing-edge stage state that the caller
    may need to propagate.
    """

    tile_worst_sat: Dict[str, float] = field(default_factory=dict)
    xy_move_pending: bool = False


# ---------------------------------------------------------------------------
# Entry point (shell only)
# ---------------------------------------------------------------------------
def acquire_single_point(ctx: SinglePointContext) -> SinglePointResult:
    """Acquire one XY position's full (T, Z, (C|angle)) stack.

    This is the contract skeleton for the Z-stack + time-lapse refactor.
    Task #1 scope lands only the signature and docstring -- no caller routes
    through this function yet. Downstream teams own the body:

      * Z-stack team: Z loop + projection hookup.
      * Time-lapse team: T loop + TimepointScheduler.
      * Coordinator: StackWriter wiring and per-modality folder layout.

    The current shell delegates back to workflow._acquire_tile_angles or
    workflow._acquire_tile_channels via the caller's acquisition_context so
    that any accidental call during development preserves the existing
    bounded-workflow behavior. Once the real body lands, this delegation is
    removed.

    Args:
        ctx: fully populated SinglePointContext.

    Returns:
        A SinglePointResult describing the acquired tile's worst-case
        saturation and trailing stage state.
    """
    # Deferred import keeps this module free of the workflow dependency
    # graph at module-load time.
    from microscope_command_server.acquisition import workflow as _workflow

    ac = ctx.acquisition_context
    if ac is None:
        raise RuntimeError(
            "SinglePointContext.acquisition_context is required for the "
            "Task #1 shell to delegate to the existing per-tile helpers."
        )

    # Shell delegation: pick channel vs angle path based on whether the
    # caller supplied a channel plan. This matches the branching already
    # present in the main workflow's tile loop.
    if ctx.channel_plan:
        logger.debug(
            "acquire_single_point shell: delegating to _acquire_tile_channels "
            "for modality=%s sample=%s",
            ctx.modality,
            ctx.sample,
        )
        # Signature: (ctx: AcquisitionContext, pos, filename, current_stage_pos)
        # Task #1 is signature-only; the caller supplying a channel plan is
        # responsible for providing the pos/filename/current_stage_pos via
        # their own adapter when they call us. Until that adapter lands we
        # cannot safely run; raise clearly.
        raise NotImplementedError(
            "acquire_single_point() is a signature-only shell in Task #1. "
            "Route your caller through workflow._acquire_tile_channels directly "
            "until the Z-stack / time-lapse teams land the real body."
        )

    logger.debug(
        "acquire_single_point shell: delegating to _acquire_tile_angles "
        "for modality=%s sample=%s",
        ctx.modality,
        ctx.sample,
    )
    raise NotImplementedError(
        "acquire_single_point() is a signature-only shell in Task #1. "
        "Route your caller through workflow._acquire_tile_angles directly "
        "until the Z-stack / time-lapse teams land the real body."
    )


__all__ = [
    "SinglePointContext",
    "SinglePointResult",
    "TParams",
    "ZParams",
    "acquire_single_point",
]
