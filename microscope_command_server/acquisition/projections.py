"""Z-stack projection operators and per-position processing hooks.

Projection operators reduce a list of Z-plane images into a single 2D
image suitable for stitching. Each operator takes a list of numpy arrays
(one per Z-plane) and returns a single array of the same spatial dimensions.

The PositionHook ABC enables custom per-position logic during acquisition
(e.g., conditional time-lapse, quality checks, adaptive protocols).

Projection functions and registry are re-exported from microscope_imageprocessing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from microscope_imageprocessing.zstack.edf import make_edf_projection
from microscope_imageprocessing.zstack.projections import (
    max_intensity_projection,
    min_intensity_projection,
    sum_projection,
    mean_projection,
    std_projection,
    PROJECTIONS,
    get_projection,
    generate_z_offsets,
)

logger = logging.getLogger(__name__)

# Re-export all projection symbols for backward compatibility
__all__ = [
    "max_intensity_projection",
    "min_intensity_projection",
    "sum_projection",
    "mean_projection",
    "std_projection",
    "PROJECTIONS",
    "get_projection",
    "generate_z_offsets",
    "make_edf_projection",
    "resolve_projection",
    "PositionHook",
]


def resolve_projection(name: str, params: Dict[str, Any]):
    """Look up a projection, honouring EDF's tuning parameters.

    ``get_projection`` returns a fixed ``List[ndarray] -> ndarray`` callable
    with no room for settings, which is right for max/min/sum/mean/std -- they
    have none. EDF has three (sharpness metric, averaging window, index
    smoothing), and their best values depend on pixel size, camera noise and
    whether the focal surface is smooth, so they are exposed to the user rather
    than fixed. This is the one place that difference is handled.

    Unknown or malformed EDF settings raise HERE, before the acquisition
    starts, rather than after the tiles have been captured.

    Args:
        name: Projection name from ``--z-projection``.
        params: Parsed command parameters; EDF reads ``edf_metric``,
            ``edf_window`` and ``edf_index_smooth`` when present.

    Returns:
        A callable taking a Z-stack and returning the projected 2D image.
    """
    if name != "edf":
        return get_projection(name)

    metric = params.get("edf_metric") or "tenengrad"
    window = params.get("edf_window")
    index_smooth = params.get("edf_index_smooth")
    kwargs = {"metric": metric}
    if window is not None:
        kwargs["window"] = int(window)
    if index_smooth is not None:
        kwargs["index_smooth"] = int(index_smooth)
    logger.info("EDF projection: %s", kwargs)
    return make_edf_projection(**kwargs)


# ============================================================
# Per-position hook interface
# ============================================================


class PositionHook(ABC):
    """Hook called at each tile position during acquisition.

    Hooks run AFTER all Z-planes and angles are acquired for a position,
    but BEFORE moving to the next tile. They receive the acquired images
    and can trigger additional operations (time-lapse, extra snaps,
    quality checks, adaptive protocols, etc.).

    To create a custom hook:
        class MyHook(PositionHook):
            def on_position_complete(self, position_index, position_xy,
                                     images, hardware, params):
                # Analyze images, trigger actions, etc.
                pass
    """

    @abstractmethod
    def on_position_complete(
        self,
        position_index: int,
        position_xy: tuple,
        images: Dict[str, Any],
        hardware,
        params: dict,
    ) -> None:
        """Called after all images at a position are acquired.

        Args:
            position_index: Tile index in the grid (0-based)
            position_xy: (x_um, y_um) stage coordinates
            images: Acquired images. Structure depends on acquisition mode:
                - 2D: {angle: image}
                - Z-stack: {angle: [z0_img, z1_img, ...]}
                - Also includes 'projected': {angle: projected_image} if Z-stack
            hardware: MicroscopeHardware instance for additional operations
            params: Full acquisition parameters dict
        """
        ...

    def should_run(self, position_index: int, images: Dict, params: dict) -> bool:
        """Whether this hook should execute at this position.

        Override to implement conditional logic (e.g., run only every Nth
        tile, or only when a quality metric exceeds a threshold).

        Default: always run.
        """
        return True
