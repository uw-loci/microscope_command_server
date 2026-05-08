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
    "PositionHook",
]


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
