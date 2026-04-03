"""Z-stack projection operators and per-position processing hooks.

Projection operators reduce a list of Z-plane images into a single 2D
image suitable for stitching. Each operator takes a list of numpy arrays
(one per Z-plane) and returns a single array of the same spatial dimensions.

The PositionHook ABC enables custom per-position logic during acquisition
(e.g., conditional time-lapse, quality checks, adaptive protocols).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Z-stack projection operators
# ============================================================

def max_intensity_projection(stack: List[np.ndarray]) -> np.ndarray:
    """Maximum intensity projection across Z planes.

    Standard projection for fluorescence and SHG imaging where the
    brightest signal at any Z depth is the desired output.
    """
    return np.max(np.stack(stack, axis=0), axis=0)


def min_intensity_projection(stack: List[np.ndarray]) -> np.ndarray:
    """Minimum intensity projection across Z planes.

    Useful for absorption/transmitted light imaging where the darkest
    (most absorbing) plane represents the feature of interest.
    """
    return np.min(np.stack(stack, axis=0), axis=0)


def sum_projection(stack: List[np.ndarray]) -> np.ndarray:
    """Sum projection across Z planes.

    Computes in float64 to prevent integer overflow, then clips to
    the input dtype range. Useful when total signal across Z matters
    (e.g., thick-section fluorescence).
    """
    arr = np.stack(stack, axis=0).astype(np.float64)
    summed = np.sum(arr, axis=0)
    dtype = stack[0].dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(summed, info.min, info.max).astype(dtype)
    return summed.astype(dtype)


def mean_projection(stack: List[np.ndarray]) -> np.ndarray:
    """Mean projection across Z planes.

    Averages all Z planes, reducing noise compared to any single plane.
    Computed in float64 for precision, cast back to input dtype.
    """
    arr = np.stack(stack, axis=0).astype(np.float64)
    averaged = np.mean(arr, axis=0)
    dtype = stack[0].dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(averaged, info.min, info.max).astype(dtype)
    return averaged.astype(dtype)


def std_projection(stack: List[np.ndarray]) -> np.ndarray:
    """Standard deviation projection across Z planes.

    Highlights regions where signal varies across Z depth. Useful for
    identifying structures that are sharply localized in Z (high std)
    versus uniform background (low std).
    """
    arr = np.stack(stack, axis=0).astype(np.float64)
    deviation = np.std(arr, axis=0)
    dtype = stack[0].dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(deviation, info.min, info.max).astype(dtype)
    return deviation.astype(dtype)


# ============================================================
# Projection registry
# ============================================================

PROJECTIONS: Dict[str, Callable[[List[np.ndarray]], np.ndarray]] = {
    "max": max_intensity_projection,
    "min": min_intensity_projection,
    "sum": sum_projection,
    "mean": mean_projection,
    "std": std_projection,
}


def get_projection(name: str) -> Callable[[List[np.ndarray]], np.ndarray]:
    """Look up a projection operator by name.

    Args:
        name: Projection name ("max", "min", "sum", "mean", "std")

    Returns:
        Projection function: List[ndarray] -> ndarray

    Raises:
        KeyError: If name is not a registered projection
    """
    if name not in PROJECTIONS:
        raise KeyError(
            f"Unknown projection '{name}'. "
            f"Available: {', '.join(sorted(PROJECTIONS.keys()))}"
        )
    return PROJECTIONS[name]


def generate_z_offsets(z_range_um: float, z_step_um: float) -> List[float]:
    """Generate symmetric Z offsets around the current focus position.

    Args:
        z_range_um: Total Z range in micrometers (e.g., 10 means +/-5)
        z_step_um: Step size in micrometers

    Returns:
        List of Z offsets relative to center (e.g., [-5, -3, -1, 1, 3, 5])
    """
    if z_step_um <= 0:
        raise ValueError(f"z_step must be positive, got {z_step_um}")
    if z_range_um <= 0:
        return [0.0]

    half = z_range_um / 2.0
    offsets = []
    z = -half
    while z <= half + z_step_um * 0.01:  # float epsilon
        offsets.append(round(z, 3))
        z += z_step_um

    if not offsets:
        return [0.0]

    logger.debug("Z-stack offsets: %d planes over +/-%.1f um (step=%.1f)",
                 len(offsets), half, z_step_um)
    return offsets


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
