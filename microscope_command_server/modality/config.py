"""Modality configuration data model.

Each imaging modality (PPM, brightfield, SHG, etc.) is described by a
ModalityConfig dataclass that captures modality-specific parameters.
Core acquisition code queries these configs via capability checks
(e.g., ``config.autofocus_angle is not None``) rather than hardcoding
modality names.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


@dataclass
class ModalityConfig:
    """Configuration for a microscope imaging modality.

    Fields default to ``None`` / empty, meaning "this modality does not
    have this capability".  Core code checks for capability presence
    rather than modality name.
    """

    # -- Autofocus --
    # Angle to rotate to before autofocus (None = no rotation needed)
    autofocus_angle: Optional[float] = None

    # -- Rotation --
    # Whether this modality uses a rotation stage
    has_rotation: bool = False

    # Default number of angles per tile (1 for non-rotation modalities)
    default_angle_count: int = 1

    # Angle name mapping: numeric angle -> canonical name
    # e.g. {0.0: "crossed", 90.0: "uncrossed", 7.0: "positive", -7.0: "negative"}
    angle_names: Dict[float, str] = field(default_factory=dict)

    # Reverse mapping built automatically from angle_names
    # e.g. {"crossed": 0.0, "uncrossed": 90.0}
    name_to_angle: Dict[str, float] = field(default_factory=dict)

    # -- White balance --
    # Key in YAML white_balance section (e.g. "ppm" -> settings["white_balance"]["ppm"])
    wb_settings_key: Optional[str] = None

    # -- Target intensity --
    # Angle-range -> target intensity mapping for exposure calibration
    # Keys are (lo, hi) inclusive ranges on abs(angle), values are target intensities
    # If empty, uses generic default (200.0)
    angle_intensity_targets: Dict[Tuple[float, float], float] = field(
        default_factory=dict
    )

    # Default target intensity when no angle-specific target matches
    default_target_intensity: float = 200.0

    # -- Post-processing --
    # Directory suffixes created by post-processing (e.g. [".biref", ".sum"])
    post_processing_suffixes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Build reverse name_to_angle mapping from angle_names."""
        if self.angle_names and not self.name_to_angle:
            self.name_to_angle = {v: k for k, v in self.angle_names.items()}

    def get_target_intensity(self, angle: float) -> float:
        """Return target intensity for a given angle.

        Checks angle_intensity_targets ranges; returns
        default_target_intensity if no range matches.
        """
        abs_angle = abs(angle)
        for (lo, hi), target in self.angle_intensity_targets.items():
            if lo <= abs_angle <= hi:
                return target
        return self.default_target_intensity
