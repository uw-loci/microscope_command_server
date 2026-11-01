"""Flat-field profile correction during acquisition.

The alternative to background images for detectors that have none. One corrector serves a run; every
acquisition path calls :meth:`FlatfieldCorrector.apply` on each raw plane, before any projection.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class FlatfieldCorrector:
    """Applies the profiles chosen for a run, by channel.

    A missing or unusable profile warns and leaves that image uncorrected, as a missing background
    does: the run goes on.
    """

    def __init__(self, profiles: Optional[Dict[Optional[str], Any]] = None):
        self._profiles = profiles or {}
        self.clipped_pixels = 0

    @classmethod
    def load(cls, folder: Optional[str], channels: Sequence[str] = ()) -> "FlatfieldCorrector":
        """Read the profiles in a folder; no folder gives a corrector that changes nothing."""
        if not folder:
            return cls()
        try:
            from microscope_imageprocessing.correction import load_profile_set
        except ImportError as e:
            logger.warning("Flat-field profiles need a newer microscope_imageprocessing: %s", e)
            return cls()

        try:
            profiles = load_profile_set(folder, channels)
        except Exception as e:
            logger.warning("Could not read flat-field profiles in %s: %s", folder, e)
            return cls()
        if not profiles:
            logger.warning("No flat-field profile in %s; acquiring uncorrected", folder)
            return cls()
        missing = [c for c in channels if c not in profiles and None not in profiles]
        if missing:
            logger.warning(
                "No flat-field profile for %s; those stay uncorrected", ", ".join(missing)
            )
        logger.info("Flat-field profiles from %s: %s", folder, sorted(map(str, profiles)))
        return cls(profiles)

    @property
    def loaded(self) -> List[str]:
        """What is covered: channel ids, or "all" for a profile covering every image."""
        return sorted(
            "all" if key is None else key
            for key, profile in self._profiles.items()
            if profile is not None
        )

    def apply(self, image, channel_id: Optional[str] = None):
        """Correct one raw plane, returning the image unchanged when no profile covers it."""
        key = channel_id if channel_id in self._profiles else None
        profile = self._profiles.get(key)
        if profile is None:
            return image
        if image.shape != profile.flatfield.shape:
            logger.warning(
                "Flat-field profile is %s but the image is %s: it was fitted for another scan "
                "size or detector; acquiring uncorrected",
                profile.flatfield.shape,
                image.shape,
            )
            # Kept as None so this channel does not fall through to the profile for every image
            self._profiles[key] = None
            return image
        corrected, clipped = profile.apply(image)
        self.clipped_pixels += clipped
        return corrected
