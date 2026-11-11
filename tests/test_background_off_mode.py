"""Background collection with white balance Off on a color camera.

2026-10-21 (demo) and 2026-11-11 (PPM): Off collections failed with "tuple index out of
range". Without white balance the red channel fills first; the exposure was capped to keep
red below saturation, so the overall median stayed at 173 against a target of 200, the
adaptive exposure gave up, and the failed angle then reached the TIFF writer as None.
"""

import pathlib
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "workflow.py"
)


@pytest.fixture
def workflow():
    mocks = {}
    for mod in [
        "pycromanager",
        "microscope_control",
        "microscope_control.hardware",
        "microscope_control.hardware.pycromanager",
        "microscope_control.autofocus",
        "microscope_control.autofocus.core",
        "ppm_library",
        "ppm_library.imaging",
        "ppm_library.imaging.writer",
        "ppm_library.imaging.background",
        "skimage",
        "skimage.filters",
    ]:
        if mod not in sys.modules:
            mocks[mod] = MagicMock()
            sys.modules[mod] = mocks[mod]
    from microscope_command_server.acquisition import workflow as wf

    yield wf
    for mod in mocks:
        del sys.modules[mod]


class RedCastCamera:
    """A color camera without white balance: red is 1.63x green, blue 1.28x green."""

    def __init__(self):
        self.exposure = 50.0
        self.camera = MagicMock()

    def set_exposure(self, ms):
        self.exposure = float(ms)

    def get_exposure(self):
        return self.exposure

    def snap_image(self):
        g = 13.0 * self.exposure
        rgb = np.array([1.63 * g, g, 1.28 * g])
        image = np.clip(np.ones((32, 32, 3)) * rgb, 0, 255).astype(np.uint8)
        return image, {}


def test_an_exposure_limited_by_one_channel_is_accepted(workflow):
    hardware = RedCastCamera()
    image, exposure = workflow.acquire_background_with_target_intensity(
        hardware=hardware,
        target_intensity=200.0,
        tolerance=2.5,
        initial_exposure_ms=50.0,
        max_iterations=10,
        logger=MagicMock(),
    )
    assert image.ndim == 3
    red = float(np.median(image[:, :, 0]))
    assert 200 <= red < 245, "red as bright as it can be without saturating"
    assert float(np.median(image)) < 200, "the overall median cannot reach the target"


def test_a_failed_angle_is_reported_not_saved():
    source = WORKFLOW.read_text()
    body = source[source.index("def simple_background_collection(") :]
    body = body[: body.index("\ndef ")]
    # Every per-angle failure records its reason and skips the save.
    handlers = body.split('logger.error(f"Failed to acquire background at angle {angle}: {e}")')[1:]
    assert len(handlers) == 4
    for after in handlers:
        head = after[:200]
        assert "failed_angles[angle] = str(e)" in head
        assert "continue" in head
    assert "Background not collected for" in body
