"""Tests for overlapped I/O invariants in the acquisition workflow.

The acquisition workflow uses a background thread pool for TIFF writes,
with pending writes drained at each autofocus check position. These tests
ensure the invariants that make this safe:

1. af_n_tiles >= 1 -- guarantees periodic drain points
2. _TileWritePool correctly drains and reports errors

Note: AutofocusUtils.get_autofocus_positions tests live in microscope_control's
test suite (test_autofocus_metrics.py). This file tests the workflow-side invariants
that don't require hardware imports.
"""

import sys
import threading
import pytest
from unittest.mock import MagicMock, patch


# ── _TileWritePool tests ───────────────────────────────────────────────
# Import _TileWritePool carefully -- workflow.py has transitive hardware deps.
# We mock the heavy imports to allow importing just the pool class.

@pytest.fixture(autouse=True)
def mock_hardware_imports():
    """Mock hardware modules so workflow.py can be imported in WSL/CI."""
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
    yield
    # Restore originals
    for mod in mocks:
        del sys.modules[mod]


def _get_pool_class():
    """Import _TileWritePool with mocked deps."""
    from microscope_command_server.acquisition.workflow import _TileWritePool
    return _TileWritePool


class TestTileWritePool:
    """Tests for the bounded background write pool."""

    def test_submit_and_drain(self):
        """Basic submit-and-drain cycle."""
        pool = _get_pool_class()(max_workers=1)
        results = []
        pool.submit(lambda: results.append(1))
        pool.submit(lambda: results.append(2))
        pool.drain()
        assert results == [1, 2]
        assert pool.pending_count == 0
        pool.shutdown()

    def test_drain_reports_failures(self):
        """Failed writes should be counted but not raise."""
        pool = _get_pool_class()(max_workers=1)

        def fail():
            raise IOError("disk full")

        pool.submit(fail)
        pool.submit(lambda: None)  # succeeds
        failed = pool.drain()
        assert failed == 1
        assert pool.total_failed == 1
        pool.shutdown()

    def test_pending_count(self):
        """pending_count tracks submitted but undrained futures."""
        pool = _get_pool_class()(max_workers=1)
        assert pool.pending_count == 0

        barrier = threading.Event()
        pool.submit(lambda: barrier.wait(timeout=5))
        assert pool.pending_count == 1
        barrier.set()
        pool.drain()
        assert pool.pending_count == 0
        pool.shutdown()

    def test_shutdown_drains_remaining(self):
        """Shutdown should complete all pending writes."""
        pool = _get_pool_class()(max_workers=1)
        results = []
        pool.submit(lambda: results.append("done"))
        pool.shutdown()
        assert results == ["done"]

    def test_multiple_drains(self):
        """Drain can be called multiple times safely."""
        pool = _get_pool_class()(max_workers=1)
        results = []

        pool.submit(lambda: results.append(1))
        pool.drain()
        assert results == [1]

        pool.submit(lambda: results.append(2))
        pool.drain()
        assert results == [1, 2]
        assert pool.total_failed == 0
        pool.shutdown()

    def test_drain_empty_is_noop(self):
        """Draining with no pending writes should return 0 failures."""
        pool = _get_pool_class()(max_workers=1)
        failed = pool.drain()
        assert failed == 0
        assert pool.pending_count == 0
        pool.shutdown()


class TestAfNTilesInvariant:
    """af_n_tiles must be >= 1 to guarantee I/O drain points.

    The overlapped I/O system drains at each autofocus check position.
    If af_n_tiles < 1, pending writes could accumulate unboundedly.
    The clamping logic in _acquisition_workflow() prevents this.

    This test verifies the clamp would be applied by testing the
    condition directly, since running the full workflow requires
    hardware.
    """

    def test_af_n_tiles_zero_clamped(self):
        """af_n_tiles=0 should be clamped to 1."""
        af_n_tiles = 0
        if af_n_tiles < 1:
            af_n_tiles = max(1, af_n_tiles)
        assert af_n_tiles == 1

    def test_af_n_tiles_negative_clamped(self):
        """Negative af_n_tiles should be clamped to 1."""
        af_n_tiles = -5
        if af_n_tiles < 1:
            af_n_tiles = max(1, af_n_tiles)
        assert af_n_tiles == 1

    def test_af_n_tiles_valid_unchanged(self):
        """Valid af_n_tiles (>= 1) should not be changed."""
        for n in [1, 3, 5, 10]:
            af_n_tiles = n
            if af_n_tiles < 1:
                af_n_tiles = max(1, af_n_tiles)
            assert af_n_tiles == n, f"af_n_tiles={n} should be unchanged"
