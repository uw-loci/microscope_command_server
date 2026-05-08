"""Tests for TimepointScheduler.

Uses an injected fake clock/sleep for fast-case and slow-case timing so
the tests do not depend on wall-clock jitter. Cancellation test uses a
real threading.Event and very short intervals.

ASCII-only per project policy.
"""

import importlib.util
import logging
import sys
import threading
import time
import types
from pathlib import Path

import pytest


def _load_timepoint_scheduler():
    """Load timepoint_scheduler.py without executing the acquisition package
    __init__ (which imports microscope_control -- not available in WSL dev).
    """
    if "microscope_control" not in sys.modules:
        sys.modules["microscope_control"] = types.ModuleType("microscope_control")
    hw_mod = sys.modules.setdefault(
        "microscope_control.hardware",
        types.ModuleType("microscope_control.hardware"),
    )
    if not hasattr(hw_mod, "Position"):
        hw_mod.Position = type("Position", (), {})
    if not hasattr(hw_mod, "PycromanagerHardware"):
        hw_mod.PycromanagerHardware = type("PycromanagerHardware", (), {})

    repo_root = Path(__file__).resolve().parent.parent
    scheduler_path = (
        repo_root / "microscope_command_server" / "acquisition" / "timepoint_scheduler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mcs_timepoint_scheduler_under_test", scheduler_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcs_timepoint_scheduler_under_test"] = module
    spec.loader.exec_module(module)
    return module


sched_mod = _load_timepoint_scheduler()
TimepointScheduler = sched_mod.TimepointScheduler


class FakeClock:
    """Deterministic clock/sleep pair for scheduler tests."""

    def __init__(self, start: float = 1000.0):
        self.now = float(start)
        self.sleep_log = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_log.append(seconds)
        self.now += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def logger():
    return logging.getLogger("test_timepoint_scheduler")


class TestFastCase:
    """interval=0.1s, 3 timepoints. Each wait_until returns ~0 delay for
    t=0, and the slept delay matches t0 + t_idx*interval."""

    def test_three_timepoints_interval_100ms(self, logger):
        fk = FakeClock(start=1000.0)
        sched = TimepointScheduler(
            t0_monotonic=1000.0,
            interval_seconds=0.1,
            logger=logger,
            clock=fk.clock,
            sleep=fk.sleep,
        )
        # t_idx = 0 is always immediate (target == t0).
        assert sched.wait_until(0) == 0.0
        assert fk.now == 1000.0

        # t_idx = 1: no prior advance -> sleep to t0 + 0.1 = 1000.1
        delay_1 = sched.wait_until(1)
        assert delay_1 == pytest.approx(0.1, abs=1e-9)
        assert fk.now == pytest.approx(1000.1, abs=1e-9)

        # t_idx = 2: still on schedule.
        delay_2 = sched.wait_until(2)
        assert delay_2 == pytest.approx(0.1, abs=1e-9)
        assert fk.now == pytest.approx(1000.2, abs=1e-9)

        assert sched.overdue_count == 0

    def test_drift_bounded_over_many_points(self, logger):
        """With variable (but fast) acq times, each start is still at
        t0 + N*interval."""
        fk = FakeClock(start=0.0)
        sched = TimepointScheduler(
            t0_monotonic=0.0,
            interval_seconds=30.0,
            logger=logger,
            clock=fk.clock,
            sleep=fk.sleep,
        )
        starts = []
        acq_times = [5.0, 10.0, 2.0, 7.0, 5.0]
        for t_idx in range(5):
            if t_idx > 0:
                sched.wait_until(t_idx)
            starts.append(fk.now)
            fk.advance(acq_times[t_idx])
        assert starts == [0.0, 30.0, 60.0, 90.0, 120.0]


class TestSlowCase:
    """When acq_time > interval, wait_until returns 0 immediately and
    logs a warning. Schedule re-anchoring is NOT done -- the scheduler
    stays anchored to t0, so drift is bounded to one interval."""

    def test_slow_acquisition_returns_zero_and_warns(self, logger, caplog):
        fk = FakeClock(start=0.0)
        sched = TimepointScheduler(
            t0_monotonic=0.0,
            interval_seconds=0.1,
            logger=logger,
            clock=fk.clock,
            sleep=fk.sleep,
        )
        # Simulate a 0.3s acquisition (3x the interval).
        fk.advance(0.3)

        with caplog.at_level(logging.WARNING):
            delay = sched.wait_until(1)
        assert delay == 0.0
        # No sleep should have been requested.
        assert fk.sleep_log == []
        assert sched.overdue_count == 1
        assert any(
            "overdue" in r.getMessage() for r in caplog.records
        ), f"no overdue warning found in records: {caplog.records}"

    def test_interval_zero_never_warns(self, logger, caplog):
        """interval=0 means 'back-to-back timepoints'; the delay=0 path
        should NOT emit a spurious overdue warning."""
        fk = FakeClock(start=0.0)
        sched = TimepointScheduler(
            t0_monotonic=0.0,
            interval_seconds=0.0,
            logger=logger,
            clock=fk.clock,
            sleep=fk.sleep,
        )
        with caplog.at_level(logging.WARNING):
            for t_idx in range(5):
                assert sched.wait_until(t_idx) == 0.0
        assert fk.sleep_log == []
        assert sched.overdue_count == 0
        # No warnings emitted for the expected interval=0 path.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []


class TestCancellation:
    """Cancellation during a long wait returns quickly."""

    def test_event_cancels_wait(self, logger):
        """Uses a real threading.Event + real time.sleep so the timing
        test is genuine. Interval is 10s; we cancel after ~0.1s and
        assert total elapsed <= ~0.7s (one poll interval of slack)."""
        sched = TimepointScheduler(
            t0_monotonic=time.monotonic(),
            interval_seconds=10.0,
            logger=logger,
        )
        cancel_event = threading.Event()

        def cancel_after_delay():
            time.sleep(0.1)
            cancel_event.set()

        t0 = time.monotonic()
        thread = threading.Thread(target=cancel_after_delay, daemon=True)
        thread.start()
        delay = sched.wait_until(1, cancel_event=cancel_event)
        elapsed = time.monotonic() - t0
        thread.join(timeout=1.0)

        assert delay == 0.0
        # 0.5s poll + a little slack. If this ever flakes, raise to 1.0s.
        assert elapsed < 0.7, f"cancellation took too long: {elapsed:.3f}s"

    def test_callable_cancel_supported(self, logger):
        fk = FakeClock(start=0.0)
        flag = {"c": False}
        sched = TimepointScheduler(
            t0_monotonic=0.0,
            interval_seconds=10.0,
            logger=logger,
            clock=fk.clock,
            sleep=fk.sleep,
        )
        # Fake sleep sets the cancel flag on the second poll.
        orig_sleep = fk.sleep
        sleep_count = [0]

        def sleep_with_cancel(s):
            sleep_count[0] += 1
            if sleep_count[0] >= 2:
                flag["c"] = True
            orig_sleep(s)

        sched._sleep = sleep_with_cancel
        delay = sched.wait_until(1, cancel_event=lambda: flag["c"])
        assert delay == 0.0


class TestValidation:
    def test_rejects_negative_interval(self, logger):
        with pytest.raises(ValueError, match="interval_seconds"):
            TimepointScheduler(t0_monotonic=0.0, interval_seconds=-1.0, logger=logger)

    def test_rejects_negative_t_idx(self, logger):
        sched = TimepointScheduler(t0_monotonic=0.0, interval_seconds=1.0, logger=logger)
        with pytest.raises(ValueError, match="t_idx"):
            sched.wait_until(-1)

    def test_rejects_bad_cancel_type(self, logger):
        # Use a fake clock so delay > 0 and the wait loop is actually
        # entered; otherwise wait_until returns before validating the
        # cancel_event type.
        fk = FakeClock(start=0.0)
        sched = TimepointScheduler(
            t0_monotonic=0.0,
            interval_seconds=1.0,
            logger=logger,
            clock=fk.clock,
            sleep=fk.sleep,
        )
        with pytest.raises(TypeError, match="cancel_event"):
            sched.wait_until(1, cancel_event=object())
