"""Tests for the time-lapse "falling behind" warning latch.

Exercises the real workflow helpers ``_format_time_lapse_warning`` and
``_maybe_report_time_lapse_warning`` (the latch invoked once per acquisition
when the first paced timepoint overruns the requested interval).

workflow.py imports microscope_control / microscope_imageprocessing, which are
not installed in the WSL dev environment, so those packages are stubbed before
the module is loaded -- the helpers under test depend on neither.

ASCII-only per project policy.
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


def _install_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    """Register a stub module so workflow.py's top-level imports succeed."""
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for attr, value in (attrs or {}).items():
        if not hasattr(mod, attr):
            setattr(mod, attr, value)
    return mod


def _load_workflow():
    """Load workflow.py with the heavy hardware packages stubbed out."""
    _install_stub("microscope_control")
    _install_stub(
        "microscope_control.hardware",
        {
            "Position": type("Position", (), {}),
            "PycromanagerHardware": type("PycromanagerHardware", (), {}),
        },
    )
    _install_stub(
        "microscope_control.hardware.pycromanager",
        {"PycromanagerHardware": type("PycromanagerHardware", (), {})},
    )
    _install_stub("microscope_control.autofocus")
    _install_stub(
        "microscope_control.autofocus.core",
        {"AutofocusUtils": type("AutofocusUtils", (), {})},
    )
    _install_stub("microscope_imageprocessing")
    _install_stub("microscope_imageprocessing.io")
    _install_stub(
        "microscope_imageprocessing.io.writer",
        {"ome_tiff_writer": lambda *a, **k: None},
    )
    _install_stub("microscope_imageprocessing.correction")
    _install_stub(
        "microscope_imageprocessing.correction.background",
        {"BackgroundCorrectionUtils": type("BackgroundCorrectionUtils", (), {})},
    )

    repo_root = Path(__file__).resolve().parent.parent
    workflow_path = repo_root / "microscope_command_server" / "acquisition" / "workflow.py"
    spec = importlib.util.spec_from_file_location("mcs_workflow_under_test", workflow_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcs_workflow_under_test"] = module
    spec.loader.exec_module(module)
    return module


workflow_mod = _load_workflow()
_format_time_lapse_warning = workflow_mod._format_time_lapse_warning
_maybe_report_time_lapse_warning = workflow_mod._maybe_report_time_lapse_warning


class FakeContext:
    """Minimal stand-in for AcquisitionContext's latch field.

    The full dataclass requires 11 hardware-bound positional args; the latch
    helper only ever reads/writes ``time_lapse_warning_fired``.
    """

    def __init__(self):
        self.time_lapse_warning_fired = False


class FakeScheduler:
    """Minimal stand-in for TimepointScheduler's warning-relevant fields."""

    def __init__(self, interval_seconds: float):
        self.interval_seconds = float(interval_seconds)
        self.overdue_count = 0
        self.last_overdue_seconds = 0.0

    def record_overrun(self, overrun_seconds: float) -> None:
        self.overdue_count += 1
        self.last_overdue_seconds = float(overrun_seconds)


@pytest.fixture
def logger():
    return logging.getLogger("test_time_lapse_warning")


class TestFormatting:
    def test_message_format(self):
        sched = FakeScheduler(interval_seconds=30.0)
        sched.record_overrun(17.3)  # 30.0 + 17.3 = 47.3s acquisition
        msg = _format_time_lapse_warning(sched)
        assert msg == (
            "Time-lapse falling behind: timepoint 1 took 47.3s "
            "but the interval is 30.0s. Remaining timepoints will start late."
        )

    def test_message_is_ascii(self):
        sched = FakeScheduler(interval_seconds=12.0)
        sched.record_overrun(5.5)
        msg = _format_time_lapse_warning(sched)
        # cp1252-safe: must encode cleanly as ASCII (no Unicode arrows/degrees).
        msg.encode("ascii")


class TestLatch:
    def test_no_overrun_does_not_fire(self, logger):
        ctx = FakeContext()
        sched = FakeScheduler(interval_seconds=10.0)
        calls = []
        _maybe_report_time_lapse_warning(ctx, sched, logger, calls.append)
        assert calls == []
        assert ctx.time_lapse_warning_fired is False

    def test_fires_exactly_once_across_two_overruns(self, logger):
        """Drive two overrunning timepoints; the callback fires once only."""
        ctx = FakeContext()
        sched = FakeScheduler(interval_seconds=30.0)
        calls = []

        # First paced timepoint overran by 17.3s (acq 47.3s).
        sched.record_overrun(17.3)
        _maybe_report_time_lapse_warning(ctx, sched, logger, calls.append)

        # Second paced timepoint also overran -- latch must suppress it.
        sched.record_overrun(22.0)
        _maybe_report_time_lapse_warning(ctx, sched, logger, calls.append)

        assert len(calls) == 1, f"expected one warning, got {calls}"
        assert ctx.time_lapse_warning_fired is True
        assert calls[0] == (
            "Time-lapse falling behind: timepoint 1 took 47.3s "
            "but the interval is 30.0s. Remaining timepoints will start late."
        )

    def test_null_callback_still_latches(self, logger):
        """A None callback must not raise and must still set the latch so a
        later non-None callback would not double-fire."""
        ctx = FakeContext()
        sched = FakeScheduler(interval_seconds=5.0)
        sched.record_overrun(3.0)
        _maybe_report_time_lapse_warning(ctx, sched, logger, None)
        assert ctx.time_lapse_warning_fired is True
