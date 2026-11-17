"""A streaming autofocus must not change the camera ROI under a running Live stream.

PPM, 2026-11-13 18:56: STRMAFZ arrived while the QuPath Live Viewer was streaming and
Micro-Manager's live mode was on. The handler read ``is_sequence_running()`` before the
crop, again before the pre-flight frame and again before the scan, and the reads
disagreed: the crop ran without stopping anything, the pre-flight frame was a snap, and
the scan's own start then failed with "This operation can not be executed while
sequence acquisition is running" -- Live had restarted in between. The ROI was restored
with the stream still running and the server stopped answering seconds later.

The handler now reads the state once (``_LiveStreamPause``), stops Live -- the core
sequence AND Micro-Manager's live mode, which restarts its sequence by itself -- before
it touches the ROI, and restarts it after the ROI is restored.

The fake core below behaves like that rig: while live mode is on, a stopped sequence
comes back by itself, so ``is_sequence_running()`` answers differently from one call to
the next; and, like the JAI, it refuses an ROI change while a sequence runs.
"""

import importlib
import importlib.util
import inspect
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

RUNNING_ERROR = "This operation can not be executed while sequence acquisition is running."
FULL = (0, 0, 2064, 1544)


def _stub_microscope_control_if_absent():
    """Make ``microscope_control`` importable when it is not installed (see
    test_streaming_af_speed_restore.py: nothing under test touches it)."""
    try:
        importlib.import_module("microscope_control.hardware")
        return
    except Exception:
        pass

    class _Any:
        def __init__(self, *args, **kwargs):
            pass

    for name in (
        "microscope_control",
        "microscope_control.hardware",
        "microscope_control.hardware.camera",
        "microscope_control.config",
        "microscope_control.autofocus",
    ):
        module = types.ModuleType(name)
        module.__getattr__ = lambda attr, _any=_Any: _any  # noqa: ARG005
        module.__path__ = []
        sys.modules[name] = module


def _load_streaming_focus():
    _stub_microscope_control_if_absent()
    path = ROOT / "microscope_command_server" / "server" / "handlers" / "streaming_focus.py"
    spec = importlib.util.spec_from_file_location("_sf_live_pause", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sf = _load_streaming_focus()


class FakeCore:
    """A camera with a sequence, an ROI, and Micro-Manager live mode behind it."""

    def __init__(self, running=False):
        self.running = running
        self.live_mode = False
        self.roi = FULL
        self.roi_changes = []
        self.roi_changes_while_running = []
        self.starts = 0
        self.stops = 0
        self.seq_reads = []

    def _live_mode_restarts(self):
        # Micro-Manager's live mode puts its sequence back when something stops it.
        if self.live_mode and not self.running:
            self.running = True

    def is_sequence_running(self):
        answer = self.running
        self.seq_reads.append(answer)
        self._live_mode_restarts()
        return answer

    def stop_sequence_acquisition(self):
        self.stops += 1
        self.running = False

    def start_continuous_sequence_acquisition(self, interval):
        if self.running:
            raise RuntimeError(RUNNING_ERROR)
        self.starts += 1
        self.running = True

    def clear_circular_buffer(self):
        pass

    def _change_roi(self, roi):
        self._live_mode_restarts()
        self.roi_changes.append(roi)
        if self.running:
            self.roi_changes_while_running.append(roi)
            raise RuntimeError(RUNNING_ERROR)
        self.roi = roi

    def clear_roi(self):
        self._change_roi(FULL)

    def set_roi(self, x, y, w, h):
        self._change_roi((x, y, w, h))

    def get_roi(self):
        return self.roi


class FakeLive:
    def __init__(self, core):
        self.core = core

    def is_live_mode_on(self):
        return self.core.live_mode

    def set_live_mode(self, on):
        self.core.live_mode = bool(on)
        if on and not self.core.running:
            self.core.running = True
        if not on:
            self.core.running = False


class FakeStudio:
    def __init__(self, core):
        self._live = FakeLive(core)

    def live(self):
        return self._live


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sf.time, "sleep", lambda s: None)


def run_scan_camera_steps(core, studio):
    """The handler's camera steps, in its order, around a scan that owns the camera.

    pause -> crop -> pre-flight -> one attempt (starts and stops its own sequence when
    none is running, as _attempt_one_scan does) -> finally: stop a leftover sequence,
    restore the ROI, resume Live.
    """
    live_pause = sf._LiveStreamPause(core, studio)
    live_pause.pause()
    saved_roi, roi_seq_was_running = sf._apply_crop_roi(core, 0.5)
    sequence_was_running = roi_seq_was_running
    try:
        if not sequence_was_running:
            core.clear_circular_buffer()
            core.start_continuous_sequence_acquisition(0)
            core.stop_sequence_acquisition()
    finally:
        if not sequence_was_running and core.is_sequence_running():
            core.stop_sequence_acquisition()
        sf._restore_roi(core, saved_roi, roi_seq_was_running)
        live_pause.resume()
    return live_pause


class TestLiveIsStoppedBeforeTheRoiChanges:
    def test_the_fake_reproduces_the_incident_without_the_pause(self):
        # Live mode on, sequence running: the crop helper stops the sequence, live mode
        # restarts it, and the ROI change lands on a running stream.
        core = FakeCore(running=True)
        FakeStudio(core).live().set_live_mode(True)
        sf._apply_crop_roi(core, 0.5)
        assert core.roi_changes_while_running, "the fake must show the 2026-11-13 failure"

    def test_live_mode_and_its_sequence_are_stopped_and_put_back(self):
        core = FakeCore(running=True)
        studio = FakeStudio(core)
        studio.live().set_live_mode(True)

        pause = run_scan_camera_steps(core, studio)

        assert core.roi_changes, "the scan did crop"
        assert core.roi_changes_while_running == []
        assert pause.sequence_was_running and pause.live_mode_was_on
        # Live is back: live mode on, its sequence running, the camera at full sensor.
        assert core.live_mode
        assert core.running
        assert core.roi == FULL

    def test_a_viewer_sequence_alone_is_stopped_and_restarted(self):
        # The QPSC Live Viewer's STRTSEQ stream, with Micro-Manager live mode off.
        core = FakeCore(running=True)
        run_scan_camera_steps(core, FakeStudio(core))
        assert core.roi_changes_while_running == []
        assert core.running, "Live is restarted after the scan"
        assert not core.live_mode, "live mode was off and stays off"
        assert core.roi == FULL
        # One start for the scan's own sequence, one to put Live back.
        assert core.starts == 2

    def test_without_a_studio_the_sequence_is_still_handled(self):
        core = FakeCore(running=True)
        run_scan_camera_steps(core, None)
        assert core.roi_changes_while_running == []
        assert core.running

    def test_nothing_running_changes_nothing(self):
        core = FakeCore(running=False)
        pause = sf._LiveStreamPause(core, FakeStudio(core))
        assert pause.pause() is False
        assert core.stops == 0
        pause.resume()
        assert core.starts == 0 and not core.running

        core = FakeCore(running=False)
        run_scan_camera_steps(core, FakeStudio(core))
        assert core.roi_changes_while_running == []
        assert not core.running, "nothing was running, so nothing is left running"
        assert core.starts == 1, "only the scan's own sequence"

    def test_resume_is_idempotent(self):
        core = FakeCore(running=True)
        pause = sf._LiveStreamPause(core, None)
        pause.pause()
        pause.resume()
        pause.resume()
        assert core.starts == 1

    def test_the_log_lines_are_ascii(self, caplog):
        core = FakeCore(running=True)
        studio = FakeStudio(core)
        studio.live().set_live_mode(True)
        with caplog.at_level("INFO", logger=sf.logger.name):
            run_scan_camera_steps(core, studio)
        lines = [r.getMessage() for r in caplog.records]
        assert any("Live was running" in m and "stopped it" in m for m in lines), lines
        assert any("restarted Live after the scan" in m for m in lines), lines
        assert all(m.isascii() for m in lines)


class TestTheHandlerReadsTheStateOnce:
    """The handler is too large to drive with a fake; these pin its wiring."""

    SRC = inspect.getsource(sf._handle_streaming_focus)

    def test_live_is_paused_before_the_crop(self):
        assert self.SRC.index("live_pause.pause()") < self.SRC.index("_apply_crop_roi(")

    def test_every_roi_restore_is_followed_by_resuming_live(self):
        restores = [m.end() for m in re.finditer(r"_restore_roi\(core, saved_roi", self.SRC)]
        assert len(restores) == 3
        for end in restores:
            following = self.SRC[end : end + 400]
            assert "live_pause.resume()" in following

    def test_the_scan_does_not_read_the_sequence_state_again_to_decide(self):
        after = self.SRC[self.SRC.index("live_pause.pause()") :]
        assert "preflight_sequence_running = roi_seq_was_running" in after
        assert "sequence_was_running = roi_seq_was_running" in after
        assert "preflight_sequence_running = bool(core.is_sequence_running())" not in after
        assert "sequence_was_running = bool(core.is_sequence_running())" not in after
