"""The streaming autofocus's Brent fallback stops on a client cancel (ABORTAF).

On 2026-10-28 single-tile refinement jumped onto bare glass, the streaming passes found
nothing and the search escalated to the Brent fallback, a stop-and-snap search over a wide
bracket that never looked at the abort Event, so Cancel was ignored until it finished and it
committed a focus. Reproduced on the demo microscope: ABORTAF acknowledged at 16:45:44.876,
then seven more evaluations and "committed final Z=5.205".
"""

import threading

import numpy as np

from microscope_command_server.server.af_events import AutofocusEventLog
from microscope_command_server.server.handlers import streaming_focus as sf


class _FakeCore:
    """A focus stage and camera whose frames are sharpest at Z = 10."""

    def __init__(self, abort_event=None, abort_after=None):
        self.z = 0.0
        self.snaps = 0
        self._abort_event = abort_event
        self._abort_after = abort_after

    def set_position(self, device, z):
        self.z = float(z)

    def get_position(self, device):
        return self.z

    def device_busy(self, device):
        return False

    def wait_for_device(self, device):
        pass

    def snap_image(self):
        self.snaps += 1
        if self._abort_after is not None and self.snaps >= self._abort_after:
            self._abort_event.set()  # the operator presses Cancel during this evaluation

    def get_image(self):
        sharpness = 50.0 * np.exp(-0.5 * ((self.z - 10.0) / 5.0) ** 2) + 1.0
        rng = np.random.default_rng(0)
        return (rng.standard_normal(32 * 32) * sharpness + 1000.0).astype(np.uint16)

    def get_image_width(self):
        return 32

    def get_image_height(self):
        return 32

    def get_number_of_components(self):
        return 1


def _run(core, abort_event, monkeypatch):
    log = AutofocusEventLog()
    monkeypatch.setattr(sf, "AF_EVENTS", log)
    log.af_started("streaming", {})
    result = sf._brent_fallback_scan(
        core,
        "Z",
        None,
        -50.0,
        70.0,
        "brenner_gradient",
        abort_event=abort_event,
    )
    return result, log.since(0)["events"]


def test_a_cancel_stops_the_fallback_and_commits_nothing(monkeypatch):
    abort = threading.Event()
    core = _FakeCore(abort_event=abort, abort_after=2)
    result, events = _run(core, abort, monkeypatch)
    assert result.status == "aborted" and result.best_z is None
    assert core.snaps == 2  # nothing after the cancel
    assert result.n_samples == 2
    assert [e["type"] for e in events].count("sample") == 2  # the operator saw both


def test_without_a_cancel_the_fallback_still_converges(monkeypatch):
    abort = threading.Event()
    core = _FakeCore()
    result, events = _run(core, abort, monkeypatch)
    assert result.status == "success"
    assert abs(result.best_z - 10.0) < 6.0
    assert [e["type"] for e in events].count("sample") == result.n_samples


def test_the_fallback_is_handed_the_abort_event():
    src = sf.__file__
    text = open(src, encoding="utf-8").read()
    call = text[text.index("brent_result = _brent_fallback_scan(") :]
    call = call[: call.index(")")]
    assert "abort_event=abort_event" in call
