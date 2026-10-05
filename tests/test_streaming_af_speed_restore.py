"""The slow scan speed must not outlive one streaming-AF scan.

A streaming scan drops the focus device to its slowest speed (MaxSpeed=1 on
Prior, about 11.5 um/s) so that streamed frames sample Z densely. Two paths used
to return from inside the scan without restoring it: the ABORTAF cancel and the
rapid-jump abort.

The cost was not subtle. The caller's first act after a cancel is to retract to
the safe Z, which on the approach-from-safe-Z scan can be the full 693 um
travel; at 11.5 um/s that is the better part of a minute of the stage still
moving after the operator pressed cancel. Observed 2026-10-04: ABORTAF at
21:45:56, "retracted to safe Z" at 21:46:27 -- 31 seconds for about 350 um,
which the operator read as the cancel having been ignored. Nothing upstream
restored it either, so every later Z move in the session crawled as well.

These tests drive the real function with a fake Core, because the property that
matters is what the stage is left holding after each kind of exit -- not any one
calculation inside the scan.
"""

import importlib
import importlib.util
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stub_microscope_control_if_absent():
    """Make ``microscope_control`` importable when it is not installed.

    CI installs the real package from git, so there it is used as-is. Locally the
    system Python is externally managed (PEP 668) and it is usually absent, which
    would otherwise make this test unrunnable on the machine where the fix is
    written -- the worst place to lose a test. Nothing under test touches
    microscope_control; it is reached only because the handlers package imports
    Position on the way in, so a permissive stub is enough.
    """
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
    """Load the handler by path; importing its package pulls in the server stack."""
    _stub_microscope_control_if_absent()
    path = ROOT / "microscope_command_server" / "server" / "handlers" / "streaming_focus.py"
    spec = importlib.util.spec_from_file_location("_sf_speed_restore", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sf = _load_streaming_focus()

SLOW = "1"
NORMAL = "100"
DEVICE = "ZStage"
SPEED_PROP = "MaxSpeed"


class FakeCore:
    """Just enough MM Core to get into and out of a scan."""

    def __init__(self):
        self.props = {SPEED_PROP: NORMAL}
        self.z = 0.0
        self.speed_history = []

    # --- properties ---
    def set_property(self, device, name, value):
        self.props[name] = str(value)
        if name == SPEED_PROP:
            self.speed_history.append(str(value))

    def get_property(self, device, name):
        return self.props[name]

    def get_allowed_property_values(self, device, name):
        return [SLOW, NORMAL]

    # --- focus device ---
    def set_position(self, device, z=None):
        self.z = float(z if z is not None else device)

    def get_position(self, device=None):
        return self.z

    def device_busy(self, device):
        return False

    def wait_for_device(self, device):
        pass

    # --- camera / buffer (unused: the scan is told a sequence is already running) ---
    def clear_circular_buffer(self):
        pass

    def start_continuous_sequence_acquisition(self, interval):
        pass

    def stop_sequence_acquisition(self):
        pass


def run_scan(core, monkeypatch, scan_outcome):
    """Run one scan whose streaming loop ends in ``scan_outcome``."""

    def fake_loop(*args, **kwargs):
        raise scan_outcome

    monkeypatch.setattr(sf, "_run_streaming_scan", fake_loop)
    return sf._attempt_one_scan(
        core,
        DEVICE,
        SPEED_PROP,
        z_center=-346.5,
        range_um=693.0,
        # True so the scan does not touch the camera: it is told the Live Viewer's
        # sequence is already running, which is the real approach-scan case.
        sequence_was_running_on_entry=True,
        attempt_label="approach",
        slow_value=SLOW,
        normal_value=NORMAL,
        z_endpoints=(0.0, -693.0),
        abort_event=threading.Event(),
    )


def test_a_cancel_leaves_the_stage_at_full_speed(monkeypatch):
    """The regression. A retract at the slow speed is what looked like a hang."""
    core = FakeCore()
    result = run_scan(core, monkeypatch, sf._AbortRequested("user cancelled"))

    assert result.status == "aborted"
    assert core.props[SPEED_PROP] == NORMAL
    # The scan did go slow at some point -- otherwise this test would pass on a
    # function that never sets the slow speed at all.
    assert SLOW in core.speed_history
    assert core.speed_history[-1] == NORMAL


def test_a_rapid_jump_abort_leaves_the_stage_at_full_speed(monkeypatch):
    core = FakeCore()
    result = run_scan(core, monkeypatch, sf._RapidJumpDetected("stage jumped to z_end"))

    assert result.status == "rapid_jump"
    assert core.props[SPEED_PROP] == NORMAL
    assert core.speed_history[-1] == NORMAL


def test_an_unexpected_error_leaves_the_stage_at_full_speed(monkeypatch):
    # The generic handler returns from inside the try as well.
    core = FakeCore()
    result = run_scan(core, monkeypatch, RuntimeError("camera fell over"))

    assert result.status == "error"
    assert core.props[SPEED_PROP] == NORMAL
    assert core.speed_history[-1] == NORMAL


def test_the_guarantee_survives_a_stage_that_refuses_the_restore(monkeypatch):
    """A stage that will not take the property must not break the cancel."""
    core = FakeCore()

    original = core.set_property
    calls = {"n": 0}

    def flaky(device, name, value):
        calls["n"] += 1
        # Refuse only the final restore, after the scan has gone slow.
        if name == SPEED_PROP and value == NORMAL and SLOW in core.speed_history:
            raise RuntimeError("serial command failed")
        original(device, name, value)

    monkeypatch.setattr(core, "set_property", flaky)
    result = run_scan(core, monkeypatch, sf._AbortRequested("user cancelled"))
    # Still reports the cancel rather than turning it into an error.
    assert result.status == "aborted"


def test_no_speed_property_is_not_an_error(monkeypatch):
    # Stages with no writable speed property route to Brent; the scan refuses
    # early and must not try to restore anything.
    core = FakeCore()

    def fake_loop(*args, **kwargs):
        raise AssertionError("should not have reached the scan loop")

    monkeypatch.setattr(sf, "_run_streaming_scan", fake_loop)
    result = sf._attempt_one_scan(
        core,
        DEVICE,
        None,
        z_center=-346.5,
        range_um=693.0,
        sequence_was_running_on_entry=True,
        slow_value=SLOW,
        normal_value=NORMAL,
        z_endpoints=(0.0, -693.0),
    )
    assert result.status == "no_slow_speed"


def test_the_cancel_is_logged_before_the_retract_moves_the_stage():
    """An operator watching the log must see the cancel land, then the retract.

    Going quiet from the abort until the stage stops is what made a working
    cancel look ignored.
    """
    source = (
        ROOT / "microscope_command_server" / "server" / "handlers" / "streaming_focus.py"
    ).read_text()
    marker = 'if scan.status == "aborted":'
    assert source.count(marker) == 1
    after = source.split(marker, 1)[1][:600]
    log_at = after.index("cancelled; retracting")
    retract_at = after.index('_retract(core, focus_device, safe_z, "aborted")')
    assert log_at < retract_at
