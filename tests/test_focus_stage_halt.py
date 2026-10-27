"""Halting a focus move on cancel (``_halt_focus_stage``).

A streaming-autofocus cancel (ABORTAF) must stop the stage where it is. ``core.stop``
on the focus device is the right call, but the Micro-Manager Prior adapter's ``ZStage``
has no ``Stop`` (its ``XYStage`` sends the controller-wide ``K`` halt), so on PPM the
call raised, was swallowed, and the slow approach scan ran to its far end before the
retract began: 2026-10-26 18:20 and 18:46, ABORTAF observed within 11 ms, the retract
done 10 s and 19 s later. The helper falls back to the XY stage's Stop when both stages
are the same adapter on the same serial port, waits for the focus stage to go idle, and
re-reads the position so a relative-move adapter does not carry the abandoned target.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stub_microscope_control_if_absent():
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
    spec = importlib.util.spec_from_file_location("_sf_focus_halt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sf = _load_streaming_focus()


class FakeCore:
    """Two Prior-like stages on one port; ZStage without Stop, XYStage with the halt."""

    def __init__(self, z_has_stop=False, shared_port=True, same_library=True):
        self.z_has_stop = z_has_stop
        self.busy = True
        self.calls = []
        self.libs = {"ZStage": "Prior", "XYStage": "Prior" if same_library else "Other"}
        self.ports = {"ZStage": "COM1", "XYStage": "COM1" if shared_port else "COM2"}

    def stop(self, label):
        self.calls.append(("stop", label))
        if label == "ZStage" and not self.z_has_stop:
            raise RuntimeError("Unsupported command")
        self.busy = False

    def get_xy_stage_device(self):
        return "XYStage"

    def get_device_library(self, label):
        return self.libs[label]

    def has_property(self, label, prop):
        return prop == "Port"

    def get_property(self, label, prop):
        return self.ports[label]

    def device_busy(self, label):
        self.calls.append(("busy", label))
        return self.busy

    def get_position(self, label):
        self.calls.append(("get_position", label))
        return -590.0


def test_prior_z_without_stop_is_halted_through_the_xy_stage():
    core = FakeCore()
    how = sf._halt_focus_stage(core, "ZStage")
    assert ("stop", "ZStage") in core.calls
    assert ("stop", "XYStage") in core.calls
    assert "XYStage" in how
    assert core.busy is False
    # The adapter's cached target is refreshed from the controller after the halt.
    assert ("get_position", "ZStage") in core.calls


def test_a_stage_with_its_own_stop_never_touches_the_xy_stage():
    core = FakeCore(z_has_stop=True)
    how = sf._halt_focus_stage(core, "ZStage")
    assert how == "focus device Stop"
    assert ("stop", "XYStage") not in core.calls
    assert ("get_position", "ZStage") in core.calls


def test_xy_stage_on_another_port_is_not_used(monkeypatch):
    monkeypatch.setattr(sf, "HALT_SETTLE_TIMEOUT_S", 0.05)
    core = FakeCore(shared_port=False)
    how = sf._halt_focus_stage(core, "ZStage")
    assert ("stop", "XYStage") not in core.calls
    assert "no halt available" in how
    # Still re-reads the position so the caller logs where the stage is.
    assert ("get_position", "ZStage") in core.calls


def test_xy_stage_from_another_adapter_is_not_used(monkeypatch):
    monkeypatch.setattr(sf, "HALT_SETTLE_TIMEOUT_S", 0.05)
    core = FakeCore(same_library=False)
    sf._halt_focus_stage(core, "ZStage")
    assert ("stop", "XYStage") not in core.calls


def test_shares_controller_rule():
    assert sf._shares_controller(FakeCore(), "ZStage", "XYStage") is True
    assert sf._shares_controller(FakeCore(shared_port=False), "ZStage", "XYStage") is False
    assert sf._shares_controller(FakeCore(same_library=False), "ZStage", "XYStage") is False
