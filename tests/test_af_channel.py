"""Dedicated focus channel: autofocus runs on a named channel at its own exposure.

workflow.py imports microscope_control (pycromanager), which is not installed off-rig,
so the function under test is extracted from the source and executed against stubs --
the same approach as test_skip_initial_autofocus.py.
"""

import ast
import pathlib

import pytest

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "workflow.py"
)
SOURCE = WORKFLOW.read_text()


def load_function(name, namespace):
    """Exec just one top-level function from workflow.py against a stub namespace."""
    tree = ast.parse(SOURCE)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(WORKFLOW), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found in workflow.py")


class FakeLogger:
    def __init__(self):
        self.messages = []

    def _log(self, level, msg, *args):
        self.messages.append((level, msg % args if args else msg))

    def info(self, msg, *a):
        self._log("info", msg, *a)

    def warning(self, msg, *a):
        self._log("warning", msg, *a)

    def error(self, msg, *a):
        self._log("error", msg, *a)

    def debug(self, msg, *a):
        self._log("debug", msg, *a)

    def text(self):
        return "\n".join(m for _, m in self.messages)


class FakeHardware:
    def __init__(self, fail=False):
        self.exposures = []
        self.fail = fail

    def set_exposure(self, ms):
        if self.fail:
            raise RuntimeError("camera busy")
        self.exposures.append(ms)


class Ctx:
    def __init__(self, **kw):
        self.af_channel = kw.get("af_channel")
        self.af_channel_exposure = kw.get("af_channel_exposure")
        self.af_channel_intensity = kw.get("af_channel_intensity")
        self.hardware = kw.get("hardware") or FakeHardware()
        self.logger = kw.get("logger") or FakeLogger()
        self.params = {"scan_type": "IF"}
        self.ppm_settings = {}
        self.channel_preset_cache = {}


def build(plan_by_id=None, applied=None, fail_apply=False):
    """Namespace with stubbed resolve_channel_plan / apply_channel_hardware_state."""
    plan_by_id = plan_by_id or {}
    calls = {"resolve": [], "apply": applied if applied is not None else []}

    def resolve_channel_plan(
        ppm_settings, scan_type, channel_ids, channel_exposures, channel_intensity_overrides=None
    ):
        calls["resolve"].append(
            (list(channel_ids), list(channel_exposures), channel_intensity_overrides)
        )
        out = []
        for i, cid in enumerate(channel_ids):
            entry = plan_by_id.get(cid)
            if entry is None:
                continue
            entry = dict(entry)
            if i < len(channel_exposures) and channel_exposures[i] > 0:
                entry["exposure_ms"] = channel_exposures[i]
            out.append(entry)
        return out

    def apply_channel_hardware_state(hardware, entry, logger, preset_cache=None):
        if fail_apply:
            raise RuntimeError("cube jammed")
        calls["apply"].append(entry["id"])

    ns = {
        "resolve_channel_plan": resolve_channel_plan,
        "apply_channel_hardware_state": apply_channel_hardware_state,
    }
    return load_function("apply_af_channel_state", ns), calls


DAPI = {"id": "DAPI", "exposure_ms": 100.0}


def test_no_focus_channel_configured_is_a_no_op():
    fn, calls = build({"DAPI": DAPI})
    ctx = Ctx()
    assert fn(ctx) is False
    assert calls["apply"] == []
    assert ctx.hardware.exposures == []


def test_focus_channel_is_applied_with_its_override_exposure():
    fn, calls = build({"DAPI": DAPI})
    ctx = Ctx(af_channel="DAPI", af_channel_exposure=20.0)
    assert fn(ctx) is True
    assert calls["apply"] == ["DAPI"]
    # The low focus exposure is used, NOT the channel's 100 ms imaging exposure.
    assert ctx.hardware.exposures == [20.0]


def test_without_an_override_the_channel_library_exposure_is_used():
    fn, calls = build({"DAPI": DAPI})
    ctx = Ctx(af_channel="DAPI")
    assert fn(ctx) is True
    assert ctx.hardware.exposures == [100.0]
    # No exposure is forwarded to the resolver, so it falls back to the library value.
    assert calls["resolve"][0][1] == []


def test_intensity_override_is_forwarded_only_when_set():
    fn, calls = build({"DAPI": DAPI})
    fn(Ctx(af_channel="DAPI", af_channel_intensity=35.0))
    assert calls["resolve"][0][2] == {"DAPI": 35.0}

    fn2, calls2 = build({"DAPI": DAPI})
    fn2(Ctx(af_channel="DAPI"))
    assert calls2["resolve"][0][2] is None


def test_zero_intensity_is_still_forwarded():
    # 0 is a legal intensity; a truthiness test would silently drop it.
    fn, calls = build({"DAPI": DAPI})
    fn(Ctx(af_channel="DAPI", af_channel_intensity=0.0))
    assert calls["resolve"][0][2] == {"DAPI": 0.0}


def test_unknown_channel_warns_and_leaves_hardware_alone():
    fn, calls = build({"DAPI": DAPI})
    ctx = Ctx(af_channel="TRITC")
    assert fn(ctx) is False
    assert calls["apply"] == []
    assert ctx.hardware.exposures == []
    assert "not in this modality" in ctx.logger.text()


def test_a_hardware_failure_does_not_abort_the_acquisition():
    fn, _ = build({"DAPI": DAPI}, fail_apply=True)
    ctx = Ctx(af_channel="DAPI", af_channel_exposure=20.0)
    assert fn(ctx) is False
    assert "Could not apply focus channel" in ctx.logger.text()


def test_an_exposure_write_failure_is_also_contained():
    fn, _ = build({"DAPI": DAPI})
    ctx = Ctx(af_channel="DAPI", af_channel_exposure=20.0, hardware=FakeHardware(fail=True))
    assert fn(ctx) is False


@pytest.mark.parametrize(
    "flag", ["--af-channel", "--af-channel-exposure", "--af-channel-intensity"]
)
def test_the_flags_are_parsed(flag):
    assert f'parts[i] == "{flag}"' in SOURCE


def test_autofocus_applies_the_focus_channel_in_both_paths():
    # Per-tile AF and the pre-acquisition search must both apply it: the acquisition
    # loop leaves the last acquired channel on the light path, so a per-tile call is
    # what keeps focus frames on the chosen channel.
    call_lines = [ln for ln in SOURCE.splitlines() if ln.strip() == "apply_af_channel_state(ctx)"]
    assert len(call_lines) == 2, call_lines
