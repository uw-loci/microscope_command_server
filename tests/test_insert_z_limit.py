"""The per-insert sample-side Z limit a STRMAFZ request can carry (``--z-limit``).

``stage.limits.z_um`` is the hardware's travel envelope; on PPM it is [-720, 1000] against a
focus range a few hundred micrometres wide, so it stopped nothing on 2026-10-26, when an
approach scan drove from safe Z 0 toward -693 through a focus plane at -109 and about
450 um past it before the operator cancelled. The insert's ``z_limit_um`` is the bound on
the sample side. The server does not know which insert is fitted, so QPSC sends it with
every scan and the handler narrows the stage limits for that request only.

These tests pin the narrowing rule, which is pure. The handler needs a live MMCore.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _stub_microscope_control_if_absent():
    """Make ``microscope_control`` importable when it is not installed (see the speed-restore test)."""
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
    spec = importlib.util.spec_from_file_location("_sf_insert_z_limit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sf = _load_streaming_focus()
narrow = sf._narrow_z_limits_to_insert


def test_ppm_2026_10_26_the_limit_becomes_the_floor():
    # Increasing Z retracts on PPM, so the sample is toward lower Z and the limit is a floor.
    z_low, z_high, side = narrow(-720.0, 1000.0, -300.0, +1.0, current_z=-121.0)
    assert (z_low, z_high, side) == (-300.0, 1000.0, "low")


def test_a_rig_that_retracts_in_negative_z_gets_a_ceiling():
    z_low, z_high, side = narrow(0.0, 2800.0, 900.0, -1.0, current_z=500.0)
    assert (z_low, z_high, side) == (0.0, 900.0, "high")


def test_it_only_ever_narrows():
    # A limit looser than the stage's own limit changes nothing.
    assert narrow(-720.0, 1000.0, -800.0, +1.0, current_z=-121.0) == (-720.0, 1000.0, "low")
    assert narrow(0.0, 2800.0, 3000.0, -1.0, current_z=500.0) == (0.0, 2800.0, "high")


def test_without_a_retract_sign_the_side_is_read_off_the_current_z():
    # The limit is by definition on the sample side of where the stage is now.
    assert narrow(-720.0, 1000.0, -300.0, None, current_z=-121.0) == (-300.0, 1000.0, "low")
    assert narrow(-720.0, 1000.0, 400.0, None, current_z=-121.0) == (-720.0, 400.0, "high")


def test_a_limit_at_the_current_z_with_no_sign_is_not_applied():
    assert narrow(-720.0, 1000.0, -121.0, None, current_z=-121.0) == (-720.0, 1000.0, None)


def test_missing_stage_limits_are_filled_rather_than_skipped():
    assert narrow(None, None, -300.0, +1.0, current_z=-121.0) == (-300.0, None, "low")
    assert narrow(None, None, 900.0, -1.0, current_z=500.0) == (None, 900.0, "high")


def test_no_limit_means_no_change():
    assert narrow(-720.0, 1000.0, None, +1.0, current_z=-121.0) == (-720.0, 1000.0, None)


@pytest.mark.parametrize("z", [-299.0, -300.0, -121.0])
def test_the_limit_itself_stays_reachable(z):
    z_low, _z_high, _side = narrow(-720.0, 1000.0, -300.0, +1.0, current_z=z)
    assert z >= z_low


def test_the_limit_is_parsed_and_ends_the_value_before_it():
    # The request from the 2026-11-13 Focus Approach Validation. With --z-limit missing from
    # the declared flags, --z-end read "-55.79999923706055 --z-limit -200.0", the server
    # dropped both endpoints and scanned 72 -> 328 around the safe Z, away from the sample,
    # and the limit itself was never applied.
    from microscope_command_server.server.handlers.utils import parse_flags

    message = (
        "--yaml D:\\cfg\\config_PPM.yml --modality ppm --range 255.79999923706055 --dump 1 "
        "--dump-label tissue --max-attempts 1 --z-start 200.0 --z-end -55.79999923706055 "
        "--z-limit -200.0"
    )
    params = parse_flags(message, sf.STRMAFZ_FLAGS)
    assert float(params["z_start"]) == 200.0
    assert float(params["z_end"]) == pytest.approx(-55.8)
    assert float(params["z_limit"]) == -200.0


def test_every_value_the_handler_reads_is_a_declared_flag():
    import inspect
    import re

    source = inspect.getsource(sf._handle_streaming_focus)
    read = set(re.findall(r'params\.get\(\s*"(\w+)"', source))
    declared = {f.lstrip("-").replace("-", "_") for f in sf.STRMAFZ_FLAGS}
    assert read <= declared, f"read but not declared: {sorted(read - declared)}"
