"""Tests for the preserve-Z-planes output path helper ``_plane_path``.

When an acquisition is run with ``--z-projection none`` the server keeps every
individual Z-plane (and, for a time series, every timepoint) so the stitcher
can assemble a 5D mosaic. ``_plane_path`` decides where each plane is written:
``{output}/[{group}/][t{tt}/]z{zz}/{filename}``.

The legacy ``--save-raw`` behavior (preserve_z_planes False) must keep the
old ``{group}/z{zz}/`` layout with no ``t{tt}`` segment, regardless of the
timepoint count -- this is the backward-compat guard.

workflow.py imports microscope_control / microscope_imageprocessing, which are
not installed in the WSL dev environment, so those packages are stubbed before
the module is loaded (same approach as test_time_lapse_warning.py).

ASCII-only per project policy.
"""

import importlib.util
import sys
import types
from pathlib import Path


def _install_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for attr, value in (attrs or {}).items():
        if not hasattr(mod, attr):
            setattr(mod, attr, value)
    return mod


def _load_workflow():
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
    spec = importlib.util.spec_from_file_location("mcs_workflow_planepath", workflow_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcs_workflow_planepath"] = module
    spec.loader.exec_module(module)
    return module


workflow_mod = _load_workflow()
_plane_path = workflow_mod._plane_path


class FakeCtx:
    """Holds only the four fields ``_plane_path`` reads."""

    def __init__(self, output_path, preserve_z_planes, n_timepoints, current_timepoint):
        self.output_path = Path(output_path)
        self.preserve_z_planes = preserve_z_planes
        self.n_timepoints = n_timepoints
        self.current_timepoint = current_timepoint


OUT = "/tmp/acq/Region"


def test_preserve_single_timepoint_has_no_t_dir():
    ctx = FakeCtx(OUT, preserve_z_planes=True, n_timepoints=1, current_timepoint=0)
    p = _plane_path(ctx, "0.0", 2, "tile_001.tif")
    assert p == Path(OUT) / "0.0" / "z002" / "tile_001.tif"


def test_preserve_time_series_has_t_dir():
    ctx = FakeCtx(OUT, preserve_z_planes=True, n_timepoints=3, current_timepoint=1)
    p = _plane_path(ctx, "DAPI", 0, "tile_005.tif")
    assert p == Path(OUT) / "DAPI" / "t001" / "z000" / "tile_005.tif"


def test_preserve_single_fov_no_group():
    ctx = FakeCtx(OUT, preserve_z_planes=True, n_timepoints=1, current_timepoint=0)
    p = _plane_path(ctx, None, 5, "tile_000.tif")
    assert p == Path(OUT) / "z005" / "tile_000.tif"


def test_save_raw_only_keeps_legacy_layout_even_with_timepoints():
    # preserve_z_planes False == the legacy --save-raw path: never add a t{tt}
    # segment, so existing forensic output is unchanged.
    ctx = FakeCtx(OUT, preserve_z_planes=False, n_timepoints=5, current_timepoint=4)
    p = _plane_path(ctx, "90.0", 3, "tile_002.tif")
    assert p == Path(OUT) / "90.0" / "z003" / "tile_002.tif"


def test_zero_padding_widths():
    ctx = FakeCtx(OUT, preserve_z_planes=True, n_timepoints=12, current_timepoint=10)
    p = _plane_path(ctx, "ch0", 7, "t.tif")
    # t and z both zero-padded to 3 digits.
    assert "t010" in p.parts
    assert "z007" in p.parts
