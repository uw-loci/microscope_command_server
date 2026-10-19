"""The demo microscope: a DemoCore behind the production hardware classes.

These tests build the real ``PycromanagerHardware`` on top of ``DemoCore`` with
a config shaped like the bundled ``config_Demo.yml`` and exercise the paths the
server handlers use: snap, live frames, stage moves with travel time, device
properties, and a focus sweep that has to peak where the virtual slide says
focus is.
"""

import logging
import time

import numpy as np
import pytest

# Imported at module level on purpose: several other test modules register
# stand-in modules for microscope_control.hardware.pycromanager when it is not
# already loaded, and they never remove them. Loading the real one during
# collection (this file sorts before them) keeps those stand-ins out of our way.
# Skipped cleanly where the hardware stack is not importable at all.
pycromanager_hw = pytest.importorskip("microscope_control.hardware.pycromanager")
PycromanagerHardware = pycromanager_hw.PycromanagerHardware

from microscope_command_server.demo.virtual_microscope import (  # noqa: E402
    DemoCore,
    DemoStudio,
    StrVector,
    VirtualSlide,
)


def demo_settings():
    """The hardware-facing shape of config_Demo.yml (the Java extension bundles the YAML)."""
    return {
        "microscope": {"name": "Demo", "type": "Brightfield"},
        "modalities": {
            "Brightfield": {
                "type": "brightfield",
                "illumination": {
                    "device": "DLamp",
                    "type": "device_property",
                    "state_property": "State",
                    "intensity_property": "Intensity",
                    "max_intensity": 100.0,
                    "label": "Demo lamp",
                },
            }
        },
        "hardware": {
            "objectives": [{"id": "DEMO_OBJECTIVE_10X", "pixel_size_xy_um": {"DEMO_CAMERA": 0.65}}],
            "detectors": ["DEMO_CAMERA"],
        },
        "id_detector": {
            "DEMO_CAMERA": {
                "name": "Demo Camera",
                "camera_type": "generic",
                "device": "DCam",
                "width_px": 512,
                "height_px": 512,
                "flip_x": False,
                "flip_y": False,
            }
        },
        "id_stage": {},
        "ppm_optics": "NA",
        "stage": {
            "stage_id": "DEMO_STAGE",
            "z_stage": "DStage",
            "limits": {
                "x_um": {"low": -25000.0, "high": 25000.0},
                "y_um": {"low": -15000.0, "high": 15000.0},
                "z_um": {"low": -500.0, "high": 500.0},
            },
        },
        "acquisition_profiles": {
            "Brightfield_10x": {
                "modality": "Brightfield",
                "detector": "DEMO_CAMERA",
                "illumination_intensity": 25,
            }
        },
    }


@pytest.fixture
def core():
    return DemoCore(xy_speed_um_per_s=200000.0, z_speed_um_per_s=50000.0)


@pytest.fixture
def hardware(core):
    return PycromanagerHardware(core, DemoStudio(core), demo_settings())


def sharpness(img: np.ndarray) -> float:
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
    return float(np.var(np.diff(gray, axis=1)) + np.var(np.diff(gray, axis=0)))


class TestCoreSurface:
    def test_loaded_devices_answer_both_java_and_python_iteration(self, core):
        devices = core.get_loaded_devices()
        assert isinstance(devices, StrVector)
        assert "Core" in list(devices)
        assert devices.get(0) == list(devices)[0]
        assert devices.size() == len(devices)

    def test_core_device_names_match_the_micro_manager_demo_adapter(self, core):
        assert core.get_camera_device() == "DCam"
        assert core.get_focus_device() == "DStage"
        assert core.get_xy_stage_device() == "XYStage"
        assert core.get_property("Core", "Camera") == "DCam"

    def test_unknown_device_and_property_raise_like_mmcore(self, core):
        with pytest.raises(RuntimeError):
            core.get_property("NoSuchDevice", "Exposure")
        with pytest.raises(RuntimeError):
            core.get_property("DCam", "NoSuchProperty")
        with pytest.raises(RuntimeError):
            core.set_property("DCam", "Binning", "3")
        with pytest.raises(RuntimeError):
            core.set_property("DCam", "OnCameraCCDXSize", "1")

    def test_property_limits_and_allowed_values(self, core):
        assert core.has_property_limits("DCam", "Exposure")
        assert core.get_property_lower_limit("DCam", "Exposure") < 1.0
        assert list(core.get_allowed_property_values("DCam", "Binning")) == ["1", "2", "4"]
        assert not core.is_property_read_only("DCam", "Exposure")
        assert core.is_property_read_only("DCam", "CameraName")

    def test_exposure_is_one_value_whichever_way_it_is_set(self, core):
        core.set_exposure(42.0)
        assert core.get_exposure() == pytest.approx(42.0)
        assert float(core.get_property("DCam", "Exposure")) == pytest.approx(42.0)
        core.set_property("DCam", "Exposure", "7")
        assert core.get_exposure() == pytest.approx(7.0)

    def test_binning_changes_image_geometry_and_pixel_size(self, core):
        assert (core.get_image_width(), core.get_image_height()) == (512, 512)
        core.set_property("DCam", "Binning", "2")
        assert (core.get_image_width(), core.get_image_height()) == (256, 256)
        assert core.get_pixel_size_um() == pytest.approx(1.3)
        assert core.get_image_buffer_size() == 256 * 256 * 4

    def test_roi_is_a_four_tuple_and_shrinks_the_frame(self, core):
        assert core.get_roi() == (0, 0, 512, 512)
        core.set_roi(100, 50, 200, 100)
        assert core.get_roi() == (100, 50, 200, 100)
        assert (core.get_image_width(), core.get_image_height()) == (200, 100)
        core.clear_roi()
        assert core.get_image_width() == 512
        with pytest.raises(RuntimeError):
            core.set_roi(0, 0, 10000, 10)

    def test_config_group_applies_its_settings(self, core):
        core.set_property("DLamp", "State", "0")
        core.set_config("Channel", "Brightfield")
        assert core.get_property("DLamp", "State") == "1"
        with pytest.raises(RuntimeError):
            core.set_config("Channel", "NoSuchPreset")


class TestStages:
    def test_stage_travel_takes_time_and_reports_busy(self):
        core = DemoCore(xy_speed_um_per_s=1000.0)
        core.set_xy_position(500.0, 0.0)
        assert core.device_busy("XYStage")
        mid = core.get_x_position()
        assert 0.0 <= mid < 500.0
        core.wait_for_device("XYStage")
        assert not core.device_busy("XYStage")
        assert core.get_x_position() == pytest.approx(500.0)

    def test_focus_moves_through_both_apis(self, core):
        core.set_position("DStage", 12.5)
        core.wait_for_device("DStage")
        assert core.get_position("DStage") == pytest.approx(12.5)
        core.set_property("DStage", "Position", "-3")
        core.wait_for_device("DStage")
        assert float(core.get_property("DStage", "Position")) == pytest.approx(-3.0)

    def test_stop_freezes_the_axis_where_it_is(self):
        core = DemoCore(xy_speed_um_per_s=1000.0)
        core.set_xy_position(1000.0, 0.0)
        time.sleep(0.05)
        core.stop("XYStage")
        x = core.get_x_position()
        assert 0.0 < x < 1000.0
        time.sleep(0.05)
        assert core.get_x_position() == pytest.approx(x)

    def test_hardware_moves_through_the_production_stage_class(self, hardware):
        from microscope_control.hardware.base import Position

        hardware.move_to_position(Position(x=1200.0, y=-300.0, z=5.0))
        pos = hardware.get_current_position()
        assert (pos.x, pos.y, pos.z) == pytest.approx((1200.0, -300.0, 5.0))


class TestCamera:
    def test_snap_through_the_production_camera_is_rgb(self, hardware):
        hardware.set_exposure(5.0)
        img, tags = hardware.snap_image()
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8
        assert tags["Width"] == 512 and tags["Height"] == 512

    def test_glass_is_bright_and_tissue_is_darker(self, hardware, core):
        from microscope_control.hardware.base import Position

        slide = core.slide
        # Find one empty and one tissue field by sampling the slide.
        empty = tissue = None
        for x in range(-18000, 18001, 1500):
            for y in range(-8000, 8001, 1500):
                frac = slide.tissue_fraction(x, y, 333.0)
                if frac < 0.02 and empty is None:
                    empty = (x, y)
                if frac > 0.9 and tissue is None:
                    tissue = (x, y)
        assert (
            empty is not None and tissue is not None
        ), "slide has no bare glass or no solid tissue field"

        hardware.move_to_position(Position(x=empty[0], y=empty[1], z=slide.focus_z(*empty)))
        bright, _ = hardware.snap_image()
        hardware.move_to_position(Position(x=tissue[0], y=tissue[1], z=slide.focus_z(*tissue)))
        dark, _ = hardware.snap_image()
        assert bright.mean() > 225
        assert dark.mean() < bright.mean() - 30

    def test_focus_sweep_peaks_at_the_slides_focal_surface(self, hardware, core):
        from microscope_control.hardware.base import Position

        slide = core.slide
        tissue = None
        for x in range(-18000, 18001, 1500):
            for y in range(-8000, 8001, 1500):
                if slide.tissue_fraction(x, y, 333.0) > 0.9:
                    tissue = (x, y)
                    break
            if tissue:
                break
        assert tissue is not None
        z_true = slide.focus_z(*tissue)
        hardware.move_to_position(Position(x=tissue[0], y=tissue[1], z=z_true))

        zs = np.linspace(z_true - 40.0, z_true + 40.0, 17)
        scores = []
        for z in zs:
            hardware.move_to_position(Position(z=float(z)))
            img, _ = hardware.snap_image()
            scores.append(sharpness(img))
        best = zs[int(np.argmax(scores))]
        assert abs(best - z_true) <= 5.5, f"sweep peaked at {best:.1f}, focus is {z_true:.1f}"
        assert scores[0] < 0.25 * max(scores) and scores[-1] < 0.25 * max(scores)

    def test_lamp_off_gives_a_dark_frame(self, hardware, core):
        core.set_property("DLamp", "State", "0")
        img, _ = hardware.snap_image()
        assert img.mean() < 8
        core.set_property("DLamp", "State", "1")

    def test_live_frames_flow_through_the_circular_buffer(self, hardware, core):
        core.set_exposure(5.0)
        hardware.start_continuous_acquisition()
        assert core.is_sequence_running()
        deadline = time.time() + 2.0
        frame = None
        while time.time() < deadline and frame is None:
            frame, _ = hardware.get_live_frame()
            time.sleep(0.01)
        assert frame is not None and frame.shape == (512, 512, 3)
        hardware.stop_continuous_acquisition()
        assert not core.is_sequence_running()

    def test_pop_consumes_and_an_empty_buffer_raises(self, core):
        core.set_exposure(1.0)
        core.start_continuous_sequence_acquisition(0)
        time.sleep(0.03)
        assert core.get_remaining_image_count() >= 1
        pix = core.pop_next_image()
        assert pix.shape == (512 * 512 * 4,)
        core.clear_circular_buffer()
        assert core.get_remaining_image_count() == 0
        with pytest.raises(RuntimeError):
            core.pop_next_image()
        core.stop_sequence_acquisition()

    def test_snap_refuses_while_streaming(self, core):
        core.start_continuous_sequence_acquisition(0)
        with pytest.raises(RuntimeError):
            core.snap_image()
        core.stop_sequence_acquisition()

    def test_eight_bit_pixel_type_is_single_component(self, hardware, core):
        core.set_property("DCam", "PixelType", "8bit")
        assert core.get_number_of_components() == 1
        img, _ = hardware.snap_image()
        assert img.shape == (512, 512)


class TestStudio:
    def test_live_mode_is_the_sequence(self, core):
        studio = DemoStudio(core)
        studio.live().set_live_mode(True)
        assert core.is_sequence_running() and studio.live().is_live_mode_on()
        studio.live().set_live_mode(False)
        assert not core.is_sequence_running()


class TestVirtualSlide:
    def test_render_is_deterministic_for_a_position(self):
        a = VirtualSlide().render(
            100.0, 200.0, 0.0, 64, 48, 0.65, 10.0, 25.0, True, np.random.default_rng(0)
        )
        b = VirtualSlide().render(
            100.0, 200.0, 0.0, 64, 48, 0.65, 10.0, 25.0, True, np.random.default_rng(0)
        )
        assert a.shape == (48, 64, 3)
        np.testing.assert_array_equal(a, b)

    def test_focal_surface_varies_across_the_slide(self):
        slide = VirtualSlide()
        zs = [slide.focus_z(x, 0.0) for x in (-15000.0, 0.0, 15000.0)]
        assert max(zs) - min(zs) > 5.0
        assert all(abs(z) < 60.0 for z in zs)

    def test_beyond_the_slide_is_glass(self):
        slide = VirtualSlide()
        assert slide.tissue_fraction(30000.0, 0.0, 333.0) == 0.0


class TestAutofocusValidationCancel:
    """The Validate AF test (TESTAFV) stops on the client's Cancel and leaves Z where it was.

    The acquisition wizard's Cancel sends ABORTAF; the handler turns it into should_abort.
    Run on the demo hardware so the real sweep and the real full autofocus are what stop.
    """

    @staticmethod
    def _run(hardware, tmp_path, should_abort):
        from microscope_control.autofocus.test import test_autofocus_validation

        return test_autofocus_validation(
            hardware=hardware,
            config_manager=None,
            yaml_file_path=str(tmp_path / "config_Demo.yml"),  # no sidecar: built-in defaults
            objective="DEMO_OBJECTIVE_10X",
            should_abort=should_abort,
        )

    @staticmethod
    def _focus_at(core, z):
        core.set_position(z)
        core.wait_for_device(core.get_focus_device())

    @staticmethod
    def _z_after(core):
        core.wait_for_device(core.get_focus_device())
        return core.get_position()

    @staticmethod
    def _abort_after(n):
        calls = {"n": 0}

        def should_abort():
            calls["n"] += 1
            return calls["n"] > n

        return should_abort, calls

    def test_uninterrupted_validation_completes(self, hardware, core, tmp_path):
        self._focus_at(core, 4.0)
        result = self._run(hardware, tmp_path, lambda: False)
        assert result["success"] is True
        assert result["cancelled"] is False
        assert self._z_after(core) == pytest.approx(4.0, abs=0.01)
        assert float(result["ground_truth_z"]) == pytest.approx(4.0, abs=0.01)

    def test_cancel_during_the_sweep_returns_z_and_reports_a_cancel(self, hardware, core, tmp_path):
        self._focus_at(core, 4.0)
        should_abort, _ = self._abort_after(2)
        result = self._run(hardware, tmp_path, should_abort)
        assert result["cancelled"] is True
        assert result["success"] is False
        assert result["recovery_z"] == "N/A", "phase 2 never started"
        assert self._z_after(core) == pytest.approx(4.0, abs=0.01)
        assert float(result["ground_truth_z"]) == pytest.approx(4.0, abs=0.01)

    def test_cancel_during_the_full_autofocus_returns_z(self, hardware, core, tmp_path):
        self._focus_at(core, 4.0)
        # The sweep polls once per step (6 by default) and once more before phase 2,
        # so a cancel after 9 polls lands inside the full autofocus.
        should_abort, calls = self._abort_after(9)
        result = self._run(hardware, tmp_path, should_abort)
        assert result["cancelled"] is True
        assert result["sweep_z"] != "N/A", "phase 1 finished"
        assert result["recovery_z"] == "N/A", "phase 2 did not finish"
        assert calls["n"] > 9
        assert self._z_after(core) == pytest.approx(4.0, abs=0.01)
        assert float(result["ground_truth_z"]) == pytest.approx(4.0, abs=0.01)


class TestFarFromFocus:
    """Autofocus that starts far from focus, the slide-4 trap of 2026-10-04.

    The demo's focal surface at the origin is +3.96 um; starting 71 um above it puts the
    true plane outside a 100 um window. See
    claude-reports/2026-10-06_slide4-autofocus-stale-seed-z.md.
    """

    START = 75.0
    TRUTH = 3.96

    @staticmethod
    def _at(core, z):
        core.set_position(z)
        core.wait_for_device(core.get_focus_device())

    def _af(self, hardware, **kw):
        from microscope_control.autofocus.core import AutofocusUtils

        # Brenner, the PPM metric; far from focus a Laplacian trend can point the wrong way.
        kw.setdefault("score_metric", AutofocusUtils.autofocus_profile_brenner_gradient)
        return hardware.autofocus(n_steps=21, search_range=100.0, move_stage_to_estimate=True, **kw)

    def test_without_edge_retries_the_window_cannot_reach_focus(self, hardware, core):
        self._at(core, self.START)
        result = self._af(hardware, edge_retries=0)
        assert isinstance(result, dict) and result["success"] is False

    def test_edge_retries_follow_the_trend_to_focus(self, hardware, core):
        # The per-tile standard AF did not pass edge_retries before 2026-10-17.
        self._at(core, self.START)
        result = self._af(hardware, edge_retries=2)
        assert isinstance(result, float)
        assert result == pytest.approx(self.TRUTH, abs=3.0)

    def test_a_flat_primary_metric_follows_the_p98_p2_trend(self, hardware, core):
        # A constant primary score models slide 4's "score range too small": only the
        # p98_p2 values from the same frames know which way focus lies.
        self._at(core, self.START)
        result = self._af(hardware, edge_retries=2, score_metric=lambda img: 1.0)
        assert isinstance(result, float)
        assert result == pytest.approx(self.TRUTH, abs=3.0)


class TestSkipFallback:
    """A skipped autofocus falls back to the last accepted focus and says it was skipped."""

    class _FailingHardware:
        """Hardware whose autofocus always fails; records how often it was asked."""

        def __init__(self):
            self.calls = 0
            self.z = 0.0

        def get_current_position(self):
            from microscope_control.hardware.base import Position

            return Position(0.0, 0.0, self.z)

        def move_to_position(self, pos):
            self.z = pos.z

        def autofocus(self, **_kw):
            self.calls += 1
            return {
                "success": False,
                "message": "flat",
                "quality_score": 0.1,
                "peak_prominence": 0.0,
                "attempted_z": -285.8,
                "original_z": -285.8,
            }

    def test_max_retries_zero_still_makes_one_attempt(self, hardware, core):
        from microscope_command_server.acquisition.workflow import autofocus_with_manual_fallback
        from microscope_control.autofocus.core import AutofocusUtils

        core.set_position(4.0)
        core.wait_for_device(core.get_focus_device())
        z = autofocus_with_manual_fallback(
            hardware=hardware,
            logger=hardware_logger(),
            request_manual_focus=None,
            max_retries=0,
            n_steps=11,
            search_range=20.0,
            score_metric=AutofocusUtils.autofocus_profile_laplacian_variance,
        )
        assert isinstance(z, float)

    def test_a_skip_is_reported_and_uses_the_fallback(self):
        from microscope_command_server.acquisition.workflow import autofocus_with_manual_fallback

        hw = self._FailingHardware()
        outcome = {}
        z = autofocus_with_manual_fallback(
            hardware=hw,
            logger=hardware_logger(),
            request_manual_focus=lambda remaining: "skip",
            max_retries=3,
            fallback_z=-350.0,
            outcome=outcome,
        )
        assert z == -350.0 and hw.z == -350.0
        assert outcome == {"skipped": True}
        assert hw.calls == 1

    def test_the_fallback_prefers_the_last_accepted_focus(self):
        from types import SimpleNamespace

        from microscope_command_server.acquisition.workflow import _skip_fallback_z

        assert _skip_fallback_z(SimpleNamespace(last_accepted_af_z=None, hint_z=-285.8)) == -285.8
        assert _skip_fallback_z(SimpleNamespace(last_accepted_af_z=-352.1, hint_z=-285.8)) == -352.1


def hardware_logger():
    import logging

    return logging.getLogger("test_far_from_focus")


# --------------------------------------------------------------------------- #
# The monochrome camera: fluorescence and LC-PolScope
# --------------------------------------------------------------------------- #

#: The demo compensator's calibrated palette, as shipped in config_Demo.yml.
DEMO_PALETTE = [
    ("State0", 0.2475, 0.5137),
    ("State1", 0.2175, 0.5137),
    ("State2", 0.2475, 0.5437),
    ("State3", 0.2775, 0.5137),
    ("State4", 0.2475, 0.4837),
]

FLUORESCENCE_CHANNELS = [
    ("DAPI", "385nm", 100, 25),
    ("FITC", "475nm", 200, 50),
    ("TRITC", "550nm", 80, 30),
    ("Cy5", "621nm", 100, 25),
]


def _lc_channel(state, a, b):
    return {
        "id": state,
        "exposure_ms": 50,
        "device_properties": [
            {"device": "MeadowlarkLC", "property": "Retardance LC-A [in waves]", "value": a},
            {"device": "MeadowlarkLC", "property": "Retardance LC-B [in waves]", "value": b},
            {"device": "DLED", "property": "State", "value": 0},
            {"device": "DLamp", "property": "State", "value": 1},
            {"device": "DLamp", "property": "Intensity", "value": 100},
        ],
    }


def _fl_channel(cid, line, exposure, power):
    props = [
        {"device": "DLamp", "property": "State", "value": 0},
        {"device": "DLED", "property": "State", "value": 1},
    ]
    for other in ("385nm", "475nm", "550nm", "621nm"):
        props.append(
            {
                "device": "DLED",
                "property": f"Enable-{other}",
                "value": "On" if other == line else "Off",
            }
        )
    props.append({"device": "DLED", "property": f"Intensity-{line}", "value": power})
    return {
        "id": cid,
        "exposure_ms": exposure,
        "intensity_property": {"device": "DLED", "property": f"Intensity-{line}"},
        "device_properties": props,
    }


def multimodal_settings():
    """config_Demo.yml's three modalities on two cameras, in the server's shape."""
    s = demo_settings()
    s["modalities"]["Fluorescence"] = {
        "type": "widefield",
        "illumination": {
            "device": "DLED",
            "type": "device_property",
            "state_property": "State",
            "intensity_property": "State",
            "max_intensity": 1.0,
            "label": "Demo epi LED",
        },
        "channels": [_fl_channel(*c) for c in FLUORESCENCE_CHANNELS],
    }
    s["modalities"]["lcpolscope"] = {
        "type": "polarized",
        "reconstruction": {
            "wavelength_nm": 549,
            "swing_waves": 0.03,
            "scheme": "5-State",
            "state_order": [p[0] for p in DEMO_PALETTE],
            "method": "openpolscope",
            "black_level": 100,
        },
        "channels": [_lc_channel(*p) for p in DEMO_PALETTE],
    }
    s["hardware"]["objectives"][0]["pixel_size_xy_um"]["DEMO_MONO_CAMERA"] = 0.586
    s["hardware"]["detectors"].append("DEMO_MONO_CAMERA")
    s["id_detector"]["DEMO_MONO_CAMERA"] = {
        "name": "Demo Mono Camera",
        "camera_type": "generic",
        "device": "DMonoCam",
        "width_px": 640,
        "height_px": 512,
        "flip_x": False,
        "flip_y": False,
    }
    s["acquisition_profiles"]["Fluorescence_10x"] = {
        "modality": "Fluorescence",
        "detector": "DEMO_MONO_CAMERA",
        "channels": [c[0] for c in FLUORESCENCE_CHANNELS],
    }
    s["acquisition_profiles"]["lcpolscope_10x"] = {
        "modality": "lcpolscope",
        "detector": "DEMO_MONO_CAMERA",
        "objective": "DEMO_OBJECTIVE_10X",
        "channels": [p[0] for p in DEMO_PALETTE],
    }
    return s


@pytest.fixture
def mm_hardware(core):
    return PycromanagerHardware(core, DemoStudio(core), multimodal_settings())


def _goto(core, x, y):
    core.set_xy_position(x, y)
    core.set_position(core.slide.focus_z(x, y))
    core.wait_for_system()


def _apply(hardware, modality, channel_id):
    from microscope_command_server.acquisition.workflow import apply_channel_hardware_state

    channels = hardware.settings["modalities"][modality]["channels"]
    entry = next(c for c in channels if c["id"] == channel_id)
    apply_channel_hardware_state(hardware, entry, logging.getLogger("test"))
    hardware.set_exposure(entry["exposure_ms"])


def _snap(hardware):
    frame, _ = hardware.snap_image()
    return frame.astype(np.float64)


class TestMonoCamera:
    def test_the_profile_switches_the_camera(self, mm_hardware, core):
        mm_hardware.apply_mode_setup("Fluorescence_10x")
        assert core.get_camera_device() == "DMonoCam"
        frame, _ = mm_hardware.snap_image()
        assert frame.shape == (512, 640) and frame.dtype == np.uint16
        assert core.get_image_bit_depth() == 12
        assert core.get_bytes_per_pixel() == 2
        assert core.get_pixel_size_um() == pytest.approx(0.586), "MM reports the active camera's"
        mm_hardware.apply_mode_setup("Brightfield_10x")
        assert core.get_camera_device() == "DCam"
        assert mm_hardware.snap_image()[0].shape == (512, 512, 3)

    def test_each_camera_keeps_its_own_exposure_and_roi(self, core):
        core.set_exposure(12.0)
        core.set_roi(0, 0, 256, 256)
        core.set_property("Core", "Camera", "DMonoCam")
        assert core.get_exposure() == 50.0
        assert core.get_roi() == (0, 0, 640, 512)
        core.set_property("Core", "Camera", "DCam")
        assert core.get_exposure() == 12.0
        assert core.get_roi() == (0, 0, 256, 256)

    def test_the_camera_checks_find_nothing_wrong_by_default_and_auto_gain_when_on(
        self, mm_hardware, core
    ):
        from microscope_command_server.calibration.camera_checks import (
            camera_setting_problems,
            camera_snapshot,
        )

        core.set_property("Core", "Camera", "DMonoCam")
        summary, props = camera_snapshot(mm_hardware)
        assert summary["significant_bits"] == 12
        assert camera_setting_problems(props, 12) == ([], [])
        core.set_property("DMonoCam", "Gain Auto", "Continuous")
        _, props = camera_snapshot(mm_hardware)
        fatal, _ = camera_setting_problems(props, 12)
        assert fatal and "Gain Auto" in fatal[0]

    def test_dark_frame_sits_at_the_offset(self, core):
        core.set_property("Core", "Camera", "DMonoCam")
        core.set_property("DLamp", "State", "0")
        core.snap_image()
        frame = core.get_image().astype(np.float64)
        assert abs(frame.mean() - 100.0) < 1.0


class TestFluorescence:
    def test_dapi_lights_the_nuclei(self, mm_hardware, core):
        mm_hardware.apply_mode_setup("Fluorescence_10x")
        _goto(core, 0.0, 0.0)
        _apply(mm_hardware, "Fluorescence", "DAPI")
        frame = _snap(mm_hardware)
        nuclei = core.slide.specimen_maps(0.0, 0.0, 640, 512, 0.586)["fl385"] > 0.9
        assert nuclei.mean() > 0.01
        assert frame[nuclei].mean() > 2000 > frame[~nuclei].mean() + 1000

    def test_channels_show_different_structures_without_saturating(self, mm_hardware, core):
        mm_hardware.apply_mode_setup("Fluorescence_10x")
        _goto(core, 0.0, 0.0)
        frames = {}
        for cid, *_ in FLUORESCENCE_CHANNELS:
            _apply(mm_hardware, "Fluorescence", cid)
            frames[cid] = _snap(mm_hardware)
            assert (frames[cid] >= 4095).mean() < 0.001, cid
            assert np.percentile(frames[cid], 99.9) > 500, cid
        corr = np.corrcoef(frames["DAPI"].ravel(), frames["TRITC"].ravel())[0, 1]
        assert corr < 0.5, "nuclei and eosin are different structures"

    def test_the_colour_camera_does_not_see_the_leds(self, mm_hardware, core):
        mm_hardware.apply_mode_setup("Fluorescence_10x")
        _apply(mm_hardware, "Fluorescence", "DAPI")
        core.set_property("Core", "Camera", "DCam")
        frame = _snap(mm_hardware)
        assert (
            frame.mean() < 5.0
        ), "the lamp is off; DAPI excitation never reaches the colour camera"


class TestLcPolScope:
    def _states(self, hardware, core, x, y):
        _goto(core, x, y)
        out = []
        for state, *_ in DEMO_PALETTE:
            _apply(hardware, "lcpolscope", state)
            out.append(_snap(hardware))
        return np.stack(out)

    def test_the_shipped_palette_is_the_extinction_minimum(self, mm_hardware, core):
        mm_hardware.apply_mode_setup("lcpolscope_10x")
        _goto(core, 22000.0, 0.0)  # bare glass
        _apply(mm_hardware, "lcpolscope", "State0")

        def mean_at(a, b):
            core.set_property("MeadowlarkLC", core.LC_A, f"{a:.4f}")
            core.set_property("MeadowlarkLC", core.LC_B, f"{b:.4f}")
            return _snap(mm_hardware).mean()

        at = mean_at(0.2475, 0.5137)
        for da, db in ((0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)):
            assert mean_at(0.2475 + da, 0.5137 + db) > at

    def test_states_reconstruct_to_the_planted_birefringence(self, mm_hardware, core):
        polscope = pytest.importorskip("polscope_library")
        mm_hardware.apply_mode_setup("lcpolscope_10x")
        background = np.mean(
            [self._states(mm_hardware, core, 22000.0, 0.0) for _ in range(4)], axis=0
        )
        states = self._states(mm_hardware, core, 0.0, 0.0)
        truth = core.slide.specimen_maps(0.0, 0.0, 640, 512, 0.586)
        for method in ("qlipp", "openpolscope"):
            result = polscope.reconstruct(
                states,
                0.03,
                549.0,
                method=method,
                black_level=100,
                background_intensities=background,
            )
            error = result.retardance_nm - truth["retardance_nm"]
            assert abs(np.median(error)) < 0.1, method
            assert np.percentile(np.abs(error), 95) < 1.0, method
            fibres = truth["retardance_nm"] > 8
            d = np.degrees(result.orientation_rad - truth["orientation_rad"])
            d = (d + 90.0) % 180.0 - 90.0
            assert np.median(np.abs(d[fibres])) < 1.0, method

    def test_without_a_background_the_residual_shows_on_glass(self, mm_hardware, core):
        polscope = pytest.importorskip("polscope_library")
        mm_hardware.apply_mode_setup("lcpolscope_10x")
        states = self._states(mm_hardware, core, 22000.0, 0.0)
        result = polscope.reconstruct(states, 0.03, 549.0, method="openpolscope", black_level=100)
        # Retardance and orientation together, as the vector a seam compares.
        r, two_theta = result.retardance_nm, 2.0 * result.orientation_rad
        vec = np.stack([r * np.cos(two_theta), r * np.sin(two_theta)])
        left = np.median(vec[:, :, :100], axis=(1, 2))
        right = np.median(vec[:, :, -100:], axis=(1, 2))
        assert np.hypot(*(right - left)) > 1.0, "the field-fixed residual differs across the field"
        assert np.median(r[:, -100:]) - np.median(r[:, :100]) > 0.3

    def test_the_adapter_wavelength_scales_what_the_crystals_make(self, core):
        core.set_property("MeadowlarkLC", core.LC_A, "0.25")
        a549, _ = core.lc_delivered_waves()
        core.set_property("MeadowlarkLC", "Wavelength", "546")
        a546, _ = core.lc_delivered_waves()
        assert a546 < a549
        assert core.get_property("MeadowlarkLC", "Retardance LC-A [in nm]") == "136.50"
        with pytest.raises(RuntimeError, match="read-only"):
            core.set_property("MeadowlarkLC", "Retardance LC-A [in nm]", "100")

    def test_lcsignal_measures_all_five_states(self, mm_hardware, core):
        from microscope_command_server.calibration.lc_state_signal import measure_lc_states

        mm_hardware.apply_mode_setup("lcpolscope_10x")
        _goto(core, 22000.0, 0.0)
        report = measure_lc_states(mm_hardware, mm_hardware.settings, "lcpolscope_10x", 50.0)
        assert "error" not in report, report
        assert [s["id"] for s in report["states"]] == [p[0] for p in DEMO_PALETTE]
        assert report["full_scale"] == 4095.0
        assert not report["clipped"]
        assert 0.25 < report["brightest_fraction"] < 0.5
        assert report["camera_fatal"] == [] and report["camera_advisory"] == []
        assert report["suggested_exposure_ms"] > 50.0, "glass alone would take a longer exposure"

    def test_the_shipped_exposure_does_not_clip_tissue(self, mm_hardware, core):
        from microscope_command_server.calibration.lc_state_signal import measure_lc_states

        mm_hardware.apply_mode_setup("lcpolscope_10x")
        for x, y in ((0.0, 0.0), (-3000.0, 2500.0), (1500.0, -800.0)):
            _goto(core, x, y)
            report = measure_lc_states(mm_hardware, mm_hardware.settings, "lcpolscope_10x", 50.0)
            assert not report["clipped"], (x, y, report["brightest_fraction"])
            assert 0.5 < report["brightest_fraction"] < 0.9, (x, y)


class TestLcCalibrationOnTheDemo:
    def test_calibration_after_a_fluorescence_profile_finds_the_shipped_palette(
        self, mm_hardware, core, tmp_path
    ):
        """A profile switch leaves the lamp off (and, from brightfield, the colour camera on);
        the calibration applies State0's hardware state first, so it still sees the field."""
        pytest.importorskip("polscope_library")
        from microscope_command_server.calibration.lc_calibration_workflow import (
            run_lc_calibration,
        )

        mm_hardware.apply_mode_setup("Brightfield_10x")
        mm_hardware.apply_mode_setup("Fluorescence_10x")
        mm_hardware.apply_mode_setup("Brightfield_10x")
        assert core.get_camera_device() == "DCam"
        _goto(core, 22000.0, 0.0)
        settings = multimodal_settings()
        settings["modalities"]["lcpolscope"]["reconstruction"]["lc_settle_ms"] = 0
        result = run_lc_calibration(
            mm_hardware, settings, output_folder=str(tmp_path), exposure_ms=50.0
        )
        assert result["success"], result.get("error")
        assert core.get_camera_device() == "DMonoCam"
        assert 200 < result["extinction_ratio"] < 330
        lca, lcb = result["palette_waves"]["State0"]
        assert lca == pytest.approx(0.2475, abs=0.002)
        assert lcb == pytest.approx(0.5137, abs=0.002)


class TestChannelBackgroundsOnTheDemo:
    def test_lc_backgrounds_are_collected_without_a_wb_mode(self, mm_hardware, core, tmp_path):
        """The client sends no --wb-mode for a per-channel collection; it was refused
        before the channel branch ran, so no LC-PolScope or fluorescence background was
        ever collected through BGACQUIRE."""
        import yaml

        from microscope_command_server.acquisition.workflow import simple_background_collection
        from microscope_control.config import ConfigManager

        settings = multimodal_settings()
        config = tmp_path / "config_Demo.yml"
        config.write_text(yaml.safe_dump(settings))
        (tmp_path / "resources").mkdir()
        (tmp_path / "resources" / "resources_LOCI.yml").write_text(
            yaml.safe_dump({"id_detector": settings["id_detector"]})
        )
        mm_hardware.apply_mode_setup("Brightfield_10x")  # the colour camera is active
        _goto(core, 22000.0, 0.0)
        out = tmp_path / "bg"
        result = simple_background_collection(
            yaml_file_path=str(config),
            output_folder_path=str(out),
            modality="lcpolscope",
            angles_str="()",
            exposures_str="()",
            hardware=mm_hardware,
            config_manager=ConfigManager(),
            logger=logging.getLogger("test"),
            update_progress=lambda *a: None,
            profile="lcpolscope_10x",
            channels=[p[0] for p in DEMO_PALETTE],
        )
        assert core.get_camera_device() == "DMonoCam", "the profile's camera, not the last one"
        assert sorted(f.name for f in out.glob("*.tif")) == sorted(
            f"{p[0]}.tif" for p in DEMO_PALETTE
        )
        assert result is not None

    def test_a_background_taken_on_tissue_is_called_out(self, mm_hardware, core, tmp_path, caplog):
        import json

        import yaml

        from microscope_command_server.acquisition.workflow import simple_background_collection
        from microscope_control.config import ConfigManager

        settings = multimodal_settings()
        config = tmp_path / "config_Demo.yml"
        config.write_text(yaml.safe_dump(settings))
        (tmp_path / "resources").mkdir()
        (tmp_path / "resources" / "resources_LOCI.yml").write_text(
            yaml.safe_dump({"id_detector": settings["id_detector"]})
        )
        spreads = {}
        for where, (x, y) in {"glass": (22000.0, 0.0), "tissue": (0.0, 0.0)}.items():
            _goto(core, x, y)
            out = tmp_path / where
            with caplog.at_level(logging.WARNING):
                simple_background_collection(
                    yaml_file_path=str(config),
                    output_folder_path=str(out),
                    modality="lcpolscope",
                    angles_str="()",
                    exposures_str="()",
                    hardware=mm_hardware,
                    config_manager=ConfigManager(),
                    logger=logging.getLogger("test"),
                    update_progress=lambda *a: None,
                    profile="lcpolscope_10x",
                    channels=[p[0] for p in DEMO_PALETTE],
                )
            spreads[where] = json.loads((out / "background_provenance.json").read_text())[
                "swing_state_spread"
            ]
        assert spreads["glass"] < 0.02, spreads
        assert spreads["tissue"] > 0.10, spreads
        assert "probably taken on specimen" in caplog.text
