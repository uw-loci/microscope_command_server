"""The demo microscope: a DemoCore behind the production hardware classes.

These tests build the real ``PycromanagerHardware`` on top of ``DemoCore`` with
a config shaped like the bundled ``config_Demo.yml`` and exercise the paths the
server handlers use: snap, live frames, stage moves with travel time, device
properties, and a focus sweep that has to peak where the virtual slide says
focus is.
"""

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
