"""An in-memory microscope that answers the Pycro-Manager ``Core`` API.

:class:`DemoCore` is what the server hands to ``PycromanagerHardware`` when it is
started with ``--demo``. It implements the subset of MMCore that
``microscope_control`` and this server call -- device properties, an XY stage and
a focus stage with simulated travel time, a camera that snaps and streams, a
circular buffer, ROI, config groups -- and nothing more. Unknown devices and
properties raise, the way MMCore does, so a typo in a config fails here too.

The camera looks at a :class:`VirtualSlide`: a procedurally generated H&E-like
section with tissue islands, texture and nuclei, lying on a tilted, gently
undulating focal surface within about +/-22 um of Z = 0. Moving the stage pans
across it; moving Z defocuses it, so autofocus sweeps produce a real focus curve
with a single peak. The slide pattern is seeded, so the same stage position
always shows the same field, up to per-frame Gaussian read noise (sigma 2 gray
levels).

Device labels follow Micro-Manager's Demo adapter where it has an obvious
counterpart (``DCam``, ``DStage``); ``XYStage`` and ``DLamp`` are this core's own
labels. Running the demo config against a real Micro-Manager Demo configuration
has not been tried.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Java-shaped return values
# --------------------------------------------------------------------------- #


class StrVector(list):
    """A list that also answers ``size()`` / ``get(i)`` like MMCore's StrVector.

    Callers in this code base reach the Java bindings both ways -- ``list(vec)``
    and ``vec.get(i) for i in range(vec.size())`` -- so this satisfies both.
    """

    def size(self) -> int:
        return len(self)

    def get(self, i: int):
        return self[i]


class TaggedImage:
    """The ``.pix`` / ``.tags`` pair that ``core.get_tagged_image()`` returns."""

    def __init__(self, pix: np.ndarray, tags: Dict[str, object]):
        self.pix = pix
        self.tags = tags


# --------------------------------------------------------------------------- #
# The specimen
# --------------------------------------------------------------------------- #


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lattice_hash(ix: np.ndarray, iy: np.ndarray, seed: float) -> np.ndarray:
    """Deterministic pseudo-random value in [0, 1) at integer lattice points."""
    v = np.sin(ix * 12.9898 + iy * 78.233 + seed * 0.7311) * 43758.5453
    return v - np.floor(v)


def _value_noise(x: np.ndarray, y: np.ndarray, cell: float, seed: float) -> np.ndarray:
    """Bilinearly interpolated lattice noise with ``cell`` spacing, in [0, 1)."""
    gx = np.floor(x / cell)
    gy = np.floor(y / cell)
    fx = x / cell - gx
    fy = y / cell - gy
    # Smooth the interpolant so cell edges do not show.
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = _lattice_hash(gx, gy, seed)
    b = _lattice_hash(gx + 1.0, gy, seed)
    c = _lattice_hash(gx, gy + 1.0, seed)
    d = _lattice_hash(gx + 1.0, gy + 1.0, seed)
    top = a + (b - a) * ux
    bottom = c + (d - c) * ux
    return top + (bottom - top) * uy


def _fractal_noise(
    x: np.ndarray, y: np.ndarray, cell: float, seed: float, octaves: int
) -> np.ndarray:
    """Sum of ``octaves`` value-noise layers, each half the cell and half the weight."""
    total = np.zeros_like(x)
    weight = 1.0
    norm = 0.0
    for i in range(octaves):
        total += weight * _value_noise(x, y, cell, seed + 17.0 * i)
        norm += weight
        cell *= 0.5
        weight *= 0.5
    return total / norm


@dataclass
class VirtualSlide:
    """A procedural H&E-like section and the focal surface it lies on.

    The slide covers ``[-half_x, half_x] x [-half_y, half_y]`` micrometres with
    the stage origin at its centre; beyond that the camera sees bare glass.
    """

    seed: float = 7.0
    half_x_um: float = 20000.0
    half_y_um: float = 10000.0
    # Focal surface: a tilt of ``tilt_um_per_mm`` plus a slow wobble, so Z
    # differs across the slide the way a real section does.
    tilt_um_per_mm: float = 0.6
    wobble_um: float = 8.0
    # Defocus blur in pixels per micrometre away from the focal surface.
    blur_px_per_um: float = 0.35
    # Illumination that renders the glass as full white.
    reference_exposure_ms: float = 10.0
    reference_intensity: float = 25.0

    def focus_z(self, x_um: float, y_um: float) -> float:
        """Z of best focus at a stage position, in micrometres."""
        tilt = self.tilt_um_per_mm * (0.001 * x_um) + 0.5 * self.tilt_um_per_mm * (0.001 * y_um)
        wobble = self.wobble_um * (
            math.sin(x_um / 3700.0 + self.seed) * math.cos(y_um / 2900.0 - self.seed)
        )
        return tilt + wobble

    def tissue_fraction(self, x_um: float, y_um: float, fov_um: float) -> float:
        """Rough fraction of a field at (x, y) covered by tissue; used by tests."""
        n = 32
        xs = np.linspace(x_um - fov_um / 2, x_um + fov_um / 2, n)
        ys = np.linspace(y_um - fov_um / 2, y_um + fov_um / 2, n)
        gx, gy = np.meshgrid(xs, ys)
        return float(self._tissue_mask(gx, gy).mean())

    def _tissue_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        islands = _fractal_noise(x, y, 2500.0, self.seed, 3)
        mask = _smoothstep(0.47, 0.56, islands)
        on_slide = (np.abs(x) < self.half_x_um) & (np.abs(y) < self.half_y_um)
        return mask * on_slide

    def render(
        self,
        x_um: float,
        y_um: float,
        z_um: float,
        width_px: int,
        height_px: int,
        pixel_size_um: float,
        exposure_ms: float,
        intensity: float,
        lamp_on: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Render the field centred on (x, y) at focus z as an RGB uint8 image."""
        cols = (np.arange(width_px, dtype=np.float64) - width_px / 2.0 + 0.5) * pixel_size_um
        rows = (np.arange(height_px, dtype=np.float64) - height_px / 2.0 + 0.5) * pixel_size_um
        wx, wy = np.meshgrid(x_um + cols, y_um + rows)

        tissue = self._tissue_mask(wx, wy)
        stroma = _fractal_noise(wx, wy, 60.0, self.seed + 101.0, 3)
        nuclei_field = _fractal_noise(wx, wy, 9.0, self.seed + 202.0, 2)
        nuclei = _smoothstep(0.70, 0.76, nuclei_field) * _smoothstep(0.2, 0.5, tissue)

        glass = np.array([243.0, 241.0, 246.0])
        eosin = np.array([226.0, 150.0, 182.0])
        eosin_dark = np.array([196.0, 110.0, 150.0])
        hematoxylin = np.array([88.0, 52.0, 128.0])

        img = np.empty((height_px, width_px, 3), dtype=np.float64)
        pink = (
            eosin[None, None, :] * (1.0 - stroma[..., None])
            + eosin_dark[None, None, :] * stroma[..., None]
        )
        img[:] = glass[None, None, :] * (1.0 - tissue[..., None]) + pink * tissue[..., None]
        img = img * (1.0 - nuclei[..., None]) + hematoxylin[None, None, :] * nuclei[..., None]

        dz = abs(z_um - self.focus_z(x_um, y_um))
        sigma_px = min(dz * self.blur_px_per_um / pixel_size_um * 0.65, 30.0)
        if sigma_px > 0.3:
            from scipy.ndimage import gaussian_filter

            for c in range(3):
                img[..., c] = gaussian_filter(img[..., c], sigma_px, mode="nearest")

        if lamp_on:
            scale = (exposure_ms / self.reference_exposure_ms) * (
                intensity / self.reference_intensity
            )
        else:
            scale = 0.0
        img *= scale
        img += rng.normal(0.0, 2.0, img.shape)
        return np.clip(img, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #


@dataclass
class _Property:
    value: str
    allowed: Optional[List[str]] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    read_only: bool = False

    @property
    def has_limits(self) -> bool:
        return self.lower is not None and self.upper is not None


@dataclass
class _Device:
    library: str
    adapter: str
    description: str
    properties: Dict[str, _Property] = field(default_factory=dict)


@dataclass
class _Axis:
    """One motorised axis with a constant-velocity travel model."""

    position: float = 0.0
    target: float = 0.0
    speed_um_per_s: float = 1000.0
    started_at: float = 0.0
    arrives_at: float = 0.0

    def start(self, target: float, now: float) -> None:
        self.position = self.at(now)
        self.target = float(target)
        travel = abs(self.target - self.position)
        self.started_at = now
        self.arrives_at = now + max(travel / self.speed_um_per_s, 0.02 if travel > 0 else 0.0)

    def at(self, now: float) -> float:
        if now >= self.arrives_at:
            self.position = self.target
            return self.target
        frac = (now - self.started_at) / (self.arrives_at - self.started_at)
        return self.position + (self.target - self.position) * frac

    def busy(self, now: float) -> bool:
        return now < self.arrives_at

    def stop(self, now: float) -> None:
        self.position = self.at(now)
        self.target = self.position
        self.arrives_at = now


class DemoCore:
    """A Pycro-Manager ``Core`` look-alike driving a :class:`VirtualSlide`.

    Thread-safe: every public method takes one re-entrant lock, because the
    server's handler threads, position pollers and acquisition thread all call
    into the core concurrently.
    """

    CAMERA = "DCam"
    FOCUS = "DStage"
    XY = "XYStage"
    LAMP = "DLamp"

    def __init__(
        self,
        slide: Optional[VirtualSlide] = None,
        pixel_size_um: float = 0.65,
        sensor_px: Tuple[int, int] = (512, 512),
        xy_speed_um_per_s: float = 4000.0,
        z_speed_um_per_s: float = 800.0,
        seed: int = 1,
    ):
        self._lock = threading.RLock()
        self._slide = slide or VirtualSlide()
        self._pixel_size_um = float(pixel_size_um)
        self._sensor_w, self._sensor_h = int(sensor_px[0]), int(sensor_px[1])
        self._rng = np.random.default_rng(seed)
        self._timeout_ms = 5000

        self._x = _Axis(speed_um_per_s=xy_speed_um_per_s)
        self._y = _Axis(speed_um_per_s=xy_speed_um_per_s)
        self._z = _Axis(speed_um_per_s=z_speed_um_per_s)

        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._snapped: Optional[np.ndarray] = None
        self._snapped_tags: Dict[str, object] = {}

        # Streaming: frames are produced on a clock at the exposure period and
        # consumed by pop_*; get_last_* reads the newest without consuming.
        self._sequence_running = False
        self._sequence_started = 0.0
        self._frames_consumed = 0
        self._buffer_capacity = 32

        self._devices: Dict[str, _Device] = {}
        self._config_groups: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}
        self._build_devices()

    # ----- construction -------------------------------------------------- #

    def _build_devices(self) -> None:
        sensor_w, sensor_h = str(self._sensor_w), str(self._sensor_h)
        self._devices["Core"] = _Device(
            "Core",
            "Core",
            "Core controller",
            {
                "Camera": _Property(self.CAMERA),
                "Focus": _Property(self.FOCUS),
                "XYStage": _Property(self.XY),
                "Shutter": _Property(""),
                "AutoShutter": _Property("0", allowed=["0", "1"]),
                "Initialize": _Property("1", read_only=True),
                "TimeoutMs": _Property("5000"),
            },
        )
        self._devices[self.CAMERA] = _Device(
            "DemoCamera",
            "DCam",
            "Demo camera (QPSC virtual slide)",
            {
                "Exposure": _Property("10.0000", lower=0.1, upper=10000.0),
                "Binning": _Property("1", allowed=["1", "2", "4"]),
                "PixelType": _Property("32bitRGB", allowed=["8bit", "32bitRGB"]),
                "Gain": _Property("0", lower=-5.0, upper=8.0),
                "OnCameraCCDXSize": _Property(sensor_w, read_only=True),
                "OnCameraCCDYSize": _Property(sensor_h, read_only=True),
                "CameraName": _Property("QPSC demo camera", read_only=True),
                "TransposeMirrorX": _Property("0", allowed=["0", "1"]),
                "TransposeMirrorY": _Property("0", allowed=["0", "1"]),
            },
        )
        self._devices[self.FOCUS] = _Device(
            "DemoCamera",
            "DStage",
            "Demo focus stage (QPSC)",
            {
                "Position": _Property("0.0000", lower=-5000.0, upper=5000.0),
                "UseSequences": _Property("No", allowed=["No", "Yes"]),
            },
        )
        self._devices[self.XY] = _Device(
            "DemoCamera",
            "DXYStage",
            "Demo XY stage (QPSC)",
            {
                "TransposeMirrorX": _Property("0", allowed=["0", "1"]),
                "TransposeMirrorY": _Property("0", allowed=["0", "1"]),
            },
        )
        self._devices[self.LAMP] = _Device(
            "DemoCamera",
            "DLightPath",
            "Demo transmitted lamp (QPSC)",
            {
                "State": _Property("1", allowed=["0", "1"]),
                "Intensity": _Property("25", lower=0.0, upper=100.0),
            },
        )
        self._config_groups["Channel"] = {
            "Brightfield": [(self.LAMP, "State", "1")],
        }

    # ----- helpers ------------------------------------------------------- #

    def _device(self, name: str) -> _Device:
        try:
            return self._devices[name]
        except KeyError:
            raise RuntimeError(f"No device with label '{name}'") from None

    def _prop(self, device: str, name: str) -> _Property:
        dev = self._device(device)
        try:
            return dev.properties[name]
        except KeyError:
            raise RuntimeError(f"Property '{name}' not found on device '{device}'") from None

    def _axis(self, device: str) -> _Axis:
        if device == self.FOCUS:
            return self._z
        raise RuntimeError(f"Device '{device}' is not a single-axis stage")

    def _binning(self) -> int:
        return int(self._prop(self.CAMERA, "Binning").value)

    def _exposure_ms(self) -> float:
        return float(self._prop(self.CAMERA, "Exposure").value)

    def _components(self) -> int:
        return 4 if self._prop(self.CAMERA, "PixelType").value == "32bitRGB" else 1

    def _frame_size(self) -> Tuple[int, int]:
        b = self._binning()
        if self._roi is not None:
            _, _, w, h = self._roi
            return max(1, w // b), max(1, h // b)
        return self._sensor_w // b, self._sensor_h // b

    def _render_frame(self) -> np.ndarray:
        """The current field as the flat byte buffer MMCore would hand back."""
        now = time.monotonic()
        w, h = self._frame_size()
        b = self._binning()
        x, y, z = self._x.at(now), self._y.at(now), self._z.at(now)
        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            # Offset the field centre by the ROI's displacement from the sensor centre.
            x += (rx + rw / 2.0 - self._sensor_w / 2.0) * self._pixel_size_um
            y += (ry + rh / 2.0 - self._sensor_h / 2.0) * self._pixel_size_um
        lamp = self._prop(self.LAMP, "State").value == "1"
        intensity = float(self._prop(self.LAMP, "Intensity").value)
        rgb = self._slide.render(
            x, y, z, w, h, self._pixel_size_um * b, self._exposure_ms(), intensity, lamp, self._rng
        )
        if self._components() == 1:
            gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
                np.uint8
            )
            return gray.reshape(-1)
        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[..., 0] = rgb[..., 2]
        bgra[..., 1] = rgb[..., 1]
        bgra[..., 2] = rgb[..., 0]
        bgra[..., 3] = 255
        return bgra.reshape(-1)

    def _tags(self) -> Dict[str, object]:
        w, h = self._frame_size()
        now = time.monotonic()
        return {
            "Width": w,
            "Height": h,
            "PixelType": "RGB32" if self._components() == 4 else "GRAY8",
            "Camera": self.CAMERA,
            "Exposure-ms": self._exposure_ms(),
            "Binning": self._binning(),
            "XPositionUm": self._x.at(now),
            "YPositionUm": self._y.at(now),
            "ZPositionUm": self._z.at(now),
        }

    def _frame_period_s(self) -> float:
        return max(self._exposure_ms() / 1000.0, 0.01)

    def _frames_produced(self) -> int:
        if not self._sequence_running:
            return 0
        return int((time.monotonic() - self._sequence_started) / self._frame_period_s())

    # ----- introspection ------------------------------------------------- #

    def get_version_info(self) -> str:
        return "MMCore version 11.0 (QPSC demo microscope -- no hardware attached)"

    def set_timeout_ms(self, ms: int) -> None:
        with self._lock:
            self._timeout_ms = int(ms)

    def get_loaded_devices(self) -> StrVector:
        with self._lock:
            return StrVector(self._devices.keys())

    def get_device_library(self, device: str) -> str:
        with self._lock:
            return self._device(device).library

    def get_device_name(self, device: str) -> str:
        with self._lock:
            return self._device(device).adapter

    def get_device_description(self, device: str) -> str:
        with self._lock:
            return self._device(device).description

    def get_camera_device(self) -> str:
        return self.get_property("Core", "Camera")

    def get_focus_device(self) -> str:
        return self.get_property("Core", "Focus")

    def set_focus_device(self, device: str) -> None:
        with self._lock:
            self._device(device)
            self._prop("Core", "Focus").value = device

    def get_xy_stage_device(self) -> str:
        return self.get_property("Core", "XYStage")

    # ----- properties ---------------------------------------------------- #

    def get_device_property_names(self, device: str) -> StrVector:
        with self._lock:
            return StrVector(self._device(device).properties.keys())

    def get_property(self, device: str, name: str) -> str:
        with self._lock:
            if device == self.FOCUS and name == "Position":
                return f"{self._z.at(time.monotonic()):.4f}"
            return self._prop(device, name).value

    def set_property(self, device: str, name: str, value) -> None:
        with self._lock:
            prop = self._prop(device, name)
            if prop.read_only:
                raise RuntimeError(f"Property '{name}' on '{device}' is read-only")
            text = str(value)
            if prop.allowed is not None and text not in prop.allowed:
                raise RuntimeError(
                    f"Invalid property value '{text}' for {device}-{name}; allowed: {prop.allowed}"
                )
            if prop.has_limits:
                number = float(text)
                if number < prop.lower or number > prop.upper:
                    raise RuntimeError(
                        f"Value {text} for {device}-{name} is outside [{prop.lower}, {prop.upper}]"
                    )
            if device == self.FOCUS and name == "Position":
                self._z.start(float(text), time.monotonic())
                return
            if device == self.CAMERA and name == "Exposure":
                text = f"{float(text):.4f}"
            prop.value = text

    def get_allowed_property_values(self, device: str, name: str) -> StrVector:
        with self._lock:
            return StrVector(self._prop(device, name).allowed or [])

    def is_property_read_only(self, device: str, name: str) -> bool:
        with self._lock:
            return self._prop(device, name).read_only

    def has_property_limits(self, device: str, name: str) -> bool:
        with self._lock:
            return self._prop(device, name).has_limits

    def get_property_lower_limit(self, device: str, name: str) -> float:
        with self._lock:
            prop = self._prop(device, name)
            return float(prop.lower) if prop.lower is not None else 0.0

    def get_property_upper_limit(self, device: str, name: str) -> float:
        with self._lock:
            prop = self._prop(device, name)
            return float(prop.upper) if prop.upper is not None else 0.0

    # ----- config groups ------------------------------------------------- #

    def set_config(self, group: str, preset: str) -> None:
        with self._lock:
            try:
                settings = self._config_groups[group][preset]
            except KeyError:
                raise RuntimeError(f"Config '{preset}' not found in group '{group}'") from None
            for device, name, value in settings:
                self.set_property(device, name, value)

    def wait_for_config(self, group: str, preset: str) -> None:
        self.wait_for_system()

    # ----- stages -------------------------------------------------------- #

    def set_position(self, *args) -> None:
        # MMCore overloads: set_position(z) targets the current focus device,
        # set_position(device, z) names one.
        if len(args) == 1:
            device, position = self.get_focus_device(), args[0]
        elif len(args) == 2:
            device, position = args
        else:
            raise TypeError("set_position expects (z) or (device, z)")
        with self._lock:
            self._axis(device).start(float(position), time.monotonic())

    def get_position(self, device: Optional[str] = None) -> float:
        with self._lock:
            return self._axis(device or self.get_focus_device()).at(time.monotonic())

    def set_xy_position(self, *args) -> None:
        # MMCore accepts both (x, y) and (device, x, y).
        if len(args) == 3:
            device, x, y = args
        elif len(args) == 2:
            device, (x, y) = self.XY, args
        else:
            raise TypeError("set_xy_position expects (x, y) or (device, x, y)")
        with self._lock:
            if device != self.XY:
                raise RuntimeError(f"Device '{device}' is not an XY stage")
            now = time.monotonic()
            self._x.start(float(x), now)
            self._y.start(float(y), now)

    def get_x_position(self, device: Optional[str] = None) -> float:
        with self._lock:
            return self._x.at(time.monotonic())

    def get_y_position(self, device: Optional[str] = None) -> float:
        with self._lock:
            return self._y.at(time.monotonic())

    def home(self, device: str) -> None:
        with self._lock:
            now = time.monotonic()
            if device == self.XY:
                self._x.start(0.0, now)
                self._y.start(0.0, now)
            else:
                self._axis(device).start(0.0, now)

    def stop(self, device: str) -> None:
        with self._lock:
            now = time.monotonic()
            if device == self.XY:
                self._x.stop(now)
                self._y.stop(now)
            else:
                self._axis(device).stop(now)

    def device_busy(self, device: str) -> bool:
        with self._lock:
            now = time.monotonic()
            if device == self.XY:
                return self._x.busy(now) or self._y.busy(now)
            if device == self.FOCUS:
                return self._z.busy(now)
            self._device(device)
            return False

    def wait_for_device(self, device: str) -> None:
        deadline = time.monotonic() + self._timeout_ms / 1000.0
        while self.device_busy(device):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Wait for device '{device}' timed out after {self._timeout_ms} ms"
                )
            time.sleep(0.002)

    def wait_for_system(self) -> None:
        self.wait_for_device(self.XY)
        self.wait_for_device(self.FOCUS)

    # ----- camera: geometry ---------------------------------------------- #

    def get_image_width(self) -> int:
        with self._lock:
            return self._frame_size()[0]

    def get_image_height(self) -> int:
        with self._lock:
            return self._frame_size()[1]

    def get_number_of_components(self) -> int:
        with self._lock:
            return self._components()

    def get_bytes_per_pixel(self) -> int:
        with self._lock:
            return self._components()

    def get_image_bit_depth(self) -> int:
        return 8

    def get_image_buffer_size(self) -> int:
        with self._lock:
            w, h = self._frame_size()
            return w * h * self._components()

    def get_pixel_size_um(self) -> float:
        with self._lock:
            return self._pixel_size_um * self._binning()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        with self._lock:
            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > self._sensor_w
                or y + h > self._sensor_h
            ):
                raise RuntimeError(
                    f"ROI ({x}, {y}, {w}, {h}) does not fit the {self._sensor_w}x{self._sensor_h} sensor"
                )
            self._roi = (int(x), int(y), int(w), int(h))

    def clear_roi(self) -> None:
        with self._lock:
            self._roi = None

    def get_roi(self) -> Tuple[int, int, int, int]:
        with self._lock:
            return self._roi if self._roi is not None else (0, 0, self._sensor_w, self._sensor_h)

    # ----- camera: exposure ---------------------------------------------- #

    def set_exposure(self, exposure_ms: float) -> None:
        self.set_property(self.CAMERA, "Exposure", exposure_ms)

    def get_exposure(self) -> float:
        return float(self.get_property(self.CAMERA, "Exposure"))

    # ----- camera: snap -------------------------------------------------- #

    def snap_image(self) -> None:
        with self._lock:
            if self._sequence_running:
                raise RuntimeError("Cannot snap while a sequence acquisition is running")
            exposure_s = self._exposure_ms() / 1000.0
        # Hold the shutter open for the exposure, like a real camera, but keep
        # the lock free so position polls keep answering.
        time.sleep(min(exposure_s, 0.5))
        with self._lock:
            self._snapped = self._render_frame()
            self._snapped_tags = self._tags()

    def get_image(self) -> np.ndarray:
        with self._lock:
            if self._snapped is None:
                raise RuntimeError("No image has been snapped")
            return self._snapped

    def get_tagged_image(self) -> TaggedImage:
        with self._lock:
            return TaggedImage(self.get_image(), dict(self._snapped_tags))

    # ----- camera: streaming --------------------------------------------- #

    def start_continuous_sequence_acquisition(self, interval_ms: float = 0.0) -> None:
        with self._lock:
            self._sequence_running = True
            self._sequence_started = time.monotonic()
            self._frames_consumed = 0

    def start_sequence_acquisition(self, *args) -> None:
        self.start_continuous_sequence_acquisition(0.0)

    def stop_sequence_acquisition(self) -> None:
        with self._lock:
            self._sequence_running = False

    def is_sequence_running(self) -> bool:
        with self._lock:
            return self._sequence_running

    def get_remaining_image_count(self) -> int:
        with self._lock:
            return max(
                0, min(self._frames_produced() - self._frames_consumed, self._buffer_capacity)
            )

    def clear_circular_buffer(self) -> None:
        with self._lock:
            self._frames_consumed = self._frames_produced()

    def pop_next_image(self) -> np.ndarray:
        with self._lock:
            if self.get_remaining_image_count() == 0:
                raise RuntimeError("Circular buffer is empty")
            self._frames_consumed += 1
            return self._render_frame()

    def pop_next_tagged_image(self) -> TaggedImage:
        with self._lock:
            pix = self.pop_next_image()
            return TaggedImage(pix, self._tags())

    def get_last_image(self) -> np.ndarray:
        with self._lock:
            if self.get_remaining_image_count() == 0 and self._frames_produced() == 0:
                raise RuntimeError("Circular buffer is empty")
            return self._render_frame()

    def get_last_tagged_image(self) -> TaggedImage:
        with self._lock:
            return TaggedImage(self.get_last_image(), self._tags())

    # ----- demo-only introspection -------------------------------------- #

    @property
    def slide(self) -> VirtualSlide:
        return self._slide


class _DemoLive:
    """``studio.live()`` -- live mode is just a sequence acquisition here."""

    def __init__(self, core: DemoCore):
        self._core = core

    def set_live_mode(self, on: bool) -> None:
        if on:
            if not self._core.is_sequence_running():
                self._core.start_continuous_sequence_acquisition(0.0)
        else:
            self._core.stop_sequence_acquisition()

    def is_live_mode_on(self) -> bool:
        return self._core.is_sequence_running()


class DemoStudio:
    """A Pycro-Manager ``Studio`` look-alike; only ``live()`` is used."""

    def __init__(self, core: DemoCore):
        self._live = _DemoLive(core)

    def live(self) -> _DemoLive:
        return self._live
