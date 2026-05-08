"""Z-stage diagnostic probe handler.

Handles PROBEZ command -- a one-shot diagnostic sweep over the focus
device to characterize its timing, non-blocking behavior, speed
sensitivity, and streaming-during-motion feasibility.

All output is written to the server log with the tag

    PROBEZ [step-N]: ...

so a client can run the probe and then hand the resulting
server_session_*.log file back for analysis via

    parse_server_log.py <log> --grep "PROBEZ" --short-time --no-level

The handler is read-mostly and idempotent: it saves and restores the
original Z position and any device properties it modifies in a finally
block, and it refuses to run if a sequence acquisition is already
active (which would indicate the rig is in use).

Nothing about this handler is performance-critical -- clarity and
thorough logging beat speed. A probe run takes ~15-40 seconds
depending on how many streaming ranges are tested.
"""

import csv
import logging
import pathlib
import time
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# Ranges to exercise in Step 4 (streaming-during-motion).
# Chosen to bracket the current sweep_range_um=6.0 setting and the
# plausible "wider, slower" operating points we might migrate to.
STREAM_RANGES_UM = [2.0, 6.0, 12.0, 20.0]

# MaxSpeed values to test in Step 3. Prior ProScan MaxSpeed is on a
# 1-100 percent scale; we descend from full speed to slowest to find
# the operating point where a 20 um move takes long enough for a
# streaming camera to collect 15+ frames. First probe run showed that
# MaxSpeed=100 and 50 both hit the same floor (~100 um/s), real
# slowdown kicks in below 25, and 5 gives ~50 um/s -- so we extend
# down to 1 this run to characterize the slow end of the curve.
MAXSPEED_VALUES = [100, 50, 25, 10, 5, 2, 1]

# Properties we deliberately never try to restore -- Prior's Port is
# pre-init-only and set_property throws at runtime even when the
# value is unchanged. Listed here so we don't emit a scary stack
# trace on the cleanup path.
NON_RESTORABLE_PROPERTIES = frozenset({"Port"})

# Relative move sizes in um for Steps 1 and 2.
MOVE_SIZES_UM = [1.0, 5.0, 10.0, 20.0, 50.0]

# Safety: don't do streaming scans wider than this in a probe run
# (the probe moves the stage, not the user). 25 um is well inside any
# realistic Z limit envelope but still larger than STREAM_RANGES_UM[-1].
MAX_STREAM_RANGE_UM = 25.0

# Ranges to exercise in Step 5 (metric validation). Smaller list than
# Step 4 since each range pays the cost of a ground-truth stepped
# sweep on top of the continuous scan.
METRIC_VALIDATION_RANGES_UM = [6.0, 12.0]

# Number of points per range for the ground-truth stepped sweep.
METRIC_STEPPED_N_STEPS = 15

# Exposures to sweep at the 6 um validation range (Step 5b). Chosen
# to bracket realistic PPM operating points:
#   0.7 ms  - real PPM acquisition exposure
#   5 ms    - typical brightfield
#   20 ms   - moderate fluorescence / low PPM angle
#   50 ms   - dark fluorescence / very low PPM angle
#   100 ms  - near the expected feasibility ceiling on Prior at MaxSpeed=1
# At each exposure we compute the expected per-frame motion blur
# (velocity * exposure) and compare it to the DOF budget to mark
# samples as in or out of the safe operating envelope.
EXPOSURE_SWEEP_MS = [0.7, 5.0, 20.0, 50.0, 100.0]

# Motion blur budget -- at or above this per-frame blur, samples are
# expected to be too smeared for a reliable focus metric. Derived
# from ~25% of a representative 20X DOF (~2 um). Conservative.
BLUR_BUDGET_UM = 0.5


def _log(step: str, msg: str) -> None:
    """Uniform tagged logging so the entire probe run is greppable."""
    logger.info("PROBEZ [%s]: %s", step, msg)


def _warn(step: str, msg: str) -> None:
    logger.warning("PROBEZ [%s]: %s", step, msg)


def _err(step: str, msg: str, exc_info: bool = False) -> None:
    logger.error("PROBEZ [%s]: %s", step, msg, exc_info=exc_info)


def _try_get_property(core, device: str, prop: str):
    """Best-effort property read -- returns (value, error_str)."""
    try:
        return core.get_property(device, prop), None
    except Exception as e:
        return None, str(e)


def _try_set_property(core, device: str, prop: str, value) -> bool:
    """Best-effort property write -- returns True on success."""
    try:
        core.set_property(device, prop, value)
        return True
    except Exception as e:
        _warn("setprop", f"{device}.{prop} <- {value!r} failed: {e}")
        return False


def _pop_image_as_numpy(core):
    """Pop one frame from the circular buffer as a numpy array.

    Handles both monochrome (ndim=2) and multi-component (ndim=3)
    cameras. Returns None if the pop failed or the buffer was empty.
    """
    try:
        pixels = core.pop_next_image()
    except Exception as e:
        logger.debug("pop_next_image failed: %s", e)
        return None
    if pixels is None:
        return None
    try:
        w = core.get_image_width()
        h = core.get_image_height()
        nch = core.get_number_of_components()
    except Exception as e:
        logger.debug("image geometry query failed: %s", e)
        return None
    arr = np.asarray(pixels)
    try:
        if nch == 1:
            return arr.reshape(h, w)
        return arr.reshape(h, w, nch)
    except Exception as e:
        logger.debug("reshape to (%dx%d x %d) failed: %s", h, w, nch, e)
        return arr


def _focus_metric(img) -> float:
    """Normalized variance on the green/first channel.

    This is the same metric family that the current sweep drift check
    uses on the acquisition path (normalized_variance in
    autofocus_PPM.yml). We implement it inline rather than importing
    the full autofocus stack so the probe stays self-contained.

    Returns 0.0 for empty / unreadable images so callers can sort
    without special-casing None.
    """
    if img is None:
        return 0.0
    a = np.asarray(img)
    if a.size == 0:
        return 0.0
    if a.ndim == 3:
        # Green channel is ch=1 in RGB; fall back to ch=0 for
        # single-component or 2-channel images.
        ch = 1 if a.shape[2] >= 2 else 0
        gray = a[:, :, ch]
    else:
        gray = a
    g = gray.astype(np.float64, copy=False)
    mean = g.mean()
    if mean <= 1e-9:
        return 0.0
    var = g.var()
    return float(var / mean)


def _str_vector_to_list(vec) -> list:
    """Convert an MMCore StrVector (mmcorej_StrVector / std::vector<string>)
    into a Python list. The Java bindings expose .size()/.get(i) but are
    not directly iterable via `for` / `list()` in pycromanager's JNI
    wrappers -- so we fall through from Python iteration to indexed
    access, then finally to an empty list.
    """
    if vec is None:
        return []
    try:
        return list(vec)
    except TypeError:
        pass
    try:
        n = int(vec.size())
        return [vec.get(i) for i in range(n)]
    except Exception:
        return []


def _snapshot_focus_device(core, step: str) -> dict:
    """Log every property on the current focus device and return a
    dict of {property_name: current_value} suitable for restoring.

    This is Step 0 data AND the snapshot used to restore state in the
    finally block. Only writable properties are captured for restore;
    read-only values are logged but not snapshotted.
    """
    focus_device = core.get_focus_device()
    _log(step, f"Focus device from core.get_focus_device() = '{focus_device}'")

    try:
        lib = core.get_device_library(focus_device)
    except Exception:
        lib = "?"
    try:
        adapter = core.get_device_name(focus_device)
    except Exception:
        adapter = "?"
    try:
        description = core.get_device_description(focus_device)
    except Exception:
        description = "?"
    _log(step, f"Adapter library='{lib}' name='{adapter}' description='{description}'")

    try:
        z_current = core.get_position(focus_device)
        _log(step, f"Current Z = {z_current:.3f} um")
    except Exception as e:
        _err(step, f"get_position({focus_device}) failed: {e}")
        z_current = None

    restore = {"__focus_device__": focus_device, "__z_original__": z_current}

    try:
        prop_names = _str_vector_to_list(core.get_device_property_names(focus_device))
    except Exception as e:
        _err(step, f"get_device_property_names({focus_device}) failed: {e}")
        return restore
    if not prop_names:
        _err(step, f"Property list for {focus_device} came back empty")
        return restore

    _log(step, f"Property count = {len(prop_names)}")

    for name in prop_names:
        value, err = _try_get_property(core, focus_device, name)
        if err is not None:
            _log(step, f"  {name} = <error: {err}>")
            continue

        try:
            read_only = core.is_property_read_only(focus_device, name)
        except Exception:
            read_only = None

        try:
            has_limits = core.has_property_limits(focus_device, name)
        except Exception:
            has_limits = False

        if has_limits:
            try:
                lo = core.get_property_lower_limit(focus_device, name)
                hi = core.get_property_upper_limit(focus_device, name)
                limits_str = f" limits=[{lo}, {hi}]"
            except Exception:
                limits_str = ""
        else:
            limits_str = ""

        try:
            allowed = _str_vector_to_list(core.get_allowed_property_values(focus_device, name))
        except Exception:
            allowed = []
        allowed_str = f" allowed={allowed}" if allowed else ""

        ro_str = "RO" if read_only else "RW"
        _log(step, f"  {name} = {value!r} [{ro_str}]{limits_str}{allowed_str}")

        if read_only is False and name not in NON_RESTORABLE_PROPERTIES:
            restore[name] = value

    return restore


def _restore_focus_device(core, restore: dict) -> None:
    """Put every property we captured back to its original value, then
    move Z back to the original position. Never raises."""
    focus_device = restore.get("__focus_device__")
    if not focus_device:
        return

    _log("cleanup", f"Restoring properties on '{focus_device}'")

    for name, value in restore.items():
        if name.startswith("__"):
            continue
        try:
            # Prior/Prior-like adapters expect string values; MM core
            # does the coercion for us if we pass a string consistently.
            core.set_property(focus_device, name, str(value))
        except Exception as e:
            _warn("cleanup", f"restore {name}={value!r} failed: {e}")

    z_original = restore.get("__z_original__")
    if z_original is not None:
        try:
            core.set_position(focus_device, z_original)
            core.wait_for_device(focus_device)
            _log("cleanup", f"Restored Z = {z_original:.3f} um")
        except Exception as e:
            _err("cleanup", f"Z restore to {z_original:.3f} failed: {e}")


def _step1_blocking_move_timing(core, focus_device: str, z0: float) -> None:
    """Baseline transit times at current speed settings.

    For each move size, measure the round-trip of a blocking move and
    of the return. These numbers are what the current stepped sweep is
    actually paying for.
    """
    _log("step-1", "Blocking move round-trip at current (default) speed")
    for dz in MOVE_SIZES_UM:
        try:
            t0 = time.perf_counter()
            core.set_position(focus_device, z0 + dz)
            core.wait_for_device(focus_device)
            t_out = (time.perf_counter() - t0) * 1000.0
            z_after_out = core.get_position(focus_device)

            t0 = time.perf_counter()
            core.set_position(focus_device, z0)
            core.wait_for_device(focus_device)
            t_back = (time.perf_counter() - t0) * 1000.0
            z_after_back = core.get_position(focus_device)

            _log(
                "step-1",
                f"dz={dz:+.1f} um  out={t_out:.0f}ms (Z={z_after_out:.3f})  "
                f"back={t_back:.0f}ms (Z={z_after_back:.3f})",
            )
        except Exception as e:
            _err("step-1", f"dz={dz} failed: {e}", exc_info=True)
            try:
                core.set_position(focus_device, z0)
                core.wait_for_device(focus_device)
            except Exception:
                pass


def _step2_nonblocking_position_readback(core, focus_device: str, z0: float) -> None:
    """Measure how long the non-blocking issue takes, how long until
    device_busy clears, and -- critically -- whether the stage reports
    mid-motion positions or jumps to the target immediately.

    If mid-motion positions ramp smoothly, we can query Z per frame
    during a continuous scan. If they snap to the target, we have to
    fall back to a time-linear velocity model.
    """
    _log("step-2", "Non-blocking issue latency + position readback during motion")
    for dz in MOVE_SIZES_UM:
        try:
            t_issue_start = time.perf_counter()
            core.set_position(focus_device, z0 + dz)
            t_issue = (time.perf_counter() - t_issue_start) * 1000.0

            samples = []
            deadline = time.perf_counter() + 3.0
            t0 = time.perf_counter()
            busy_cleared_at = None

            while time.perf_counter() < deadline:
                t_rel = (time.perf_counter() - t0) * 1000.0
                try:
                    z_now = core.get_position(focus_device)
                except Exception as e:
                    z_now = float("nan")
                    _warn("step-2", f"get_position during motion failed: {e}")

                try:
                    busy = core.device_busy(focus_device)
                except Exception:
                    busy = None

                samples.append((t_rel, z_now, busy))

                if busy is False and busy_cleared_at is None:
                    busy_cleared_at = t_rel
                    # Give one more sample after busy clears then stop.
                    break

                time.sleep(0.005)

            # Ensure fully arrived (defensive).
            try:
                core.wait_for_device(focus_device)
            except Exception:
                pass
            z_final = core.get_position(focus_device)

            # Compact summary of the position trace.
            trace = ", ".join(
                f"({t:.0f}ms,{z:.3f},{'B' if busy else 'I' if busy is False else '?'})"
                for (t, z, busy) in samples[:12]
            )
            if len(samples) > 12:
                trace += f", ... ({len(samples)} total)"

            _log(
                "step-2",
                f"dz={dz:+.1f} um  issue={t_issue:.1f}ms  "
                f"busy_clear={busy_cleared_at if busy_cleared_at is not None else 'never'}ms  "
                f"z_final={z_final:.3f}",
            )
            _log("step-2", f"         trace: {trace}")

            # Return for next iteration.
            core.set_position(focus_device, z0)
            core.wait_for_device(focus_device)
        except Exception as e:
            _err("step-2", f"dz={dz} failed: {e}", exc_info=True)
            try:
                core.set_position(focus_device, z0)
                core.wait_for_device(focus_device)
            except Exception:
                pass


def _step3_maxspeed_sensitivity(core, focus_device: str, z0: float, speed_prop: str) -> None:
    """Measure a 20 um blocking move round-trip across the MaxSpeed
    range. The goal is to find a (speed, move_time) operating point
    where a ~6 um sweep takes long enough for the camera to produce
    ~15-30 frames at 38 fps (0.4-0.8 seconds)."""
    _log("step-3", f"MaxSpeed sensitivity sweep via property '{speed_prop}'")
    dz = 20.0
    for speed in MAXSPEED_VALUES:
        # Prior property values are strings -- pass as str to avoid
        # MM Java binding type-coercion quirks.
        if not _try_set_property(core, focus_device, speed_prop, str(speed)):
            continue
        try:
            t0 = time.perf_counter()
            core.set_position(focus_device, z0 + dz)
            core.wait_for_device(focus_device)
            t_out = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            core.set_position(focus_device, z0)
            core.wait_for_device(focus_device)
            t_back = (time.perf_counter() - t0) * 1000.0

            inferred_v_out = (dz / t_out) * 1000.0 if t_out > 0 else float("nan")
            inferred_v_back = (dz / t_back) * 1000.0 if t_back > 0 else float("nan")

            _log(
                "step-3",
                f"{speed_prop}={speed}  dz={dz:+.1f} um  "
                f"out={t_out:.0f}ms (~{inferred_v_out:.1f} um/s)  "
                f"back={t_back:.0f}ms (~{inferred_v_back:.1f} um/s)",
            )
        except Exception as e:
            _err("step-3", f"speed={speed} failed: {e}", exc_info=True)
            try:
                core.set_position(focus_device, z0)
                core.wait_for_device(focus_device)
            except Exception:
                pass


def _step4_stream_during_motion(core, focus_device: str, z0: float, speed_prop: str) -> None:
    """The critical feasibility test.

    At a slow MaxSpeed (so the move takes hundreds of ms), start a
    continuous sequence acquisition, issue a non-blocking move, and
    pop frames from the circular buffer as fast as possible while
    recording (pop_time, core.get_position()) for each. Repeat for
    several range values so we can see how frame count scales with
    range and whether sample density is adequate.

    The camera exposure and FrameRateHz are left at whatever MM is
    currently set to. The first probe run showed that running the
    probe with JAICamera FrameRateHz=1 produced only 3 fps streaming
    and made Step 4 unusable -- so this step now explicitly measures
    free-run frame cadence BEFORE motion and reports it prominently,
    so the user can tell immediately if they need to bump the frame
    rate and re-run.
    """
    # Pick the slowest speed from Step 3. MAXSPEED_VALUES is sorted
    # fast-to-slow so [-1] is the slowest we sweeped.
    slow_speed = MAXSPEED_VALUES[-1]
    if not _try_set_property(core, focus_device, speed_prop, str(slow_speed)):
        _err("step-4", f"Cannot set {speed_prop}={slow_speed}; skipping streaming test")
        return
    _log("step-4", f"Using {speed_prop}={slow_speed} for streaming runs")

    # Report current camera state for context. The three things that
    # matter for streaming feasibility are: what camera is active,
    # what its exposure is, and what its nominal frame rate property
    # is set to. If the camera is JAI and the rate is 1 Hz, Step 4's
    # streaming trials will be garbage -- the user has to bump it
    # from MicroManager or via the QPSC camera control dialog.
    try:
        cam = core.get_camera_device()
        _log("step-4", f"Camera device = {cam}")
    except Exception as e:
        _log("step-4", f"get_camera_device failed: {e}")
        cam = None

    try:
        exposure_ms = core.get_exposure()
        _log("step-4", f"Camera exposure (current) = {exposure_ms:.2f} ms")
    except Exception as e:
        _log("step-4", f"Could not query exposure: {e}")

    if cam:
        for rate_prop in ("FrameRateHz", "FrameRate", "TargetFrameRate"):
            try:
                rate_val = core.get_property(cam, rate_prop)
                _log("step-4", f"{cam}.{rate_prop} = {rate_val}")
                break
            except Exception:
                continue

    # Characterize free-run frame cadence BEFORE any motion. Measures
    # inter-frame spacing for up to 30 frames over up to 2 seconds.
    _measure_free_run_frame_rate(core)

    for range_um in STREAM_RANGES_UM:
        if range_um > MAX_STREAM_RANGE_UM:
            _warn("step-4", f"Skipping range={range_um} (exceeds MAX_STREAM_RANGE_UM)")
            continue
        _stream_one_range(core, focus_device, z0, range_um)


def _measure_free_run_frame_rate(core) -> None:
    """Stream the camera without any motion and measure inter-frame
    spacing. This reveals whether the camera is actually running at
    its nominal max rate or is throttled by FrameRateHz / exposure /
    something else.
    """
    tag = "step-4 freerun"
    try:
        if core.is_sequence_running():
            _warn(tag, "Sequence already running; skipping free-run measurement")
            return
    except Exception:
        pass

    try:
        core.clear_circular_buffer()
        core.start_continuous_sequence_acquisition(0)
    except Exception as e:
        _err(tag, f"start_continuous_sequence_acquisition failed: {e}")
        return

    # Warm-up so we skip any first-frame priming delay.
    time.sleep(0.1)
    try:
        while core.get_remaining_image_count() > 0:
            core.pop_next_image()
    except Exception:
        pass

    # Collect up to 30 frames or 2 seconds, whichever comes first.
    timestamps = []
    t0 = time.perf_counter()
    try:
        while len(timestamps) < 30 and (time.perf_counter() - t0) < 2.0:
            try:
                if core.get_remaining_image_count() > 0:
                    core.pop_next_image()
                    timestamps.append((time.perf_counter() - t0) * 1000.0)
                    continue
            except Exception as e:
                _warn(tag, f"pop failed: {e}")
                break
            time.sleep(0.002)
    finally:
        try:
            core.stop_sequence_acquisition()
        except Exception:
            pass
        try:
            core.clear_circular_buffer()
        except Exception:
            pass

    if len(timestamps) < 2:
        _warn(
            tag, f"only captured {len(timestamps)} frames in 2 s -- camera may be idle or very slow"
        )
        return

    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    avg = sum(deltas) / len(deltas)
    fps = 1000.0 / avg if avg > 0 else float("nan")
    d_min = min(deltas)
    d_max = max(deltas)

    _log(
        tag,
        f"captured {len(timestamps)} frames in {timestamps[-1]:.0f}ms  "
        f"inter-frame avg={avg:.1f}ms  min={d_min:.1f}ms  max={d_max:.1f}ms  "
        f"measured_rate={fps:.1f} fps",
    )
    if fps < 20.0:
        _warn(
            tag,
            f"measured frame rate {fps:.1f} fps is well below a normal JAI max "
            f"(~38 fps) -- Step 4 streaming results will be frame-rate-limited, "
            f"not stage-limited. Bump JAICamera.FrameRateHz to its max and re-run.",
        )


def _stream_one_range(core, focus_device: str, z0: float, range_um: float) -> None:
    """Run a single streaming-during-motion trial for one sweep range."""
    half = range_um / 2.0
    z_start = z0 - half
    z_end = z0 + half
    tag = f"step-4 range={range_um:.1f}"

    _log(tag, f"Start trial: [{z_start:.3f} -> {z_end:.3f}]")

    # Seed position (blocking).
    try:
        core.set_position(focus_device, z_start)
        core.wait_for_device(focus_device)
    except Exception as e:
        _err(tag, f"Seed move to {z_start:.3f} failed: {e}", exc_info=True)
        return

    # Refuse to start streaming if a sequence is already running --
    # we do NOT want to disturb an active acquisition.
    try:
        if core.is_sequence_running():
            _err(tag, "Sequence already running; aborting streaming trial")
            return
    except Exception as e:
        _warn(tag, f"is_sequence_running check failed: {e}")

    # Start streaming at max rate (interval_ms=0 = free run).
    try:
        core.clear_circular_buffer()
        core.start_continuous_sequence_acquisition(0)
    except Exception as e:
        _err(tag, f"start_continuous_sequence_acquisition failed: {e}", exc_info=True)
        return

    # Small settle so the first frame is meaningful.
    time.sleep(0.05)

    try:
        # Drain any frames that arrived during the settle so our
        # t=0 matches the motion start, not the settle start.
        drained = 0
        try:
            while core.get_remaining_image_count() > 0:
                core.pop_next_image()
                drained += 1
        except Exception as e:
            _warn(tag, f"Pre-motion drain failed after {drained} frames: {e}")
        _log(tag, f"Pre-motion drained {drained} stale frames from buffer")

        # Fire the move non-blocking. t0 is the moment we return from
        # the issue call.
        t0 = time.perf_counter()
        try:
            core.set_position(focus_device, z_end)
        except Exception as e:
            _err(tag, f"Non-blocking set_position failed: {e}", exc_info=True)
            return

        # Pull frames from the circular buffer as fast as they arrive.
        # We only query stage Z when we actually pop a frame -- that
        # keeps the polling loop from hammering the serial line and
        # frees the main bottleneck seen in the first probe run, where
        # the per-iteration get_position()+device_busy() pair was
        # itself limiting the loop rate to ~3 fps regardless of the
        # camera's actual frame cadence.
        #
        # device_busy() is also serial-bound, so we only call it every
        # few iterations and use it only to exit the loop when the
        # stage arrives. The hard deadline guarantees we eventually
        # exit even if something goes wrong.
        frame_samples = []  # (t_ms_at_pop, z_at_pop)
        stage_done_at = None
        busy_check_every = 6  # ~12 ms between device_busy calls at 2 ms sleep
        loop_count = 0
        # Hard deadline: pessimistic upper bound on stage motion at
        # the slowest speed we tested, + 500 ms tail for post-motion
        # frames. 25 ms/um * range + 500 ms gives lots of headroom.
        deadline = time.perf_counter() + (range_um * 0.05 + 1.0)

        while time.perf_counter() < deadline:
            t_rel = (time.perf_counter() - t0) * 1000.0

            # 1) Pop frames (non-blocking). Stamp each pop with its
            # own perf_counter() reading -- the outer t_rel at the top
            # of the loop is stale by the time we fetch the 2nd, 3rd,
            # 4th frame from a burst, which made the first probe run's
            # traces misleading (4 frames at the same t_rel).
            popped_this_tick = 0
            try:
                while core.get_remaining_image_count() > 0:
                    core.pop_next_image()
                    t_pop = (time.perf_counter() - t0) * 1000.0
                    try:
                        z_at_pop = core.get_position(focus_device)
                    except Exception:
                        z_at_pop = float("nan")
                    frame_samples.append((t_pop, z_at_pop))
                    popped_this_tick += 1
                    if popped_this_tick >= 4:
                        break  # don't starve the busy check
            except Exception as e:
                _warn(tag, f"pop loop failed: {e}")

            # 2) Periodically check if the stage is done. After it is,
            # keep looping for a small tail so we catch any frames
            # that arrive between stage_done and the post-move
            # handshake settling.
            loop_count += 1
            if loop_count % busy_check_every == 0:
                try:
                    busy = core.device_busy(focus_device)
                except Exception:
                    busy = None
                if busy is False and stage_done_at is None:
                    stage_done_at = t_rel

            if stage_done_at is not None and (t_rel - stage_done_at) > 300.0:
                break  # 300 ms of post-motion tail is plenty

            time.sleep(0.002)

        # Final defensive wait + end-position read.
        try:
            core.wait_for_device(focus_device)
        except Exception:
            pass
        z_final = core.get_position(focus_device)
    finally:
        try:
            core.stop_sequence_acquisition()
        except Exception as e:
            _warn(tag, f"stop_sequence_acquisition failed: {e}")
        try:
            core.clear_circular_buffer()
        except Exception:
            pass

    # Return to z0 for next trial.
    try:
        core.set_position(focus_device, z0)
        core.wait_for_device(focus_device)
    except Exception as e:
        _err(tag, f"Return to z0 failed: {e}")

    # Summarize.
    total_frames = len(frame_samples)
    if stage_done_at is not None:
        in_motion = sum(1 for (t, _z) in frame_samples if t <= stage_done_at)
    else:
        in_motion = total_frames

    z_values = [z for (_t, z) in frame_samples if z == z]  # NaN-safe
    if z_values:
        z_min = min(z_values)
        z_max = max(z_values)
        z_span = z_max - z_min
    else:
        z_min = z_max = z_span = float("nan")

    # Inter-frame timing stats over the first N frames.
    if len(frame_samples) >= 2:
        deltas = [
            frame_samples[i][0] - frame_samples[i - 1][0] for i in range(1, len(frame_samples))
        ]
        avg_delta = sum(deltas) / len(deltas)
        fps = 1000.0 / avg_delta if avg_delta > 0 else float("nan")
    else:
        avg_delta = float("nan")
        fps = float("nan")

    _log(
        tag,
        f"done: stage_done_at={stage_done_at}  frames={total_frames}  "
        f"in_motion={in_motion}  z_span={z_span:.3f}um  z_final={z_final:.3f}  "
        f"frame_avg_dt={avg_delta:.1f}ms (~{fps:.1f} fps)",
    )

    # Dump every (frame_time, z) pair so we can reconstruct the z(t)
    # curve offline. Capped at 40 to keep the log readable.
    for t_rel, z_at_pop in frame_samples[:40]:
        _log(tag, f"   frame t={t_rel:>7.1f}ms  z={z_at_pop:.3f}")
    if len(frame_samples) > 40:
        _log(tag, f"   ... ({len(frame_samples) - 40} more frames not logged)")


def _step5_metric_validation(
    core, focus_device: str, z0: float, speed_prop: str, log_dir: pathlib.Path
) -> None:
    """Real feasibility gate: do frames captured during a smooth scan
    produce a usable focus metric curve?

    Second-probe-run findings that changed this design:

    - JAI streaming mode has an ~800 ms 'settling' transient after
      startContinuousSequenceAcquisition where the normalized_variance
      metric decays by ~25% at a constant Z position. This makes every
      in-motion pop sample useless as a focus indicator; the motion-
      phase metrics are inflated relative to steady-state by an
      unknown, time-varying amount. We abandoned the pop-from-stream
      approach in favor of fresh snap_image() calls during motion.

    - At MaxSpeed=1 the Prior's non-blocking set_position leaves the
      stage with accumulated positioning error (observed 2-5 um
      overshoot on Step 4 trials). We now keep MaxSpeed at the
      default (100) for positioning moves and only drop to the slow
      value for the actual scan motion, which bounds the weirdness
      to the scan window.

    - Before any motion at all, we now run a static metric stability
      check: 20 consecutive snaps at the same Z, just to catch any
      camera-side drift (auto-gain, auto-WB, thermal) that would
      invalidate ALL metric comparisons independent of motion.

    Each validation range runs:
      0) Static stability (before any motion, once per range)
      1) Snap-during-motion at slow MaxSpeed: fire non-blocking move,
         loop calling snap_image + get_position during motion, record
         (t_ms, z_avg, metric) per snap. z_avg = (z_before + z_after)/2
         brackets the per-exposure motion so we have a known z window
         per sample.
      2) Stepped sweep at full MaxSpeed ground truth: blocking
         step + snap + metric, 15 points.
      3) Argmax comparison, VERDICT, CSV dump.

    Nothing is committed: Z is returned to z0 at the end.
    """
    if not _try_set_property(core, focus_device, speed_prop, str(MAXSPEED_VALUES[-1])):
        _err(
            "step-5",
            f"Cannot set {speed_prop}={MAXSPEED_VALUES[-1]}; " f"skipping metric validation",
        )
        return
    _log("step-5", f"Metric validation -- slow scan speed {speed_prop}={MAXSPEED_VALUES[-1]}")

    # Save original exposure so we can restore it after the exposure
    # sweep. Everything in Step 5 is exposure-aware from here on.
    original_exposure_ms = None
    try:
        original_exposure_ms = core.get_exposure()
        _log(
            "step-5",
            f"Camera exposure (original) = {original_exposure_ms:.2f} ms "
            "-- will be restored at end of step-5",
        )
    except Exception as e:
        _warn("step-5", f"get_exposure failed: {e}")

    # Estimate the minimum achievable stage velocity at slow_speed so
    # we can annotate each exposure with an expected blur budget.
    # Uses the Step 3 curve if we can find it in the log, else a
    # safe 11.5 um/s default (forward Prior MaxSpeed=1 measurement).
    min_velocity_um_s = _estimate_min_velocity(
        core, focus_device, speed_prop, MAXSPEED_VALUES[-1], z0
    )
    _log(
        "step-5",
        f"Min achievable velocity at {speed_prop}={MAXSPEED_VALUES[-1]} "
        f"= {min_velocity_um_s:.2f} um/s",
    )
    _log("step-5", f"Blur budget (25% of ~2um DOF): {BLUR_BUDGET_UM:.2f} um")
    _log("step-5", "Expected blur per exposure at this velocity:")
    for exp_ms in EXPOSURE_SWEEP_MS:
        blur = min_velocity_um_s * (exp_ms / 1000.0)
        status = "OK" if blur <= BLUR_BUDGET_UM else "OVER-BUDGET"
        _log("step-5", f"   exposure={exp_ms:>6.1f} ms  blur={blur:.3f} um  {status}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Static stability check at the ORIGINAL exposure. If the
        # pipeline is unstable at the user's current exposure the
        # whole validation is pointless.
        if not _step5_static_stability(core, focus_device, z0):
            _err(
                "step-5",
                "Static stability FAILED at original exposure -- " "aborting metric validation",
            )
            return

        # 5a. Range sweep at the original exposure. Each trial runs
        # THREE scans: snap-motion, streaming-motion, stepped. We
        # compare all three against the stepped ground truth so we
        # can tell whether streaming (high sample density) or snap
        # (slow but no buffer latency) is the right mechanism for
        # smooth focus on this specific camera+stage combo.
        _log("step-5", "=== 5a: three-way comparison (snap-motion, streaming-motion, stepped) ===")
        for range_um in METRIC_VALIDATION_RANGES_UM:
            csv_name = f"probez_metric_range{int(range_um)}_{timestamp}.csv"
            csv_path = log_dir / csv_name
            _metric_validate_one_range(
                core,
                focus_device,
                z0,
                range_um,
                speed_prop,
                MAXSPEED_VALUES[-1],
                csv_path,
                exposure_ms=original_exposure_ms or 0.0,
            )

        # 5b. Streaming-mode static stability check at multiple
        # exposures. The previous run at 16.86 ms showed streaming-
        # mode metric drift of ~25% over 800 ms. This run at 0.73 ms
        # showed SNAP-mode is stable. Now we need to know: at which
        # exposure does STREAMING become stable? That determines the
        # exposure ceiling for using streaming as the sample
        # mechanism (which is ~10x denser than snap on this JAI).
        _log("step-5", "=== 5b: streaming-mode stability across exposures ===")
        streaming_summary = []  # (exp_ms, mean, cv_pct, drift_pct, verdict)
        for exp_ms in EXPOSURE_SWEEP_MS:
            _log("step-5", f"--- streaming stability: set exposure = {exp_ms} ms ---")
            try:
                core.set_exposure(exp_ms)
            except Exception as e:
                _warn("step-5", f"set_exposure({exp_ms}) failed: {e}")
                continue
            result = _streaming_stability_at_exposure(core, focus_device, z0, exp_ms)
            if result is not None:
                streaming_summary.append((exp_ms, *result))

        # Summary table.
        if streaming_summary:
            _log("step-5", "=== Streaming-mode stability summary ===")
            _log("step-5", "exposure_ms   n_frames   mean_metric   CV_pct   drift_pct   verdict")
            for exp_ms, n, mean, cv, drift, verdict in streaming_summary:
                _log(
                    "step-5",
                    f"  {exp_ms:>6.1f}        {n:>5}    {mean:>8.3f}     "
                    f"{cv:>5.2f}    {drift:>6.2f}     {verdict}",
                )

    finally:
        # Always restore original exposure so the probe doesn't leave
        # the user staring at a different camera state than they
        # started with.
        if original_exposure_ms is not None:
            try:
                core.set_exposure(original_exposure_ms)
                _log("step-5", f"Restored camera exposure to {original_exposure_ms:.2f} ms")
            except Exception as e:
                _err("step-5", f"Failed to restore exposure: {e}")


def _streaming_stability_at_exposure(core, focus_device, z0, exposure_ms):
    """Start continuous sequence acquisition, pop ~30 frames at a
    constant Z, compute the focus metric on each, and return stats.

    Returns (n_frames, mean_metric, cv_pct, drift_pct, verdict_string)
    or None on failure. verdict is 'STABLE' / 'DRIFT' / 'NOISY' /
    'FAIL' based on CV and first/last-quarter drift thresholds.

    This is the same kind of stability check as the snap-mode
    _step5_static_stability, but exercised through the streaming
    path so we can see whether streaming-mode drift is exposure-
    dependent.
    """
    tag = f"step-5 stream-stable exp={exposure_ms}"

    # Make sure we're at z0 at full speed.
    _try_set_property(core, focus_device, "MaxSpeed", "100")
    try:
        core.set_position(focus_device, z0)
        core.wait_for_device(focus_device)
    except Exception as e:
        _err(tag, f"Failed to seat at z0: {e}")
        return None

    # Abort any running sequence.
    try:
        if core.is_sequence_running():
            _warn(tag, "Sequence already running; stopping first")
            core.stop_sequence_acquisition()
    except Exception:
        pass

    try:
        core.clear_circular_buffer()
        core.start_continuous_sequence_acquisition(0)
    except Exception as e:
        _err(tag, f"start_continuous_sequence_acquisition failed: {e}")
        return None

    metrics = []
    timestamps = []
    try:
        # Warm-up: wait 100 ms then drain any frames accumulated
        # before our loop starts. We WANT the potentially unstable
        # early frames for this test because they are exactly what
        # streaming-based smooth focus would see. Re-drain at the
        # same time so we can time the first real pop from a clean
        # starting point.
        time.sleep(0.1)
        try:
            while core.get_remaining_image_count() > 0:
                _pop_image_as_numpy(core)
        except Exception:
            pass

        t0 = time.perf_counter()
        deadline = t0 + 2.5
        while len(metrics) < 30 and time.perf_counter() < deadline:
            try:
                if core.get_remaining_image_count() > 0:
                    img = _pop_image_as_numpy(core)
                    m = _focus_metric(img)
                    t_ms = (time.perf_counter() - t0) * 1000.0
                    metrics.append(m)
                    timestamps.append(t_ms)
                    _log(
                        tag,
                        f"   stream snap {len(metrics) - 1:>2}  "
                        f"t={t_ms:>7.1f}ms  metric={m:.4f}",
                    )
                else:
                    time.sleep(0.003)
            except Exception as e:
                _warn(tag, f"pop failed: {e}")
                break
    finally:
        try:
            core.stop_sequence_acquisition()
        except Exception:
            pass
        try:
            core.clear_circular_buffer()
        except Exception:
            pass

    if len(metrics) < 5:
        _err(tag, f"only {len(metrics)} frames captured; insufficient for stats")
        return None

    mean = sum(metrics) / len(metrics)
    var = sum((m - mean) ** 2 for m in metrics) / len(metrics)
    std = var**0.5
    cv = (std / mean * 100.0) if mean > 1e-9 else float("inf")
    q = max(1, len(metrics) // 4)
    first_q = sum(metrics[:q]) / q
    last_q = sum(metrics[-q:]) / q
    drift = abs(last_q - first_q) / mean * 100.0 if mean > 1e-9 else 0.0

    if cv <= 5.0 and drift <= 5.0:
        verdict = "STABLE"
    elif cv > 5.0 and drift > 5.0:
        verdict = "DRIFT+NOISY"
    elif drift > 5.0:
        verdict = "DRIFT"
    else:
        verdict = "NOISY"

    _log(
        tag,
        f"n={len(metrics)}  mean={mean:.3f}  CV={cv:.2f}%  drift={drift:.2f}%  "
        f"verdict={verdict}",
    )
    return (len(metrics), mean, cv, drift, verdict)


def _estimate_min_velocity(core, focus_device, speed_prop, slow_speed, z0) -> float:
    """Time one 10 um move at slow_speed to derive a fresh velocity
    estimate for blur-budget reporting. Uses the new busy-poll wait
    path via wait_for_device (close enough for a one-off measurement).
    Falls back to a conservative Prior-MaxSpeed=1 value if anything
    goes wrong."""
    fallback = 11.5  # from PROBEZ second-run Step 3 data on PPM
    try:
        _try_set_property(core, focus_device, speed_prop, str(slow_speed))
        # Positioning at full speed, scan at slow speed.
        _try_set_property(core, focus_device, speed_prop, "100")
        core.set_position(focus_device, z0)
        core.wait_for_device(focus_device)
        _try_set_property(core, focus_device, speed_prop, str(slow_speed))
        t0 = time.perf_counter()
        core.set_position(focus_device, z0 + 10.0)
        core.wait_for_device(focus_device)
        elapsed = time.perf_counter() - t0
        velocity = 10.0 / elapsed if elapsed > 0 else fallback
        # Return to z0 at full speed.
        _try_set_property(core, focus_device, speed_prop, "100")
        core.set_position(focus_device, z0)
        core.wait_for_device(focus_device)
        return max(velocity, 1.0)
    except Exception as e:
        _warn("step-5", f"Velocity estimate failed, using fallback {fallback}: {e}")
        return fallback


def _step5_static_stability(core, focus_device: str, z0: float, n_frames: int = 20) -> bool:
    """Snap n_frames images at a constant Z and log the metric of
    each. Returns True if the metric is stable (CV < 5%), False if
    it drifts -- drift would invalidate every subsequent motion-based
    metric comparison, so we abort downstream tests if we can't trust
    the pipeline.
    """
    tag = "step-5 static"
    _log(tag, f"Static stability: {n_frames} snaps at z={z0:.3f} (no motion)")

    # Make sure we're at z0 and not on a bogus adapter speed.
    if not _try_set_property(core, focus_device, "MaxSpeed", "100"):
        _warn(tag, "Could not set MaxSpeed=100 for static prep; continuing")
    try:
        core.set_position(focus_device, z0)
        core.wait_for_device(focus_device)
    except Exception as e:
        _err(tag, f"Failed to seat at z0: {e}")
        return False

    metrics = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        try:
            core.snap_image()
            img = _snap_get_image_as_numpy(core)
            m = _focus_metric(img)
        except Exception as e:
            _warn(tag, f"snap {i} failed: {e}")
            continue
        t_ms = (time.perf_counter() - t0) * 1000.0
        metrics.append(m)
        _log(tag, f"   static snap {i:>2}  t={t_ms:>7.1f}ms  metric={m:.4f}")

    if len(metrics) < 3:
        _err(tag, f"only {len(metrics)} static snaps usable; aborting")
        return False

    mean = sum(metrics) / len(metrics)
    var = sum((m - mean) ** 2 for m in metrics) / len(metrics)
    std = var**0.5
    cv = (std / mean * 100.0) if mean > 1e-9 else float("inf")

    _log(
        tag,
        f"static metric: mean={mean:.4f}  std={std:.4f}  CV={cv:.2f}%  "
        f"min={min(metrics):.4f}  max={max(metrics):.4f}",
    )

    if cv > 5.0:
        _err(
            tag,
            f"static metric CV = {cv:.1f}% exceeds 5% threshold -- "
            "camera pipeline is not stable enough for motion-based focus",
        )
        return False

    # Additional check: is there a monotonic trend indicating drift?
    # Compare first quarter mean to last quarter mean.
    q = max(1, len(metrics) // 4)
    first_q = sum(metrics[:q]) / q
    last_q = sum(metrics[-q:]) / q
    drift_pct = abs(last_q - first_q) / mean * 100.0 if mean > 1e-9 else 0.0
    _log(
        tag,
        f"first-quarter mean={first_q:.4f}  last-quarter mean={last_q:.4f}  "
        f"drift={drift_pct:.2f}%",
    )

    if drift_pct > 5.0:
        _err(
            tag,
            f"static metric drift = {drift_pct:.1f}% across capture window -- "
            "camera has adaptive behavior that invalidates motion metrics",
        )
        return False

    _log(tag, "static stability PASSED -- metric pipeline is usable")
    return True


def _snap_get_image_as_numpy(core):
    """Pull the last snapped image out of MMCore as numpy. Handles
    mono and multi-component cameras."""
    pixels = core.get_image()
    if pixels is None:
        return None
    w = core.get_image_width()
    h = core.get_image_height()
    nch = core.get_number_of_components()
    arr = np.asarray(pixels)
    try:
        if nch == 1:
            return arr.reshape(h, w)
        return arr.reshape(h, w, nch)
    except Exception:
        return arr


def _metric_validate_one_range(
    core,
    focus_device: str,
    z0: float,
    range_um: float,
    speed_prop: str,
    slow_speed: int,
    csv_path: pathlib.Path,
    exposure_ms: float = 0.0,
):
    """Run snap-during-motion + stepped scans at one range and write a
    CSV comparing the two curves.

    Returns (delta_um_or_None, max_z_span_um, verdict_string) so the
    exposure-sweep driver can build a summary table. Returns None if
    the trial failed before producing usable samples.

    CSV columns:
        scan_type, t_ms, z_um, metric
    where scan_type is one of 'snap_motion' (new) or 'stepped'.

    Design notes motivated by the second probe run:

    - 'snap_motion' replaces the old 'continuous' streaming approach.
      We snap_image() in a tight loop while the stage is moving
      non-blocking. Each sample uses the same camera pipeline as
      snap-mode ground truth, avoiding the ~800 ms streaming
      startup transient that corrupted the previous run.

    - Z is bracketed per sample: we read get_position() before and
      after the snap so z_avg and z_span tell us exactly how far the
      stage moved during exposure (the motion-blur window).

    - Positioning moves (seed to z_start, return to z0) use MaxSpeed
      = 100, not slow_speed. At slow_speed the Prior left ~3 um of
      accumulated error between Step 4 trials; keeping positioning
      at full speed and only dropping to slow for the scan itself
      bounds the quirky behavior to the scan window.
    """
    tag = f"step-5 range={range_um:.1f}"
    half = range_um / 2.0
    z_start = z0 - half
    z_end = z0 + half

    _log(tag, f"Scan window [{z_start:.3f} -> {z_end:.3f}]  csv={csv_path.name}")

    rows = []  # (scan_type, t_ms, z_um, metric)
    snap_motion_samples = []  # (t_ms, z_avg, metric, z_span)
    stepped_samples = []  # (t_ms, z_actual, metric)

    # ---- Part 1: snap-during-motion at slow scan speed ----
    # Positioning move uses MaxSpeed=100 to avoid the Prior's
    # accumulated-error behavior at MaxSpeed=1.
    if not _try_set_property(core, focus_device, speed_prop, "100"):
        _warn(tag, "Could not restore MaxSpeed=100 for positioning")
    try:
        core.set_position(focus_device, z_start)
        core.wait_for_device(focus_device)
    except Exception as e:
        _err(tag, f"Seed move to {z_start:.3f} failed: {e}")
        return None

    # Now switch to slow speed for the actual scan.
    if not _try_set_property(core, focus_device, speed_prop, str(slow_speed)):
        _err(tag, f"Cannot set {speed_prop}={slow_speed}; skipping snap scan")
        return None

    t0 = time.perf_counter()
    try:
        core.set_position(focus_device, z_end)
    except Exception as e:
        _err(tag, f"Non-blocking move to z_end failed: {e}")
        _try_set_property(core, focus_device, speed_prop, "100")
        return None

    # Snap loop: run until stage not-busy or hard deadline.
    # Deadline is generous at the slow speed (~25 ms/um plus setup).
    deadline = time.perf_counter() + (range_um * 0.12 + 2.0)
    consecutive_not_busy = 0
    try:
        while time.perf_counter() < deadline:
            try:
                z_before = core.get_position(focus_device)
            except Exception:
                z_before = float("nan")
            try:
                core.snap_image()
                img = _snap_get_image_as_numpy(core)
            except Exception as e:
                _warn(tag, f"snap during motion failed: {e}")
                break
            try:
                z_after = core.get_position(focus_device)
            except Exception:
                z_after = float("nan")

            t_ms = (time.perf_counter() - t0) * 1000.0
            metric = _focus_metric(img)
            if z_before == z_before and z_after == z_after:
                z_avg = (z_before + z_after) / 2.0
                z_span = abs(z_after - z_before)
            else:
                z_avg = z_after if z_after == z_after else z_before
                z_span = float("nan")
            snap_motion_samples.append((t_ms, z_avg, metric, z_span))

            # Check if we've stopped. Need two consecutive not-busy
            # reads so a fleeting ready-signal doesn't exit early.
            try:
                busy = core.device_busy(focus_device)
            except Exception:
                busy = None
            if busy is False:
                consecutive_not_busy += 1
                if consecutive_not_busy >= 2:
                    break
            else:
                consecutive_not_busy = 0
    finally:
        # Restore MaxSpeed=100 for the stepped-sweep phase.
        _try_set_property(core, focus_device, speed_prop, "100")
        try:
            core.wait_for_device(focus_device)
        except Exception:
            pass

    # ---- Part 2: stepped sweep at full speed (ground truth) ----
    step_size = range_um / (METRIC_STEPPED_N_STEPS - 1)
    try:
        t0_step = time.perf_counter()
        for i in range(METRIC_STEPPED_N_STEPS):
            z = z_start + i * step_size
            try:
                core.set_position(focus_device, z)
                core.wait_for_device(focus_device)
                core.snap_image()
                img = _snap_get_image_as_numpy(core)
                metric = _focus_metric(img)
                t_ms = (time.perf_counter() - t0_step) * 1000.0
                z_actual = core.get_position(focus_device)
                stepped_samples.append((t_ms, z_actual, metric))
            except Exception as e:
                _warn(tag, f"stepped point i={i} failed: {e}")
    except Exception as e:
        _err(tag, f"stepped sweep failed: {e}")

    # Return to z0 cleanly.
    try:
        core.set_position(focus_device, z0)
        core.wait_for_device(focus_device)
    except Exception as e:
        _warn(tag, f"Return to z0 failed: {e}")

    # ---- Analysis + CSV ----
    # For snap-motion samples, the argmax uses z_avg (column 1) and
    # metric (column 2).
    snap_peak = _argmax_z_generic([(s[1], s[2]) for s in snap_motion_samples])
    step_peak = _argmax_z_generic([(s[1], s[2]) for s in stepped_samples])
    delta = (snap_peak - step_peak) if (snap_peak is not None and step_peak is not None) else None

    # Max z_span during motion snaps -- tells us how much the stage
    # moved during a single exposure (motion-blur window).
    valid_spans = [s[3] for s in snap_motion_samples if s[3] == s[3]]  # NaN-safe
    max_span = max(valid_spans) if valid_spans else float("nan")
    avg_span = sum(valid_spans) / len(valid_spans) if valid_spans else float("nan")

    _log(
        tag,
        f"snap_motion: {len(snap_motion_samples)} samples, peak Z={_fmt(snap_peak)}  "
        f"stepped: {len(stepped_samples)} samples, peak Z={_fmt(step_peak)}  "
        f"delta={_fmt(delta)} um",
    )
    _log(
        tag,
        f"per-exposure z span: avg={avg_span:.3f} um  max={max_span:.3f} um "
        f"(motion-blur window per snap)",
    )

    verdict = "no_data"
    if delta is not None:
        if abs(delta) <= 0.5:
            verdict = "GOOD"
            _log(tag, "VERDICT: snap-motion peak matches stepped within 0.5 um -- metric is viable")
        elif abs(delta) <= 1.0:
            verdict = "BORDERLINE"
            _log(tag, "VERDICT: snap-motion peak within 1 um of stepped -- borderline, inspect CSV")
        else:
            verdict = "BAD"
            _warn(
                tag,
                f"VERDICT: snap-motion peak off stepped by {delta:.2f} um -- "
                "check CSV for shape; may need slower speed or wider range",
            )

    for t_ms, z_avg, m, z_span in snap_motion_samples:
        rows.append(("snap_motion", t_ms, z_avg, m, z_span))
        _log(tag, f"   SNAP t={t_ms:>7.1f}ms  z={z_avg:.3f}  metric={m:.2f}  span={z_span:.3f}")
    for t_ms, z, m in stepped_samples:
        rows.append(("stepped", t_ms, z, m, 0.0))
        _log(tag, f"   STEP t={t_ms:>7.1f}ms  z={z:.3f}  metric={m:.2f}")

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scan_type", "t_ms", "z_um", "metric", "z_span_um", "exposure_ms"])
            for row in rows:
                w.writerow(
                    [
                        row[0],
                        f"{row[1]:.3f}",
                        f"{row[2]:.3f}",
                        f"{row[3]:.6f}",
                        f"{row[4]:.3f}",
                        f"{exposure_ms:.2f}",
                    ]
                )
        _log(tag, f"Wrote CSV: {csv_path}")
    except Exception as e:
        _err(tag, f"CSV write failed: {e}")

    return (delta, max_span, verdict)


def _argmax_z_generic(zm_pairs):
    """Return the Z with the maximum metric from a list of (z, metric)
    pairs, or None if empty/NaN.

    Uses a simple argmax rather than a parabolic fit for this
    validation pass -- we just want to know 'do snap-motion and
    stepped agree on where the peak is'. A parabolic fit would
    smooth out real disagreements.
    """
    best_z = None
    best_m = float("-inf")
    for z, m in zm_pairs:
        if z != z or m != m:  # NaN-safe
            continue
        if m > best_m:
            best_m = m
            best_z = z
    return best_z


def _fmt(v):
    if v is None:
        return "None"
    try:
        return f"{v:.3f}"
    except Exception:
        return str(v)


def handle_probez(conn, client, hardware, settings, **kwargs):
    """PROBEZ -- one-shot Z-stage diagnostic probe.

    No payload. Runs Steps 0-4 (see module docstring), logs every
    observation with the "PROBEZ [step-N]:" tag, and responds with
    PROBEZ_OK or PROBEZ_FAIL. All hardware state is restored in a
    finally block.

    Intended for exploratory characterization of the focus device on
    a given rig. Safe to run multiple times; not safe to run while
    an acquisition or other client operation is in progress -- the
    server already serializes commands per-client, but this handler
    also checks core.is_sequence_running() defensively.
    """
    addr = getattr(client, "addr", client)
    _log("start", f"PROBEZ requested by {addr}")

    # Safety: refuse if server isn't configured. We want the session
    # log to be writing to the real microscope-specific log dir.
    server_configured = kwargs.get("server_configured", False)
    if not server_configured:
        _err("start", "Server not configured; send CONFIG first")
        try:
            conn.sendall(b"PROBEZFL")
        except Exception:
            pass
        return

    core = hardware.core
    restore = None

    try:
        # Defensive: refuse if a sequence is already running.
        try:
            if core.is_sequence_running():
                _err("start", "Sequence acquisition already running; aborting")
                conn.sendall(b"PROBEZFL")
                return
        except Exception as e:
            _warn("start", f"is_sequence_running check failed: {e}")

        # Step 0: snapshot device state. This is also our restore dict.
        restore = _snapshot_focus_device(core, "step-0")
        focus_device = restore.get("__focus_device__")
        z0 = restore.get("__z_original__")
        if not focus_device or z0 is None:
            _err("step-0", "Could not obtain focus device or Z0; aborting")
            conn.sendall(b"PROBEZFL")
            return

        # Step 1: blocking round-trip timing.
        _step1_blocking_move_timing(core, focus_device, z0)

        # Step 2: non-blocking issue + position readback.
        _step2_nonblocking_position_readback(core, focus_device, z0)

        # Pick the speed property name for Step 3. Prior ProScan calls
        # it "MaxSpeed"; some adapters use "SpeedX" or "Velocity". We
        # probe by checking which of the common names exists among the
        # captured property list.
        speed_prop = None
        for candidate in ("MaxSpeed", "Velocity", "Speed", "MaxVelocity"):
            if candidate in restore:
                speed_prop = candidate
                break
        if speed_prop is None:
            _warn(
                "step-3",
                "No MaxSpeed/Velocity/Speed/MaxVelocity property found on focus "
                "device; skipping steps 3, 4, 5",
            )
        else:
            _step3_maxspeed_sensitivity(core, focus_device, z0, speed_prop)
            _step4_stream_during_motion(core, focus_device, z0, speed_prop)

            # Step 5: pixel + metric validation. Writes a CSV per
            # range into the same directory as the active session
            # log so it's easy to find.
            active_config_path = kwargs.get("active_connection_config_path")
            if active_config_path:
                csv_dir = pathlib.Path(active_config_path).resolve().parent / "logs"
            else:
                csv_dir = pathlib.Path.cwd()
            _step5_metric_validation(core, focus_device, z0, speed_prop, csv_dir)

        _log("done", "Probe completed successfully")
        try:
            conn.sendall(b"PROBEZOK")
        except Exception:
            pass

    except Exception as e:
        _err("error", f"Probe raised: {e}", exc_info=True)
        try:
            conn.sendall(b"PROBEZFL")
        except Exception:
            pass
    finally:
        if restore is not None:
            _restore_focus_device(core, restore)
