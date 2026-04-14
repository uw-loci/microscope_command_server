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

import logging
import time

logger = logging.getLogger(__name__)


# Ranges to exercise in Step 4 (streaming-during-motion).
# Chosen to bracket the current sweep_range_um=6.0 setting and the
# plausible "wider, slower" operating points we might migrate to.
STREAM_RANGES_UM = [2.0, 6.0, 12.0, 20.0]

# MaxSpeed values to test in Step 3. Prior ProScan MaxSpeed is on a
# 0-100 percent scale; we descend from full speed to slow to find the
# point where a 20 um move takes ~500-1000 ms (a realistic streaming
# target at 38 fps).
MAXSPEED_VALUES = [100, 50, 25, 10, 5]

# Relative move sizes in um for Steps 1 and 2.
MOVE_SIZES_UM = [1.0, 5.0, 10.0, 20.0, 50.0]

# Safety: don't do streaming scans wider than this in a probe run
# (the probe moves the stage, not the user). 25 um is well inside any
# realistic Z limit envelope but still larger than STREAM_RANGES_UM[-1].
MAX_STREAM_RANGE_UM = 25.0


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
        prop_names = list(core.get_device_property_names(focus_device))
    except Exception as e:
        _err(step, f"get_device_property_names({focus_device}) failed: {e}")
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
            allowed = list(core.get_allowed_property_values(focus_device, name))
        except Exception:
            allowed = []
        allowed_str = f" allowed={allowed}" if allowed else ""

        ro_str = "RO" if read_only else "RW"
        _log(step, f"  {name} = {value!r} [{ro_str}]{limits_str}{allowed_str}")

        if read_only is False:
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
            core.set_property(focus_device, name, value)
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


def _step3_maxspeed_sensitivity(
    core, focus_device: str, z0: float, speed_prop: str
) -> None:
    """Measure a 20 um blocking move round-trip across the MaxSpeed
    range. The goal is to find a (speed, move_time) operating point
    where a ~6 um sweep takes long enough for the camera to produce
    ~15-30 frames at 38 fps (0.4-0.8 seconds)."""
    _log("step-3", f"MaxSpeed sensitivity sweep via property '{speed_prop}'")
    dz = 20.0
    for speed in MAXSPEED_VALUES:
        if not _try_set_property(core, focus_device, speed_prop, speed):
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


def _step4_stream_during_motion(
    core, focus_device: str, z0: float, speed_prop: str
) -> None:
    """The critical feasibility test.

    At a slow MaxSpeed (so the move takes hundreds of ms), start a
    continuous sequence acquisition, issue a non-blocking move, and
    pop frames from the circular buffer as fast as possible while
    recording (pop_time, core.get_position()) for each. Repeat for
    several range values so we can see how frame count scales with
    range and whether sample density is adequate.

    The camera exposure is left at whatever MM is currently set to.
    Users of the probe should set an appropriate exposure via the
    normal camera-control UI before running.
    """
    # Pick the slowest speed from Step 3 that actually produced a
    # measurable move, then use it for all streaming runs. "5" matches
    # the slowest MAXSPEED_VALUES entry and is the most likely to give
    # usable streaming duration on a Prior ProScan.
    slow_speed = MAXSPEED_VALUES[-1]
    if not _try_set_property(core, focus_device, speed_prop, slow_speed):
        _err("step-4", f"Cannot set {speed_prop}={slow_speed}; skipping streaming test")
        return
    _log("step-4", f"Using {speed_prop}={slow_speed} for streaming runs")

    # Report current exposure for context.
    try:
        exposure_ms = core.get_exposure()
        _log("step-4", f"Camera exposure (current) = {exposure_ms:.2f} ms")
    except Exception as e:
        _log("step-4", f"Could not query exposure: {e}")

    for range_um in STREAM_RANGES_UM:
        if range_um > MAX_STREAM_RANGE_UM:
            _warn("step-4", f"Skipping range={range_um} (exceeds MAX_STREAM_RANGE_UM)")
            continue
        _stream_one_range(core, focus_device, z0, range_um)


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

        # Pull frames + Z + busy until the stage reports done, plus a
        # short tail so we can see post-motion frames for sanity.
        samples = []  # (t_ms, z_um, is_busy, frame_seen)
        tail_frames = 0
        stage_done_at = None
        deadline = time.perf_counter() + 5.0  # hard cap

        while time.perf_counter() < deadline:
            t_rel = (time.perf_counter() - t0) * 1000.0

            # Pop one frame if available. We don't look at pixels -- the
            # only thing that matters for this probe is that the pop
            # succeeded and we logged the Z at that moment.
            frame_this_tick = False
            try:
                if core.get_remaining_image_count() > 0:
                    core.pop_next_image()
                    frame_this_tick = True
            except Exception as e:
                _warn(tag, f"pop_next_image failed: {e}")

            try:
                z_now = core.get_position(focus_device)
            except Exception:
                z_now = float("nan")

            try:
                busy = core.device_busy(focus_device)
            except Exception:
                busy = None

            samples.append((t_rel, z_now, busy, frame_this_tick))

            if busy is False and stage_done_at is None:
                stage_done_at = t_rel
                # Keep pulling frames for a short tail so we see any
                # drift between frame arrival and stage arrival.
                tail_frames = 0

            if stage_done_at is not None:
                tail_frames += 1
                if tail_frames >= 10:
                    break

            # Tight-ish sampling rate. Too fast hammers the serial
            # layer; too slow misses frames. 3 ms is a good compromise
            # between responsiveness and overhead.
            time.sleep(0.003)

        # Make absolutely sure the stage is done before we continue.
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
    total_frames = sum(1 for s in samples if s[3])
    in_motion_frames = sum(
        1 for s in samples if s[3] and (stage_done_at is None or s[0] <= stage_done_at)
    )
    if samples:
        z_min = min(s[1] for s in samples if s[1] == s[1])  # NaN-safe
        z_max = max(s[1] for s in samples if s[1] == s[1])
        z_span = z_max - z_min
    else:
        z_min = z_max = z_span = float("nan")

    _log(
        tag,
        f"done: stage_done_at={stage_done_at}ms  frames_total={total_frames}  "
        f"frames_in_motion={in_motion_frames}  "
        f"z_reported_span={z_span:.3f} um  z_final={z_final:.3f}",
    )

    # Log a compact trace: every 3rd sample up to the first 30.
    trace_samples = samples[::3][:30]
    for t_rel, z_now, busy, frame in trace_samples:
        b = "B" if busy else ("I" if busy is False else "?")
        f = "F" if frame else "."
        _log(tag, f"   t={t_rel:>6.1f}ms  z={z_now:.3f}  {b}{f}")


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
                "device; skipping steps 3 and 4",
            )
        else:
            _step3_maxspeed_sensitivity(core, focus_device, z0, speed_prop)
            _step4_stream_during_motion(core, focus_device, z0, speed_prop)

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
