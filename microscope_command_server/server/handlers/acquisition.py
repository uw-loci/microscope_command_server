"""Acquisition command handlers.

Handles image acquisition commands:
ACQUIRE, BGACQUIRE, ZSTACK, TLAPSE

CRITICAL: The ACQUIRE handler spawns a background acquisition_thread that
references global acquisition state dicts (acquisition_states, acquisition_progress,
etc.). These are passed via **kwargs for now.
TODO: Migrate to a proper AcquisitionManager class that encapsulates this state.
"""

import socket
import pathlib
import threading
import time
import logging

from microscope_command_server.server.protocol import END_MARKER
from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)


def handle_acquire(conn, client, hardware, settings, **kwargs):
    """Main acquisition workflow (spawns background thread).

    Reads the acquisition message, clears cancellation event, and
    spawns a daemon thread running acquisitionWorkflow. Sends a
    STARTED acknowledgment immediately.

    CRITICAL: This handler spawns an acquisition_thread. The thread
    function (acquisitionWorkflow) references global state dicts passed
    via kwargs. The caller must store the returned thread reference
    for cleanup on disconnect.

    Required kwargs:
        addr: Client address tuple
        acquisition_locks: Dict[addr -> Lock]
        acquisition_states: Dict[addr -> AcquisitionState]
        acquisition_progress: Dict[addr -> (current, total)]
        acquisition_cancel_events: Dict[addr -> Event]
        AcquisitionState: The AcquisitionState enum class
        acquisitionWorkflow: The workflow function to run in the thread

    Returns:
        The started acquisition Thread, or None if startup failed.
        The caller should store this for cleanup.
    """
    addr = kwargs["addr"]
    acquisition_locks = kwargs["acquisition_locks"]
    acquisition_states = kwargs["acquisition_states"]
    acquisition_progress = kwargs["acquisition_progress"]
    acquisition_cancel_events = kwargs["acquisition_cancel_events"]
    AcquisitionState = kwargs["AcquisitionState"]
    acquisitionWorkflow = kwargs["acquisitionWorkflow"]

    logger.info("Client %s requested acquisition workflow", addr)

    # Check if already running
    with acquisition_locks[addr]:
        if acquisition_states[addr] == AcquisitionState.RUNNING:
            logger.warning("Acquisition already running for %s", addr)
            return None
        # Set state to RUNNING immediately
        acquisition_states[addr] = AcquisitionState.RUNNING
        acquisition_progress[addr] = (0, 0)

    # Read the full message immediately
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    # Set a timeout for reading
    conn.settimeout(5.0)

    acquisition_thread = None

    try:
        while True:
            # Read in chunks
            chunk = conn.recv(1024)
            if not chunk:
                logger.error(
                    "Connection closed while reading acquisition message from %s",
                    addr,
                )
                with acquisition_locks[addr]:
                    acquisition_states[addr] = AcquisitionState.FAILED
                break

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)

            # Check if we have the end marker
            full_message = "".join(message_parts)
            if END_MARKER in full_message:
                # Remove the end marker
                message = full_message.replace("," + END_MARKER, "").replace(END_MARKER, "")
                logger.debug(
                    "Received complete acquisition message (%d bytes) " "in %.2fs",
                    total_bytes,
                    time.time() - start_time,
                )

                # Clear cancellation event
                acquisition_cancel_events[addr].clear()

                # Start acquisition in separate thread
                acquisition_thread = threading.Thread(
                    target=acquisitionWorkflow,
                    args=(message, addr),
                    daemon=True,
                    name=f"Acquisition-{addr}",
                )
                acquisition_thread.start()

                logger.info("Acquisition thread started for %s", addr)

                # Send acknowledgment to prevent client timeout
                # Format matches BGACQUIRE pattern for consistency
                ack_response = "STARTED:ACQUIRE".ljust(16)[:16].encode()
                conn.sendall(ack_response)
                logger.debug("Sent ACQUIRE acknowledgment to %s", addr)
                break

            # Safety check for message size
            if total_bytes > 10000:  # 10KB max
                logger.error(
                    "Acquisition message too large from %s: %d bytes",
                    addr,
                    total_bytes,
                )
                with acquisition_locks[addr]:
                    acquisition_states[addr] = AcquisitionState.FAILED
                break

            # Timeout check
            if time.time() - start_time > 10:
                logger.error("Timeout reading acquisition message from %s", addr)
                with acquisition_locks[addr]:
                    acquisition_states[addr] = AcquisitionState.FAILED
                break

    except socket.timeout:
        logger.error("Socket timeout reading acquisition message from %s", addr)
        with acquisition_locks[addr]:
            acquisition_states[addr] = AcquisitionState.FAILED
    except Exception as e:
        logger.error("Error reading acquisition message from %s: %s", addr, e)
        with acquisition_locks[addr]:
            acquisition_states[addr] = AcquisitionState.FAILED
    finally:
        # Reset socket to blocking mode
        conn.settimeout(None)

    return acquisition_thread


def handle_bgacquire(conn, client, hardware, settings, **kwargs):
    """Background acquisition (synchronous, blocking).

    Reads the background acquisition message with flags for --yaml,
    --output, --modality, --angles, --exposures, --wb-mode,
    --objective, --detector, and optional --use_per_angle_wb boolean.

    Executes simple_background_collection synchronously and returns
    the final exposures.

    Response: STARTED:<output_path> then SUCCESS:<output_path>|<exposures>
              or FAILED:<reason>
    """
    addr = kwargs.get(
        "addr", client if isinstance(client, tuple) else getattr(client, "addr", client)
    )
    config_manager = kwargs.get("config_manager")
    acquisition_locks = kwargs.get("acquisition_locks", {})
    acquisition_progress = kwargs.get("acquisition_progress", {})
    active_connection_config_path = kwargs.get("active_connection_config_path")

    logger.info("Client %s requested background acquisition", addr)

    # Read the message using the same pattern as ACQUIRE command
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    conn.settimeout(5.0)

    try:
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                logger.error("Connection closed while reading background acquisition message")
                conn.sendall(b"FAILED:Connection closed")
                break

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)

            full_message = "".join(message_parts)

            if END_MARKER in full_message:
                message = full_message.replace(END_MARKER, "").strip()

                # Parse the message
                params = {}

                # Check for boolean flags first (no value)
                use_per_angle_wb = "--use_per_angle_wb" in message

                # Split by known flags to avoid issues with spaces in paths
                # Include --wb-mode as a valued flag
                flags = [
                    "--yaml",
                    "--output",
                    "--modality",
                    "--angles",
                    "--exposures",
                    "--wb-mode",
                    "--objective",
                    "--detector",
                    "--target-intensity",
                    "--profile",
                    "--channels",
                ]

                for i, flag in enumerate(flags):
                    if flag in message:
                        # Find where this flag starts
                        start_idx = message.index(flag) + len(flag)

                        # Find where the next flag starts (or use end of string)
                        end_idx = len(message)
                        for next_flag in flags[i + 1 :]:
                            if next_flag in message[start_idx:]:
                                next_pos = message.index(next_flag, start_idx)
                                if next_pos < end_idx:
                                    end_idx = next_pos
                                    break
                        # Also check for boolean flag --use_per_angle_wb
                        if "--use_per_angle_wb" in message[start_idx:]:
                            wb_pos = message.index("--use_per_angle_wb", start_idx)
                            if wb_pos < end_idx:
                                end_idx = wb_pos

                        # Extract the value and clean it up
                        value = message[start_idx:end_idx].strip()

                        # Map to the parameter name
                        if flag == "--yaml":
                            params["yaml_file_path"] = value
                        elif flag == "--output":
                            params["output_folder_path"] = value
                        elif flag == "--modality":
                            params["modality"] = value
                        elif flag == "--angles":
                            params["angles_str"] = value
                        elif flag == "--exposures":
                            params["exposures_str"] = value
                        elif flag == "--wb-mode":
                            params["wb_mode"] = value.lower()
                        elif flag == "--objective":
                            params["objective"] = value
                        elif flag == "--detector":
                            params["detector"] = value
                        elif flag == "--target-intensity":
                            try:
                                params["target_intensity"] = float(value)
                            except ValueError:
                                logger.warning("Invalid --target-intensity value: %s", value)
                        elif flag == "--profile":
                            params["profile"] = value
                        elif flag == "--channels":
                            params["channels"] = [
                                c.strip() for c in value.split(",") if c.strip()
                            ]

                # Resolve wb_mode: prefer explicit --wb-mode, fall back to boolean flag
                if "wb_mode" in params:
                    logger.info("WB mode for background acquisition: %s", params["wb_mode"])
                elif use_per_angle_wb:
                    params["wb_mode"] = "per_angle"
                    logger.info(
                        "Per-angle white balance enabled for background acquisition (legacy flag)"
                    )
                # If neither --wb-mode nor --use_per_angle_wb, leave wb_mode unset
                # and let simple_background_collection use its default

                # Keep legacy flag for backward compat
                params["use_per_angle_wb"] = use_per_angle_wb

                # Validate required parameters
                required = ["yaml_file_path", "output_folder_path", "modality"]
                missing = [key for key in required if key not in params]
                if missing:
                    error_msg = f"Missing required parameters: {missing}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                    break

                # SAFETY WARNING: Check if ACQUIRE yaml differs from CONFIG
                if active_connection_config_path:
                    acquire_yaml = pathlib.Path(params["yaml_file_path"]).resolve()
                    connection_yaml = pathlib.Path(active_connection_config_path).resolve()
                    if acquire_yaml != connection_yaml:
                        logger.warning("=" * 80)
                        logger.warning("CONFIG MISMATCH WARNING")
                        logger.warning("Connection CONFIG:  %s", connection_yaml)
                        logger.warning("ACQUIRE --yaml:     %s", acquire_yaml)
                        logger.warning(
                            "ACQUIRE yaml will override connection config for this acquisition"
                        )
                        logger.warning(
                            "This may cause unexpected behavior or hardware misconfiguration!"
                        )
                        logger.warning("=" * 80)

                # Send immediate acknowledgment to prevent client timeout
                try:
                    ack_response = f"STARTED:{params['output_folder_path']}".encode()
                    conn.sendall(ack_response)
                    logger.info("Sent STARTED acknowledgment for background acquisition")

                    # Execute background acquisition using simplified collection
                    from microscope_command_server.acquisition.workflow import (
                        simple_background_collection,
                    )

                    # Create progress update function for this client
                    def update_progress(current, total):
                        if addr in acquisition_locks:
                            with acquisition_locks[addr]:
                                acquisition_progress[addr] = (current, total)

                    bg_result = simple_background_collection(
                        yaml_file_path=params["yaml_file_path"],
                        output_folder_path=params["output_folder_path"],
                        modality=params["modality"],
                        angles_str=params.get("angles_str", "()"),
                        exposures_str=params.get("exposures_str", "()"),
                        hardware=hardware,
                        config_manager=config_manager,
                        logger=logger,
                        update_progress=update_progress,
                        use_per_angle_wb=params.get("use_per_angle_wb", False),
                        wb_mode=params.get("wb_mode"),
                        objective=params.get("objective"),
                        detector=params.get("detector"),
                        target_intensity_override=params.get("target_intensity"),
                        profile=params.get("profile"),
                        channels=params.get("channels"),
                    )
                    final_exposures = bg_result.get("final_exposures", {})
                    applied_lamp = bg_result.get("applied_lamp_intensity")
                    lamp_device = bg_result.get("lamp_device_label")
                    resolved_profile = bg_result.get("resolved_profile")
                    channel_intensities = bg_result.get("channel_intensities") or {}

                    # Format exposures as angle:exposure pairs over the wire
                    # (e.g. "90:137.1,7:245.8,-7:155.2"). Wire format is fixed
                    # because the Java client parses it; only the diagnostic
                    # log line varies between rotation and non-rotation.
                    exposures_formatted = ",".join(
                        f"{angle}:{exposure:.2f}"
                        for angle, exposure in sorted(final_exposures.items())
                    )

                    # Third pipe-field: lamp/device/profile metadata. Java tolerates
                    # its absence (old server), so new clients always get it.
                    lamp_str = f"{applied_lamp:.2f}" if applied_lamp is not None else "none"
                    device_str = lamp_device if lamp_device else "none"
                    profile_str = resolved_profile if resolved_profile else "none"
                    meta = f"lamp={lamp_str};device={device_str};profile={profile_str}"
                    if channel_intensities:
                        chint = ",".join(
                            f"{cid}:{val:.2f}" for cid, val in channel_intensities.items()
                        )
                        meta += f";chint={chint}"

                    # Send success response with output path, exposures, and metadata
                    response = (
                        f"SUCCESS:{params['output_folder_path']}"
                        f"|{exposures_formatted}|{meta}".encode()
                    )
                    conn.sendall(response)
                    requested_angles = params.get("angles_str", "").strip()
                    is_non_rotation = not requested_angles or requested_angles in (
                        "()",
                        "(0)",
                        "(0.0)",
                    )
                    if is_non_rotation and len(final_exposures) == 1:
                        only_exposure = next(iter(final_exposures.values()))
                        logger.info(
                            "Background acquisition completed successfully (exposure=%.2fms)",
                            only_exposure,
                        )
                    else:
                        logger.info(
                            "Background acquisition completed successfully with exposures: %s",
                            exposures_formatted,
                        )

                except Exception as e:
                    logger.error("Background acquisition failed: %s", str(e), exc_info=True)
                    response = f"FAILED:{str(e)}".encode()
                    conn.sendall(response)

                # We found and processed the END_MARKER, so break the while loop
                break

            # Safety checks for the while loop
            if total_bytes > 10000:  # 10KB max
                logger.error(
                    "Background acquisition message too large: %d bytes",
                    total_bytes,
                )
                conn.sendall(b"FAILED:Message too large")
                break

            if time.time() - start_time > 10:
                logger.error("Timeout reading background acquisition message")
                conn.sendall(b"FAILED:Timeout waiting for complete message")
                break

    except socket.timeout:
        logger.error("Timeout reading background acquisition message from %s", addr)
        conn.sendall(b"FAILED:Timeout reading message")
    except Exception as e:
        logger.error("Error in background acquisition: %s", str(e), exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
    finally:
        conn.settimeout(None)  # Reset to blocking mode


def handle_zstack(conn, client, hardware, settings, **kwargs):
    """Z-stack acquisition at current XY position.

    Reads --output, --z-start, --z-end, --z-step, --modality,
    --angles, --wb-mode, --yaml, --objective, --detector flags.

    Response: SUCCESS:<output>|planes:<n>|files:<n>|elapsed:<s>
              or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = kwargs.get(
        "addr", client if isinstance(client, tuple) else getattr(client, "addr", client)
    )
    logger.info("Client %s requested Z-stack acquisition", addr)

    try:
        message = read_message_string(conn, chunk_size=4096)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read ZSTACK message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    logger.info("ZSTACK message: %s", message)

    # Parse parameters: --output --z-start --z-end --z-step
    #   --modality --angles --wb-mode --yaml --objective --detector
    params = {}
    parts = message.split()
    i = 0
    while i < len(parts):
        if parts[i] == "--output" and i + 1 < len(parts):
            params["output"] = parts[i + 1]
            i += 2
        elif parts[i] == "--z-start" and i + 1 < len(parts):
            params["z_start"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--z-end" and i + 1 < len(parts):
            params["z_end"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--z-step" and i + 1 < len(parts):
            params["z_step"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--modality" and i + 1 < len(parts):
            params["modality"] = parts[i + 1]
            i += 2
        elif parts[i] == "--angles" and i + 1 < len(parts):
            params["angles"] = parts[i + 1]
            i += 2
        elif parts[i] == "--wb-mode" and i + 1 < len(parts):
            params["wb_mode"] = parts[i + 1]
            i += 2
        elif parts[i] == "--yaml" and i + 1 < len(parts):
            params["yaml"] = parts[i + 1]
            i += 2
        elif parts[i] == "--objective" and i + 1 < len(parts):
            params["objective"] = parts[i + 1]
            i += 2
        elif parts[i] == "--detector" and i + 1 < len(parts):
            params["detector"] = parts[i + 1]
            i += 2
        elif parts[i] == "--projection" and i + 1 < len(parts):
            params["projection"] = parts[i + 1]
            i += 2
        elif parts[i] == "--bg-correction" and i + 1 < len(parts):
            params["bg_correction"] = parts[i + 1].lower() == "true"
            i += 2
        elif parts[i] == "--bg-folder" and i + 1 < len(parts):
            params["bg_folder"] = parts[i + 1]
            i += 2
        elif parts[i] == "--bg-method" and i + 1 < len(parts):
            params["bg_method"] = parts[i + 1]
            i += 2
        elif parts[i] == "--timepoints" and i + 1 < len(parts):
            params["timepoints"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "--interval" and i + 1 < len(parts):
            params["interval"] = float(parts[i + 1])
            i += 2
        else:
            i += 1

    # Validate required params
    for req in ["output", "z_start", "z_end", "z_step"]:
        if req not in params:
            conn.sendall(f"FAILED:Missing --{req.replace('_', '-')}".encode())
            return

    conn.sendall(f"STARTED:{params['output']}".encode())
    try:
        from microscope_command_server.acquisition.stack_timelapse import acquire_z_stack

        result = acquire_z_stack(
            hardware=hardware,
            output_folder=params["output"],
            z_start=params["z_start"],
            z_end=params["z_end"],
            z_step=params["z_step"],
            modality=params.get("modality", "brightfield"),
            angles_str=params.get("angles", "(0)"),
            config_manager=config_manager,
            wb_mode=params.get("wb_mode", "off"),
            objective=params.get("objective"),
            detector=params.get("detector"),
            yaml_file_path=params.get("yaml"),
            projection=params.get("projection", "none"),
            background_correction_enabled=params.get("bg_correction", False),
            background_folder=params.get("bg_folder"),
            background_correction_method=params.get("bg_method", "divide"),
            n_timepoints=params.get("timepoints", 1),
            interval_seconds=params.get("interval", 0.0),
        )
        response = (
            f"SUCCESS:{params['output']}|"
            f"planes:{result['n_planes']}|"
            f"timepoints:{result.get('n_timepoints', 1)}|"
            f"files:{len(result['files'])}|"
            f"elapsed:{result['elapsed_seconds']:.1f}s"
        )
        conn.sendall(response.encode())
        logger.info(
            "ZSTACK complete: T=%d, Z=%d",
            result.get("n_timepoints", 1),
            result["n_planes"],
        )
    except Exception as e:
        logger.error("ZSTACK failed: %s", e, exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())


def handle_tlapse(conn, client, hardware, settings, **kwargs):
    """Time-lapse acquisition at current position.

    Reads --output, --timepoints, --interval, --modality, --angles,
    --wb-mode, --yaml, --objective, --detector flags.

    Response: SUCCESS:<output>|timepoints:<n>|files:<n>|elapsed:<s>
              or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = kwargs.get(
        "addr", client if isinstance(client, tuple) else getattr(client, "addr", client)
    )
    logger.info("Client %s requested time-lapse acquisition", addr)

    try:
        message = read_message_string(conn, chunk_size=4096)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read TLAPSE message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    logger.info("TLAPSE message: %s", message)

    # Parse parameters: --output --timepoints --interval
    #   --modality --angles --wb-mode --yaml --objective --detector
    params = {}
    parts = message.split()
    i = 0
    while i < len(parts):
        if parts[i] == "--output" and i + 1 < len(parts):
            params["output"] = parts[i + 1]
            i += 2
        elif parts[i] == "--timepoints" and i + 1 < len(parts):
            params["timepoints"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "--interval" and i + 1 < len(parts):
            params["interval"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--modality" and i + 1 < len(parts):
            params["modality"] = parts[i + 1]
            i += 2
        elif parts[i] == "--angles" and i + 1 < len(parts):
            params["angles"] = parts[i + 1]
            i += 2
        elif parts[i] == "--wb-mode" and i + 1 < len(parts):
            params["wb_mode"] = parts[i + 1]
            i += 2
        elif parts[i] == "--yaml" and i + 1 < len(parts):
            params["yaml"] = parts[i + 1]
            i += 2
        elif parts[i] == "--objective" and i + 1 < len(parts):
            params["objective"] = parts[i + 1]
            i += 2
        elif parts[i] == "--detector" and i + 1 < len(parts):
            params["detector"] = parts[i + 1]
            i += 2
        elif parts[i] == "--bg-correction" and i + 1 < len(parts):
            params["bg_correction"] = parts[i + 1].lower() == "true"
            i += 2
        elif parts[i] == "--bg-folder" and i + 1 < len(parts):
            params["bg_folder"] = parts[i + 1]
            i += 2
        elif parts[i] == "--bg-method" and i + 1 < len(parts):
            params["bg_method"] = parts[i + 1]
            i += 2
        else:
            i += 1

    for req in ["output", "timepoints", "interval"]:
        if req not in params:
            conn.sendall(f"FAILED:Missing --{req}".encode())
            return

    conn.sendall(f"STARTED:{params['output']}".encode())
    try:
        from microscope_command_server.acquisition.stack_timelapse import acquire_time_lapse

        result = acquire_time_lapse(
            hardware=hardware,
            output_folder=params["output"],
            n_timepoints=params["timepoints"],
            interval_seconds=params["interval"],
            modality=params.get("modality", "brightfield"),
            angles_str=params.get("angles", "(0)"),
            config_manager=config_manager,
            wb_mode=params.get("wb_mode", "off"),
            objective=params.get("objective"),
            detector=params.get("detector"),
            yaml_file_path=params.get("yaml"),
            background_correction_enabled=params.get("bg_correction", False),
            background_folder=params.get("bg_folder"),
            background_correction_method=params.get("bg_method", "divide"),
        )
        response = (
            f"SUCCESS:{params['output']}|"
            f"timepoints:{result['n_timepoints']}|"
            f"files:{len(result['files'])}|"
            f"elapsed:{result['elapsed_seconds']:.1f}s"
        )
        conn.sendall(response.encode())
        logger.info("TLAPSE complete: %d timepoints", result["n_timepoints"])
    except Exception as e:
        logger.error("TLAPSE failed: %s", e, exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
