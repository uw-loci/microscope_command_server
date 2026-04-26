"""System and alignment command handlers.

Handles connection management, configuration, and alignment commands:
CONFIG, RECONFG, DISCONNECT, SHUTDOWN, SIFTAL
"""

import struct
import socket
import time
import logging

import yaml

from microscope_command_server.server.protocol import END_MARKER
from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)


def handle_disconnect(conn, client, hardware, settings, **kwargs):
    """Handle client disconnect request.

    Returns 'DISCONNECT' sentinel so the caller can break the
    command loop.

    Returns:
        str: 'DISCONNECT' to signal the caller to close the connection.
    """
    addr = kwargs.get("addr", client if isinstance(client, tuple) else getattr(client, "addr", client))
    logger.info("Client %s requested to disconnect", addr)
    return "DISCONNECT"


def handle_shutdown(conn, client, hardware, settings, **kwargs):
    """Handle server shutdown request.

    Sets the shutdown_event so the main server loop exits after
    all clients disconnect.

    Returns 'SHUTDOWN' sentinel so the caller can break the command loop.

    Required kwargs:
        shutdown_event: threading.Event to signal shutdown.

    Returns:
        str: 'SHUTDOWN' to signal the caller to close the connection.
    """
    addr = kwargs.get("addr", client if isinstance(client, tuple) else getattr(client, "addr", client))
    shutdown_event = kwargs.get("shutdown_event")
    logger.warning("Client %s requested server shutdown", addr)
    if shutdown_event is not None:
        shutdown_event.set()
    return "SHUTDOWN"


def handle_config(conn, client, hardware, settings, **kwargs):
    """Handle CONFIG command -- connection setup and config loading.

    This is the most complex system command. It:
    1. Reads the config file path from the socket
    2. Validates connection locking (single active client)
    3. Loads and validates the YAML config
    4. Reinitializes hardware components from the new config
    5. Starts session logging
    6. Sends version info back to the client

    Modifies global server state via kwargs:
        server_configured, active_connection_addr, active_connection_config_path

    Required kwargs:
        addr: Client address tuple
        config_manager: ConfigManager instance
        connection_state_lock: Lock protecting connection state
        server_configured: bool (current state, returned modified)
        active_connection_addr: tuple or None (current, returned modified)
        active_connection_config_path: str or None (current, returned modified)
        start_session_logging: callable(config_path)

    Returns:
        dict with updated global state keys:
            server_configured, active_connection_addr, active_connection_config_path
        or None if the command was handled without state changes (e.g., blocked).
    """
    addr = kwargs["addr"]
    config_manager = kwargs["config_manager"]
    connection_state_lock = kwargs["connection_state_lock"]
    current_server_configured = kwargs.get("server_configured", False)
    current_active_addr = kwargs.get("active_connection_addr")
    current_active_config_path = kwargs.get("active_connection_config_path")
    start_session_logging = kwargs.get("start_session_logging")

    logger.info("Client %s sent CONFIG command", addr)

    try:
        # Read config file path: 4 bytes length + path string
        path_length_bytes = conn.recv(4)
        if not path_length_bytes:
            logger.error("CONFIG: No path length received")
            conn.sendall(b"CFG_FAIL")
            return None

        path_length = struct.unpack("!I", path_length_bytes)[0]
        logger.debug("CONFIG: Expecting config path of %d bytes", path_length)

        config_path_bytes = conn.recv(path_length)
        config_path = config_path_bytes.decode("utf-8")
        logger.info("CONFIG: Received config path: %s", config_path)

        # Check connection locking
        with connection_state_lock:
            if current_active_addr is not None and current_active_addr != addr:
                # Another connection exists - check if same IP (likely reconnect)
                # addr is (ip, port) tuple - compare IP only
                active_ip = current_active_addr[0]
                new_ip = addr[0]

                if active_ip == new_ip:
                    # Same IP reconnecting - usually the previous connection
                    # crashed and we should let the new one take over.
                    # CRITICAL EXCEPTION: if the previous addr is currently
                    # running an acquisition workflow, taking over kills it
                    # mid-flight. This was observed 2026-04-25 on PPM:
                    # during a multi-annotation existing-image workflow, a
                    # transient status-query timeout on the Java side caused
                    # MicroscopeSocketClient to auto-reconnect. The reconnect
                    # opened a new socket and re-sent CONFIG. Without this
                    # guard the takeover closed the still-acquiring primary
                    # socket, aborted the workflow with WinError 10053, and
                    # drained 5 pending writes. The Java side then surfaced
                    # this as "Read timed out" -> "Acquisition failed".
                    #
                    # Reject the new CONFIG (CFG_BLCK) when an acquisition
                    # is active for the existing addr -- the original
                    # connection keeps running, the spurious reconnect dies
                    # cleanly. The Java client will see a CFG_BLCK response
                    # rather than a hijacked socket; future change should
                    # have it not auto-reconnect during acquisitions, but
                    # this server-side guard is the load-bearing fix.
                    acquisition_states = kwargs.get("acquisition_states", {})
                    AcquisitionState = kwargs.get("AcquisitionState")
                    active_state = acquisition_states.get(current_active_addr)
                    is_actively_acquiring = (
                        active_state is not None
                        and AcquisitionState is not None
                        and active_state in (
                            AcquisitionState.RUNNING,
                            AcquisitionState.CANCELLING,
                        )
                    )
                    if is_actively_acquiring:
                        logger.warning(
                            "CONFIG: Refusing same-IP takeover from %s -- previous "
                            "connection is actively running acquisition (state=%s). "
                            "New connection %s will be rejected to protect the "
                            "in-flight workflow.",
                            current_active_addr, active_state, addr,
                        )
                        error_msg = (
                            f"BLOCKED: Active acquisition on {current_active_addr}; "
                            f"refusing to take over."
                        ).encode("utf-8")
                        error_length = struct.pack("!I", len(error_msg))
                        conn.sendall(b"CFG_BLCK" + error_length + error_msg)
                        return None

                    logger.warning("CONFIG: Same IP reconnecting - taking over from %s", current_active_addr)
                    logger.warning("CONFIG: Previous connection may have been improperly closed")
                    # Stop any orphaned sequence acquisition left running by
                    # the dead client. Without this the camera can stay
                    # streaming into MM's circular buffer and eventually
                    # hard-lock the Hamamatsu sCMOS driver + MicroManager
                    # itself (observed on OWS3 2026-04-09).
                    try:
                        if hardware.core.is_sequence_running():
                            logger.warning(
                                "CONFIG: Stopping orphaned sequence acquisition from dead client"
                            )
                            hardware.core.stop_sequence_acquisition()
                    except Exception as stop_err:
                        logger.error(
                            "CONFIG: Failed to stop orphaned sequence acquisition: %s",
                            stop_err,
                        )
                    # Clear the old addr (will be set to new addr below).
                    # KEEP current_active_config_path so the downstream
                    # path_changed check can skip rebuilding hardware when
                    # the reconnecting client is using the same config.
                    # Rebuilding during live acquisition hangs on
                    # _detect_camera_name() core calls (see OWS3 incident
                    # 2026-04-09).
                    current_active_addr = None
                else:
                    # Different IP - reject this CONFIG
                    logger.warning("CONFIG: Rejected - connection %s already active", current_active_addr)
                    error_msg = f"BLOCKED: Active connection from {current_active_addr}".encode("utf-8")
                    error_length = struct.pack("!I", len(error_msg))
                    conn.sendall(b"CFG_BLCK" + error_length + error_msg)
                    return None

        # Load the config file
        new_settings = config_manager.load_config_file(config_path)

        # Validate essential config sections exist
        # Note: id_detector specs come from resources file, not main config
        # Main config has hardware.detectors which lists detector IDs
        required_sections = ["microscope", "stage"]
        missing = [s for s in required_sections if s not in new_settings or not new_settings[s]]
        if missing:
            error_msg = f"Config missing required sections: {', '.join(missing)}"
            logger.error("CONFIG: %s", error_msg)
            error_bytes = error_msg.encode("utf-8")
            error_length = struct.pack("!I", len(error_bytes))
            conn.sendall(b"CFG_FAIL" + error_length + error_bytes)
            return None

        # Update hardware with new configuration. Always refresh the
        # settings dict (cheap). Only rebuild composed components when
        # the config path actually changes -- aux connections re-send
        # CONFIG on every reconnect with the same path, and
        # re-enumerating all MM device properties via
        # _detect_camera_name() while a sequence acquisition is running
        # can exceed the client's 5s read timeout and corrupt both
        # connections (see OWS3 incident 2026-04-09).
        hardware.settings = new_settings
        path_changed = (current_active_config_path != config_path)
        if path_changed:
            logger.info(
                "CONFIG: Config path changed (%s -> %s), rebuilding hardware",
                current_active_config_path, config_path,
            )
            hardware._camera_name = hardware._detect_camera_name()
            hardware._camera_registry = hardware._build_camera_registry()
            hardware._active_detector_id = hardware._find_detector_id(hardware._camera_name)
            hardware._stage = hardware._create_stage()
            hardware._rotation_stage = hardware._create_rotation_stage()
            hardware._illumination = hardware._create_illumination()
            hardware._detector = hardware._create_detector()
        else:
            logger.debug(
                "CONFIG: Same config path (%s), skipping hardware rebuild",
                config_path,
            )

        # Build updated state to return
        updated_state = {
            "server_configured": True,
            "active_connection_addr": addr,
            "active_connection_config_path": config_path,
            "settings": new_settings,
        }

        microscope_name = new_settings.get("microscope", {}).get("name", "Unknown")
        logger.info("CONFIG: Successfully loaded config for microscope: %s", microscope_name)
        logger.info("CONFIG: Server now configured and ready for operations")

        # Start session logging to <config_dir>/logs/
        if start_session_logging:
            start_session_logging(config_path)

        # Send success response with version info payload
        import json
        from microscope_command_server.version_info import collect_versions
        version_json = json.dumps(collect_versions()).encode("utf-8")
        version_length = struct.pack("!I", len(version_json))
        conn.sendall(b"CFG___OK" + version_length + version_json)

        return updated_state

    except FileNotFoundError:
        error_msg = f"Config file not found: {config_path}"
        logger.error("CONFIG: %s", error_msg)
        error_bytes = error_msg.encode("utf-8")
        error_length = struct.pack("!I", len(error_bytes))
        conn.sendall(b"CFG_FAIL" + error_length + error_bytes)
        return None
    except Exception as e:
        error_msg = f"Failed to load config: {str(e)}"
        logger.error("CONFIG: %s", error_msg, exc_info=True)
        error_bytes = error_msg.encode("utf-8")
        error_length = struct.pack("!I", len(error_bytes))
        conn.sendall(b"CFG_FAIL" + error_length + error_bytes)
        return None


def handle_reconfig(conn, client, hardware, settings, **kwargs):
    """Re-read YAML config files from disk after a calibration write.

    Java sends this after WB calibration, polarizer calibration, or
    background collection writes new values to the YAML.  Without it
    the Python server keeps stale cached values until restart.

    No payload.  Response: ACK_____ on success, FAILED:<reason> on error.

    Does NOT rebuild hardware objects (camera, stage, etc.) -- only the
    cached settings dict and any derived config caches are refreshed.
    """
    addr = kwargs.get(
        "addr",
        client if isinstance(client, tuple) else getattr(client, "addr", client),
    )
    config_path = kwargs.get("active_connection_config_path")
    config_manager = kwargs.get("config_manager")

    logger.info("Client %s requested RECONFIG", addr)

    if not config_path:
        msg = "No config path set -- send CONFIG first"
        logger.error("RECONFIG: %s", msg)
        conn.sendall(f"FAILED:{msg}".encode())
        return None

    try:
        from pathlib import Path

        config_dir = Path(config_path).parent
        microscope_name = Path(config_path).stem.replace("config_", "")

        # 1. Re-read the main config YAML
        new_settings = config_manager.load_config_file(config_path)
        hardware.settings = new_settings
        logger.info("RECONFIG: Reloaded %s", config_path)

        # 2. Re-read imageprocessing YAML (calibration results live here)
        imgproc_path = config_dir / f"imageprocessing_{microscope_name}.yml"
        if imgproc_path.exists():
            with open(imgproc_path) as f:
                imgproc_data = yaml.safe_load(f) or {}
            # Store for workflow.py's load_jai_calibration_from_imageprocessing()
            # which re-reads from disk anyway, but update the settings cache
            # in case anything caches the old values.
            new_settings["_imageprocessing_data"] = imgproc_data
            logger.info("RECONFIG: Reloaded %s", imgproc_path)
        else:
            logger.debug("RECONFIG: No imageprocessing file at %s", imgproc_path)

        # 3. Re-read autofocus YAML
        af_path = config_dir / f"autofocus_{microscope_name}.yml"
        if af_path.exists():
            with open(af_path) as f:
                af_data = yaml.safe_load(f) or {}
            new_settings["_autofocus_data"] = af_data
            logger.info("RECONFIG: Reloaded %s", af_path)
        else:
            logger.debug("RECONFIG: No autofocus file at %s", af_path)

        conn.sendall(b"ACK_____")
        logger.info("RECONFIG: Complete -- server now using latest YAML values")

        # Return updated settings for the server to store
        return {"settings": new_settings}

    except Exception as e:
        msg = str(e)
        logger.error("RECONFIG failed: %s", msg, exc_info=True)
        conn.sendall(f"FAILED:{msg}".encode())
        return None


def handle_siftal(conn, client, hardware, settings, **kwargs):
    """SIFT auto-alignment: snap microscope image and match against WSI region.

    Reads --wsi-region, --micro-px, --wsi-px, --min-px, --flip-x,
    --flip-y flags. Snaps a microscope image, runs SIFT feature
    matching against the provided WSI region file, and returns
    the offset in micrometers.

    Response: SUCCESS:<offset_x>,<offset_y>|inliers:<n>|confidence:<f>
              or FAILED:<reason>
    """
    addr = kwargs.get("addr", client if isinstance(client, tuple) else getattr(client, "addr", client))
    logger.info("Client %s requested SIFT auto-alignment", addr)

    try:
        message = read_message_string(conn, chunk_size=4096)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read SIFTAL message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    logger.info("SIFTAL message: %s", message)

    # Parse SIFT parameters from message
    params = {}
    parts = message.split()
    i = 0
    while i < len(parts):
        if parts[i] == "--wsi-region" and i + 1 < len(parts):
            params["wsi_region_path"] = parts[i + 1]; i += 2
        elif parts[i] == "--micro-px" and i + 1 < len(parts):
            params["micro_px"] = float(parts[i + 1]); i += 2
        elif parts[i] == "--wsi-px" and i + 1 < len(parts):
            params["wsi_px"] = float(parts[i + 1]); i += 2
        elif parts[i] == "--min-px" and i + 1 < len(parts):
            params["min_px"] = float(parts[i + 1]); i += 2
        elif parts[i] == "--ratio" and i + 1 < len(parts):
            params["ratio_threshold"] = float(parts[i + 1]); i += 2
        elif parts[i] == "--min-matches" and i + 1 < len(parts):
            params["min_match_count"] = int(parts[i + 1]); i += 2
        elif parts[i] == "--contrast" and i + 1 < len(parts):
            params["contrast_threshold"] = float(parts[i + 1]); i += 2
        elif parts[i] == "--nfeatures" and i + 1 < len(parts):
            params["nfeatures"] = int(parts[i + 1]); i += 2
        elif parts[i] == "--flip-x":
            params["flip_x"] = True; i += 1
        elif parts[i] == "--flip-y":
            params["flip_y"] = True; i += 1
        else:
            i += 1

    if "wsi_region_path" not in params:
        conn.sendall(b"FAILED:Missing --wsi-region")
        return

    try:
        import cv2
        from microscope_command_server.alignment.sift_matcher import match_sift

        # Read WSI region from file
        wsi_path = params["wsi_region_path"]
        wsi_region = cv2.imread(wsi_path)
        if wsi_region is None:
            conn.sendall(f"FAILED:Could not read WSI region: {wsi_path}".encode())
            return

        # Snap microscope image
        image, metadata = hardware.snap_image()
        if image is None:
            conn.sendall(b"FAILED:Could not snap microscope image")
            return

        # Convert RGB to BGR for OpenCV if needed
        if image.ndim == 3 and image.shape[2] == 3:
            micro_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            micro_bgr = image

        micro_px = params.get("micro_px", 0.173)
        wsi_px = params.get("wsi_px", 0.25)
        min_px = params.get("min_px", 1.0)
        flip_x = params.get("flip_x", False)
        flip_y = params.get("flip_y", False)
        ratio_threshold = params.get("ratio_threshold", 0.7)
        min_match_count = params.get("min_match_count", 10)
        contrast_threshold = params.get("contrast_threshold", 0.04)
        nfeatures = params.get("nfeatures", 0)

        logger.info(
            "SIFT: micro_px=%s, wsi_px=%s, min_px=%s, flip=(%s,%s), "
            "ratio=%s, min_matches=%s, contrast=%s, nfeatures=%s",
            micro_px, wsi_px, min_px, flip_x, flip_y,
            ratio_threshold, min_match_count, contrast_threshold, nfeatures,
        )

        result = match_sift(
            microscope_image=micro_bgr,
            wsi_region=wsi_region,
            microscope_pixel_size_um=micro_px,
            wsi_pixel_size_um=wsi_px,
            flip_x=flip_x,
            flip_y=flip_y,
            min_match_count=min_match_count,
            ratio_threshold=ratio_threshold,
            min_pixel_size_um=min_px,
            contrast_threshold=contrast_threshold,
            nfeatures=nfeatures,
        )

        if result is None:
            conn.sendall(b"FAILED:SIFT matching failed - insufficient features or matches")
        else:
            offset_x, offset_y, n_inliers, confidence = result
            response = (f"SUCCESS:{offset_x:.2f},{offset_y:.2f}|"
                        f"inliers:{n_inliers}|confidence:{confidence:.3f}")
            conn.sendall(response.encode())
            logger.info(
                "SIFTAL complete: offset=(%.1f, %.1f) um, inliers=%d, confidence=%.2f",
                offset_x, offset_y, n_inliers, confidence,
            )

    except ImportError as e:
        conn.sendall(f"FAILED:OpenCV not available: {e}".encode())
    except Exception as e:
        logger.error("SIFTAL failed: %s", e, exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
