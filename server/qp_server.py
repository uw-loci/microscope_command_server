"""
QuPath Microscope Server - Enhanced Version
===========================================

A socket-based server that provides remote control of a microscope through Micro-Manager.
Handles stage movement, image acquisition, and multi-angle imaging workflows.

Enhanced Features:
- Acquisition status monitoring
- Real-time progress updates
- Acquisition cancellation support
- Non-blocking socket communication during acquisition
- Improved state management and logging
"""

import socket
import threading
import struct
import sys
import pathlib
import time
import enum
from threading import Lock
import logging
from datetime import datetime

from microscope_control.config import ConfigManager


def check_for_existing_server(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Check if a server is already running on the specified host and port.

    Attempts to connect to the port and send a simple query command.
    If successful, another server instance is already running.

    Args:
        host: The host to check (typically 127.0.0.1 for localhost)
        port: The port to check
        timeout: Connection timeout in seconds

    Returns:
        True if a server is already running, False otherwise
    """
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(timeout)
        test_socket.connect((host, port))

        # Try to send a simple command to verify it's our server
        # GETXY command is safe and quick
        test_socket.sendall(b"getxy___")

        # If we get here without exception, a server is responding
        test_socket.close()
        return True

    except (ConnectionRefusedError, socket.timeout, OSError):
        # Connection refused or timeout means no server is running
        return False
    except Exception:
        # Any other error - assume no server running
        return False
    finally:
        try:
            test_socket.close()
        except Exception:
            pass
from microscope_control.hardware import Position
from microscope_control.hardware.pycromanager import PycromanagerHardware, init_pycromanager
from microscope_command_server.server.protocol import ExtendedCommand, TCP_PORT, END_MARKER
from microscope_command_server.acquisition.workflow import _acquisition_workflow


# Configure logging
current_file_path = pathlib.Path(__file__).resolve()
base_dir = current_file_path.parent  # e.g., smart-wsi-scanner/src/smart_wsi_scanner
log_dir = base_dir / "server_logfiles"
log_dir.mkdir(parents=True, exist_ok=True)  # Create it if it doesn't exist
filename = log_dir / f'qp_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,  # Changed from INFO to DEBUG to see [TIMING-INTERNAL] logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = TCP_PORT  # Default: 5000

# Threading events for coordination
shutdown_event = threading.Event()


# Global acquisition state management
class AcquisitionState(enum.Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Global acquisition tracking
acquisition_states = {}  # addr -> AcquisitionState
acquisition_progress = {}  # addr -> (current, total)
acquisition_locks = {}  # addr -> Lock
acquisition_cancel_events = {}  # addr -> Event
acquisition_failure_messages = {}  # addr -> str (error message when FAILED)
acquisition_final_z = {}  # addr -> float (final Z position when COMPLETED, for tilt model)
manual_focus_request_events = {}  # addr -> Event (set when manual focus needed)
manual_focus_complete_events = {}  # addr -> Event (set when user acknowledges)
manual_focus_user_choice = {}  # addr -> str ("retry", "skip", "cancel")
manual_focus_retries_remaining = {}  # addr -> int (number of retries remaining)


def init_pycromanager_with_logger():
    """Initialize Pycro-Manager connection to Micro-Manager."""
    logger.info("Initializing Pycro-Manager connection...")
    core, studio = init_pycromanager()
    if not core:
        logger.error("Failed to initialize Micro-Manager connection")
        sys.exit(1)
    logger.info("Pycro-Manager initialized successfully")
    return core, studio


# Initialize hardware connections
logger.info("Loading generic startup configuration...")
config_manager = ConfigManager()

## GENERIC CONFIG loaded for exploratory XYZ movements
# Actual microscope-specific config loaded during ACQUIRE command via --yaml parameter
package_dir = pathlib.Path(__file__).parent.parent

# Try to load generic config
generic_config_path = package_dir / "configurations" / "config_generic.yml"
if generic_config_path.exists():
    logger.info(f"Loading generic startup config from {generic_config_path}")
    startup_settings = config_manager.load_config_file(str(generic_config_path))
else:
    # Fallback to hardcoded minimal config if file doesn't exist
    logger.warning("Generic config file not found, using hardcoded minimal defaults")
    startup_settings = {
        "microscope": {"name": "Generic", "type": "Unconfigured"},
        "stage": {
            "stage_id": "GENERIC_STAGE",
            "limits": {
                "x_um": {"low": -100000, "high": 100000},
                "y_um": {"low": -100000, "high": 100000},
                "z_um": {"low": -20000, "high": 20000}
            }
        },
        "ppm_optics": "NA",
        "modalities": {},
        "hardware": {},
        "id_stage": {},
        "id_detector": {},
        "id_camera": {}
    }

# Load LOCI resources if available (for device lookup during ACQUIRE)
loci_rsc_file = package_dir / "configurations" / "resources" / "resources_LOCI.yml"
if loci_rsc_file.exists():
    loci_resources = config_manager.load_config_file(str(loci_rsc_file))
    startup_settings.update(loci_resources)
    logger.info("Loaded LOCI resources for hardware device lookup")
else:
    logger.warning("LOCI resources file not found - device lookups may fail during ACQUIRE")

# Initialize hardware with generic config (will be replaced during ACQUIRE)
logger.info("Initializing Micro-Manager connection...")
core, studio = init_pycromanager_with_logger()
hardware = PycromanagerHardware(core, studio, startup_settings)
logger.info("Hardware initialized with generic config")
logger.info("Server ready - microscope-specific config will be loaded from ACQUIRE --yaml parameter")


def acquisitionWorkflow(message, client_addr):
    """Wrapper for acquisition workflow with state management."""

    def _update_progress(current: int, total: int):
        with acquisition_locks[client_addr]:
            acquisition_progress[client_addr] = (current, total)

    def _set_state(state_str: str, error_message: str = None, final_z: float = None):
        with acquisition_locks[client_addr]:
            try:
                new_state = AcquisitionState[state_str]
                acquisition_states[client_addr] = new_state
                # Store error message if state is FAILED
                if new_state == AcquisitionState.FAILED and error_message:
                    acquisition_failure_messages[client_addr] = error_message
                # Store final Z position if state is COMPLETED (for tilt correction model)
                if new_state == AcquisitionState.COMPLETED and final_z is not None:
                    acquisition_final_z[client_addr] = final_z
            except KeyError:
                acquisition_states[client_addr] = AcquisitionState.FAILED
                if error_message:
                    acquisition_failure_messages[client_addr] = error_message

    def _is_cancelled() -> bool:
        return acquisition_cancel_events[client_addr].is_set()

    def _request_manual_focus(retries_remaining: int):
        """Signal manual focus needed and wait for user acknowledgment.

        Args:
            retries_remaining: Number of autofocus retries remaining after this

        Returns:
            str: User's choice - "retry", "skip", or "cancel"
        """
        logger.info(f"Manual focus requested for client {client_addr} (retries remaining: {retries_remaining})")
        # Store retries remaining so REQMANF can return it
        manual_focus_retries_remaining[client_addr] = retries_remaining
        # Set request event to signal client
        manual_focus_request_events[client_addr].set()
        # Clear previous choice
        manual_focus_user_choice[client_addr] = None
        # Wait for user to acknowledge (blocks acquisition thread)
        logger.info("Waiting for manual focus acknowledgment from user...")
        manual_focus_complete_events[client_addr].wait()
        # Get user's choice
        user_choice = manual_focus_user_choice[client_addr] or "cancel"
        # Clear events for next potential use
        manual_focus_request_events[client_addr].clear()
        manual_focus_complete_events[client_addr].clear()
        manual_focus_user_choice[client_addr] = None
        manual_focus_retries_remaining[client_addr] = 0
        logger.info(f"Manual focus acknowledged, user chose: {user_choice}")
        return user_choice

    return _acquisition_workflow(
        message=message,
        client_addr=client_addr,
        hardware=hardware,
        config_manager=config_manager,
        logger=logger,
        update_progress=_update_progress,
        set_state=_set_state,
        is_cancelled=_is_cancelled,
        request_manual_focus=_request_manual_focus,
    )


def handle_client(conn, addr):
    """
    Handle commands from a connected QuPath client with enhanced acquisition control.
    """
    logger.info(f">>> New client connected from {addr}")

    # Initialize client state
    acquisition_locks[addr] = Lock()
    acquisition_states[addr] = AcquisitionState.IDLE
    acquisition_progress[addr] = (0, 0)
    acquisition_cancel_events[addr] = threading.Event()
    acquisition_failure_messages[addr] = None
    manual_focus_request_events[addr] = threading.Event()
    manual_focus_complete_events[addr] = threading.Event()
    manual_focus_user_choice[addr] = None
    manual_focus_retries_remaining[addr] = 0

    acquisition_thread = None

    try:
        while True:
            # All commands are 8 bytes
            data = conn.recv(8)
            if not data:
                logger.info(f"Client {addr} disconnected (no data)")
                break

            logger.debug(f"Received command from {addr}: {data}")

            # Connection management commands
            if data == ExtendedCommand.DISCONNECT:
                logger.info(f"Client {addr} requested to disconnect")
                break

            if data == ExtendedCommand.SHUTDOWN:
                logger.warning(f"Client {addr} requested server shutdown")
                shutdown_event.set()
                break

            # Position query commands
            if data == ExtendedCommand.GETXY:
                logger.debug(f"Client {addr} requested XY position")
                try:
                    current_position_xyz = hardware.get_current_position()
                    response = struct.pack("!ff", current_position_xyz.x, current_position_xyz.y)
                    conn.sendall(response)
                    logger.debug(
                        f"Sent XY position to {addr}: ({current_position_xyz.x}, {current_position_xyz.y})"
                    )
                except Exception as e:
                    logger.error(f"Failed to get XY position: {e}", exc_info=True)
                    # Send error message (8 bytes to match expected response size)
                    error_msg = f"HW_ERROR".ljust(8)[:8]
                    conn.sendall(error_msg.encode("utf-8"))
                continue

            if data == ExtendedCommand.GETZ:
                logger.debug(f"Client {addr} requested Z position")
                try:
                    current_position_xyz = hardware.get_current_position()
                    response = struct.pack("!f", current_position_xyz.z)
                    conn.sendall(response)
                    logger.debug(f"Sent Z position to {addr}: {current_position_xyz.z}")
                except Exception as e:
                    logger.error(f"Failed to get Z position: {e}", exc_info=True)
                    # Send error message (4 bytes to match expected response size)
                    error_msg = "HWERR"[:4]
                    conn.sendall(error_msg.encode("utf-8"))
                continue

            if data == ExtendedCommand.GETFOV:
                logger.debug(f"Client {addr} requested Field of View")
                try:
                    current_fov_x, current_fov_y = hardware.get_fov()
                    response = struct.pack("!ff", current_fov_x, current_fov_y)
                    conn.sendall(response)
                    logger.debug(f"Sent FOV to {addr}: ({current_fov_x}, {current_fov_y})")
                except Exception as e:
                    logger.error(f"Failed to get FOV: {e}")
                    # Send error response or default values
                    response = struct.pack("!ff", 0.0, 0.0)  # or some error indicator
                    conn.sendall(response)
                continue

            if data == ExtendedCommand.GETR:
                logger.debug(f"Client {addr} requested rotation angle")
                try:
                    angle = hardware.get_psg_ticks()
                    response = struct.pack("!f", angle)
                    conn.sendall(response)
                    logger.debug(f"Sent rotation angle to {addr}: {angle}°")
                except Exception as e:
                    logger.error(f"Failed to get rotation angle: {e}", exc_info=True)
                    # Send error message (4 bytes to match expected response size)
                    error_msg = "HWERR"[:4]
                    conn.sendall(error_msg.encode("utf-8"))
                continue

            # Movement commands
            if data == ExtendedCommand.MOVE:
                coords = conn.recv(8)
                if len(coords) == 8:
                    x, y = struct.unpack("!ff", coords)
                    logger.info(f"Client {addr} requested move to: X={x}, Y={y}")
                    try:
                        hardware.move_to_position(Position(x, y))
                        logger.info(f"Move completed to X={x}, Y={y}")
                    except Exception as e:
                        logger.error(f"Failed to move to XY position: {e}", exc_info=True)
                        # No response expected for movement commands, but log the error
                else:
                    logger.error(f"Client {addr} sent incomplete move coordinates")
                continue

            if data == ExtendedCommand.MOVEZ:
                z = conn.recv(4)
                z_position = struct.unpack("!f", z)[0]
                logger.info(f"Client {addr} requested move to Z={z_position}")
                try:
                    hardware.move_to_position(Position(z=z_position))
                    logger.info(f"Move completed to Z={z_position}")
                except Exception as e:
                    logger.error(f"Failed to move to Z position: {e}", exc_info=True)
                continue

            if data == ExtendedCommand.MOVER:
                coords = conn.recv(4)
                angle = struct.unpack("!f", coords)[0]
                logger.info(f"Client {addr} requested rotation to {angle}°")
                try:
                    hardware.set_psg_ticks(
                        angle
                    )  # , is_sequence_start=True)  # Single rotation command
                    logger.info(f"Rotation completed to {angle}°")
                except Exception as e:
                    logger.error(f"Failed to rotate stage: {e}", exc_info=True)
                continue

            # ============ ACQUISITION STATUS COMMANDS ============

            # Status query command
            if data == ExtendedCommand.STATUS:
                with acquisition_locks[addr]:
                    state = acquisition_states[addr]
                    # If state is FAILED and we have an error message, send it
                    if state == AcquisitionState.FAILED and addr in acquisition_failure_messages:
                        # Send "FAILED: <message>" format (truncated to fit in response)
                        error_msg = acquisition_failure_messages[addr]
                        # Java client expects to parse this format
                        state_str = f"FAILED: {error_msg}"[:250]  # Reasonable limit for error message
                        # Pad to 16 bytes minimum for compatibility, but can be longer
                        response = state_str.encode('utf-8')
                        conn.sendall(response)
                        logger.debug(f"Sent FAILED status with message to {addr}: {error_msg[:50]}...")
                    # If state is COMPLETED and we have final_z, include it for tilt model
                    elif state == AcquisitionState.COMPLETED and addr in acquisition_final_z:
                        final_z = acquisition_final_z[addr]
                        # Send "COMPLETED|final_z:<value>" format
                        state_str = f"COMPLETED|final_z:{final_z:.2f}"
                        response = state_str.encode('utf-8')
                        conn.sendall(response)
                        logger.debug(f"Sent COMPLETED status with final_z to {addr}: {final_z:.2f}")
                    else:
                        # Send state as 16-byte string (padded)
                        state_str = state.value.ljust(16)[:16]
                        conn.sendall(state_str.encode())
                        logger.debug(f"Sent acquisition status to {addr}: {state.value}")
                continue

            # Progress query command
            if data == ExtendedCommand.PROGRESS:
                with acquisition_locks[addr]:
                    current, total = acquisition_progress[addr]
                # Send as two integers
                response = struct.pack("!II", current, total)
                conn.sendall(response)
                logger.debug(f"Sent progress to {addr}: {current}/{total}")
                continue

            # Cancel acquisition command
            if data == ExtendedCommand.CANCEL:
                logger.warning(f"Client {addr} requested acquisition cancellation")
                with acquisition_locks[addr]:
                    if acquisition_states[addr] == AcquisitionState.RUNNING:
                        acquisition_states[addr] = AcquisitionState.CANCELLING
                        acquisition_cancel_events[addr].set()
                        logger.info(f"Cancellation initiated for {addr}")
                # Send acknowledgment
                conn.sendall(b"ACK")
                continue

            # ============ MANUAL FOCUS REQUEST/ACKNOWLEDGMENT ============

            if data == ExtendedCommand.REQMANF:
                # Check if manual focus is requested
                if manual_focus_request_events[addr].is_set():
                    # Manual focus needed - send request status with retries remaining (8 bytes exactly)
                    retries = manual_focus_retries_remaining.get(addr, 0)
                    # Format: "NEEDEDnn" where nn is 00-99
                    response = f"NEEDED{retries:02d}".encode('utf-8')
                    conn.sendall(response)
                    logger.debug(f"Sent manual focus request to {addr} (retries remaining: {retries})")
                else:
                    # No manual focus needed (8 bytes exactly)
                    conn.sendall(b"IDLE____")
                    logger.debug(f"Manual focus not needed for {addr}")
                continue

            # Manual focus acknowledgment - retry autofocus
            if data == ExtendedCommand.ACKMF:
                # Client chose to retry autofocus after manual adjustment
                manual_focus_user_choice[addr] = "retry"
                manual_focus_complete_events[addr].set()
                conn.sendall(b"ACK")
                logger.info(f"Manual focus acknowledged by client {addr} - will retry autofocus")
                continue

            # Skip autofocus retry - use current focus
            if data == ExtendedCommand.SKIPAF:
                # Client chose to use current focus position
                manual_focus_user_choice[addr] = "skip"
                manual_focus_complete_events[addr].set()
                conn.sendall(b"ACK")
                logger.info(f"Manual focus acknowledged by client {addr} - using current focus")
                continue

            # ============ ACQUISITION COMMAND ============

            if data == ExtendedCommand.ACQUIRE:
                logger.info(f"Client {addr} requested acquisition workflow")

                # Check if already running
                with acquisition_locks[addr]:
                    if acquisition_states[addr] == AcquisitionState.RUNNING:
                        logger.warning(f"Acquisition already running for {addr}")
                        continue
                    # Set state to RUNNING immediately
                    acquisition_states[addr] = AcquisitionState.RUNNING
                    acquisition_progress[addr] = (0, 0)

                # Read the full message immediately
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                # Set a timeout for reading
                conn.settimeout(5.0)

                try:
                    while True:
                        # Read in chunks
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                f"Connection closed while reading acquisition message from {addr}"
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
                            message = full_message.replace("," + END_MARKER, "").replace(
                                END_MARKER, ""
                            )
                            logger.debug(
                                f"Received complete acquisition message ({total_bytes} bytes) "
                                f"in {time.time() - start_time:.2f}s"
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

                            logger.info(f"Acquisition thread started for {addr}")

                            # Send acknowledgment to prevent client timeout
                            # Format matches BGACQUIRE pattern for consistency
                            ack_response = "STARTED:ACQUIRE".ljust(16)[:16].encode()
                            conn.sendall(ack_response)
                            logger.debug(f"Sent ACQUIRE acknowledgment to {addr}")
                            break

                        # Safety check for message size
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Acquisition message too large from {addr}: {total_bytes} bytes"
                            )
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break

                        # Timeout check
                        if time.time() - start_time > 10:
                            logger.error(f"Timeout reading acquisition message from {addr}")
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break

                except socket.timeout:
                    logger.error(f"Socket timeout reading acquisition message from {addr}")
                    with acquisition_locks[addr]:
                        acquisition_states[addr] = AcquisitionState.FAILED
                except Exception as e:
                    logger.error(f"Error reading acquisition message from {addr}: {e}")
                    with acquisition_locks[addr]:
                        acquisition_states[addr] = AcquisitionState.FAILED
                finally:
                    # Reset socket to blocking mode
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.BGACQUIRE:
                logger.info(f"Client {addr} requested background acquisition")

                # Read the message using the same pattern as ACQUIRE command
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading background acquisition message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags to avoid issues with spaces in paths
                            flags = ["--yaml", "--output", "--modality", "--angles", "--exposures"]

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

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path", "modality"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for background acquisition"
                                )

                                # Execute background acquisition using simplified collection
                                from microscope_command_server.acquisition.workflow import (
                                    simple_background_collection,
                                )

                                # Create progress update function for this client
                                def update_progress(current, total):
                                    with acquisition_locks[addr]:
                                        acquisition_progress[addr] = (current, total)

                                final_exposures = simple_background_collection(
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    modality=params["modality"],
                                    angles_str=params.get("angles_str", "()"),
                                    exposures_str=params.get("exposures_str", "()"),
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    logger=logger,
                                    update_progress=update_progress,
                                )

                                # Format exposures as angle:exposure pairs
                                # e.g., "90:137.1,7:245.8,-7:155.2"
                                exposures_formatted = ",".join(
                                    f"{angle}:{exposure:.2f}"
                                    for angle, exposure in sorted(final_exposures.items())
                                )

                                # Send success response with output path and final exposures
                                response = f"SUCCESS:{params['output_folder_path']}|{exposures_formatted}".encode()
                                conn.sendall(response)
                                logger.info(
                                    f"Background acquisition completed successfully with exposures: {exposures_formatted}"
                                )

                            except Exception as e:
                                logger.error(
                                    f"Background acquisition failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Background acquisition message too large: {total_bytes} bytes"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading background acquisition message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading background acquisition message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in background acquisition: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            if data == ExtendedCommand.SNAP:
                logger.info(f"Client {addr} requested simple snap (fixed exposure)")
                snap_start_time = time.time()

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading snap message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)
                        logger.debug(f"SNAP: received {total_bytes} bytes so far")

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Parse flags: --angle, --exposure, --output, --debayer
                            flags = ["--angle", "--exposure", "--output", "--debayer"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--angle":
                                        params["angle"] = float(value)
                                    elif flag == "--exposure":
                                        params["exposure_ms"] = float(value)
                                    elif flag == "--output":
                                        params["output_path"] = value
                                    elif flag == "--debayer":
                                        # Support "auto", "true"/"1"/"yes", "false"/"0"/"no"
                                        val = value.lower()
                                        if val == "auto":
                                            params["debayer"] = "auto"
                                        else:
                                            params["debayer"] = val in ("true", "1", "yes")

                            # Validate required parameters
                            required = ["angle", "exposure_ms", "output_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            try:
                                import tifffile
                                from pathlib import Path

                                angle = params["angle"]
                                exposure_ms = params["exposure_ms"]
                                output_path = Path(params["output_path"])
                                debayer = params.get("debayer", "auto")

                                # Create output directory if needed
                                output_path.parent.mkdir(parents=True, exist_ok=True)

                                # Set exposure (fixed - no adaptive adjustment!)
                                hardware.set_exposure(exposure_ms)
                                logger.info(f"Set exposure to {exposure_ms:.2f} ms (FIXED)")

                                # Set rotation angle
                                if hasattr(hardware, "set_psg_ticks"):
                                    hardware.set_psg_ticks(angle)
                                    logger.info(f"Set rotation angle to {angle:.2f} deg")

                                # Snap image with simple acquisition
                                image, metadata = hardware.snap_image(debayering=debayer)

                                if image is None:
                                    raise RuntimeError("snap_image returned None")

                                # Save the image
                                tifffile.imwrite(
                                    str(output_path),
                                    image,
                                    compression="zlib",
                                    compressionargs={"level": 6},
                                )

                                elapsed = time.time() - snap_start_time
                                logger.info(
                                    f"SNAP complete: {output_path.name}, "
                                    f"angle={angle:.2f}deg, exposure={exposure_ms:.2f}ms, "
                                    f"shape={image.shape}, median={float(image.mean()):.1f}, "
                                    f"total_time={elapsed:.2f}s"
                                )

                                # Send success response
                                response = f"SUCCESS:{output_path}".encode()
                                conn.sendall(response)
                                logger.debug(f"SNAP: sent SUCCESS response")

                            except Exception as e:
                                logger.error(f"SNAP failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"SNAP message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading SNAP message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading SNAP message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in SNAP: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.TESTAF:
                logger.info(f"Client {addr} requested autofocus test")

                # Read the message using the same pattern as BGACQUIRE
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading autofocus test message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags to avoid issues with spaces in paths
                            flags = ["--yaml", "--output", "--objective"]

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

                                    # Extract the value and clean it up
                                    value = message[start_idx:end_idx].strip()

                                    # Map to the parameter name
                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path", "objective"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for standard autofocus test"
                                )

                                # Execute STANDARD autofocus test
                                from microscope_control.autofocus.test import (
                                    test_standard_autofocus_at_current_position,
                                )

                                result = test_standard_autofocus_at_current_position(
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    objective=params["objective"],
                                    logger=logger,
                                )

                                if result["success"]:
                                    # Format result as: SUCCESS:plot_path|initial_z:final_z:z_shift
                                    result_data = f"{result['initial_z']:.2f}:{result['final_z']:.2f}:{result['z_shift']:.2f}"
                                    response = (
                                        f"SUCCESS:{result['plot_path']}|{result_data}".encode()
                                    )
                                    conn.sendall(response)
                                    logger.info(f"Autofocus test completed: {result['message']}")
                                else:
                                    response = f"FAILED:{result['message']}".encode()
                                    conn.sendall(response)
                                    logger.error(f"Autofocus test failed: {result['message']}")

                            except Exception as e:
                                logger.error(f"Autofocus test failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(f"Autofocus test message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading autofocus test message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading autofocus test message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in autofocus test: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            if data == ExtendedCommand.TESTADAF:
                logger.info(f"Client {addr} requested adaptive autofocus test")

                # Read the message using the same pattern as TESTAF
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading adaptive autofocus test message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags to avoid issues with spaces in paths
                            flags = ["--yaml", "--output", "--objective"]

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

                                    # Extract the value and clean it up
                                    value = message[start_idx:end_idx].strip()

                                    # Map to the parameter name
                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path", "objective"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for adaptive autofocus test"
                                )

                                # Execute ADAPTIVE autofocus test
                                from microscope_control.autofocus.test import (
                                    test_adaptive_autofocus_at_current_position,
                                )

                                result = test_adaptive_autofocus_at_current_position(
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    objective=params["objective"],
                                    logger=logger,
                                )

                                if result["success"]:
                                    # Format result as: SUCCESS:message|initial_z:final_z:z_shift
                                    result_data = f"{result['initial_z']:.2f}:{result['final_z']:.2f}:{result['z_shift']:.2f}"
                                    response = f"SUCCESS:{result['message']}|{result_data}".encode()
                                    conn.sendall(response)
                                    logger.info(
                                        f"Adaptive autofocus test completed: {result['message']}"
                                    )
                                else:
                                    response = f"FAILED:{result['message']}".encode()
                                    conn.sendall(response)
                                    logger.error(
                                        f"Adaptive autofocus test failed: {result['message']}"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Adaptive autofocus test failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Adaptive autofocus test message too large: {total_bytes} bytes"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading adaptive autofocus test message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading adaptive autofocus test message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in adaptive autofocus test: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            if data == ExtendedCommand.AFBENCH:
                logger.info(f"Client {addr} requested autofocus benchmark")

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading benchmark message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags
                            flags = ["--reference_z", "--output", "--distances", "--quick", "--objective"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--reference_z":
                                        params["reference_z"] = float(value)
                                    elif flag == "--output":
                                        params["output_folder"] = value
                                    elif flag == "--distances":
                                        # Parse comma-separated distances
                                        params["test_distances"] = [float(d.strip()) for d in value.split(",")]
                                    elif flag == "--quick":
                                        params["quick_mode"] = value.lower() in ("true", "1", "yes")
                                    elif flag == "--objective":
                                        params["objective"] = value

                            # Validate required parameters
                            required = ["reference_z", "output_folder"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment
                            try:
                                ack_response = f"STARTED:{params['output_folder']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for autofocus benchmark")

                                # Create progress callback that sends socket updates
                                # This keeps the connection alive during long benchmarks
                                # Format: PROGRESS:current:total:message (consistent with PPMBIREF)
                                def send_progress(current: int, total: int, status_msg: str):
                                    """Send progress update to keep connection alive."""
                                    try:
                                        progress_msg = f"PROGRESS:{current}:{total}:{status_msg}"
                                        conn.sendall(progress_msg.encode())
                                    except Exception as e:
                                        logger.warning(f"Failed to send progress update: {e}")

                                # Execute benchmark with progress callback
                                from microscope_control.autofocus.benchmark import (
                                    run_autofocus_benchmark_from_server,
                                )

                                result = run_autofocus_benchmark_from_server(
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    reference_z=params["reference_z"],
                                    output_folder=params["output_folder"],
                                    test_distances=params.get("test_distances"),
                                    quick_mode=params.get("quick_mode", False),
                                    objective=params.get("objective"),
                                    logger=logger,
                                    progress_callback=send_progress,
                                )

                                # Check for safety violation
                                if result.get("safety_violation"):
                                    error_msg = result.get("error", "Safety limit exceeded")
                                    response = f"FAILED:SAFETY:{error_msg}".encode()
                                    conn.sendall(response)
                                    logger.error(f"Autofocus benchmark SAFETY VIOLATION: {error_msg}")
                                else:
                                    # Format response
                                    success_rate = result.get("success_rate", 0)
                                    total_trials = result.get("total_trials", 0)
                                    results_dir = result.get("results_directory", "")

                                    response = f"SUCCESS:Benchmark complete. {total_trials} trials, {success_rate:.1%} success rate|{results_dir}".encode()
                                    conn.sendall(response)
                                    logger.info(f"Autofocus benchmark completed: {total_trials} trials")

                            except Exception as e:
                                logger.error(
                                    f"Autofocus benchmark failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"Benchmark message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading benchmark message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading benchmark message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in autofocus benchmark: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.POLCAL:
                logger.info(f"Client {addr} requested polarizer calibration")

                # Read the message using the same pattern as BGACQUIRE
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading polarizer calibration message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags
                            flags = [
                                "--yaml",
                                "--output",
                                "--start",
                                "--end",
                                "--step",
                                "--exposure",
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

                                    # Extract the value and clean it up
                                    value = message[start_idx:end_idx].strip()

                                    # Map to the parameter name
                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--start":
                                        params["start_angle"] = float(value)
                                    elif flag == "--end":
                                        params["end_angle"] = float(value)
                                    elif flag == "--step":
                                        params["step_size"] = float(value)
                                    elif flag == "--exposure":
                                        params["exposure_ms"] = float(value)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Set defaults for optional parameters
                            params.setdefault("start_angle", 0.0)
                            params.setdefault("end_angle", 360.0)
                            params.setdefault("step_size", 5.0)
                            params.setdefault("exposure_ms", 10.0)

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for polarizer calibration")

                                # Execute polarizer calibration workflow
                                from microscope_command_server.acquisition.workflow import (
                                    polarizer_calibration_workflow,
                                )

                                report_path = polarizer_calibration_workflow(
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    start_angle=params["start_angle"],
                                    end_angle=params["end_angle"],
                                    step_size=params["step_size"],
                                    exposure_ms=params["exposure_ms"],
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    logger=logger,
                                )

                                # Send success response with report path
                                response = f"SUCCESS:{report_path}".encode()
                                conn.sendall(response)

                                logger.info(
                                    f"Polarizer calibration completed. Report: {report_path}"
                                )

                            except Exception as e:
                                logger.error(
                                    f"Polarizer calibration failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Polarizer calibration message too large: {total_bytes} bytes"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading polarizer calibration message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading polarizer calibration message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in polarizer calibration: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            # ============ PPM TESTING COMMANDS ============

            if data == ExtendedCommand.PPMSENS:
                logger.info(f"Client {addr} requested PPM rotation sensitivity test")

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading PPMSENS message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse parameters
                            params = {}
                            flags = ["--yaml", "--output", "--test-type", "--base-angle", "--repeats"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break
                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--test-type":
                                        params["test_type"] = value
                                    elif flag == "--base-angle":
                                        params["base_angle"] = float(value)
                                    elif flag == "--repeats":
                                        params["n_repeats"] = int(value)

                            # Set defaults
                            params.setdefault("test_type", "repeatability")
                            params.setdefault("base_angle", 7.0)
                            params.setdefault("n_repeats", 10)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for PPM sensitivity test")

                                # Run PPM sensitivity test using programmatic interface
                                from ppm_library.ppm.sensitivity_test import run_ppm_sensitivity_test

                                result_dir = run_ppm_sensitivity_test(
                                    config_yaml=params["yaml_file_path"],
                                    output_dir=params["output_folder_path"],
                                    host="127.0.0.1",  # Connect back to ourselves
                                    port=PORT,
                                    test_type=params["test_type"],
                                    base_angle=params["base_angle"],
                                    n_repeats=params["n_repeats"],
                                    keep_images=True,
                                )

                                if result_dir:
                                    response = f"SUCCESS:{result_dir}".encode()
                                    conn.sendall(response)
                                    logger.info(f"PPM sensitivity test completed: {result_dir}")
                                else:
                                    response = b"FAILED:Test did not complete successfully"
                                    conn.sendall(response)
                                    logger.error("PPM sensitivity test failed")

                            except ImportError as e:
                                logger.error(f"PPM sensitivity test module not available: {e}")
                                response = f"FAILED:Module not available - {e}".encode()
                                conn.sendall(response)
                            except Exception as e:
                                logger.error(f"PPM sensitivity test failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"PPMSENS message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading PPMSENS message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading PPMSENS message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in PPMSENS: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.PPMBIREF:
                logger.info(f"Client {addr} requested PPM birefringence maximization test")

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading PPMBIREF message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse parameters
                            params = {}
                            flags = ["--yaml", "--output", "--mode", "--min-angle", "--max-angle",
                                    "--step", "--exposure", "--target-intensity"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break
                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--mode":
                                        params["exposure_mode"] = value
                                    elif flag == "--min-angle":
                                        params["min_angle"] = float(value)
                                    elif flag == "--max-angle":
                                        params["max_angle"] = float(value)
                                    elif flag == "--step":
                                        params["angle_step"] = float(value)
                                    elif flag == "--exposure":
                                        params["fixed_exposure_ms"] = float(value)
                                    elif flag == "--target-intensity":
                                        params["target_intensity"] = int(value)

                            # Set defaults
                            params.setdefault("exposure_mode", "interpolate")
                            params.setdefault("min_angle", -10.0)
                            params.setdefault("max_angle", 10.0)
                            params.setdefault("angle_step", 0.5)  # Coarser default for server
                            params.setdefault("target_intensity", 128)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Validate fixed mode requires exposure
                            if params["exposure_mode"] == "fixed" and "fixed_exposure_ms" not in params:
                                error_msg = "fixed_exposure_ms required when mode=fixed"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for PPM birefringence test")

                                # Create progress callback to send updates through socket
                                def send_progress(current: int, total: int):
                                    """Send progress update through socket."""
                                    try:
                                        progress_msg = f"PROGRESS:{current}:{total}".encode()
                                        conn.sendall(progress_msg)
                                        logger.debug(f"Sent progress: {current}/{total}")
                                    except Exception as e:
                                        logger.warning(f"Failed to send progress: {e}")

                                # Run PPM birefringence test using programmatic interface
                                from ppm_library.ppm.birefringence_test import run_birefringence_maximization_test

                                result_dir = run_birefringence_maximization_test(
                                    config_yaml=params["yaml_file_path"],
                                    output_dir=params["output_folder_path"],
                                    host="127.0.0.1",  # Connect back to ourselves
                                    port=PORT,
                                    angle_range=(params["min_angle"], params["max_angle"]),
                                    angle_step=params["angle_step"],
                                    exposure_mode=params["exposure_mode"],
                                    fixed_exposure_ms=params.get("fixed_exposure_ms"),
                                    keep_images=True,
                                    target_intensity=params["target_intensity"],
                                    progress_callback=send_progress,
                                )

                                if result_dir:
                                    response = f"SUCCESS:{result_dir}".encode()
                                    conn.sendall(response)
                                    logger.info(f"PPM birefringence test completed: {result_dir}")
                                else:
                                    response = b"FAILED:Test did not complete successfully"
                                    conn.sendall(response)
                                    logger.error("PPM birefringence test failed")

                            except ImportError as e:
                                logger.error(f"PPM birefringence test module not available: {e}")
                                response = f"FAILED:Module not available - {e}".encode()
                                conn.sendall(response)
                            except Exception as e:
                                logger.error(f"PPM birefringence test failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"PPMBIREF message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading PPMBIREF message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading PPMBIREF message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in PPMBIREF: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            # Legacy GET/SET commands (not implemented)
            if data == ExtendedCommand.GET:
                logger.debug("GET property not yet implemented")
                continue

            if data == ExtendedCommand.SET:
                logger.debug("SET property not yet implemented")
                continue

            # Unknown command
            logger.warning(f"Unknown command from {addr}: {data}")

    except Exception as e:
        logger.error(f"Error handling client {addr}: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        if acquisition_thread and acquisition_thread.is_alive():
            logger.info(f"Cancelling acquisition for disconnected client {addr}")
            acquisition_cancel_events[addr].set()
            acquisition_thread.join(timeout=10)

        # Remove client state
        if addr in acquisition_locks:
            del acquisition_locks[addr]
        if addr in acquisition_states:
            del acquisition_states[addr]
        if addr in acquisition_progress:
            del acquisition_progress[addr]
        if addr in acquisition_cancel_events:
            del acquisition_cancel_events[addr]
        if addr in acquisition_failure_messages:
            del acquisition_failure_messages[addr]
        if addr in acquisition_final_z:
            del acquisition_final_z[addr]

        conn.close()
        logger.info(f"<<< Client {addr} disconnected and cleaned up")


def main():
    """Main server loop that accepts client connections and spawns handler threads."""
    logger.info("=" * 60)
    logger.info("QuPath Microscope Server - Enhanced Version")
    logger.info("=" * 60)

    # Check for existing server instance BEFORE attempting to bind
    logger.info("Checking for existing server instance...")
    if check_for_existing_server("127.0.0.1", PORT):
        logger.error("=" * 60)
        logger.error("ANOTHER SERVER INSTANCE IS ALREADY RUNNING!")
        logger.error("=" * 60)
        logger.error(f"A server is already listening on port {PORT}.")
        logger.error("Please close the existing server before starting a new one.")
        logger.error("")
        logger.error("To find the existing server:")
        logger.error("  Windows: Use Task Manager to find python.exe processes")
        logger.error("  Linux: Run 'lsof -i :5000' or 'netstat -tlnp | grep 5000'")
        logger.error("=" * 60)
        print("\n" + "=" * 60)
        print("ERROR: Another server instance is already running on port {}!".format(PORT))
        print("Please close the existing server before starting a new one.")
        print("=" * 60 + "\n")
        sys.exit(1)

    logger.info("No existing server instance found. Proceeding with startup...")

    logger.info(f"Server configuration:")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Micro-Manager core: {'Connected' if core else 'Not connected'}")
    logger.info(f"  Hardware: {'Initialized' if hardware else 'Not initialized'}")

    # Log loaded configuration
    microscope_info = ppm_settings.get("microscope", {})
    logger.info(f"  Microscope: {microscope_info.get('name', 'Unknown')}")
    logger.info(f"  Type: {microscope_info.get('type', 'Unknown')}")

    logger.info("Features:")
    logger.info("  - Status monitoring")
    logger.info("  - Progress tracking")
    logger.info("  - Cancellation support")
    logger.info("  - Enhanced logging")
    logger.info("  - Multi-instance detection")
    logger.info("=" * 60)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f"Server listening on {HOST}:{PORT}")
        logger.info("Ready for connections...")

        threads = []

        while not shutdown_event.is_set():
            try:
                s.settimeout(1.0)
                conn, addr = s.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                thread.start()
                threads.append(thread)
            except socket.timeout:
                continue
            except OSError:
                break

        logger.info("Server shutting down. Waiting for client threads to finish...")
        shutdown_event.set()

        for t in threads:
            t.join(timeout=5.0)

        logger.info("Server has shut down.")


if __name__ == "__main__":
    main()
