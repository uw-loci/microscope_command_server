import socket
import struct
import argparse
import sys
import time
import logging
from typing import Optional, Tuple
from microscope_command_server.server.protocol import Command, ExtendedCommand, TCP_PORT, END_MARKER

HOST = "127.0.0.1"  # Server address (localhost by default)
PORT = TCP_PORT  # Must match server


def get_stageXY():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.GETXY.value)
        data = s.recv(8)
        if len(data) == 8:
            x, y = struct.unpack("!ff", data)
            print(f"{x,y}")
        else:
            print("Failed to receive stage location.")


def get_stageZ():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.GETZ.value)
        data = s.recv(4)
        if len(data) == 4:
            z = struct.unpack("!f", data)
            print(f"{z}")
        else:
            print("Failed to receive stage location.")


def move_stageZ():
    parser = argparse.ArgumentParser(description="Move Z stage")
    parser.add_argument("-z", "--z", type=float, required=True, help="Z position in microns")
    args = parser.parse_args()
    packed = struct.pack("!f", args.z)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.MOVEZ.value + packed)


def move_stageXY():

    parser = argparse.ArgumentParser(description="Move XYZ stage")

    # All arguments use flags and are not positional
    parser.add_argument("-x", "--x", type=float, required=True, help="X position")
    parser.add_argument("-y", "--y", type=float, required=True, help="Y position")

    args = parser.parse_args()

    x, y = args.x, args.y
    packed = struct.pack("!ff", x, y)
    # print("Asking to move to", x, y)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.MOVE.value + packed)


def get_stageR():
    """Get the current rotation angle of the stage."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.GETR.value)
        data = s.recv(4)
        if len(data) == 4:
            angle = struct.unpack("!f", data)[0]
            print(f"Current rotation angle: {angle:.2f} degrees")
        else:
            print("Failed to receive rotation angle.")


def move_stageR():
    """Move rotation stage to specified angle."""
    parser = argparse.ArgumentParser(description="Move rotation stage")
    parser.add_argument("angle", type=float, help="Rotation angle in degrees")
    args = parser.parse_args(sys.argv[2:])

    packed = struct.pack("!f", args.angle)
    # print("Asking to move to", x, y)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.MOVER.value + packed)


def shutdown_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.SHUTDOWN.value)
        print("Sent server shutdown command. Disconnected.")


def disconnect():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.DISCONNECT.value)
        print("Disconnected from server.")


def get():
    message = ",".join(["MicroPublisher6", "Color - Blue scale"]) + "," + END_MARKER
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.GET.value + message.encode())
        print("Getting", message)
        data = s.recv(4)
        if len(data) == 4:
            prop_value = struct.unpack("!f", data)[0]
            print(f"C: {prop_value:.2f} ms")
        else:
            print("Failed to receive exposure")


class QuPathTestClient:
    """Test client for QuPath microscope server with persistent connection."""

    def __init__(self, host: str = "127.0.0.1", port: int = TCP_PORT):
        """Initialize test client with server connection parameters."""
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            self.socket.connect((self.host, self.port))
            self.logger.info(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            try:
                self.socket.send(ExtendedCommand.DISCONNECT)
                time.sleep(0.1)
            except:
                pass
            finally:
                self.socket.close()
                self.socket = None
                self.logger.info("Disconnected from server")

    def get_xy(self) -> Tuple[float, float]:
        """Get current XY position."""
        self.socket.send(ExtendedCommand.GETXY)
        response = self.socket.recv(8)
        x, y = struct.unpack("!ff", response)
        return x, y

    def get_z(self) -> float:
        """Get current Z position."""
        self.socket.send(ExtendedCommand.GETZ)
        response = self.socket.recv(4)
        return struct.unpack("!f", response)[0]

    def get_rotation(self) -> float:
        """Get current rotation angle."""
        self.socket.send(ExtendedCommand.GETR)
        response = self.socket.recv(4)
        return struct.unpack("!f", response)[0]

    def move_xy(self, x: float, y: float):
        """Move to XY position."""
        self.socket.send(ExtendedCommand.MOVE)
        self.socket.send(struct.pack("!ff", x, y))
        time.sleep(0.5)

    def move_z(self, z: float):
        """Move to Z position."""
        self.socket.send(ExtendedCommand.MOVEZ)
        self.socket.send(struct.pack("!f", z))
        time.sleep(0.5)

    def move_rotation(self, angle: float):
        """Move rotation stage to angle."""
        self.socket.send(ExtendedCommand.MOVER)
        self.socket.send(struct.pack("!f", angle))
        time.sleep(0.5)

    def snap_image(
        self,
        output_path: str,
        angle: float,
        exposure_ms: float,
        white_balance: bool = False,
        yaml_path: str = None,
        objective: str = None,
        detector: str = None,
        wb_reference_angle: float = None,
    ) -> str:
        """
        Snap a single image using SNAP command.

        Args:
            output_path: Directory to save the image
            angle: Rotation angle (for filename)
            exposure_ms: Exposure time in milliseconds
            white_balance: If True, apply per-angle white balance calibration from YAML
            yaml_path: Path to config YAML file (required if white_balance=True)
            objective: Objective ID for calibration lookup (optional, uses hardware.settings if not provided)
            detector: Detector ID for calibration lookup (optional, uses hardware.settings if not provided)
            wb_reference_angle: If provided, use this angle for WB calibration lookup instead
                               of the actual capture angle. Useful for calibration mode where
                               consistent color is needed across all angles.

        Returns:
            Path to saved image file
        """
        message = f"--angle {angle} --exposure {exposure_ms} --output {output_path}"
        if white_balance and yaml_path:
            message += f" --white_balance true --yaml {yaml_path}"
            if objective:
                message += f" --objective {objective}"
            if detector:
                message += f" --detector {detector}"
            if wb_reference_angle is not None:
                message += f" --wb_ref_angle {wb_reference_angle}"
        message += f" {END_MARKER}"
        self.socket.send(ExtendedCommand.SNAP)
        self.socket.send(message.encode())

        # Wait for response
        self.socket.settimeout(30.0)
        response = self.socket.recv(4096).decode()
        self.socket.settimeout(10.0)

        if response.startswith("SUCCESS:"):
            return response[8:].strip()
        elif response.startswith("FAILED:"):
            raise RuntimeError(f"SNAP failed: {response[7:]}")
        else:
            raise RuntimeError(f"Unexpected response: {response}")

    # ===== test_ prefixed methods for compatibility with sensitivity_test.py =====

    def test_status(self) -> str:
        """Test STATUS command - returns acquisition status."""
        self.socket.send(ExtendedCommand.STATUS)
        response = self.socket.recv(16)
        return response.decode().strip()

    def test_get_rotation(self) -> float:
        """Get current rotation angle (test_ prefix for compatibility)."""
        return self.get_rotation()

    def test_move_rotation(self, angle: float):
        """Move rotation stage (test_ prefix for compatibility)."""
        self.move_rotation(angle)

    def test_get_xy(self) -> Tuple[float, float]:
        """Get XY position (test_ prefix for compatibility)."""
        return self.get_xy()

    def test_get_z(self) -> float:
        """Get Z position (test_ prefix for compatibility)."""
        return self.get_z()

    def test_snap(
        self,
        angle: float,
        exposure_ms: float,
        output_path: str,
        white_balance: bool = False,
        yaml_path: str = None,
        objective: str = None,
        detector: str = None,
        wb_reference_angle: float = None,
    ) -> str:
        """
        Snap image (test_ prefix for compatibility with sensitivity_test.py).

        Args:
            angle: Rotation angle
            exposure_ms: Exposure time in ms
            output_path: Output directory path
            white_balance: If True, apply per-angle white balance calibration from YAML
            yaml_path: Path to config YAML file (required if white_balance=True)
            objective: Objective ID for calibration lookup (optional)
            detector: Detector ID for calibration lookup (optional)
            wb_reference_angle: If provided, use this angle for WB calibration lookup
                               instead of the actual capture angle. For calibration mode.

        Returns:
            Path to saved image
        """
        return self.snap_image(
            output_path,
            angle,
            exposure_ms,
            white_balance=white_balance,
            yaml_path=yaml_path,
            objective=objective,
            detector=detector,
            wb_reference_angle=wb_reference_angle,
        )


def main():
    while True:
        user_input = input("Enter Q (quit) or D(disconnect) : ")
        if user_input == "Q":
            shutdown_server()
            break
        elif user_input == "D":
            disconnect()
            break
        elif user_input == "XY":
            get_stageXY()
            continue
        elif user_input == "Z":
            get_stageZ()
            continue
        elif user_input == "R":
            get_stageR()
            continue
        elif user_input == "E":
            get()
            continue
        else:
            print("Invalid commands. Please try again.")


if __name__ == "__main__":

    # Uncomment the following lines to test individual functions
    # move_stageR()
    # move_stageZ()
    # move_stageXY()
    # acquisitionWorkflow()

    ## all others are available via command line args
    main()
