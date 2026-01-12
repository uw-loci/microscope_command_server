"""
Microscope Command Server - Remote Microscope Control via Socket
=================================================================

A socket-based command server for remote microscope control, designed for
QuPath integration. Provides:

- Socket server for remote control commands
- Client library for sending commands
- Acquisition workflow orchestration
- Multi-threaded command handling
- Real-time progress monitoring
- Acquisition cancellation support

This server coordinates between QuPath (Java) and the microscope hardware
(Python/Micro-Manager), enabling automated acquisition workflows driven by
annotations in QuPath.

Example Usage:
-------------
# Server side:
from microscope_command_server.server.qp_server import run_server
run_server(host='0.0.0.0', port=5000)

# Client side:
from microscope_command_server.client.client import get_stageXY, move_stageXY
x, y = get_stageXY()
move_stageXY(x + 1000, y + 1000)
"""

__version__ = "1.0.0"
__author__ = "Mike Nelson, Bin Li, Jenu Chacko"

# Note: We use 'microscope_server' as the package name internally
# This file is at microscope_command_server/__init__.py for the repository folder
# but the Python package is named 'microscope_server'
