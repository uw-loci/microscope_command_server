"""
Microscope Command Server - Remote Microscope Control via Socket
=================================================================

A socket-based command server for remote microscope control. Provides:

- Socket server for remote control commands
- Client library for sending commands
- Acquisition workflow orchestration
- Multi-threaded command handling
- Real-time progress monitoring
- Acquisition cancellation support

This server coordinates between client applications and microscope hardware
(Python/Micro-Manager), enabling automated acquisition workflows.

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

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("microscope-command-server")
except Exception:
    __version__ = "0.0.0.dev"
__author__ = "Mike Nelson, Bin Li, Jenu Chacko"

# Note: We use 'microscope_server' as the package name internally
# This file is at microscope_command_server/__init__.py for the repository folder
# but the Python package is named 'microscope_server'
