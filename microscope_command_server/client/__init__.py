"""
Microscope Server Client Library
=================================

Client library for communicating with the microscope command server.
Provides Python functions for remote stage control and acquisition.
"""

from microscope_command_server.client.client import (
    get_stageXY,
    get_stageZ,
    move_stageXY,
    move_stageZ,
    get_stageR,
    move_stageR,
)

__all__ = [
    "get_stageXY",
    "get_stageZ",
    "move_stageXY",
    "move_stageZ",
    "get_stageR",
    "move_stageR",
]
