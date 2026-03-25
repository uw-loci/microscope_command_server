@echo off
REM ============================================================
REM  Start QPSC Microscope Command Server
REM
REM  Prerequisites:
REM    1. Micro-Manager must be running with hardware loaded
REM    2. Python virtual environment with QPSC packages installed
REM
REM  Usage:
REM    Double-click this file, or run from command prompt.
REM    The server will start and wait for QuPath connections.
REM
REM  Tip: Place a shortcut to this file on your desktop alongside
REM       shortcuts for Micro-Manager and QuPath. Launch all three
REM       in order: Micro-Manager -> this server -> QuPath.
REM ============================================================

setlocal

REM -- Find and activate the virtual environment --
REM    Looks for venv_qpsc in the parent directory (QPSC_Project root)
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

if exist "%PROJECT_ROOT%\venv_qpsc\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv_qpsc\Scripts\activate.bat"
) else if exist "%SCRIPT_DIR%\venv\Scripts\activate.bat" (
    call "%SCRIPT_DIR%\venv\Scripts\activate.bat"
) else if "%VIRTUAL_ENV%"=="" (
    echo WARNING: No virtual environment found. Trying system Python...
    echo If this fails, activate your venv first or edit this script.
    echo.
)

echo ============================================================
echo  QPSC Microscope Command Server
echo ============================================================
echo.
echo  Make sure Micro-Manager is running before proceeding.
echo  Press Ctrl+C to stop the server.
echo.

REM -- Start the server --
python -m microscope_command_server.server.qp_server

REM -- If server exits, pause so the user can see any error --
echo.
echo Server stopped.
pause
