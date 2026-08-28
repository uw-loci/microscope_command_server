@echo off
REM ============================================================
REM  Start QPSC Microscope Command Server
REM
REM  Prerequisites:
REM    1. Micro-Manager must be running with hardware loaded
REM    2. A Python environment with the QPSC packages installed
REM
REM  Usage:
REM    Double-click this file, or run from command prompt.
REM    The server will start and wait for QuPath connections.
REM
REM  Tip: Place a shortcut to this file on your desktop alongside
REM       shortcuts for Micro-Manager and QuPath. Launch all three
REM       in order: Micro-Manager -> this server -> QuPath.
REM
REM  WHICH PYTHON DOES THIS USE?
REM  The rigs do not agree -- some use a venv, the LC-PolScope
REM  machine uses conda and has no system Python at all -- so the
REM  interpreter is resolved in this order, first match wins:
REM
REM    1. server_env.bat next to this script, if it exists.
REM       Your per-rig escape hatch. Git ignores it, so whatever
REM       you put there survives a pull. Use it for anything the
REM       cases below cannot express.
REM    2. An environment that is already active (CONDA_PREFIX or
REM       VIRTUAL_ENV). Launching from an Anaconda Prompt lands here.
REM    3. QPSC_CONDA_ENV -- the name (or full path) of a conda env
REM       to activate. Set it once with:
REM           setx QPSC_CONDA_ENV qpsc
REM    4. venv_qpsc in the project root, or venv beside this script.
REM
REM  Conda is activated properly rather than by calling the env's
REM  python.exe directly: conda packages resolve DLLs out of the
REM  env's Library\bin, which only activation puts on PATH.
REM ============================================================

setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

REM -- 1. Per-rig override --------------------------------------
if exist "%SCRIPT_DIR%server_env.bat" (
    echo Environment: server_env.bat ^(per-rig override^)
    call "%SCRIPT_DIR%server_env.bat"
    goto :preflight
)

REM -- 2. Something is already active ---------------------------
if defined CONDA_PREFIX (
    echo Environment: active conda env at %CONDA_PREFIX%
    goto :preflight
)
if defined VIRTUAL_ENV (
    echo Environment: active venv at %VIRTUAL_ENV%
    goto :preflight
)

REM -- 3. A named conda environment -----------------------------
if defined QPSC_CONDA_ENV goto :activate_conda

REM -- 4. The historical venv layout ----------------------------
if exist "%PROJECT_ROOT%\venv_qpsc\Scripts\activate.bat" (
    echo Environment: venv_qpsc
    call "%PROJECT_ROOT%\venv_qpsc\Scripts\activate.bat"
    goto :preflight
)
if exist "%SCRIPT_DIR%venv\Scripts\activate.bat" (
    echo Environment: venv
    call "%SCRIPT_DIR%venv\Scripts\activate.bat"
    goto :preflight
)

echo WARNING: No QPSC environment found. Trying whatever python is on PATH.
echo          If this machine uses conda, set the env name once:
echo              setx QPSC_CONDA_ENV your_env_name
echo.
goto :preflight

:activate_conda
set "CONDA_BAT="
for %%I in (conda.bat) do if not defined CONDA_BAT set "CONDA_BAT=%%~$PATH:I"
if not defined CONDA_BAT (
    for %%R in (
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\anaconda3"
        "%USERPROFILE%\miniforge3"
        "%LOCALAPPDATA%\miniconda3"
        "%LOCALAPPDATA%\anaconda3"
        "C:\ProgramData\miniconda3"
        "C:\ProgramData\Anaconda3"
        "C:\miniconda3"
        "C:\Anaconda3"
    ) do if not defined CONDA_BAT if exist "%%~R\condabin\conda.bat" set "CONDA_BAT=%%~R\condabin\conda.bat"
)
if not defined CONDA_BAT (
    echo ERROR: QPSC_CONDA_ENV is set to "%QPSC_CONDA_ENV%" but conda.bat could
    echo        not be found on PATH or in the usual install locations.
    echo.
    echo        Either run this from an Anaconda Prompt, or create
    echo        server_env.bat next to this script containing:
    echo            call "C:\path\to\conda\condabin\conda.bat" activate %QPSC_CONDA_ENV%
    echo.
    pause
    exit /b 1
)
echo Environment: conda env "%QPSC_CONDA_ENV%" via %CONDA_BAT%
call "%CONDA_BAT%" activate "%QPSC_CONDA_ENV%"
if errorlevel 1 (
    echo.
    echo ERROR: could not activate conda environment "%QPSC_CONDA_ENV%".
    echo        Check the name with:  conda env list
    echo.
    pause
    exit /b 1
)

:preflight
REM -- Report the interpreter, and say plainly what is missing.
REM    A missing optional package does not stop the server, but it
REM    disables that modality's processing at the point of use --
REM    which on a long acquisition means finding out far too late.
python -c "import sys; print('Python:     ' + sys.version.split()[0]); print('Executable: ' + sys.executable)"
if errorlevel 1 (
    echo.
    echo ERROR: no usable 'python' was found.
    echo        This machine may have no system Python -- see the header
    echo        of this file for how to point it at a conda env.
    echo.
    pause
    exit /b 1
)

set "QPSC_MISSING="
for %%P in (microscope_command_server microscope_control microscope_imageprocessing) do call :require %%P
for %%P in (ppm_library polscope_library) do call :optional %%P
if defined QPSC_MISSING (
    echo.
    echo ERROR: required packages are not installed in this environment.
    echo        Having the repositories cloned is not the same thing -- they
    echo        have to be installed into the env that runs the server.
    echo.
    echo        Fix it once, from an Anaconda Prompt:
    echo            conda activate ^<env^>
    echo            cd /d "%SCRIPT_DIR%"
    echo            update_env.bat
    echo.
    pause
    exit /b 1
)

echo.
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
exit /b 0


REM ============================================================
REM  Subroutines. Kept out of the for-loop bodies deliberately:
REM  a multi-line block inside a for body needs delayed expansion
REM  to read a variable it just set, and getting that subtly wrong
REM  is how a preflight check ends up silently passing.
REM ============================================================

:require
call :probe %~1
if errorlevel 1 (
    echo   MISSING ^(required^): %~1
    set "QPSC_MISSING=1"
)
exit /b 0

:optional
call :probe %~1
if errorlevel 1 echo   absent ^(optional^): %~1 -- that modality will not process
exit /b 0

:probe
REM  Drop sys.path[0] before importing. For 'python -c' that entry is the
REM  working directory, which is this repo -- and this repo contains a folder
REM  named microscope_command_server. Without the pop, the server package
REM  imports straight off the source tree and reports itself installed no
REM  matter what the environment actually holds, while its siblings (which
REM  are NOT subfolders here) correctly report missing. That combination is
REM  worse than no check: it points the blame at the wrong packages.
REM  An editable install is importable from anywhere, so this stays true for
REM  a correctly set up environment.
python -c "import sys; sys.path.pop(0); import %~1" 2>nul
exit /b %errorlevel%
