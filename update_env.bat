@echo off
REM ============================================================
REM  Update QPSC Python Environment
REM
REM  Reinstalls all three local packages (ppm-library,
REM  microscope-control, microscope-command-server) from their
REM  pyproject.toml into the active virtual environment.
REM
REM  Usage:
REM    1. Activate venv first:  call venv_qpsc\Scripts\activate.bat
REM    2. Run this script:      update_env.bat
REM
REM  Optional extras (e.g. pandas for PPM analysis):
REM    update_env.bat --analysis
REM ============================================================

setlocal

REM -- Verify we're inside a virtual environment --
if "%VIRTUAL_ENV%"=="" (
    echo ERROR: No virtual environment is active.
    echo Run:  call venv_qpsc\Scripts\activate.bat
    exit /b 1
)

echo.
echo Active venv: %VIRTUAL_ENV%
echo.

REM -- Resolve project root (parent of microscope_command_server) --
REM    Assumes this script lives in microscope_command_server/
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

set "PPM_LIB=%PROJECT_ROOT%\ppm_library"
set "MICRO_CTRL=%PROJECT_ROOT%\microscope_control"
set "MICRO_SRV=%PROJECT_ROOT%\microscope_command_server"

REM -- Verify directories exist --
if not exist "%PPM_LIB%\pyproject.toml" (
    echo ERROR: ppm_library not found at %PPM_LIB%
    exit /b 1
)
if not exist "%MICRO_CTRL%\pyproject.toml" (
    echo ERROR: microscope_control not found at %MICRO_CTRL%
    exit /b 1
)
if not exist "%MICRO_SRV%\pyproject.toml" (
    echo ERROR: microscope_command_server not found at %MICRO_SRV%
    exit /b 1
)

REM -- Check for --analysis flag --
set "PPM_EXTRAS="
if "%~1"=="--analysis" set "PPM_EXTRAS=[analysis]"

REM -- Install in dependency order --
echo [1/3] Installing ppm-library%PPM_EXTRAS% ...
pip install -e "%PPM_LIB%%PPM_EXTRAS%" --quiet
if errorlevel 1 (
    echo FAILED: ppm-library install
    exit /b 1
)

echo [2/3] Installing microscope-control ...
pip install -e "%MICRO_CTRL%" --quiet
if errorlevel 1 (
    echo FAILED: microscope-control install
    exit /b 1
)

echo [3/3] Installing microscope-command-server ...
pip install -e "%MICRO_SRV%" --quiet
if errorlevel 1 (
    echo FAILED: microscope-command-server install
    exit /b 1
)

echo.
echo All packages updated successfully.
echo.

REM -- Show installed versions --
pip show ppm-library microscope-control microscope-command-server 2>nul | findstr /i "Name: Version:"

endlocal
