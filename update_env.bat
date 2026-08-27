@echo off
REM ============================================================
REM  Update QPSC Python Environment
REM
REM  Reinstalls the local QPSC packages from their pyproject.toml
REM  into the active environment, in dependency order.
REM
REM  Works with either a venv or a conda env -- the LC-PolScope
REM  machine has no system Python and uses conda, so requiring
REM  VIRTUAL_ENV would lock that rig out.
REM
REM  Usage:
REM    1. Activate the environment first, whichever kind it is:
REM         call venv_qpsc\Scripts\activate.bat
REM         conda activate qpsc
REM    2. Run this script:      update_env.bat
REM
REM  Optional extras (e.g. pandas for PPM analysis):
REM    update_env.bat --analysis
REM
REM  WHICH PACKAGES GET INSTALLED
REM  The two required ones always. ppm_library and polscope_library
REM  are installed only if their folder is present next to this
REM  repo, because no rig runs every modality -- but a modality
REM  whose library is absent will not process its data, so what was
REM  skipped is reported rather than passed over in silence.
REM
REM  Note these are installed from local checkouts, not PyPI.
REM  polscope-library in particular is not published, so the
REM  [polscope] extra in pyproject.toml cannot resolve on its own;
REM  installing the checkout here is what satisfies it.
REM ============================================================

setlocal

REM -- Verify some environment is active ------------------------
set "QPSC_ENV="
if defined VIRTUAL_ENV set "QPSC_ENV=%VIRTUAL_ENV%"
if defined CONDA_PREFIX set "QPSC_ENV=%CONDA_PREFIX%"

if not defined QPSC_ENV (
    echo ERROR: No Python environment is active.
    echo.
    echo   venv:   call venv_qpsc\Scripts\activate.bat
    echo   conda:  conda activate ^<env name^>
    echo.
    echo Installing into a base or system interpreter is refused: it is
    echo how one rig's dependency versions end up silently applied to
    echo another modality.
    exit /b 1
)

echo.
echo Active environment: %QPSC_ENV%
python -c "import sys; print('Python:             ' + sys.version.split()[0])"
if errorlevel 1 (
    echo ERROR: no usable 'python' in this environment.
    exit /b 1
)
echo.

REM -- Resolve project root (parent of microscope_command_server) --
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

set "PPM_LIB=%PROJECT_ROOT%\ppm_library"
set "POLSCOPE_LIB=%PROJECT_ROOT%\polscope_library"
set "MICRO_IP=%PROJECT_ROOT%\microscope_imageprocessing"
set "MICRO_CTRL=%PROJECT_ROOT%\microscope_control"
set "MICRO_SRV=%PROJECT_ROOT%\microscope_command_server"

REM -- Verify the required directories exist --------------------
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

set "SKIPPED="

REM -- Optional modality libraries, only if checked out ---------
if exist "%MICRO_IP%\pyproject.toml" (
    echo [1/5] Installing microscope-imageprocessing ...
    pip install -e "%MICRO_IP%" --quiet
    if errorlevel 1 goto :failed_ip
) else (
    echo [1/5] microscope_imageprocessing checkout not found -- relying on the installed copy.
)

if exist "%PPM_LIB%\pyproject.toml" (
    echo [2/5] Installing ppm-library%PPM_EXTRAS% ...
    pip install -e "%PPM_LIB%%PPM_EXTRAS%" --quiet
    if errorlevel 1 goto :failed_ppm
) else (
    echo [2/5] ppm_library not found -- SKIPPED. PPM processing will be unavailable.
    set "SKIPPED=1"
)

if exist "%POLSCOPE_LIB%\pyproject.toml" (
    echo [3/5] Installing polscope-library ...
    pip install -e "%POLSCOPE_LIB%" --quiet
    if errorlevel 1 goto :failed_polscope
) else (
    echo [3/5] polscope_library not found -- SKIPPED. LC-PolScope reconstruction
    echo       and LC calibration will be unavailable.
    set "SKIPPED=1"
)

echo [4/5] Installing microscope-control ...
pip install -e "%MICRO_CTRL%" --quiet
if errorlevel 1 goto :failed_ctrl

echo [5/5] Installing microscope-command-server ...
pip install -e "%MICRO_SRV%" --quiet
if errorlevel 1 goto :failed_srv

echo.
echo All available packages installed.
if defined SKIPPED (
    echo.
    echo NOTE: one or more modality libraries were skipped above. That is
    echo       correct if this rig does not run that modality. If it does,
    echo       clone the missing repo next to this one and re-run.
)
echo.

REM -- Show installed versions --
pip show ppm-library polscope-library microscope-control microscope-command-server 2>nul | findstr /i "Name: Version:"

endlocal
exit /b 0

:failed_ip
echo FAILED: microscope-imageprocessing install
exit /b 1
:failed_ppm
echo FAILED: ppm-library install
exit /b 1
:failed_polscope
echo FAILED: polscope-library install
exit /b 1
:failed_ctrl
echo FAILED: microscope-control install
exit /b 1
:failed_srv
echo FAILED: microscope-command-server install
exit /b 1
