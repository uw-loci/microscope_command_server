# Microscope Command Server

Socket-based command server for remote microscope control and QuPath integration.

> **Part of the QPSC (QuPath Scope Control) system**
> For complete installation instructions, see: https://github.com/uw-loci/QPSC

## Features

- **Socket Server**: TCP/IP server for remote microscope control
- **QuPath Integration**: Designed for QuPath annotation-driven acquisition
- **Client Library**: Python functions for stage control and acquisition
- **Acquisition Workflows**: Multi-tile, multi-modality acquisition orchestration
- **Real-time Monitoring**: Progress tracking and cancellation support

## Installation

**Part of [QPSC (QuPath Scope Control)](https://github.com/uw-loci/QPSC)**

**Requirements:**
- Python 3.9 or later
- pip (Python package installer)
- Git (for `pip install git+https://...` commands)

⚠️ **Important**: This package depends on `microscope-control` and `ppm-library`.
See the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#automated-installation-windows) for complete setup instructions.

### Quick Install (from GitHub)

**Install dependencies first:**
```bash
# 1. Install ppm-library
pip install git+https://github.com/uw-loci/ppm_library.git

# 2. Install microscope-control
pip install git+https://github.com/uw-loci/microscope_control.git

# 3. Then install microscope_command_server
pip install git+https://github.com/uw-loci/microscope_command_server.git
```

### Development Install (editable mode)

```bash
git clone https://github.com/uw-loci/microscope_command_server.git
cd microscope_command_server
pip install -e .
```

**For automated setup**, use the [QPSC setup script](https://github.com/uw-loci/QPSC/blob/main/PPM-QuPath.ps1).

### Troubleshooting Installation

#### Problem: `ModuleNotFoundError: No module named 'microscope_command_server'`

**Cause:** Package not installed correctly or virtual environment not activated.

**Solution:**

1. **Ensure virtual environment is activated:**
   ```bash
   # Windows
   path\to\venv_qpsc\Scripts\Activate.ps1

   # Linux/macOS
   source path/to/venv_qpsc/bin/activate
   ```

2. **Reinstall the package:**
   ```bash
   pip install -e . --force-reinstall
   ```

3. **Verify installation:**
   ```bash
   pip show microscope-command-server
   ```

#### Problem: Entry point `microscope-command-server` command not found

**Cause:** Entry points not registered or PATH not updated.

**Solution:**

Try running the server directly:
```bash
# Using Python module
python -m microscope_command_server.server.qp_server

# Or with PYTHONPATH set (if needed)
export PYTHONPATH="/path/to/parent/directory:$PYTHONPATH"
microscope-command-server
```

#### Problem: Port 5000 already in use

**Symptom:** `OSError: [Errno 48] Address already in use`

**Cause:** Another server instance or application is using port 5000.

**Solution:**
```bash
# Find process using port 5000
# Windows:
netstat -ano | findstr :5000
# macOS/Linux:
lsof -i :5000

# Kill the process if safe
```

For more troubleshooting, see the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#troubleshooting-python-package-installation).

## Quick Start

### Server Side

```python
from microscope_command_server.server.qp_server import run_server

# Start server
run_server(host='0.0.0.0', port=5000)
```

Or run from command line:
```bash
# Option 1: Entry point command (NOTE: uses hyphens, not underscores!)
microscope-command-server

# Option 2: Python module syntax
python -m microscope_command_server.server.qp_server
```

**Common mistake:** The command is `microscope-command-server` (with **hyphens**), not `microscope_command_server` (with underscores).

### Client Side

```python
from microscope_command_server.client import get_stageXY, move_stageXY

# Get current position
x, y = get_stageXY()

# Move stage
move_stageXY(x + 1000, y + 1000)
```

## Architecture

The server coordinates between QuPath (Java) and the microscope hardware (Python/Micro-Manager):

```
QuPath Extension → Socket Client → Microscope Server
                                        ↓
                              Microscope Control
                                   +
                               PPM Library
                                        ↓
                              Micro-Manager Hardware
```

## Server Configuration

The microscope command server uses a **dynamic configuration** approach:

### Startup
- Server loads a minimal generic configuration (`config_generic.yml`)
- Connects to Micro-Manager (hardware must be available)
- Waits for client connections

### During Acquisition
- Client sends ACQUIRE command with `--yaml /path/to/config.yml` parameter
- Server loads microscope-specific config from the provided path
- Hardware settings are updated dynamically
- Microscope-specific methods (e.g., PPM rotation) are initialized

### Exploratory Commands
Commands like GETXY, MOVE, GETZ use the most recently loaded config:
- Before first ACQUIRE: Uses generic startup config with permissive stage limits
- After ACQUIRE: Uses the microscope-specific config from that acquisition

**Note**: Always provide the `--yaml` parameter in ACQUIRE commands to ensure correct microscope configuration.

## Testing

This package includes automated unit tests for components that can be tested without hardware.

### Automated Unit Tests

Pytest-compatible unit tests are located in the `tests/` directory:
- **`tests/test_tiles.py`** - Tests for TileConfiguration.txt parsing and generation

These tests:
- Run without hardware (use synthetic test data and temp files)
- Can be integrated into CI/CD pipelines
- Test protocol handling, tile configuration, and utility functions

**Running Unit Tests:**

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_tiles.py

# Run with coverage report
pytest --cov=microscope_command_server --cov-report=html

# View coverage report
open htmlcov/index.html  # or xdg-open on Linux
```

**Test Coverage:**

Current automated tests achieve ~60-70% coverage for testable components:
- ✅ TileConfiguration parsing (coordinates extraction)
- ✅ TileConfiguration generation (2D pixel coordinates and 3D stage coordinates)
- ⏸️ Socket protocol (future test expansion)
- ⏸️ Server communication (requires integration testing)

**Hardware Diagnostic Tools:**

This package does not include standalone diagnostic tools. Hardware testing is performed via:
- The `TESTAF` and `TESTADAF` server commands (call diagnostic functions from `microscope_control`)
- The `PPMSENS` and `PPMBIREF` server commands (call diagnostic functions from `ppm_library`)

See the `microscope_control` and `ppm_library` documentation for details on these diagnostic tools.

## License

MIT License
