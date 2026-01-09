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

## Quick Start

### Server Side

```python
from microscope_server.server.qp_server import run_server

# Start server
run_server(host='0.0.0.0', port=5000)
```

Or run from command line:
```bash
microscope-server
```

### Client Side

```python
from microscope_server.client import get_stageXY, move_stageXY

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

## License

MIT License
