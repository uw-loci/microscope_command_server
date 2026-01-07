# Microscope Command Server

Socket-based command server for remote microscope control and QuPath integration.

## Features

- **Socket Server**: TCP/IP server for remote microscope control
- **QuPath Integration**: Designed for QuPath annotation-driven acquisition
- **Client Library**: Python functions for stage control and acquisition
- **Acquisition Workflows**: Multi-tile, multi-modality acquisition orchestration
- **Real-time Monitoring**: Progress tracking and cancellation support

## Installation

```bash
pip install microscope-server
```

Requires both `microscope-control` and `ppm-library` packages.

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
