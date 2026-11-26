"""``python -m microscope_command_server.trace replay TRACE_DIR [options]``."""

import sys

from microscope_command_server.trace.replayer import main as replay_main

USAGE = "usage: python -m microscope_command_server.trace replay TRACE_DIR [options]"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] == "replay":
        return replay_main(argv[1:])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
