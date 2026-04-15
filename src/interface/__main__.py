# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Dispatch to eval or grad entry points.

Usage:
    python -m interface {eval|grad} <network.mininn> [<input1.bin> ...]
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("eval", "grad"):
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv.pop(1)
    if cmd == "eval":
        from interface.eval import main as eval_main
        eval_main()
    else:
        from interface.grad import main as grad_main
        grad_main()


if __name__ == "__main__":
    main()
