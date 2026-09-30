"""Repository-level entry point for HCL experiments."""

from __future__ import annotations

import sys

from hcl.cli import main as cli_main


def main() -> int:
    """Run HCL while keeping ``python main.py --config ...`` convenient."""
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["run", *argv]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
