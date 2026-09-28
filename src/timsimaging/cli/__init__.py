"""Command-line interface for timsimaging."""

import argparse

from . import process


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``timsimaging`` console command."""
    parser = argparse.ArgumentParser(prog="timsimaging")
    subparsers = parser.add_subparsers(dest="command", required=True)
    process.add_subparser(subparsers)
    args = parser.parse_args(argv)
    return args.func(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
