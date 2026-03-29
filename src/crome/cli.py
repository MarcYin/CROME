"""Top-level CLI for package-safe migration commands."""

from __future__ import annotations

import argparse

from crome.acquisition import alphaearth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crome", description="CROME migration utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    alphaearth_parser = alphaearth.build_parser()
    alphaearth_help = alphaearth_parser.description or "Download AlphaEarth annual embeddings."
    subparsers.add_parser(
        "download-alphaearth",
        help=alphaearth_help,
        parents=[alphaearth_parser],
        add_help=False,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download-alphaearth":
        forwarded = argv[1:] if argv is not None else None
        return alphaearth.main(forwarded)
    raise ValueError(f"Unsupported command: {args.command}")
