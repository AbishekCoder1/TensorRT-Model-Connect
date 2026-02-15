"""CLI entry point: trtf-build <model-dir> -o <out.trtfb> [options]."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from . import __version__


def _cmd_build(args: argparse.Namespace) -> int:
    from .engine_builder import build_bundle

    if not args.model_dir:
        print("Error: model directory required", file=sys.stderr)
        return 1
    if not args.output:
        print("Error: -o / --output required", file=sys.stderr)
        return 1

    try:
        build_bundle(
            model_dir=args.model_dir,
            output_path=args.output,
            max_cache_length=args.max_cache_length,
            verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    bundle_path = args.bundle_path
    if not bundle_path:
        print("Error: bundle path required", file=sys.stderr)
        return 1

    try:
        with open(bundle_path, "rb") as f:
            magic = f.read(8)
            if magic != b"TRTFB\x00\x01\x00":
                print(f"Error: not a valid .trtfb bundle: {bundle_path}",
                      file=sys.stderr)
                return 1

            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_len).decode("utf-8")

        header = json.loads(header_json)
        fields = [
            ("Model ID", "model_id"),
            ("Model type", "model_type"),
            ("Family", "family"),
            ("TRT version", "trt_version"),
            ("GPU", "gpu_name"),
            ("Created", "created_at"),
            ("Vocab size", "vocab_size"),
            ("Hidden size", "hidden_size"),
            ("Layers", "num_layers"),
            ("Attention heads", "num_attention_heads"),
            ("KV heads", "num_key_value_heads"),
            ("Max cache length", "max_cache_length"),
        ]
        for label, key in fields:
            print(f"{label + ':':<20} {header.get(key, '')}")

        sections = header.get("sections", {})
        if sections:
            print(f"{'Sections:':<20}")
            for name, meta in sections.items():
                size_mb = meta.get("size", 0) / (1024 * 1024)
                print(f"  {name}: {size_mb:.1f} MB")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"trtf-build {__version__}")
    try:
        import tensorrt as trt
        print(f"TensorRT:  {trt.__version__}")
    except ImportError:
        print("TensorRT:  not installed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trtf-build",
        description="Build .trtfb bundles from HuggingFace models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # trtf-build build <model-dir> -o <out.trtfb>
    build_p = subparsers.add_parser("build", help="Build a .trtfb bundle")
    build_p.add_argument("model_dir", help="HF model directory")
    build_p.add_argument("-o", "--output", required=True,
                         help="Output .trtfb file path")
    build_p.add_argument("--max-cache-length", type=int, default=256,
                         help="KV cache length (default: 256)")
    build_p.add_argument("--verbose", action="store_true",
                         help="Verbose TRT builder output")

    # trtf-build inspect <bundle.trtfb>
    inspect_p = subparsers.add_parser("inspect",
                                      help="Inspect a .trtfb bundle")
    inspect_p.add_argument("bundle_path", help=".trtfb file to inspect")

    # trtf-build version
    subparsers.add_parser("version", help="Show version info")

    # Allow bare positional as sugar: `trtf-build <model-dir> -o out.trtfb`
    # (i.e., default subcommand is "build")
    args, remaining = parser.parse_known_args()

    if args.command is None:
        # Try to parse as implicit "build"
        if remaining or (len(sys.argv) > 1 and sys.argv[1] not in
                         ("--help", "-h")):
            build_args = ["build"] + sys.argv[1:]
            args = parser.parse_args(build_args)
        else:
            parser.print_help()
            sys.exit(0)

    dispatch = {
        "build": _cmd_build,
        "inspect": _cmd_inspect,
        "version": _cmd_version,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))
