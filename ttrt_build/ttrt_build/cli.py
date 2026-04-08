"""CLI entry point for ttrt-build.

Usage:
    ttrt-build build Qwen/Qwen3-0.6B -o qwen3.trtfb [--max-cache-length 256] [--precision fp16]
    ttrt-build inspect qwen3.trtfb
    ttrt-build version
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_build(args: argparse.Namespace) -> int:
    import ttrt_build
    try:
        ttrt_build.build(
            args.model,
            args.output,
            max_cache_length=args.max_cache_length,
            precision=args.precision,
            verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_inspect(args: argparse.Namespace) -> int:
    from .bundle_reader import read_bundle_header
    try:
        header = read_bundle_header(args.bundle)
        print(json.dumps(header, indent=2))
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def cmd_version(_args: argparse.Namespace) -> int:
    import torch
    torch_ver = torch.__version__
    try:
        import torch_tensorrt
        ttrt_ver = torch_tensorrt.__version__
    except ImportError:
        ttrt_ver = "not installed"
    print(f"ttrt-build 0.1.0 (torch={torch_ver}, torch_tensorrt={ttrt_ver})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ttrt-build",
        description="Torch-TRT model builder — compile HuggingFace models into .trtfb bundles",
    )
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="Build a .trtfb bundle from a HF model")
    p_build.add_argument("model", help="HF repo ID or local model directory")
    p_build.add_argument("-o", "--output", required=True, help="Output .trtfb path")
    p_build.add_argument("--max-cache-length", type=int, default=256,
                         help="KV cache length (default: 256)")
    p_build.add_argument("--precision", default="fp16",
                         choices=["fp16", "bf16", "fp32"],
                         help="Compute precision (default: fp16)")
    p_build.add_argument("--verbose", action="store_true")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect a .trtfb bundle")
    p_inspect.add_argument("bundle", help="Path to .trtfb file")

    # version
    sub.add_parser("version", help="Print version info")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {"build": cmd_build, "inspect": cmd_inspect, "version": cmd_version}
    sys.exit(handlers[args.command](args))
