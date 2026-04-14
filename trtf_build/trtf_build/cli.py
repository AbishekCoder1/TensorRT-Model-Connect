"""CLI entry point: trtf-build <model-dir> -o <out.trtfb> [options]."""

from __future__ import annotations

import argparse
import json
import struct
import sys


def _get_version() -> str:
    """Get package version, trying importlib.metadata first, then __init__."""
    try:
        from importlib.metadata import version
        return version("trtf-build")
    except Exception:
        pass
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "0.1.0"


__version__ = _get_version()


def _cmd_build(args: argparse.Namespace) -> int:
    if not args.model:
        print("Error: model (HF repo ID or local directory) required",
              file=sys.stderr)
        return 1
    if not args.output:
        print("Error: -o / --output required", file=sys.stderr)
        return 1

    # Method dispatch: non-default methods go through the registry
    method_name = getattr(args, 'method', 'trt')
    if method_name != 'trt':
        from .engine_defs import get_engine_def
        engine_def = get_engine_def(method_name)
        if engine_def is None:
            print(f"Error: method '{method_name}' is not available. "
                  f"Install its dependencies (e.g., pip install torch_tensorrt).",
                  file=sys.stderr)
            return 1
        try:
            engine_def.build(
                args.model,
                args.output,
                max_cache_length=args.max_cache_length,
                precision=args.precision,
                verbose=args.verbose,
            )
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # Default TRT Network API path
    from .engine_builder import build

    # FP8 quantization: --fp8-scales (pre-computed) or --fp8 (auto-calibrate)
    fp8_scales = None
    fp8_auto = getattr(args, 'fp8', False)
    if getattr(args, 'fp8_scales', None):
        import json as _json
        with open(args.fp8_scales) as _f:
            fp8_scales = _json.load(_f)
        print(f"[trtf-build] Loaded FP8 scales from {args.fp8_scales} "
              f"({len(fp8_scales)} layers)", file=sys.stderr)
    elif fp8_auto:
        # Sentinel: engine_builder will call plugin.fp8_calibrate()
        fp8_scales = "auto"
        print("[trtf-build] FP8 auto-calibration enabled", file=sys.stderr)

    save_fp8_scales = getattr(args, 'save_fp8_scales', None)

    try:
        build(
            model_id_or_path=args.model,
            output_path=args.output,
            max_cache_length=args.max_cache_length,
            precision=args.precision,
            quantize=args.quantize,
            quant_scales=args.quant_scales,
            quant_calibration_samples=args.quant_calibration_samples,
            verbose=args.verbose,
            fp8_scales=fp8_scales,
            save_fp8_scales=save_fp8_scales,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _read_bundle_header(bundle_path: str) -> dict:
    """Read and return the JSON header from a .trtfb bundle."""
    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
    return json.loads(header_json)


def list_engine_sections(bundle_path: str) -> list[dict]:
    """List all TRT engine plan sections in a bundle.

    Returns list of dicts: [{name, size_bytes, size_mb, role}]
    where role is 'primary', 'vision', 'text_encoder', 'denoiser', 'vae', etc.
    """
    header = _read_bundle_header(bundle_path)
    sections = header.get("sections", {})

    engines = []
    for name, meta in sections.items():
        if not name.endswith("_plan") and name != "engine_plan":
            continue
        size_bytes = meta.get("size", 0)

        # Infer role from section name
        if name == "engine_plan":
            role = "primary"
        elif "vision" in name:
            role = "vision"
        elif "text_encoder" in name:
            role = "text_encoder"
        elif "denoiser" in name:
            role = "denoiser"
        elif "vae" in name:
            role = "vae"
        elif "lt_" in name or "local_transformer" in name:
            role = "local_transformer"
        else:
            role = name.replace("_plan", "")

        engines.append({
            "name": name,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "role": role,
        })

    return engines


def _cmd_inspect(args: argparse.Namespace) -> int:
    bundle_path = args.bundle_path
    if not bundle_path:
        print("Error: bundle path required", file=sys.stderr)
        return 1

    try:
        header = _read_bundle_header(bundle_path)

        if getattr(args, 'list_engines', False):
            # Engine-only listing mode
            engines = list_engine_sections(bundle_path)
            if not engines:
                print("No engine sections found.", file=sys.stderr)
                return 1
            print(f"{'Section':<30} {'Size':>10} {'Role':<16}")
            print(f"{'-'*30} {'-'*10} {'-'*16}")
            for e in engines:
                print(f"{e['name']:<30} {e['size_mb']:>8.1f} MB {e['role']:<16}")
            return 0

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
            ("Precision", "precision"),
            ("Quantization", "quantization"),
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

    # trtf-build build <model> -o <out.trtfb>
    build_p = subparsers.add_parser("build", help="Build a .trtfb bundle")
    build_p.add_argument("model",
                         help="HF repo ID (e.g. Qwen/Qwen3-0.6B) or local directory")
    build_p.add_argument("-o", "--output", required=True,
                         help="Output .trtfb file path")
    build_p.add_argument("--max-cache-length", type=int, default=256,
                         help="KV cache length (default: 256)")
    build_p.add_argument("--precision", choices=["fp32", "fp16", "bf16"],
                         default="fp32",
                         help="Engine precision (default: fp32)")
    build_p.add_argument("--quantize",
                         choices=["fp8", "int8", "int4", "nvfp4", "w4a8"],
                         default=None,
                         help="Quantization format (default: none)")
    build_p.add_argument("--quant-scales",
                         default=None,
                         help="Path to pre-computed quantization scales JSON (skips calibration)")
    build_p.add_argument("--quant-calibration-samples",
                         type=int, default=512,
                         help="Number of calibration samples for PTQ (default: 512)")
    build_p.add_argument("--method", type=str, default="trt",
                         choices=["trt", "torchtrt"],
                         help="Engine definition method: trt (default, TRT Network API) "
                              "or torchtrt (torch-trt)")
    build_p.add_argument("--verbose", action="store_true",
                         help="Verbose TRT builder output")
    build_p.add_argument("--fp8", action="store_true",
                         help="Enable FP8 quantization (auto-calibrate via ModelOpt)")
    build_p.add_argument("--fp8-scales", default=None,
                         help="Path to pre-computed FP8 scales JSON (skips calibration)")
    build_p.add_argument("--save-fp8-scales", default=None,
                         help="Save calibrated FP8 scales to JSON (reuse with --fp8-scales)")

    # trtf-build inspect <bundle.trtfb>
    inspect_p = subparsers.add_parser("inspect",
                                      help="Inspect a .trtfb bundle")
    inspect_p.add_argument("bundle_path", help=".trtfb file to inspect")
    inspect_p.add_argument("--list-engines", action="store_true",
                           help="List only TRT engine plan sections with roles")

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
