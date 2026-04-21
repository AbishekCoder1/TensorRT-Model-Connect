"""CLI entry point: trtf-build <model-dir> -o <out.trtfb> [options]."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import struct
import sys
from pathlib import Path


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

    build_model_ref = args.model

    # Backend dispatch: default to auto-selection that prefers raw TRT and
    # falls back to Torch-TRT when raw support is unavailable.
    method_name = getattr(args, 'method', 'auto')
    if method_name == 'auto':
        try:
            method_name, build_model_ref = _auto_select_build_backend(args.model)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    if not getattr(args, "_skip_profile_resolution", False):
        try:
            build_model_ref, build_family = _resolve_build_model_metadata(
                build_model_ref,
                method_name,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()
            return 1

        reexec_rc = _maybe_reexec_build_in_profile(
            args,
            build_model_ref,
            build_family,
        )
        if reexec_rc is not None:
            return reexec_rc

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
                build_model_ref,
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
    from .quantization import canonicalize_quant_format

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
    quantize = canonicalize_quant_format(args.quantize)

    try:
        build(
            model_id_or_path=build_model_ref,
            output_path=args.output,
            max_cache_length=args.max_cache_length,
            precision=args.precision,
            quantize=quantize,
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


def _resolve_build_model_metadata(model_ref: str, method_name: str) -> tuple[str, str]:
    """Return (resolved_model_ref, family_name) for the selected build backend."""
    from .config import ModelConfig
    from .engine_builder import _resolve_model, find_diffusion_plugin, find_plugin

    resolved_model_ref = _resolve_model(model_ref)
    model_dir = Path(resolved_model_ref)

    if method_name != "trt":
        from .engine_defs.torch_trt.compiler import _parse_model_config
        from .engine_defs.torch_trt.families import find_plugin as find_torchtrt_plugin

        plugin = find_torchtrt_plugin(_parse_model_config(model_dir))
        return resolved_model_ref, getattr(plugin, "name", "")

    if (model_dir / "model_index.json").exists():
        model_index = json.loads((model_dir / "model_index.json").read_text())
        plugin = find_diffusion_plugin(str(model_index.get("_class_name", "") or ""))
        return resolved_model_ref, getattr(plugin, "name", "")

    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    return resolved_model_ref, getattr(plugin, "name", "")


def _resolve_build_profile_name(family_name: str) -> str:
    from .python_profiles import normalize_execution_profiles

    return normalize_execution_profiles(None, family=family_name).get("build", "base")


def _maybe_reexec_build_in_profile(
    args: argparse.Namespace,
    build_model_ref: str,
    build_family: str,
) -> int | None:
    """Re-exec build under a declared Python profile when the family requires it."""
    from .python_profiles import (
        ACTIVE_PROFILE_ENV,
        DEFAULT_PROFILE,
        resolve_profile_python,
    )

    required_profile = _resolve_build_profile_name(build_family)
    if required_profile == DEFAULT_PROFILE:
        return None

    active_profile = os.environ.get(ACTIVE_PROFILE_ENV, "").strip()
    if active_profile == required_profile:
        return None

    target_python = resolve_profile_python(required_profile, sys.executable)
    current_python = str(Path(sys.executable).absolute())
    if current_python == target_python:
        return None

    env = os.environ.copy()
    env[ACTIVE_PROFILE_ENV] = required_profile
    cmd = [target_python, "-m", "trtf_build.__main__"] + sys.argv[1:]
    print(
        f"[trtf-build] Switching build to Python profile {required_profile!r}: "
        f"{target_python}",
        file=sys.stderr,
    )
    return subprocess.run(cmd, env=env).returncode


def _auto_select_build_backend(model_ref: str) -> tuple[str, str]:
    """Return (method_name, resolved_model_ref) for the best available backend.

    The selection rule is:
      1. Prefer the raw TensorRT Network API backend when a native family plugin
         exists for the model.
      2. Fall back to the Torch-TRT backend when no raw plugin exists but a
         Torch-TRT family plugin does.
    """
    from .config import ModelConfig
    from .engine_builder import _resolve_model, find_plugin, find_diffusion_plugin
    from .engine_defs import get_engine_def

    resolved_model_ref = _resolve_model(model_ref)
    model_dir = Path(resolved_model_ref)

    torchtrt_backend = get_engine_def("torchtrt")
    torchtrt_available = torchtrt_backend is not None
    torchtrt_supported = False

    if (model_dir / "model_index.json").exists():
        model_index = json.loads((model_dir / "model_index.json").read_text())
        pipeline_class = str(model_index.get("_class_name", "") or "")
        raw_supported = (
            find_diffusion_plugin(pipeline_class) is not None
            or find_plugin(pipeline_class.lower()) is not None
        )
    else:
        config = ModelConfig.from_dir(model_dir)
        raw_plugin = find_plugin(config.model_type)
        raw_supported = raw_plugin is not None
        if torchtrt_available:
            from .engine_defs.torch_trt.families import find_plugin as find_torchtrt_plugin

            torchtrt_plugin = find_torchtrt_plugin(config)
            torchtrt_supported = torchtrt_plugin is not None

            # Prefer a config-aware Torch-TRT family over a generic raw
            # model_type match. This avoids routing specialized checkpoints
            # like Chronos-Bolt through a generic T5 raw builder.
            matches_config = getattr(torchtrt_plugin, "matches_config", None)
            raw_matches_config = getattr(raw_plugin, "matches_config", None)
            if (
                raw_supported
                and torchtrt_supported
                and callable(matches_config)
                and matches_config(config)
                and not (callable(raw_matches_config) and raw_matches_config(config))
            ):
                raw_supported = False

    if raw_supported:
        print("[trtf-build] Auto-selected backend: trt", file=sys.stderr)
        return "trt", resolved_model_ref

    if torchtrt_available and not torchtrt_supported and (model_dir / "model_index.json").exists():
        from .engine_defs.torch_trt.compiler import _parse_model_config
        from .engine_defs.torch_trt.families import find_plugin as find_torchtrt_plugin

        torchtrt_supported = find_torchtrt_plugin(_parse_model_config(model_dir)) is not None

    if torchtrt_supported:
        print("[trtf-build] Auto-selected backend: torchtrt", file=sys.stderr)
        return "torchtrt", resolved_model_ref

    if not raw_supported and not torchtrt_available:
        raise RuntimeError(
            "No raw TRT family plugin matched this model, and the Torch-TRT "
            "backend is not available. Install torch_tensorrt or choose a "
            "model with native TRT support."
        )

    if not raw_supported and not torchtrt_supported:
        raise RuntimeError(
            "No supported build backend matched this model. "
            "It is not implemented on the raw TRT path, and no Torch-TRT "
            "family plugin matched it either."
        )

    print("[trtf-build] Auto-selected backend: trt", file=sys.stderr)
    return "trt", resolved_model_ref


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
                         choices=["fp8", "int8", "int8_sq", "int4", "int4_awq", "nvfp4", "w4a8"],
                         default=None,
                         help="Quantization format (default: none)")
    build_p.add_argument("--quant-scales",
                         default=None,
                         help="Path to pre-computed quantization scales JSON (skips calibration)")
    build_p.add_argument("--quant-calibration-samples",
                         type=int, default=512,
                         help="Number of calibration samples for PTQ (default: 512)")
    build_p.add_argument("--method", type=str, default="auto",
                         choices=["auto", "trt", "torchtrt"],
                         help="Engine definition method: auto (default, prefer raw TRT and "
                              "fall back to torchtrt), trt, or torchtrt")
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
