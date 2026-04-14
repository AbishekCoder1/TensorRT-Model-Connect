#!/usr/bin/env python3
"""Unified 4-way inference performance comparison.

Compares all four inference backends for the same HF model:
  1. Torch-TRT  (StatelessCacheWrapper -> raw TRT engine via trtf_build.engine_defs.torch_trt)
  2. Raw TRT    (trtf_build graph API -> raw TRT engine)
  3. torch.compile (PyTorch 2.x compiler, default backend)
  4. HF eager   (baseline, no compilation)

All backends use token-by-token autoregressive decoding with KV cache.
Backends run serially to avoid GPU memory contention.

Usage:
    # Full 4-way comparison (builds both TRT engines on the fly)
    python3 tools/perf_compare_all.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "The capital of France is" \
      --max-new-tokens 50

    # Use pre-built bundles
    python3 tools/perf_compare_all.py \
      --model Qwen/Qwen3-0.6B \
      --torchtrt-bundle /tmp/qwen3_torchtrt.trtfb \
      --rawtrt-bundle /tmp/qwen3_raw.trtfb \
      --prompt "The capital of France is" \
      --max-new-tokens 50

    # Skip specific backends
    python3 tools/perf_compare_all.py \
      --model Qwen/Qwen3-0.6B \
      --skip-compile --skip-rawtrt \
      --prompt "Hello" --max-new-tokens 20

    # Save results as JSON
    python3 tools/perf_compare_all.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "Hello" --max-new-tokens 20 \
      --json results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np

from perf_utils import (
    get_gpu_name, get_trt_version, stats, bench_hf_eager, bench_torch_compile,
    bench_torchtrt_bundle, build_torchtrt_bundle,
    build_json_output,
    precision_to_torch_dtype, dtype_label,
)


# ---------------------------------------------------------------------------
# Backend: Raw TRT (trtf_build graph API)
# ---------------------------------------------------------------------------

def bench_rawtrt(engine_plan: bytes, num_layers: int, max_cache_length: int,
                 tokenizer, prompt: str, max_new_tokens: int,
                 warmup: int, iterations: int, verbose: bool) -> dict:
    from trtf_build.debug_runner import TrtRunner

    runner = TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=max_cache_length,
        num_layers=num_layers,
    )

    input_ids = tokenizer.encode(prompt)
    prefill_times, decode_times, gen_ids_last = [], [], []

    for run_idx in range(warmup + iterations):
        is_warmup = run_idx < warmup
        gen_ids = []
        runner.reset()

        t0 = time.perf_counter()
        for tid in input_ids:
            result = runner.step(tid)
        logits = result["logits"].flatten()
        prefill_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits))
            gen_ids.append(next_token)
            result = runner.step(next_token)
            logits = result["logits"].flatten()
        decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            gen_ids_last = gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [raw-trt {tag}] prefill={prefill_ms:.1f}ms "
                  f"decode={decode_ms:.1f}ms", file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "gen_ids": gen_ids_last,
        "num_tokens": max_new_tokens,
    }


def build_rawtrt_engine(model_id: str, max_cache_length: int,
                         verbose: bool):
    """Returns (engine_plan, num_layers, max_cache_length)."""
    from trtf_build.engine_builder import _resolve_model as _trtf_resolve
    from trtf_build.config import ModelConfig
    from trtf_build.families import find_plugin

    model_dir = _trtf_resolve(model_id)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None:
        raise ValueError(f"No plugin for model_type={config.model_type!r}")

    print("  Building Raw TRT engine ...", file=sys.stderr)
    t0 = time.perf_counter()
    weights = plugin.load_weights(model_dir, config)
    engine_plan = plugin.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    build_s = time.perf_counter() - t0
    print(f"  Raw TRT engine built in {build_s:.1f}s "
          f"({len(engine_plan) / 1e6:.1f} MB)", file=sys.stderr)
    return engine_plan, config.num_hidden_layers, max_cache_length


# ---------------------------------------------------------------------------
# Pretty report with ANSI colors and visual bars
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RED = "\033[31m"
_RESET = "\033[0m"

_BACKEND_LABELS = {
    "torch_trt": "Torch-TRT",
    "raw_trt": "Raw TRT",
    "torch_compile": "torch.compile",
    "hf_eager": "HF Eager",
}


def _bar(value: float, max_val: float, width: int = 20) -> str:
    if max_val <= 0:
        return ""
    filled = int(round(value / max_val * width))
    filled = min(filled, width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def print_report(model_name: str, prompt: str, num_input_tokens: int,
                 max_new_tokens: int, iterations: int, warmup: int,
                 results: dict[str, dict],
                 backend_dtypes: dict[str, str] | None = None):
    gpu = get_gpu_name()
    trt_ver = get_trt_version()

    backends = list(results.keys())
    eager_decode = stats(results["hf_eager"]["decode_times"])["mean"] \
        if "hf_eager" in results else None

    # Compute metrics per backend
    metrics = {}
    for b in backends:
        res = results[b]
        prefill = stats(res["prefill_times"])
        decode = stats(res["decode_times"])
        n = res.get("num_tokens", max_new_tokens)
        per_tok = decode["mean"] / n if n > 0 else 0
        tps = 1000.0 * n / decode["mean"] if decode["mean"] > 0 else 0
        total_vals = [p + d for p, d in zip(res["prefill_times"],
                                             res["decode_times"])]
        total = stats(total_vals)
        sp = eager_decode / decode["mean"] \
            if eager_decode and decode["mean"] > 0 else 0

        metrics[b] = {
            "prefill": prefill,
            "decode": decode,
            "per_tok": per_tok,
            "tps": tps,
            "total": total,
            "speedup": sp,
        }

    best_backend = min(backends,
                       key=lambda b: metrics[b]["decode"]["mean"])

    # Header
    prompt_display = prompt[:50] + ("..." if len(prompt) > 50 else "")
    print()
    print(f"{_BOLD}{'=' * 76}{_RESET}")
    print(f"{_BOLD}  INFERENCE PERFORMANCE COMPARISON{_RESET}")
    print(f"{'=' * 76}")
    print(f"  Model:      {_CYAN}{model_name}{_RESET}")
    print(f"  GPU:        {gpu}")
    print(f"  TensorRT:   {trt_ver}")
    if backend_dtypes:
        dtype_parts = [f"{_BACKEND_LABELS.get(k, k)}: {v}"
                       for k, v in backend_dtypes.items()]
        print(f"  Precision:  {', '.join(dtype_parts)}")
    print(f"  Prompt:     \"{prompt_display}\" ({num_input_tokens} tokens)")
    print(f"  Generation: {max_new_tokens} tokens, "
          f"{iterations} iterations, {warmup} warmup")
    print(f"{'=' * 76}")

    label_w = 18
    col_w = 16

    header = f"  {'':>{label_w}}"
    for b in backends:
        header += f"  {_BOLD}{_BACKEND_LABELS.get(b, b):>{col_w}}{_RESET}"
    print(header)
    print(f"  {'\u2500' * (label_w + (col_w + 2) * len(backends))}")

    # Prefill
    row = f"  {'Prefill (ms)':>{label_w}}"
    for b in backends:
        m = metrics[b]["prefill"]
        row += f"  {m['mean']:>{col_w - 8}.1f} \u00b1 {m['std']:<.1f}ms"
    print(row)

    # Decode
    row = f"  {'Decode (ms)':>{label_w}}"
    for b in backends:
        m = metrics[b]["decode"]
        val = f"{m['mean']:.1f} \u00b1 {m['std']:.1f}"
        if b == best_backend:
            row += f"  {_GREEN}{val:>{col_w}}{_RESET}"
        else:
            row += f"  {val:>{col_w}}"
    print(row)

    # Per-token
    row = f"  {'Per-token (ms)':>{label_w}}"
    for b in backends:
        val = f"{metrics[b]['per_tok']:.2f}"
        if b == best_backend:
            row += f"  {_GREEN}{val:>{col_w}}{_RESET}"
        else:
            row += f"  {val:>{col_w}}"
    print(row)

    # Throughput
    row = f"  {'Throughput (t/s)':>{label_w}}"
    for b in backends:
        val = f"{metrics[b]['tps']:.1f}"
        if b == best_backend:
            row += f"  {_GREEN}{val:>{col_w}}{_RESET}"
        else:
            row += f"  {val:>{col_w}}"
    print(row)

    # Total
    row = f"  {'Total (ms)':>{label_w}}"
    for b in backends:
        m = metrics[b]["total"]
        val = f"{m['mean']:.1f} \u00b1 {m['std']:.1f}"
        row += f"  {val:>{col_w}}"
    print(row)

    print(f"  {'\u2500' * (label_w + (col_w + 2) * len(backends))}")

    # Speedup row
    if eager_decode:
        row = f"  {_BOLD}{'vs Eager':>{label_w}}{_RESET}"
        for b in backends:
            sp = metrics[b]["speedup"]
            if b == "hf_eager":
                val = "baseline"
            elif sp >= 2.0:
                val = f"{_GREEN}{sp:.2f}x{_RESET}"
            elif sp >= 1.0:
                val = f"{_YELLOW}{sp:.2f}x{_RESET}"
            else:
                val = f"{_RED}{sp:.2f}x{_RESET}"
            raw_val = f"{sp:.2f}x" if b != "hf_eager" else "baseline"
            pad = col_w - len(raw_val)
            row += f"  {' ' * pad}{val}"
        print(row)

    # Visual bars
    print()
    print(f"  {_BOLD}Decode latency (lower is better):{_RESET}")
    max_decode = max(metrics[b]["decode"]["mean"] for b in backends)
    for b in backends:
        d = metrics[b]["decode"]["mean"]
        bar = _bar(d, max_decode, width=30)
        label = _BACKEND_LABELS.get(b, b)
        if b == best_backend:
            print(f"    {label:<14s} {_GREEN}{bar}{_RESET}  {d:.1f}ms")
        else:
            print(f"    {label:<14s} {bar}  {d:.1f}ms")

    # Token agreement
    print()
    if "hf_eager" in results:
        ref_ids = results["hf_eager"]["gen_ids"]
        for b in backends:
            if b == "hf_eager":
                continue
            b_ids = results[b]["gen_ids"]
            match = ref_ids == b_ids
            label = _BACKEND_LABELS.get(b, b)
            if match:
                print(f"  {_GREEN}\u2713{_RESET} {label}: tokens MATCH eager")
            else:
                n_match = sum(1 for a, c in zip(ref_ids, b_ids) if a == c)
                total = max(len(ref_ids), len(b_ids))
                print(f"  {_RED}\u2717{_RESET} {label}: {n_match}/{total} tokens match")

    # Compile time
    if "torch_compile" in results:
        ct = results["torch_compile"].get("compile_time_s", 0)
        if ct:
            print(f"\n  {_DIM}torch.compile setup: {ct:.1f}s "
                  f"(not included in timings){_RESET}")

    print(f"\n  {_DIM}* Prefill: HF/compile batch all tokens; "
          f"TRT pipelines are token-by-token{_RESET}")
    print(f"  {_DIM}* Excludes: model loading, tokenization, "
          f"engine build{_RESET}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified 4-way inference performance comparison")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--torchtrt-bundle",
                        help="Pre-built Torch-TRT .trtfb bundle")
    parser.add_argument("--rawtrt-bundle",
                        help="Pre-built Raw TRT .trtfb bundle")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--max-cache-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--precision", default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="Compute precision for Torch-TRT and HF models (default: fp16)")
    parser.add_argument("--skip-torchtrt", action="store_true",
                        help="Skip Torch-TRT benchmark")
    parser.add_argument("--skip-rawtrt", action="store_true",
                        help="Skip Raw TRT benchmark")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip torch.compile benchmark")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = precision_to_torch_dtype(args.precision)
    precision_label = dtype_label(args.precision)

    results = {}
    backend_dtypes = {}
    model_id = args.model

    print(f"\n{_BOLD}[perf] Starting 4-way benchmark: {model_id} "
          f"(precision={args.precision}){_RESET}",
          file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    print(f"[perf] Prompt: {len(input_ids)} tokens", file=sys.stderr)

    # 1) Torch-TRT
    if not args.skip_torchtrt:
        print(f"\n{_CYAN}[1/4] Torch-TRT ({precision_label}){_RESET}",
              file=sys.stderr)
        bundle_path = args.torchtrt_bundle
        if not bundle_path:
            bundle_path = build_torchtrt_bundle(
                model_id, args.max_cache_length, args.verbose,
                precision=args.precision)

        print(f"  Benchmarking ({args.warmup} warmup + "
              f"{args.iterations} iter) ...", file=sys.stderr)
        results["torch_trt"] = bench_torchtrt_bundle(
            bundle_path, tokenizer, args.prompt, args.max_new_tokens,
            args.warmup, args.iterations, args.verbose)
        backend_dtypes["torch_trt"] = precision_label
        gc.collect()
        torch.cuda.empty_cache()

    # 2) Raw TRT
    if not args.skip_rawtrt:
        print(f"\n{_CYAN}[2/4] Raw TRT (float32){_RESET}", file=sys.stderr)
        if args.rawtrt_bundle:
            from trtf_build.debug_runner import load_engine_from_bundle
            engine_plan, header = load_engine_from_bundle(args.rawtrt_bundle)
            num_layers = header["num_layers"]
            max_cache = header.get("max_cache_length", args.max_cache_length)
        else:
            engine_plan, num_layers, max_cache = build_rawtrt_engine(
                model_id, args.max_cache_length, args.verbose)

        print(f"  Benchmarking ({args.warmup} warmup + "
              f"{args.iterations} iter) ...", file=sys.stderr)
        results["raw_trt"] = bench_rawtrt(
            engine_plan, num_layers, max_cache,
            tokenizer, args.prompt, args.max_new_tokens,
            args.warmup, args.iterations, args.verbose)
        backend_dtypes["raw_trt"] = "float32"
        del engine_plan
        gc.collect()
        torch.cuda.empty_cache()

    # 3) HF eager
    print(f"\n{_CYAN}[3/4] HF Eager ({precision_label}){_RESET}",
          file=sys.stderr)
    print(f"  Loading HF model ({precision_label}) ...", file=sys.stderr)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, device_map="cuda",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    backend_dtypes["hf_eager"] = precision_label

    print(f"  Benchmarking ({args.warmup} warmup + "
          f"{args.iterations} iter) ...", file=sys.stderr)
    results["hf_eager"] = bench_hf_eager(
        hf_model, tokenizer, args.prompt, args.max_new_tokens,
        args.warmup, args.iterations, args.verbose)

    # 4) torch.compile
    if not args.skip_compile:
        print(f"\n{_CYAN}[4/4] torch.compile ({precision_label}){_RESET}",
              file=sys.stderr)
        try:
            results["torch_compile"] = bench_torch_compile(
                hf_model, tokenizer, args.prompt, args.max_new_tokens,
                args.warmup, args.iterations, args.verbose)
            backend_dtypes["torch_compile"] = precision_label
        except Exception as e:
            print(f"  {_RED}torch.compile failed: {e}{_RESET}",
                  file=sys.stderr)

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # Report
    print_report(
        model_id, args.prompt, len(input_ids), args.max_new_tokens,
        args.iterations, args.warmup, results,
        backend_dtypes=backend_dtypes)

    if args.json_path:
        data = build_json_output(
            model_id, args.prompt, len(input_ids), args.max_new_tokens,
            args.iterations, args.warmup, results)
        data["metadata"]["precision"] = args.precision
        data["metadata"]["backend_dtypes"] = backend_dtypes
        with open(args.json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[perf] Results saved to {args.json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
