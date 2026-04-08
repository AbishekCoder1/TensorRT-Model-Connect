#!/usr/bin/env python3
"""Torch-TRT vs torch.compile vs HF eager performance comparison.

Compares three inference backends for the same HF model:
  1. Torch-TRT (StatelessCacheWrapper -> raw TRT engine via ttrt_build)
  2. torch.compile (PyTorch 2.x compiler with default backend)
  3. HF eager (baseline, no compilation)

All three use token-by-token autoregressive decoding with KV cache.
Backends run serially to avoid GPU memory contention.

Usage:
    python3 tools/perf_compare_torchtrt.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "The capital of France is" \
      --max-new-tokens 50

    # Use a pre-built Torch-TRT bundle
    python3 tools/perf_compare_torchtrt.py \
      --model Qwen/Qwen3-0.6B \
      --bundle /tmp/qwen3.trtfb \
      --prompt "The capital of France is" \
      --max-new-tokens 50

    # Save results as JSON
    python3 tools/perf_compare_torchtrt.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "Hello" --max-new-tokens 20 \
      --json results.json

    # Skip torch.compile
    python3 tools/perf_compare_torchtrt.py \
      --model Qwen/Qwen3-0.6B --skip-compile
"""

from __future__ import annotations

import argparse
import gc
import json
import sys

from perf_utils import (
    get_gpu_name, stats, fmt, speedup,
    bench_hf_eager, bench_torch_compile,
    bench_torchtrt_bundle, build_torchtrt_bundle,
    build_json_output,
    precision_to_torch_dtype, dtype_label,
)


# ---------------------------------------------------------------------------
# Reporting (simple table format for this tool)
# ---------------------------------------------------------------------------

def print_report(model_name: str, prompt: str, num_input_tokens: int,
                 max_new_tokens: int, iterations: int, warmup: int,
                 results: dict[str, dict],
                 backend_dtypes: dict[str, str] | None = None):
    """Print formatted comparison table."""
    gpu = get_gpu_name()
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"Performance Comparison: {model_name}")
    print(f"GPU: {gpu}")
    print(f'Prompt: "{prompt[:60]}{"..." if len(prompt) > 60 else ""}" '
          f'({num_input_tokens} tokens)')
    print(f"Max new tokens: {max_new_tokens}, "
          f"{iterations} iterations, {warmup} warmup")
    if backend_dtypes:
        dtype_strs = [f"{k}: {v}" for k, v in backend_dtypes.items()]
        print(f"Precision: {', '.join(dtype_strs)}")
    print(sep)

    backends = list(results.keys())
    header = f"{'':>20s}"
    for b in backends:
        header += f"  {b:>16s}"
    if "hf_eager" in results and len(backends) > 1:
        header += f"  {'Speedup':>8s}"
    print(header)
    print("-" * len(header))

    # Prefill
    row = f"  {'Prefill (ms)':>18s}:"
    prefill_means = {}
    for b in backends:
        s = stats(results[b]["prefill_times"])
        prefill_means[b] = s["mean"]
        row += f"  {fmt(s['mean'], s['std']):>16s}"
    if "hf_eager" in results and len(backends) > 1:
        ref = prefill_means.get("hf_eager", 0)
        best_other = min(v for k, v in prefill_means.items() if k != "hf_eager")
        row += f"  {speedup(ref, best_other):>8s}"
    print(row)

    # Decode
    row = f"  {'Decode (ms)':>18s}:"
    decode_means = {}
    for b in backends:
        s = stats(results[b]["decode_times"])
        decode_means[b] = s["mean"]
        row += f"  {fmt(s['mean'], s['std']):>16s}"
    if "hf_eager" in results and len(backends) > 1:
        ref = decode_means.get("hf_eager", 0)
        best_other = min(v for k, v in decode_means.items() if k != "hf_eager")
        row += f"  {speedup(ref, best_other):>8s}"
    print(row)

    # Per-token
    row = f"  {'Per-token (ms)':>18s}:"
    per_tok = {}
    for b in backends:
        d = stats(results[b]["decode_times"])
        n = results[b].get("num_tokens", max_new_tokens)
        pt = d["mean"] / n if n > 0 else 0
        per_tok[b] = pt
        row += f"  {f'{pt:.2f}':>16s}"
    if "hf_eager" in results and len(backends) > 1:
        ref = per_tok.get("hf_eager", 0)
        best_other = min(v for k, v in per_tok.items() if k != "hf_eager" and v > 0)
        row += f"  {speedup(ref, best_other):>8s}"
    print(row)

    # Throughput
    row = f"  {'Throughput (t/s)':>18s}:"
    tps = {}
    for b in backends:
        d = stats(results[b]["decode_times"])
        n = results[b].get("num_tokens", max_new_tokens)
        t = 1000.0 * n / d["mean"] if d["mean"] > 0 else 0
        tps[b] = t
        row += f"  {f'{t:.1f}':>16s}"
    if "hf_eager" in results and len(backends) > 1:
        best_tps = max(v for k, v in tps.items() if k != "hf_eager")
        ref_tps = tps.get("hf_eager", 0)
        row += f"  {speedup(best_tps, ref_tps):>8s}"
    print(row)

    # Total
    row = f"  {'Total (ms)':>18s}:"
    totals = {}
    for b in backends:
        ps = results[b]["prefill_times"]
        ds = results[b]["decode_times"]
        total_vals = [p + d for p, d in zip(ps, ds)]
        s = stats(total_vals)
        totals[b] = s["mean"]
        row += f"  {fmt(s['mean'], s['std']):>16s}"
    if "hf_eager" in results and len(backends) > 1:
        ref = totals.get("hf_eager", 0)
        best_other = min(v for k, v in totals.items() if k != "hf_eager")
        row += f"  {speedup(ref, best_other):>8s}"
    print(row)

    print()

    # Token agreement
    if "hf_eager" in results:
        ref_ids = results["hf_eager"]["gen_ids"]
        for b in backends:
            if b == "hf_eager":
                continue
            b_ids = results[b]["gen_ids"]
            match = ref_ids == b_ids
            if match:
                print(f"  {b} vs hf_eager: tokens MATCH")
            else:
                n_match = sum(1 for a, b_ in zip(ref_ids, b_ids) if a == b_)
                total = max(len(ref_ids), len(b_ids))
                print(f"  {b} vs hf_eager: {n_match}/{total} tokens match")

    # Compile time
    if "torch_compile" in results:
        ct = results["torch_compile"].get("compile_time_s", 0)
        if ct:
            print(f"  torch.compile setup time: {ct:.1f}s (not included in timings)")

    print("\n  * Prefill: HF batches all tokens; TRT/compile are token-by-token")
    print("  * Excludes: model loading, tokenization, engine build")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Torch-TRT vs torch.compile vs HF eager performance comparison")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--bundle",
                        help="Pre-built Torch-TRT .trtfb bundle (skips build)")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--max-cache-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--precision", default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="Compute precision for Torch-TRT and HF models (default: fp16)")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip torch.compile benchmark")
    parser.add_argument("--skip-torchtrt", action="store_true",
                        help="Skip Torch-TRT benchmark")
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
    model_id = args.model
    print(f"[perf] Model: {model_id} (precision={args.precision})",
          file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    print(f"[perf] Prompt: {len(input_ids)} tokens", file=sys.stderr)

    # Track per-backend dtypes
    backend_dtypes = {}

    # 1) Torch-TRT
    if not args.skip_torchtrt:
        bundle_path = args.bundle
        if not bundle_path:
            bundle_path = build_torchtrt_bundle(
                model_id, args.max_cache_length, args.verbose,
                precision=args.precision)

        print(f"[perf] Benchmarking Torch-TRT ({args.warmup} warmup + "
              f"{args.iterations} iter) ...", file=sys.stderr)
        results["torch_trt"] = bench_torchtrt_bundle(
            bundle_path, tokenizer, args.prompt, args.max_new_tokens,
            args.warmup, args.iterations, args.verbose)
        backend_dtypes["torch_trt"] = precision_label
        gc.collect()
        torch.cuda.empty_cache()

    # 2) HF eager
    print(f"[perf] Loading HF model ({precision_label}) ...", file=sys.stderr)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, device_map="cuda",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    backend_dtypes["hf_eager"] = precision_label

    print(f"[perf] Benchmarking HF eager ({args.warmup} warmup + "
          f"{args.iterations} iter) ...", file=sys.stderr)
    results["hf_eager"] = bench_hf_eager(
        hf_model, tokenizer, args.prompt, args.max_new_tokens,
        args.warmup, args.iterations, args.verbose)

    # 3) torch.compile
    if not args.skip_compile:
        print(f"[perf] Benchmarking torch.compile ({args.warmup} warmup + "
              f"{args.iterations} iter) ...", file=sys.stderr)
        try:
            results["torch_compile"] = bench_torch_compile(
                hf_model, tokenizer, args.prompt, args.max_new_tokens,
                args.warmup, args.iterations, args.verbose)
            backend_dtypes["torch_compile"] = precision_label
        except Exception as e:
            print(f"[perf] torch.compile failed: {e}", file=sys.stderr)

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
        print(f"\n[perf] Results saved to {args.json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
