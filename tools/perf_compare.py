#!/usr/bin/env python3
"""TRT vs HuggingFace inference performance comparison.

Runs both backends in-process Python for a controlled, apples-to-apples
comparison. TRT uses TrtRunner (debug_runner.py); HF uses
AutoModelForCausalLM on CUDA with KV cache enabled.

TRT and HF run serially (not simultaneously), so large models that
exceed GPU memory when loaded together are supported.  Use --dtype
float16 (default) to reduce HF memory usage.

Usage:
    # Build TRT engine on the fly from HF model
    python3 tools/perf_compare.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "The capital of France is" \
      --max-new-tokens 20 \
      --max-cache-length 256 \
      --warmup 2 --iterations 5

    # Use a pre-built bundle (skips engine build)
    python3 tools/perf_compare.py \
      --model Qwen/Qwen3-0.6B \
      --bundle /path/to/qwen3.trtfb \
      --prompt "The capital of France is" \
      --max-new-tokens 20

    # Save results as JSON
    python3 tools/perf_compare.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "Hello" --max-new-tokens 20 \
      --json results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_gpu_name() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def _get_trt_version() -> str:
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return "unknown"


def build_trt_engine(model_id_or_path: str, max_cache_length: int,
                     verbose: bool):
    """Build TRT engine and return (engine_plan_bytes, config, model_dir)."""
    from trtf_build.engine_builder import _resolve_model
    from trtf_build.config import ModelConfig
    from trtf_build.families import find_plugin

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None:
        raise ValueError(f"No plugin for model_type={config.model_type!r}")

    # Reject unsupported model types
    rt = getattr(plugin, "runtime_strategy", None)
    if rt == "vision_language":
        raise SystemExit(
            "ERROR: Vision-language models are not supported by perf_compare. "
            "Use tools/diff_vl.py instead.")

    print(f"[perf] Loading weights ({config.model_type}) ...", file=sys.stderr)
    weights = plugin.load_weights(model_dir, config)
    print(f"[perf] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = plugin.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    print(f"[perf] Engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    is_mamba = rt == "ssm_recurrent"
    return engine_plan, config, model_dir, is_mamba


def load_trt_from_bundle(bundle_path: str):
    """Load TRT engine from a pre-built bundle.

    Returns (engine_plan_bytes, num_layers, max_cache_length, bundle_config,
             is_mamba).
    """
    from trtf_build.debug_runner import load_engine_from_bundle, \
        load_config_from_bundle

    engine_plan, header = load_engine_from_bundle(bundle_path)
    bundle_config = load_config_from_bundle(bundle_path)

    # Reject unsupported runtime strategies
    rt = bundle_config.get("runtime_strategy", "decoder_kv_cache")
    if rt == "vision_language":
        raise SystemExit(
            "ERROR: VL bundles are not supported. Use tools/diff_vl.py.")

    is_mamba = rt == "ssm_recurrent"
    return (engine_plan, header["num_layers"],
            header.get("max_cache_length", 0), bundle_config, is_mamba)


def load_hf_model(model_dir: str, dtype: str, trust_remote_code: bool):
    """Load HF model on CUDA with the specified dtype."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype!r}. "
                         f"Choose from: {list(dtype_map)}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=False,
            torch_dtype=torch_dtype).to("cuda")
    except (ValueError, KeyError, ImportError) as e:
        if trust_remote_code:
            print(f"[perf] Native loading failed ({e}), "
                  f"retrying with trust_remote_code=True ...",
                  file=sys.stderr)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True,
                torch_dtype=torch_dtype).to("cuda")
        else:
            raise ValueError(
                f"Failed to load model from {model_dir}. "
                f"If custom code is required, use --trust-remote-code. "
                f"Error: {e}"
            ) from e

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Timing stats helper
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    """Return mean/std/values dict for a list of measurements."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": s, "values": values}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_trt(engine_plan: bytes, num_layers: int, max_cache_length: int,
              input_ids: list[int], max_new_tokens: int,
              warmup: int, iterations: int, eos_token_id: int | None,
              verbose: bool) -> dict:
    """Benchmark TRT inference via PerfTrtRunner (device-resident KV cache).

    Returns dict with timing lists and generated token IDs.
    """
    from trtf_build.debug_runner import PerfTrtRunner

    # Create runner once (deserialization outside timing)
    runner = PerfTrtRunner(
        engine_plan=engine_plan,
        max_cache_length=max_cache_length,
        num_layers=num_layers,
    )

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_token_counts: list[int] = []
    gen_ids: list[int] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup

        # Reset device-side cache
        runner.reset()

        # -- Prefill --
        t0 = time.perf_counter()
        for tid in input_ids:
            logits = runner.step(tid)
        prefill_ms = (time.perf_counter() - t0) * 1000

        # -- Decode --
        tokens_generated = 0
        run_gen_ids: list[int] = []
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits))
            run_gen_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            logits = runner.step(next_token)
            tokens_generated += 1
        decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            decode_token_counts.append(tokens_generated)
            gen_ids = run_gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [trt {tag}] prefill={prefill_ms:.2f}ms "
                  f"decode={decode_ms:.2f}ms ({tokens_generated} tokens)",
                  file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "decode_token_counts": decode_token_counts,
        "gen_ids": gen_ids,
    }


def bench_trt_mamba(engine_plan: bytes, num_layers: int,
                    input_ids: list[int], max_new_tokens: int,
                    warmup: int, iterations: int, eos_token_id: int | None,
                    verbose: bool) -> dict:
    """Benchmark TRT inference for Mamba/SSM via PerfMambaTrtRunner.

    Returns dict with timing lists and generated token IDs.
    """
    from trtf_build.debug_runner import PerfMambaTrtRunner

    # Create runner once (deserialization outside timing)
    runner = PerfMambaTrtRunner(
        engine_plan=engine_plan,
        num_layers=num_layers,
    )

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_token_counts: list[int] = []
    gen_ids: list[int] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup

        # Reset device-side recurrent state
        runner.reset()

        # -- Prefill --
        t0 = time.perf_counter()
        for tid in input_ids:
            logits = runner.step(tid)
        prefill_ms = (time.perf_counter() - t0) * 1000

        # -- Decode --
        tokens_generated = 0
        run_gen_ids: list[int] = []
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits))
            run_gen_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            logits = runner.step(next_token)
            tokens_generated += 1
        decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            decode_token_counts.append(tokens_generated)
            gen_ids = run_gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [trt {tag}] prefill={prefill_ms:.2f}ms "
                  f"decode={decode_ms:.2f}ms ({tokens_generated} tokens)",
                  file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "decode_token_counts": decode_token_counts,
        "gen_ids": gen_ids,
    }


def bench_hf(model, input_ids: list[int], max_new_tokens: int,
             warmup: int, iterations: int, eos_token_id: int | None,
             verbose: bool) -> dict:
    """Benchmark HF inference with KV cache.

    Returns dict with timing lists and generated token IDs.
    """
    import torch

    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_token_counts: list[int] = []
    gen_ids: list[int] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup

        with torch.no_grad():
            # -- Prefill --
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(ids_tensor, use_cache=True)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000

            # HF decoders use past_key_values; Mamba uses cache_params.
            is_mamba_hf = hasattr(outputs, "cache_params")
            if is_mamba_hf:
                past = outputs.cache_params
                seq_len = ids_tensor.shape[1]
            else:
                past = outputs.past_key_values
            logits = outputs.logits  # (1, seq_len, vocab)

            # -- Decode --
            tokens_generated = 0
            run_gen_ids: list[int] = []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for step in range(max_new_tokens):
                next_token = int(logits[0, -1].argmax())
                run_gen_ids.append(next_token)
                if eos_token_id is not None and next_token == eos_token_id:
                    break
                next_input = torch.tensor(
                    [[next_token]], dtype=torch.long, device="cuda")
                if is_mamba_hf:
                    cache_pos = torch.tensor(
                        [seq_len + step], dtype=torch.long, device="cuda")
                    outputs = model(
                        next_input, cache_params=past,
                        cache_position=cache_pos, use_cache=True)
                    past = outputs.cache_params
                else:
                    outputs = model(
                        next_input, past_key_values=past,
                        use_cache=True)
                    past = outputs.past_key_values
                logits = outputs.logits
                tokens_generated += 1
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            decode_token_counts.append(tokens_generated)
            gen_ids = run_gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [hf  {tag}] prefill={prefill_ms:.2f}ms "
                  f"decode={decode_ms:.2f}ms ({tokens_generated} tokens)",
                  file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "decode_token_counts": decode_token_counts,
        "gen_ids": gen_ids,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(mean: float, std: float) -> str:
    """Format mean +/- std."""
    return f"{mean:.1f} +/- {std:.1f}"


def _speedup(hf_mean: float, trt_mean: float) -> str:
    """Compute and format speedup (HF/TRT)."""
    if trt_mean <= 0:
        return "N/A"
    return f"{hf_mean / trt_mean:.2f}x"


def print_report(model_name: str, prompt: str, num_input_tokens: int,
                 max_new_tokens: int, iterations: int, warmup: int,
                 hf_dtype: str, trt_res: dict, hf_res: dict,
                 is_mamba: bool = False):
    """Print formatted comparison table to stdout."""
    gpu = _get_gpu_name()
    trt_ver = _get_trt_version()

    trt_prefill = _stats(trt_res["prefill_times"])
    trt_decode = _stats(trt_res["decode_times"])
    hf_prefill = _stats(hf_res["prefill_times"])
    hf_decode = _stats(hf_res["decode_times"])

    # Per-token and throughput (from decode phase)
    trt_avg_tokens = (statistics.mean(trt_res["decode_token_counts"])
                      if trt_res["decode_token_counts"] else 0)
    hf_avg_tokens = (statistics.mean(hf_res["decode_token_counts"])
                     if hf_res["decode_token_counts"] else 0)

    if trt_avg_tokens > 0 and trt_decode["mean"] > 0:
        trt_per_tok = trt_decode["mean"] / trt_avg_tokens
        trt_per_tok_std = trt_decode["std"] / trt_avg_tokens
        trt_tps = 1000.0 * trt_avg_tokens / trt_decode["mean"]
        trt_tps_std = (1000.0 * trt_avg_tokens * trt_decode["std"]
                       / trt_decode["mean"] ** 2)
    else:
        trt_per_tok = trt_per_tok_std = 0.0
        trt_tps = trt_tps_std = 0.0

    if hf_avg_tokens > 0 and hf_decode["mean"] > 0:
        hf_per_tok = hf_decode["mean"] / hf_avg_tokens
        hf_per_tok_std = hf_decode["std"] / hf_avg_tokens
        hf_tps = 1000.0 * hf_avg_tokens / hf_decode["mean"]
        hf_tps_std = (1000.0 * hf_avg_tokens * hf_decode["std"]
                      / hf_decode["mean"] ** 2)
    else:
        hf_per_tok = hf_per_tok_std = 0.0
        hf_tps = hf_tps_std = 0.0

    trt_total = _stats([p + d for p, d in zip(trt_res["prefill_times"],
                                              trt_res["decode_times"])])
    hf_total = _stats([p + d for p, d in zip(hf_res["prefill_times"],
                                             hf_res["decode_times"])])

    # Check token agreement
    trt_gen = trt_res["gen_ids"]
    hf_gen = hf_res["gen_ids"]
    text_match = trt_gen == hf_gen

    prompt_display = prompt[:60] + ("..." if len(prompt) > 60 else "")
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"Perf Comparison: {model_name}")
    print(f"GPU: {gpu}, TRT: {trt_ver}")
    print(f'Prompt: "{prompt_display}" ({num_input_tokens} tokens)')
    print(f"Max new tokens: {max_new_tokens}, "
          f"{iterations} iterations, {warmup} warmup")
    print(f"Token match: {text_match}"
          + ("" if text_match
             else f" (TRT={len(trt_gen)} tokens, HF={len(hf_gen)} tokens)"))
    print(sep)

    hdr = f"{'':>20s}  {'TRT':>16s}  {'HF':>16s}  {'Speedup':>8s}"
    print(hdr)

    rows = [
        ("Prefill (ms)",
         _fmt(trt_prefill["mean"], trt_prefill["std"]),
         _fmt(hf_prefill["mean"], hf_prefill["std"]),
         _speedup(hf_prefill["mean"], trt_prefill["mean"]) + "  *"),
        ("Decode (ms)",
         _fmt(trt_decode["mean"], trt_decode["std"]),
         _fmt(hf_decode["mean"], hf_decode["std"]),
         _speedup(hf_decode["mean"], trt_decode["mean"])),
    ]

    if trt_per_tok > 0 and hf_per_tok > 0:
        rows.append(("Per-token (ms)",
                      f"{trt_per_tok:.2f} +/- {trt_per_tok_std:.2f}",
                      f"{hf_per_tok:.2f} +/- {hf_per_tok_std:.2f}",
                      _speedup(hf_per_tok, trt_per_tok)))
        rows.append(("Throughput (t/s)",
                      f"{trt_tps:.1f} +/- {trt_tps_std:.1f}",
                      f"{hf_tps:.1f} +/- {hf_tps_std:.1f}",
                      _speedup(trt_tps, hf_tps)))

    rows.append(("Total (ms)",
                  _fmt(trt_total["mean"], trt_total["std"]),
                  _fmt(hf_total["mean"], hf_total["std"]),
                  _speedup(hf_total["mean"], trt_total["mean"])))

    for label, trt_val, hf_val, sp in rows:
        print(f"  {label:>18s}:  {trt_val:>16s}  {hf_val:>16s}  {sp:>8s}")

    print()
    if is_mamba:
        print("* Prefill: HF processes full sequence; TRT is token-by-token")
        print("  Decode: both token-by-token with recurrent state")
    else:
        print("* Prefill: HF batches all tokens; TRT processes token-by-token")
        print("  Decode: both token-by-token with KV cache (apples-to-apples)")
    print(f"  Excludes: model loading, tokenization, engine build")
    print(f"  HF dtype: {hf_dtype}")


def build_json_output(model_name: str, prompt: str, num_input_tokens: int,
                      max_new_tokens: int, iterations: int, warmup: int,
                      hf_dtype: str, trt_res: dict, hf_res: dict) -> dict:
    """Build structured JSON output."""
    trt_prefill = _stats(trt_res["prefill_times"])
    trt_decode = _stats(trt_res["decode_times"])
    hf_prefill = _stats(hf_res["prefill_times"])
    hf_decode = _stats(hf_res["decode_times"])

    trt_avg_tokens = (statistics.mean(trt_res["decode_token_counts"])
                      if trt_res["decode_token_counts"] else 0)
    hf_avg_tokens = (statistics.mean(hf_res["decode_token_counts"])
                     if hf_res["decode_token_counts"] else 0)

    def _per_token_stats(decode_stat: dict, avg_tokens: float) -> dict:
        if avg_tokens > 0 and decode_stat["mean"] > 0:
            pt = decode_stat["mean"] / avg_tokens
            pt_std = decode_stat["std"] / avg_tokens
            tps = 1000.0 * avg_tokens / decode_stat["mean"]
            tps_std = (1000.0 * avg_tokens * decode_stat["std"]
                       / decode_stat["mean"] ** 2)
        else:
            pt = pt_std = tps = tps_std = 0.0
        return {
            "per_token_ms": {"mean": pt, "std": pt_std},
            "throughput_tps": {"mean": tps, "std": tps_std},
        }

    trt_tok = _per_token_stats(trt_decode, trt_avg_tokens)
    hf_tok = _per_token_stats(hf_decode, hf_avg_tokens)

    trt_total = _stats([p + d for p, d in zip(trt_res["prefill_times"],
                                              trt_res["decode_times"])])
    hf_total = _stats([p + d for p, d in zip(hf_res["prefill_times"],
                                             hf_res["decode_times"])])

    def _safe_div(a: float, b: float) -> float | None:
        return round(a / b, 3) if b > 0 else None

    return {
        "metadata": {
            "model": model_name,
            "gpu": _get_gpu_name(),
            "trt_version": _get_trt_version(),
            "prompt": prompt,
            "num_input_tokens": num_input_tokens,
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "iterations": iterations,
            "hf_dtype": hf_dtype,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "trt": {
            "prefill_ms": trt_prefill,
            "decode_ms": trt_decode,
            "per_token_ms": trt_tok["per_token_ms"],
            "throughput_tps": trt_tok["throughput_tps"],
            "total_ms": trt_total,
            "num_decode_tokens": int(trt_avg_tokens),
        },
        "hf": {
            "prefill_ms": hf_prefill,
            "decode_ms": hf_decode,
            "per_token_ms": hf_tok["per_token_ms"],
            "throughput_tps": hf_tok["throughput_tps"],
            "total_ms": hf_total,
            "num_decode_tokens": int(hf_avg_tokens),
        },
        "speedup": {
            "prefill": _safe_div(hf_prefill["mean"], trt_prefill["mean"]),
            "decode": _safe_div(hf_decode["mean"], trt_decode["mean"]),
            "per_token": _safe_div(hf_tok["per_token_ms"]["mean"],
                                   trt_tok["per_token_ms"]["mean"]),
            "throughput": _safe_div(trt_tok["throughput_tps"]["mean"],
                                    hf_tok["throughput_tps"]["mean"]),
            "total": _safe_div(hf_total["mean"], trt_total["mean"]),
        },
        "token_match": trt_res["gen_ids"] == hf_res["gen_ids"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TRT vs HuggingFace inference performance comparison")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--bundle",
                        help="Pre-built .trtfb bundle (skips engine build)")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-cache-length", type=int, default=256,
                        help="TRT KV cache length (ignored with --bundle)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup iterations (not counted)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Timed iterations")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="HF model dtype (default: float16)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom code from the HF model repo")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # -- Resolve model directory --
    from trtf_build.engine_builder import _resolve_model
    model_dir = _resolve_model(args.model)

    # -- Tokenize --
    from transformers import AutoTokenizer
    print("[perf] Loading tokenizer ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    print(f"[perf] Prompt: {len(input_ids)} tokens", file=sys.stderr)

    # Determine EOS token ID
    eos_token_id = None
    if tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id

    # -- Load / build TRT engine --
    if args.bundle:
        print(f"[perf] Loading bundle: {args.bundle}", file=sys.stderr)
        engine_plan, num_layers, max_cache_length, _, is_mamba = \
            load_trt_from_bundle(args.bundle)
    else:
        engine_plan, config, _, is_mamba = build_trt_engine(
            args.model, args.max_cache_length, args.verbose)
        num_layers = config.num_hidden_layers
        max_cache_length = args.max_cache_length

    # -- Bench TRT (GPU-exclusive) --
    backend_label = "TRT-Mamba" if is_mamba else "TRT"
    print(f"[perf] Benchmarking {backend_label} ({args.warmup} warmup + "
          f"{args.iterations} iterations) ...", file=sys.stderr)
    if is_mamba:
        trt_res = bench_trt_mamba(
            engine_plan, num_layers,
            input_ids, args.max_new_tokens,
            args.warmup, args.iterations, eos_token_id, args.verbose)
    else:
        trt_res = bench_trt(
            engine_plan, num_layers, max_cache_length,
            input_ids, args.max_new_tokens,
            args.warmup, args.iterations, eos_token_id, args.verbose)
    del engine_plan

    # Free TRT GPU memory before loading HF
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    # -- Bench HF (GPU-exclusive) --
    print(f"[perf] Loading HF model (dtype={args.dtype}) ...", file=sys.stderr)
    hf_model = load_hf_model(model_dir, args.dtype, args.trust_remote_code)
    print(f"[perf] Benchmarking HF ({args.warmup} warmup + "
          f"{args.iterations} iterations) ...", file=sys.stderr)
    hf_res = bench_hf(
        hf_model, input_ids, args.max_new_tokens,
        args.warmup, args.iterations, eos_token_id, args.verbose)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # -- Report --
    print_report(
        args.model, args.prompt, len(input_ids),
        args.max_new_tokens, args.iterations, args.warmup,
        args.dtype, trt_res, hf_res, is_mamba=is_mamba)

    # -- JSON output --
    if args.json_path:
        data = build_json_output(
            args.model, args.prompt, len(input_ids),
            args.max_new_tokens, args.iterations, args.warmup,
            args.dtype, trt_res, hf_res)
        with open(args.json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[perf] Results saved to {args.json_path}", file=sys.stderr)

    # Warn if tokens differ
    if trt_res["gen_ids"] != hf_res["gen_ids"]:
        print("\nWARNING: TRT and HF generated different tokens. "
              "Per-token metrics may not be directly comparable.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
