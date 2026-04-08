"""Shared utilities for performance comparison tools.

Contains helpers, benchmark backends, and reporting used by both
perf_compare_torchtrt.py and perf_compare_all.py.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------

PRECISION_DTYPE_MAP: dict[str, str] = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}


def precision_to_torch_dtype(precision: str):
    """Convert a precision string (fp16/bf16/fp32) to a torch.dtype."""
    import torch
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if precision not in mapping:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown precision {precision!r}. Valid: {valid}")
    return mapping[precision]


def dtype_label(precision: str) -> str:
    """Human-readable label for a precision string."""
    return PRECISION_DTYPE_MAP.get(precision, precision)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gpu_name() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def get_trt_version() -> str:
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return "unknown"


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": s, "values": values}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.1f} +/- {std:.1f}"


def speedup(baseline: float, candidate: float) -> str:
    if candidate <= 0:
        return "N/A"
    return f"{baseline / candidate:.2f}x"


def resolve_model(model_id: str) -> str:
    """Resolve HF repo ID or local path to a local directory."""
    from pathlib import Path
    p = Path(model_id)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id)
    except Exception:
        pass
    try:
        from trtf_build.engine_builder import _resolve_model as _trtf_resolve
        return _trtf_resolve(model_id)
    except Exception:
        raise FileNotFoundError(
            f"Cannot resolve model: {model_id!r}. "
            "Pass a local directory or valid HF repo ID.")


# ---------------------------------------------------------------------------
# Benchmark backends
# ---------------------------------------------------------------------------

def bench_hf_eager(model, tokenizer, prompt: str, max_new_tokens: int,
                   warmup: int, iterations: int, verbose: bool) -> dict:
    """Benchmark HF eager inference (no compilation)."""
    import torch

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    prefill_times, decode_times, gen_ids_last = [], [], []

    for run_idx in range(warmup + iterations):
        is_warmup = run_idx < warmup
        gen_ids = []

        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids, use_cache=True)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000

            logits = out.logits[:, -1, :]
            next_token = int(logits[0].argmax())
            gen_ids.append(next_token)
            past = out.past_key_values

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                next_input = torch.tensor([[next_token]], device="cuda")
                out = model(next_input, past_key_values=past, use_cache=True)
                logits = out.logits[:, -1, :]
                next_token = int(logits[0].argmax())
                gen_ids.append(next_token)
                past = out.past_key_values
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            gen_ids_last = gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [eager {tag}] prefill={prefill_ms:.1f}ms "
                  f"decode={decode_ms:.1f}ms", file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "gen_ids": gen_ids_last,
        "num_tokens": max_new_tokens,
    }


def bench_torch_compile(model, tokenizer, prompt: str, max_new_tokens: int,
                         warmup: int, iterations: int, verbose: bool) -> dict:
    """Benchmark torch.compile inference."""
    import torch

    print("  Compiling model with torch.compile ...", file=sys.stderr)
    t0 = time.perf_counter()
    compiled = torch.compile(model)
    compile_time = time.perf_counter() - t0

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    prefill_times, decode_times, gen_ids_last = [], [], []
    compile_warmup = max(warmup, 3)

    for run_idx in range(compile_warmup + iterations):
        is_warmup = run_idx < compile_warmup
        gen_ids = []

        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = compiled(input_ids, use_cache=True)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000

            logits = out.logits[:, -1, :]
            next_token = int(logits[0].argmax())
            gen_ids.append(next_token)
            past = out.past_key_values

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                next_input = torch.tensor([[next_token]], device="cuda")
                out = compiled(next_input, past_key_values=past, use_cache=True)
                logits = out.logits[:, -1, :]
                next_token = int(logits[0].argmax())
                gen_ids.append(next_token)
                past = out.past_key_values
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            gen_ids_last = gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - compile_warmup + 1}"
            print(f"  [compile {tag}] prefill={prefill_ms:.1f}ms "
                  f"decode={decode_ms:.1f}ms", file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "gen_ids": gen_ids_last,
        "num_tokens": max_new_tokens,
        "compile_time_s": compile_time,
    }


# ---------------------------------------------------------------------------
# Torch-TRT runner and benchmark
# ---------------------------------------------------------------------------

class TorchTrtRunner:
    """TRT runner for Torch-TRT bundles (cache_kv_N / outputN naming).

    Mirrors TrtRunner from debug_runner.py but uses the Torch-TRT tensor naming
    convention: cache_kv_{2i}/cache_kv_{2i+1} for K/V cache inputs, and
    output0 (logits) / output{2i+1}/output{2i+2} for outputs.
    """

    def __init__(self, engine_plan: bytes, max_cache_length: int, num_layers: int):
        import tensorrt as trt
        from cuda import cudart

        self.max_cache_length = max_cache_length
        self.num_layers = num_layers
        self._cudart = cudart

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()

        cache_shape = tuple(self.engine.get_tensor_shape("cache_kv_0"))
        self.attention_size = cache_shape[1]

        err, self.stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess

        self.cache_length = 0
        attention_window = max_cache_length + 1
        row_bytes = self.attention_size * 4
        cache_bytes = max_cache_length * row_bytes

        self._d_cache = []
        for _ in range(num_layers * 2):
            err, ptr = cudart.cudaMalloc(cache_bytes)
            assert err == cudart.cudaError_t.cudaSuccess
            self._d_cache.append(ptr)

        self._d_present = []
        for _ in range(num_layers * 2):
            err, ptr = cudart.cudaMalloc(row_bytes)
            assert err == cudart.cudaError_t.cudaSuccess
            self._d_present.append(ptr)

        self._h_token_id = np.zeros((1,), dtype=np.int32)
        self._h_position_id = np.zeros((1,), dtype=np.int32)
        self._h_mask = np.zeros((1, attention_window), dtype=np.float32)

        err, self._d_token_id = cudart.cudaMalloc(4)
        assert err == cudart.cudaError_t.cudaSuccess
        err, self._d_position_id = cudart.cudaMalloc(4)
        assert err == cudart.cudaError_t.cudaSuccess
        err, self._d_mask = cudart.cudaMalloc(attention_window * 4)
        assert err == cudart.cudaError_t.cudaSuccess

        logits_shape = tuple(self.engine.get_tensor_shape("output0"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        assert err == cudart.cudaError_t.cudaSuccess

        for ptr in self._d_cache:
            cudart.cudaMemsetAsync(ptr, 0, cache_bytes, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def reset(self):
        cache_bytes = self.max_cache_length * self.attention_size * 4
        for ptr in self._d_cache:
            self._cudart.cudaMemsetAsync(ptr, 0, cache_bytes, self.stream)
        self._cudart.cudaStreamSynchronize(self.stream)
        self.cache_length = 0

    def step(self, token_id: int) -> np.ndarray:
        from cuda import cudart
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        attention_window = self.max_cache_length + 1

        position_id = min(self.cache_length, self.max_cache_length)
        self._h_mask[:] = -1e9
        valid = min(self.cache_length, self.max_cache_length)
        self._h_mask[0, :valid] = 0.0
        self._h_mask[0, -1] = 0.0

        self._h_token_id[0] = token_id
        self._h_position_id[0] = position_id

        cudart.cudaMemcpyAsync(self._d_token_id, self._h_token_id.ctypes.data,
                                4, H2D, stream)
        cudart.cudaMemcpyAsync(self._d_position_id, self._h_position_id.ctypes.data,
                                4, H2D, stream)
        cudart.cudaMemcpyAsync(self._d_mask, self._h_mask.ctypes.data,
                                attention_window * 4, H2D, stream)

        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("position_id", self._d_position_id)
        self.context.set_tensor_address("attention_mask", self._d_mask)
        self.context.set_tensor_address("output0", self._d_logits)

        for i in range(self.num_layers * 2):
            self.context.set_tensor_address(f"cache_kv_{i}", self._d_cache[i])
            self.context.set_tensor_address(f"output{i + 1}", self._d_present[i])

        self.context.execute_async_v3(stream)

        row_bytes = self.attention_size * 4
        for i in range(self.num_layers * 2):
            cache_buf = self._d_cache[i]
            present_buf = self._d_present[i]
            if self.cache_length < self.max_cache_length:
                offset = self.cache_length * row_bytes
                cudart.cudaMemcpyAsync(cache_buf + offset, present_buf,
                                        row_bytes, D2D, stream)
            else:
                cudart.cudaMemcpyAsync(cache_buf, cache_buf + row_bytes,
                                        (self.max_cache_length - 1) * row_bytes,
                                        D2D, stream)
                offset = (self.max_cache_length - 1) * row_bytes
                cudart.cudaMemcpyAsync(cache_buf + offset, present_buf,
                                        row_bytes, D2D, stream)

        cudart.cudaMemcpyAsync(self._h_logits.ctypes.data, self._d_logits,
                                self._logits_numel * 4, D2H, stream)
        cudart.cudaStreamSynchronize(stream)
        self.cache_length = min(self.cache_length + 1, self.max_cache_length)
        return self._h_logits.copy()


def bench_torchtrt_bundle(bundle_path: str, tokenizer, prompt: str,
                           max_new_tokens: int, warmup: int, iterations: int,
                           verbose: bool) -> dict:
    """Benchmark Torch-TRT bundle via dedicated TRT runner."""
    from trtf_build.debug_runner import load_engine_from_bundle

    engine_plan, header = load_engine_from_bundle(bundle_path)
    num_layers = header["num_layers"]
    max_cache_length = header.get("max_cache_length", 256)

    runner = TorchTrtRunner(
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
            logits = runner.step(tid)
        logits = logits.flatten()
        prefill_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits))
            gen_ids.append(next_token)
            logits = runner.step(next_token).flatten()
        decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            gen_ids_last = gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [torch-trt {tag}] prefill={prefill_ms:.1f}ms "
                  f"decode={decode_ms:.1f}ms", file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "gen_ids": gen_ids_last,
        "num_tokens": max_new_tokens,
    }


# ---------------------------------------------------------------------------
# Engine builders
# ---------------------------------------------------------------------------

def build_torchtrt_bundle(model_id: str, max_cache_length: int,
                           verbose: bool,
                           precision: str = "fp16") -> str:
    """Build a Torch-TRT bundle and return the path."""
    import tempfile
    out_path = tempfile.mktemp(suffix=".trtfb")
    model_dir = resolve_model(model_id)
    print(f"  Building Torch-TRT engine (precision={precision}) ...",
          file=sys.stderr)
    t0 = time.perf_counter()
    from ttrt_build.compiler import build_bundle
    build_bundle(model_dir, out_path, max_cache_length=max_cache_length,
                 precision=precision, verbose=verbose)
    build_s = time.perf_counter() - t0
    print(f"  Torch-TRT engine built in {build_s:.1f}s", file=sys.stderr)
    return out_path


# ---------------------------------------------------------------------------
# JSON output builder
# ---------------------------------------------------------------------------

def build_json_output(model_name: str, prompt: str, num_input_tokens: int,
                      max_new_tokens: int, iterations: int, warmup: int,
                      results: dict[str, dict]) -> dict:
    """Build structured JSON output from benchmark results."""
    out = {
        "metadata": {
            "model": model_name,
            "gpu": get_gpu_name(),
            "trt_version": get_trt_version(),
            "prompt": prompt,
            "num_input_tokens": num_input_tokens,
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "iterations": iterations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "backends": {},
    }

    for name, res in results.items():
        prefill = stats(res["prefill_times"])
        decode = stats(res["decode_times"])
        n = res.get("num_tokens", max_new_tokens)
        total_vals = [p + d for p, d in zip(res["prefill_times"],
                                             res["decode_times"])]
        total = stats(total_vals)

        entry = {
            "prefill_ms": prefill,
            "decode_ms": decode,
            "total_ms": total,
            "per_token_ms": decode["mean"] / n if n > 0 else 0,
            "throughput_tps": 1000.0 * n / decode["mean"] if decode["mean"] > 0 else 0,
            "num_tokens": n,
        }
        if "compile_time_s" in res:
            entry["compile_time_s"] = res["compile_time_s"]
        if "gen_ids" in res:
            entry["gen_ids"] = res["gen_ids"]
        out["backends"][name] = entry

    # Speedups vs eager
    if "hf_eager" in results:
        ref_decode = stats(results["hf_eager"]["decode_times"])["mean"]
        speedups = {}
        for name in results:
            if name == "hf_eager":
                continue
            cand_decode = stats(results[name]["decode_times"])["mean"]
            speedups[name] = round(ref_decode / cand_decode, 3) \
                if cand_decode > 0 else None
        out["decode_speedup_vs_eager"] = speedups

    # Token agreement
    if "hf_eager" in results:
        ref_ids = results["hf_eager"].get("gen_ids", [])
        agreement = {}
        for name in results:
            if name == "hf_eager":
                continue
            agreement[name] = results[name].get("gen_ids", []) == ref_ids
        out["token_match_vs_eager"] = agreement

    return out
