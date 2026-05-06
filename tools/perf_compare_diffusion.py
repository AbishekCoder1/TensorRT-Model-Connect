#!/usr/bin/env python3
"""Diffusion pipeline performance comparison: HF eager vs torch.compile vs Torch-TRT.

Compares three inference backends for diffusion models (e.g., PixArt-Sigma):
  1. HF eager (diffusers pipeline, baseline)
  2. torch.compile (compiled UNet/transformer)
  3. Torch-TRT (pre-built .trtfb bundle via C++ runtime)

All backends run the full diffusion pipeline: T5 encode → denoising loop → VAE decode.
Timing is broken down by stage. Outputs are saved as PNG for visual comparison.

Usage:
    # All three backends
    python3 tools/perf_compare_diffusion.py \
      --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
      --bundle /workspace/tensorrt-model-connect/engines/pixart_sigma_v8.trtfb \
      --prompt "A photo of a dog chewing on a bone" \
      --output-dir /workspace/tensorrt-model-connect/outputs/pixart/perf

    # Skip torch.compile (slow compilation)
    python3 tools/perf_compare_diffusion.py \
      --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
      --bundle /workspace/tensorrt-model-connect/engines/pixart_sigma_v8.trtfb \
      --prompt "A photo of a dog chewing on a bone" \
      --skip-compile

    # Save results as JSON
    python3 tools/perf_compare_diffusion.py \
      --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
      --bundle /workspace/tensorrt-model-connect/engines/pixart_sigma_v8.trtfb \
      --prompt "A red sports car on a highway" \
      --json results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Helpers (inline to avoid dependency on perf_utils for diffusion-specific
# metrics — perf_utils is decoder-focused)
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


def stats(values: list[float]) -> dict:
    import statistics
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": s}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.0f} +/- {std:.0f}"


def speedup(baseline: float, candidate: float) -> str:
    if candidate <= 0:
        return "N/A"
    return f"{baseline / candidate:.2f}x"


def save_image(pixels: np.ndarray, path: str):
    """Save HWC float32 [0,1] image as PNG."""
    from PIL import Image
    img = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"  Saved {path}", file=sys.stderr)


def image_stats(pixels: np.ndarray) -> dict:
    """Compute basic image statistics for comparison."""
    return {
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "min": float(pixels.min()),
        "max": float(pixels.max()),
    }


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two [0,1] images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM (mean over channels, no windowing)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    s1, s2 = img1.var(), img2.var()
    s12 = np.mean((img1 - mu1) * (img2 - mu2))
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2)
    return float(num / den)


# ---------------------------------------------------------------------------
# HF eager benchmark
# ---------------------------------------------------------------------------

def bench_hf_eager(model_id: str, prompt: str, num_steps: int,
                   guidance_scale: float, seed: int,
                   warmup: int, iterations: int,
                   verbose: bool) -> dict:
    """Benchmark HF diffusers pipeline (eager mode).

    Uses pipe() directly — this handles prompt encoding, CFG, scheduling,
    and VAE decode internally with correct dtype handling.
    """
    import torch
    from diffusers import PixArtSigmaPipeline

    print("[perf] Loading HF PixArt-Sigma pipeline (fp16) ...", file=sys.stderr)
    t_load = time.perf_counter()
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    model_load_s = time.perf_counter() - t_load
    print(f"[perf] HF model load: {model_load_s:.1f}s", file=sys.stderr)

    total_times = []
    pixels_last = None

    for run_idx in range(warmup + iterations):
        is_warmup = run_idx < warmup
        gen = torch.Generator("cuda").manual_seed(seed)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        output = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=1024, width=1024,
            generator=gen,
        )

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        img_np = np.array(output.images[0], dtype=np.float32) / 255.0

        if not is_warmup:
            total_times.append(total_ms)
            pixels_last = img_np

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [eager {tag}] total={total_ms:.0f}ms", file=sys.stderr)

    del pipe
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    return {
        "total_times": total_times,
        "pixels": pixels_last,
        "image_stats": image_stats(pixels_last),
        "model_load_s": model_load_s,
    }


# ---------------------------------------------------------------------------
# torch.compile benchmark
# ---------------------------------------------------------------------------

def bench_torch_compile(model_id: str, prompt: str, num_steps: int,
                        guidance_scale: float, seed: int,
                        warmup: int, iterations: int,
                        verbose: bool) -> dict:
    """Benchmark HF diffusers with torch.compile on the transformer.

    Uses pipe() directly (same as eager) but with torch.compile applied to
    the transformer submodule. This ensures correct dtype handling through
    the pipeline's internal caption projection and scheduling.
    """
    import torch
    from diffusers import PixArtSigmaPipeline

    print("[perf] Loading HF PixArt-Sigma pipeline (fp16, torch.compile) ...",
          file=sys.stderr)
    t_load = time.perf_counter()
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    model_load_s = time.perf_counter() - t_load

    print("[perf] Compiling transformer with torch.compile ...", file=sys.stderr)
    t0 = time.perf_counter()
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
    compile_setup_s = time.perf_counter() - t0

    total_times = []
    pixels_last = None

    # Extra warmup for torch.compile (first runs trigger compilation)
    effective_warmup = max(warmup, 3)

    for run_idx in range(effective_warmup + iterations):
        is_warmup = run_idx < effective_warmup
        gen = torch.Generator("cuda").manual_seed(seed)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        output = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=1024, width=1024,
            generator=gen,
        )

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        img_np = np.array(output.images[0], dtype=np.float32) / 255.0

        if not is_warmup:
            total_times.append(total_ms)
            pixels_last = img_np

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - effective_warmup + 1}"
            print(f"  [compile {tag}] total={total_ms:.0f}ms", file=sys.stderr)

    del pipe
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    return {
        "total_times": total_times,
        "pixels": pixels_last,
        "image_stats": image_stats(pixels_last),
        "compile_setup_s": compile_setup_s,
        "model_load_s": model_load_s,
    }


# ---------------------------------------------------------------------------
# Torch-TRT benchmark (via C++ binary)
# ---------------------------------------------------------------------------

def _load_trt_engine(engine_bytes: bytes):
    """Deserialize a TRT engine from bytes. Returns (engine, context)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    return engine, context


def _run_trt_engine(engine, context, inputs: dict, stream) -> dict:
    """Run a TRT engine with named inputs, return named outputs as tensors."""
    import torch
    import tensorrt as trt

    # Classify I/O tensors
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            if name not in inputs:
                raise RuntimeError(f"Missing input: {name}")
            tensor = inputs[name]
            context.set_input_shape(name, tuple(tensor.shape))
            context.set_tensor_address(name, tensor.data_ptr())

    # Allocate outputs
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            dtype_trt = engine.get_tensor_dtype(name)
            import torch
            dtype_map = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
            }
            dtype = dtype_map.get(dtype_trt, torch.float32)
            out = torch.empty(list(shape), dtype=dtype, device="cuda")
            context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out

    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


def bench_torchtrt(bundle_path: str, prompt: str, num_steps: int,
                   guidance_scale: float, seed: int,
                   warmup: int, iterations: int,
                   verbose: bool) -> dict:
    """Benchmark Torch-TRT bundle in-process (engines loaded once).

    Loads the 3 TRT engines (T5, DiT, VAE) from the bundle, then runs
    the full diffusion pipeline in Python — matching the C++ pipeline.
    This gives a fair apples-to-apples comparison with HF eager/compile.
    """
    import torch
    from tensorrt_model_connect.debug_runner import load_section_from_bundle, \
        load_config_from_bundle

    print("[perf] Loading TRT engines from bundle ...", file=sys.stderr)
    t_load_start = time.perf_counter()
    config = load_config_from_bundle(bundle_path)

    t5_bytes = load_section_from_bundle(bundle_path, "text_encoder_0_plan")
    dit_bytes = load_section_from_bundle(bundle_path, "denoiser_plan")
    vae_bytes = load_section_from_bundle(bundle_path, "vae_decoder_plan")
    if not t5_bytes or not dit_bytes or not vae_bytes:
        raise RuntimeError("Bundle missing required engine sections")

    t5_engine, t5_ctx = _load_trt_engine(t5_bytes)
    dit_engine, dit_ctx = _load_trt_engine(dit_bytes)
    vae_engine, vae_ctx = _load_trt_engine(vae_bytes)
    stream = torch.cuda.Stream()
    engine_load_s = time.perf_counter() - t_load_start
    print(f"[perf] Engine deserialization: {engine_load_s:.1f}s", file=sys.stderr)

    # Config
    seq_len = config.get("text_seq_len", 120)
    z_dim = config.get("z_dim", 4)
    sf = config.get("scale_factor_spatial", 8)
    h_lat = 1024 // sf
    w_lat = 1024 // sf

    # Tokenize — extract tokenizer files from bundle to a temp dir
    print("[perf] Tokenizing prompt ...", file=sys.stderr)
    import tempfile
    tokenizer_dir = tempfile.mkdtemp(prefix="trt_tokenizer_")
    for section_name in ["tokenizer_config.json", "spiece.model",
                         "special_tokens_map.json"]:
        data = load_section_from_bundle(bundle_path, section_name)
        if data:
            with open(os.path.join(tokenizer_dir, section_name), "wb") as f:
                f.write(data)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    token_ids = tokenizer.encode(prompt)
    # Ensure EOS (T5 EOS = 1)
    if not token_ids or token_ids[-1] != 1:
        token_ids.append(1)
    num_real = len(token_ids)

    del t5_bytes, dit_bytes, vae_bytes
    print(f"[perf] Engines loaded. tokens={num_real}, seq_len={seq_len}",
          file=sys.stderr)

    # --- Helper: run T5 encoder ---
    def run_t5(ids, zero_padding=True):
        padded = ids[:seq_len] + [0] * max(0, seq_len - len(ids))
        if zero_padding:
            mask = [1 if padded[i] != 0 else 0 for i in range(seq_len)]
        else:
            mask = [1] * seq_len
        inp = {
            "input_ids": torch.tensor([padded], dtype=torch.int32, device="cuda"),
            "attention_mask": torch.tensor([mask], dtype=torch.int32, device="cuda"),
        }
        out = _run_trt_engine(t5_engine, t5_ctx, inp, stream)
        return out["output0"]  # [1, seq_len, te_dim] fp32

    # --- DPM-Solver++ scheduler (simplified, matches C++ impl) ---
    def make_scheduler(ns):
        """Build alpha/sigma/lambda tables and timesteps."""
        T = 1000
        alpha_t = []
        sigma_t = []
        cum = 1.0
        for i in range(T):
            beta = 0.0001 + i / (T - 1) * (0.02 - 0.0001)
            cum *= (1.0 - beta)
            alpha_t.append(cum ** 0.5)
            sigma_t.append((1.0 - cum) ** 0.5)
        import math
        lambda_t = [math.log(a / s) for a, s in zip(alpha_t, sigma_t)]
        # linspace timesteps
        timesteps = []
        for i in range(1, ns + 1):
            val = i / ns * (T - 1)
            timesteps.append(round(val))
        timesteps.reverse()
        return alpha_t, sigma_t, lambda_t, timesteps

    def dpm_step(eps_pred, sample, step_idx, num_s, alpha_t, sigma_t,
                 lambda_t, timesteps, model_outputs, lower_order_nums):
        """One DPM-Solver++ step. Returns (new_sample, model_outputs, lower_order_nums)."""
        t_s0 = timesteps[step_idx]
        t_t = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else 0

        # eps → x0
        a = alpha_t[t_s0]
        s = sigma_t[t_s0]
        x0 = (sample - s * eps_pred) / a

        model_outputs.append(x0)
        if len(model_outputs) > 2:
            model_outputs.pop(0)

        order = 2
        if lower_order_nums < 1 or step_idx == num_s - 1:
            order = 1

        import math
        if order == 1 or len(model_outputs) < 2:
            h = lambda_t[t_t] - lambda_t[t_s0]
            result = (sigma_t[t_t] / sigma_t[t_s0]) * sample - \
                     alpha_t[t_t] * (math.exp(-h) - 1.0) * model_outputs[-1]
        else:
            t_s1 = timesteps[step_idx - 1]
            h = lambda_t[t_t] - lambda_t[t_s0]
            h_0 = lambda_t[t_s0] - lambda_t[t_s1]
            r0 = h_0 / h
            exp_neg_h = math.exp(-h)
            base = alpha_t[t_t] * (exp_neg_h - 1.0)
            d0 = model_outputs[-1]
            d1 = (1.0 / r0) * (model_outputs[-1] - model_outputs[-2])
            result = (sigma_t[t_t] / sigma_t[t_s0]) * sample - base * d0 - 0.5 * base * d1

        new_lon = min(lower_order_nums + 1, 2)
        return result, model_outputs, new_lon

    total_times = []
    pixels_last = None

    for run_idx in range(warmup + iterations):
        is_warmup = run_idx < warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1) T5 encode (conditioned + unconditional)
        cond_emb = run_t5(token_ids, zero_padding=True)       # fp32
        uncond_emb = run_t5([1], zero_padding=False)           # fp32

        # 2) Init latents
        gen = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn(1, z_dim, h_lat, w_lat,
                              dtype=torch.float32, device="cuda",
                              generator=gen)

        # 3) Scheduler setup
        alpha_t, sigma_t, lambda_t, timesteps = make_scheduler(num_steps)

        # 4) Denoising loop
        model_outputs = []
        lower_order_nums = 0

        # Build masks (fp16)
        cond_mask = torch.zeros(1, seq_len, dtype=torch.float16, device="cuda")
        cond_mask[0, :num_real] = 1.0
        uncond_mask = torch.ones(1, seq_len, dtype=torch.float16, device="cuda")

        for si, ts in enumerate(timesteps):
            ts_tensor = torch.tensor([float(ts)], dtype=torch.float16, device="cuda")
            sample_fp16 = latents.half()

            # Conditioned prediction
            dit_inp = {
                "sample": sample_fp16,
                "encoder_hidden_states": cond_emb.half(),
                "timestep": ts_tensor,
                "encoder_attention_mask": cond_mask,
            }
            cond_out = _run_trt_engine(dit_engine, dit_ctx, dit_inp, stream)
            noise_cond = cond_out["output0"][:, :z_dim].float()

            # Unconditioned prediction
            dit_inp["encoder_hidden_states"] = uncond_emb.half()
            dit_inp["encoder_attention_mask"] = uncond_mask
            uncond_out = _run_trt_engine(dit_engine, dit_ctx, dit_inp, stream)
            noise_uncond = uncond_out["output0"][:, :z_dim].float()

            # CFG
            eps = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Scheduler step
            latents, model_outputs, lower_order_nums = dpm_step(
                eps.squeeze(0).flatten(), latents.squeeze(0).flatten(),
                si, num_steps, alpha_t, sigma_t, lambda_t, timesteps,
                model_outputs, lower_order_nums)
            latents = latents.view(1, z_dim, h_lat, w_lat)

        # 5) VAE decode
        vae_inp = {
            "latent": latents.half(),
        }
        vae_out = _run_trt_engine(vae_engine, vae_ctx, vae_inp, stream)
        image_raw = vae_out["output0"]  # [1, 3, H, W] fp32

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        # CHW [-1,1] → HWC [0,1]
        img_np = image_raw[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1.0) * 0.5
        img_np = np.clip(img_np, 0, 1)

        if not is_warmup:
            total_times.append(total_ms)
            pixels_last = img_np

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [trt {tag}] total={total_ms:.0f}ms", file=sys.stderr)

    return {
        "total_times": total_times,
        "pixels": pixels_last,
        "image_stats": image_stats(pixels_last) if pixels_last is not None else {},
        "engine_load_s": engine_load_s,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(model_name: str, prompt: str, num_steps: int,
                 guidance_scale: float, iterations: int, warmup: int,
                 results: dict[str, dict]):
    """Print formatted comparison table."""
    gpu = get_gpu_name()
    sep = "=" * 76

    print(f"\n{sep}")
    print(f"Diffusion Performance Comparison: {model_name}")
    print(f"GPU: {gpu}")
    print(f'Prompt: "{prompt[:60]}{"..." if len(prompt) > 60 else ""}"')
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}, "
          f"Precision: fp16, Image: 1024x1024")
    print(f"Iterations: {iterations} (+ {warmup} warmup)")
    print(sep)

    backends = list(results.keys())

    # Stage breakdown (only for backends that have it)
    has_stages = any("encode_times" in results[b] for b in backends)

    if has_stages:
        header = f"{'':>22s}"
        for b in backends:
            header += f"  {b:>16s}"
        if len(backends) > 1 and "hf_eager" in results:
            header += f"  {'Speedup':>8s}"
        print(header)
        print("-" * len(header))

        # T5 Encode
        row = f"  {'T5 Encode (ms)':>20s}:"
        encode_means = {}
        for b in backends:
            if "encode_times" in results[b]:
                s = stats(results[b]["encode_times"])
                encode_means[b] = s["mean"]
                row += f"  {fmt(s['mean'], s['std']):>16s}"
            else:
                row += f"  {'(included)':>16s}"
        print(row)

        # Denoise
        row = f"  {'Denoise (ms)':>20s}:"
        denoise_means = {}
        for b in backends:
            if "denoise_times" in results[b]:
                s = stats(results[b]["denoise_times"])
                denoise_means[b] = s["mean"]
                row += f"  {fmt(s['mean'], s['std']):>16s}"
            else:
                row += f"  {'(included)':>16s}"
        if len(denoise_means) > 1 and "hf_eager" in denoise_means:
            ref = denoise_means["hf_eager"]
            best = min(v for k, v in denoise_means.items() if k != "hf_eager")
            row += f"  {speedup(ref, best):>8s}"
        print(row)

        # Per-step
        row = f"  {'Per-step (ms)':>20s}:"
        for b in backends:
            if "denoise_times" in results[b]:
                s = stats(results[b]["denoise_times"])
                ps = s["mean"] / num_steps
                row += f"  {f'{ps:.1f}':>16s}"
            else:
                row += f"  {'—':>16s}"
        print(row)

        # VAE Decode
        row = f"  {'VAE Decode (ms)':>20s}:"
        for b in backends:
            if "vae_times" in results[b]:
                s = stats(results[b]["vae_times"])
                row += f"  {fmt(s['mean'], s['std']):>16s}"
            else:
                row += f"  {'(included)':>16s}"
        print(row)

    # Total (all backends have this)
    print()
    header = f"{'':>22s}"
    for b in backends:
        header += f"  {b:>16s}"
    if len(backends) > 1 and "hf_eager" in results:
        header += f"  {'Speedup':>8s}"
    print(header)
    print("-" * len(header))

    row = f"  {'Total (ms)':>20s}:"
    total_means = {}
    for b in backends:
        s = stats(results[b]["total_times"])
        total_means[b] = s["mean"]
        row += f"  {fmt(s['mean'], s['std']):>16s}"
    if len(backends) > 1 and "hf_eager" in total_means:
        ref = total_means["hf_eager"]
        best = min(v for k, v in total_means.items() if k != "hf_eager")
        row += f"  {speedup(ref, best):>8s}"
    print(row)

    row = f"  {'Images/min':>20s}:"
    for b in backends:
        s = stats(results[b]["total_times"])
        ips = 60000.0 / s["mean"] if s["mean"] > 0 else 0
        row += f"  {f'{ips:.2f}':>16s}"
    print(row)

    # Image quality comparison
    print()
    ref_pixels = None
    if "hf_eager" in results and results["hf_eager"].get("pixels") is not None:
        ref_pixels = results["hf_eager"]["pixels"]

    for b in backends:
        ist = results[b].get("image_stats", {})
        if ist:
            line = f"  {b}: mean={ist.get('mean', 0):.3f} std={ist.get('std', 0):.3f}"
            if ref_pixels is not None and b != "hf_eager":
                other = results[b].get("pixels")
                if other is not None and ref_pixels.shape == other.shape:
                    p = psnr(ref_pixels, other)
                    s = ssim_simple(ref_pixels, other)
                    line += f"  PSNR={p:.1f}dB  SSIM={s:.4f}"
            print(line)

    # Startup times
    print("\n  Startup (not included in per-iteration timings):")
    for b in backends:
        parts = []
        ml = results[b].get("model_load_s", 0)
        if ml:
            parts.append(f"model load {ml:.1f}s")
        cs = results[b].get("compile_setup_s", 0)
        if cs:
            parts.append(f"compile {cs:.1f}s")
        el = results[b].get("engine_load_s", 0)
        if el:
            parts.append(f"engine deser {el:.1f}s")
        total_startup = ml + cs + el
        if parts:
            print(f"    {b}: {' + '.join(parts)} = {total_startup:.1f}s")

    print("\n  * All backends: fp16 compute, 1024x1024 output")
    print("  * Excludes: model loading, engine build, torch.compile compilation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diffusion perf: HF eager vs torch.compile vs Torch-TRT")
    parser.add_argument("--model", required=True,
                        help="HF repo ID (e.g. PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)")
    parser.add_argument("--bundle", required=True,
                        help="Torch-TRT .trtfb bundle path (container path)")
    parser.add_argument("--trtmc-binary", default="./build/trtmc",
                        help="Path to trtmc binary (default: ./build/trtmc)")
    parser.add_argument("--hf-python", default="/opt/venv/bin/python",
                        help="Path to Python with HF tokenizers")
    parser.add_argument("--prompt",
                        default="A photo of a dog chewing on a bone")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip torch.compile benchmark (slow setup)")
    parser.add_argument("--skip-eager", action="store_true",
                        help="Skip HF eager benchmark")
    parser.add_argument("--skip-trt", action="store_true",
                        help="Skip Torch-TRT benchmark")
    parser.add_argument("--output-dir", default="/tmp/perf_diffusion",
                        help="Directory for output images")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # 1) HF eager
    if not args.skip_eager:
        print("\n[perf] === HF Eager ===", file=sys.stderr)
        results["hf_eager"] = bench_hf_eager(
            args.model, args.prompt, args.num_steps, args.guidance_scale,
            args.seed, args.warmup, args.iterations, args.verbose)
        if results["hf_eager"]["pixels"] is not None:
            save_image(results["hf_eager"]["pixels"],
                       os.path.join(args.output_dir, "hf_eager.png"))
        gc.collect()

    # 2) torch.compile
    if not args.skip_compile:
        print("\n[perf] === torch.compile ===", file=sys.stderr)
        try:
            results["torch_compile"] = bench_torch_compile(
                args.model, args.prompt, args.num_steps, args.guidance_scale,
                args.seed, args.warmup, args.iterations, args.verbose)
            if results["torch_compile"]["pixels"] is not None:
                save_image(results["torch_compile"]["pixels"],
                           os.path.join(args.output_dir, "torch_compile.png"))
        except Exception as e:
            print(f"[perf] torch.compile failed: {e}", file=sys.stderr)
        gc.collect()

    # 3) Torch-TRT
    if not args.skip_trt:
        print("\n[perf] === Torch-TRT ===", file=sys.stderr)
        results["torch_trt"] = bench_torchtrt(
            args.bundle, args.prompt, args.num_steps, args.guidance_scale,
            args.seed,
            args.warmup, args.iterations, args.verbose)
        if results["torch_trt"].get("pixels") is not None:
            save_image(results["torch_trt"]["pixels"],
                       os.path.join(args.output_dir, "torch_trt.png"))

    # Report
    print_report(args.model, args.prompt, args.num_steps,
                 args.guidance_scale, args.iterations, args.warmup, results)

    # JSON output
    if args.json_path:
        json_data = {
            "metadata": {
                "model": args.model,
                "prompt": args.prompt,
                "num_steps": args.num_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "precision": "fp16",
                "resolution": "1024x1024",
                "gpu": get_gpu_name(),
                "iterations": args.iterations,
                "warmup": args.warmup,
            },
            "results": {},
        }
        for backend, data in results.items():
            entry = {}
            for key in ["encode_times", "denoise_times", "vae_times",
                        "total_times", "image_stats", "compile_setup_s",
                        "engine_load_s", "model_load_s"]:
                if key in data:
                    entry[key] = data[key]
            json_data["results"][backend] = entry

        # Add cross-backend quality metrics
        if "hf_eager" in results and results["hf_eager"].get("pixels") is not None:
            ref = results["hf_eager"]["pixels"]
            quality = {}
            for b, data in results.items():
                if b == "hf_eager":
                    continue
                other = data.get("pixels")
                if other is not None and ref.shape == other.shape:
                    quality[b] = {
                        "psnr": psnr(ref, other),
                        "ssim": ssim_simple(ref, other),
                    }
            json_data["quality_vs_eager"] = quality

        with open(args.json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n[perf] Results saved to {args.json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
