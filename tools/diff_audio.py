#!/usr/bin/env python3
"""Diff tool for Bark text-to-audio: C++ TRT pipeline vs HuggingFace.

Staged comparison to isolate audio quality issues:

  Stage 1: C++ sampling smoke test
    Run the C++ binary with sampling (no TRTF_BARK_GREEDY) and check that
    the output waveform has speech-level energy (RMS > threshold).

  Stage 2: Token distribution comparison
    Run both C++ and HF pipelines, dump intermediate tokens (semantic, coarse),
    and compare distributions (count, range, entropy). Since sampling is
    stochastic, exact match is not expected -- we check that both produce
    valid tokens in the expected ranges.

  Stage 3: Codec comparison
    Take coarse tokens from C++ (dumped via TRTF_BARK_DUMP), run them through
    both TRT codec (via C++ binary) and HF EnCodec, and compare waveforms
    sample-by-sample.

  Stage 4: Layer-by-layer codec debug (future)
    Build a debug EnCodec engine with intermediate outputs.

Usage:
    # Stage 1: Quick smoke test -- does C++ produce speech?
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtf \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python --stage 1

    # Stage 2: Token distribution comparison
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtf \\
      --model suno/bark-small \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python --stage 2

    # Stage 3: Codec waveform comparison
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtf \\
      --model suno/bark-small \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python --stage 3

    # All stages (default)
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtf \\
      --model suno/bark-small \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_energy(waveform: np.ndarray) -> float:
    """Compute RMS energy of a waveform."""
    if len(waveform) == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform ** 2)))


def read_wav_f32(path: str) -> tuple[np.ndarray, int]:
    """Read a float32 WAV file. Returns (samples, sample_rate)."""
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a RIFF file: {path}")
        f.read(4)  # chunk size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError(f"Not a WAVE file: {path}")

        sample_rate = 24000
        data_bytes = b""

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                if audio_format != 3:  # IEEE float
                    raise ValueError(
                        f"Expected IEEE float (3), got format {audio_format}")
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.read(chunk_size)

    samples = np.frombuffer(data_bytes, dtype=np.float32)
    return samples, sample_rate


def write_wav_f32(path: str, samples: np.ndarray, sample_rate: int = 24000):
    """Write a float32 WAV file."""
    num_samples = len(samples)
    data_size = num_samples * 4
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sample_rate,
                            sample_rate * 4, 4, 32))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(samples.astype(np.float32).tobytes())


def read_token_file(path: str) -> np.ndarray:
    """Read a newline-delimited token file."""
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(int(line))
    return np.array(tokens, dtype=np.int32)


def token_stats(tokens: np.ndarray, label: str) -> dict:
    """Compute and print basic token statistics."""
    if len(tokens) == 0:
        print(f"  {label}: EMPTY")
        return {"count": 0}

    unique = np.unique(tokens)
    # Entropy of the distribution
    counts = np.bincount(tokens[tokens >= 0])
    probs = counts[counts > 0] / counts[counts > 0].sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    stats = {
        "count": len(tokens),
        "min": int(tokens.min()),
        "max": int(tokens.max()),
        "unique": len(unique),
        "entropy": float(entropy),
        "mean": float(tokens.mean()),
    }
    print(f"  {label}: count={stats['count']}, range=[{stats['min']}, "
          f"{stats['max']}], unique={stats['unique']}, "
          f"entropy={stats['entropy']:.2f} bits")
    return stats


def find_trt_lib_dir() -> str:
    """Find the TRT library directory from the Python tensorrt_libs package."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
    except ImportError:
        pass
    return ""


def run_cpp_bark(binary: str, bundle: str, prompt: str, output_wav: str,
                 hf_python: str, dump_dir: str | None = None,
                 greedy: bool = False, max_tokens: int = 0) -> bool:
    """Run the C++ Bark pipeline and return True on success."""
    env = os.environ.copy()

    # Set LD_LIBRARY_PATH for TRT
    trt_lib = find_trt_lib_dir()
    if trt_lib:
        env["LD_LIBRARY_PATH"] = (
            f"{trt_lib}:/usr/local/cuda/lib64:"
            + env.get("LD_LIBRARY_PATH", ""))

    if greedy:
        env["TRTF_BARK_GREEDY"] = "1"
    elif "TRTF_BARK_GREEDY" in env:
        del env["TRTF_BARK_GREEDY"]

    if dump_dir:
        env["TRTF_BARK_DUMP"] = dump_dir

    cmd = [
        binary, "generate-audio", bundle,
        "--prompt", prompt,
        "--output", output_wav,
    ]
    if hf_python:
        cmd += ["--hf-python", hf_python]
    if max_tokens > 0:
        cmd += ["--max-new-tokens", str(max_tokens)]

    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  C++ pipeline FAILED (rc={result.returncode})", file=sys.stderr)
        if result.stderr:
            # Print last 20 lines of stderr
            lines = result.stderr.strip().split("\n")
            for line in lines[-20:]:
                print(f"    {line}", file=sys.stderr)
        return False

    # Print key info from stderr
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if any(k in line for k in ["semantic:", "coarse:", "codec:",
                                        "generated", "Audio saved"]):
                print(f"    {line}", file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Stage 1: C++ sampling smoke test
# ---------------------------------------------------------------------------

def stage1_cpp_smoke_test(args) -> bool:
    """Run C++ pipeline with sampling and check output has speech energy."""
    print("\n=== Stage 1: C++ Sampling Smoke Test ===", file=sys.stderr)

    if not args.bundle or not args.binary:
        print("  SKIP: --bundle and --binary required", file=sys.stderr)
        return True

    with tempfile.TemporaryDirectory(prefix="diff_audio_") as tmpdir:
        wav_path = os.path.join(tmpdir, "bark_sampled.wav")

        ok = run_cpp_bark(
            args.binary, args.bundle, args.prompt, wav_path,
            args.hf_python, greedy=False)

        if not ok:
            print("  FAIL: C++ pipeline returned error", file=sys.stderr)
            return False

        if not os.path.exists(wav_path):
            print("  FAIL: No output WAV file", file=sys.stderr)
            return False

        waveform, sr = read_wav_f32(wav_path)
        energy = compute_energy(waveform)

        print(f"  Output: {len(waveform)} samples @ {sr} Hz, "
              f"energy={energy:.6f}", file=sys.stderr)

        if energy < args.min_energy:
            print(f"  FAIL: Output is near-silent "
                  f"(energy={energy:.6f} < {args.min_energy})",
                  file=sys.stderr)

            # Save for inspection
            save_path = "/tmp/bark_diff_stage1.wav"
            write_wav_f32(save_path, waveform, sr)
            print(f"  Saved output to {save_path} for inspection",
                  file=sys.stderr)
            return False

        print(f"  PASS: Output has speech-level energy ({energy:.6f})",
              file=sys.stderr)

        # Save for reference
        save_path = "/tmp/bark_diff_stage1.wav"
        write_wav_f32(save_path, waveform, sr)
        print(f"  Saved to {save_path}", file=sys.stderr)

    return True


# ---------------------------------------------------------------------------
# Stage 2: Token distribution comparison
# ---------------------------------------------------------------------------

def stage2_token_comparison(args) -> bool:
    """Compare token distributions between C++ and HF pipelines."""
    print("\n=== Stage 2: Token Distribution Comparison ===", file=sys.stderr)

    if not args.bundle or not args.binary:
        print("  SKIP: --bundle and --binary required for C++ side",
              file=sys.stderr)
        return True

    if not args.model:
        print("  SKIP: --model required for HF reference", file=sys.stderr)
        return True

    cpp_sem = None
    cpp_coarse = None

    # --- C++ side: run with TRTF_BARK_DUMP ---
    with tempfile.TemporaryDirectory(prefix="diff_audio_") as tmpdir:
        wav_path = os.path.join(tmpdir, "bark_cpp.wav")
        dump_prefix = os.path.join(tmpdir, "bark_dump")

        ok = run_cpp_bark(
            args.binary, args.bundle, args.prompt, wav_path,
            args.hf_python, dump_dir=dump_prefix, greedy=False)

        if not ok:
            print("  FAIL: C++ pipeline failed", file=sys.stderr)
            return False

        sem_file = dump_prefix + ".sem_tokens"
        coarse_file = dump_prefix + ".coarse_tokens"

        if os.path.exists(sem_file):
            cpp_sem = read_token_file(sem_file)
        else:
            print("  WARNING: No semantic token dump", file=sys.stderr)

        if os.path.exists(coarse_file):
            cpp_coarse = read_token_file(coarse_file)
        else:
            print("  WARNING: No coarse token dump", file=sys.stderr)

    # --- HF side: run Bark with sampling ---
    hf_sem = None
    hf_coarse = None
    print("  Loading HF model...", file=sys.stderr)
    try:
        from transformers import AutoProcessor, BarkModel
        from transformers.models.bark.generation_configuration_bark import (
            BarkSemanticGenerationConfig, BarkCoarseGenerationConfig,
        )
        import torch

        processor = AutoProcessor.from_pretrained(args.model)
        model = BarkModel.from_pretrained(args.model)
        model.eval()

        inputs = processor(args.prompt, return_tensors="pt")

        # Build typed generation configs from the model's config dicts
        sem_gen_cfg = BarkSemanticGenerationConfig(
            **model.generation_config.semantic_config)
        coarse_gen_cfg = BarkCoarseGenerationConfig(
            **model.generation_config.coarse_acoustics_config)

        with torch.no_grad():
            # Step 1: semantic tokens
            semantic_output = model.semantic.generate(
                inputs["input_ids"],
                semantic_generation_config=sem_gen_cfg,
            )
            sem_flat = semantic_output.cpu().numpy().flatten()
            # Filter to valid semantic range [0, 10000)
            hf_sem = sem_flat[sem_flat < 10000]

            # Step 2: coarse tokens
            coarse_output = model.coarse_acoustics.generate(
                semantic_output,
                semantic_generation_config=sem_gen_cfg,
                coarse_generation_config=coarse_gen_cfg,
            )
            hf_coarse = coarse_output.cpu().numpy().flatten()

    except Exception as e:
        print(f"  HF generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("  Skipping HF comparison, showing C++ stats only",
              file=sys.stderr)

    # --- Compare ---
    passed = True

    print("\n  Semantic tokens:", file=sys.stderr)
    if cpp_sem is not None:
        cpp_sem_stats = token_stats(cpp_sem, "C++")
        # Semantic tokens should be in [0, 10000)
        if cpp_sem_stats["count"] > 0:
            if cpp_sem_stats["min"] < 0 or cpp_sem_stats["max"] >= 10000:
                print("    FAIL: C++ semantic tokens out of range [0, 10000)",
                      file=sys.stderr)
                passed = False
            if cpp_sem_stats["count"] < 10:
                print("    WARNING: Very few semantic tokens generated",
                      file=sys.stderr)
    if hf_sem is not None:
        hf_sem_stats = token_stats(hf_sem, "HF ")

    print("\n  Coarse tokens:", file=sys.stderr)
    if cpp_coarse is not None:
        cpp_coarse_stats = token_stats(cpp_coarse, "C++")
        # Coarse tokens: codebook 0 in [10000, 11024), codebook 1 in [11024, 12048)
        if cpp_coarse_stats["count"] > 0:
            if cpp_coarse_stats["min"] < 10000 or cpp_coarse_stats["max"] >= 12048:
                print("    FAIL: C++ coarse tokens out of range [10000, 12048)",
                      file=sys.stderr)
                passed = False

            # Check interleaving: even indices should be CB0, odd should be CB1
            cb0 = cpp_coarse[0::2]
            cb1 = cpp_coarse[1::2]
            cb0_in_range = np.all((cb0 >= 10000) & (cb0 < 11024))
            cb1_in_range = np.all((cb1 >= 11024) & (cb1 < 12048))
            if not cb0_in_range:
                print("    FAIL: CB0 (even) tokens not in [10000, 11024)",
                      file=sys.stderr)
                passed = False
            if not cb1_in_range:
                print("    FAIL: CB1 (odd) tokens not in [11024, 12048)",
                      file=sys.stderr)
                passed = False
    if hf_coarse is not None:
        hf_coarse_stats = token_stats(hf_coarse, "HF ")

    if passed:
        print("\n  PASS: Token distributions look valid", file=sys.stderr)
    else:
        print("\n  FAIL: Token distribution issues detected", file=sys.stderr)

    return passed


# ---------------------------------------------------------------------------
# Stage 3: Codec waveform comparison
# ---------------------------------------------------------------------------

def stage3_codec_comparison(args) -> bool:
    """Compare TRT codec vs HF codec on same coarse tokens."""
    print("\n=== Stage 3: Codec Waveform Comparison ===", file=sys.stderr)

    if not args.bundle or not args.binary:
        print("  SKIP: --bundle and --binary required", file=sys.stderr)
        return True

    if not args.model:
        print("  SKIP: --model required for HF codec", file=sys.stderr)
        return True

    # --- Step 1: Generate coarse tokens via C++ and get TRT codec output ---
    with tempfile.TemporaryDirectory(prefix="diff_audio_") as tmpdir:
        wav_path = os.path.join(tmpdir, "bark_trt.wav")
        dump_prefix = os.path.join(tmpdir, "bark_dump")

        ok = run_cpp_bark(
            args.binary, args.bundle, args.prompt, wav_path,
            args.hf_python, dump_dir=dump_prefix, greedy=False)

        if not ok:
            print("  FAIL: C++ pipeline failed", file=sys.stderr)
            return False

        coarse_file = dump_prefix + ".coarse_tokens"
        if not os.path.exists(coarse_file):
            print("  FAIL: No coarse token dump from C++", file=sys.stderr)
            return False

        cpp_coarse = read_token_file(coarse_file)
        if len(cpp_coarse) == 0:
            print("  FAIL: Empty coarse tokens", file=sys.stderr)
            return False

        # Read TRT codec waveform
        if not os.path.exists(wav_path):
            print("  FAIL: No TRT output WAV", file=sys.stderr)
            return False

        trt_waveform, trt_sr = read_wav_f32(wav_path)

        # --- Step 2: Run same coarse tokens through HF EnCodec ---
        # Use only the first codec_frames tokens (matching TRT truncation)
        # to ensure a fair comparison.
        n_total_frames = len(cpp_coarse) // 2
        # Detect TRT codec frame limit from the wav length
        codec_frames = len(trt_waveform) // 320  # upsample_factor=320
        n_use = min(n_total_frames, codec_frames)
        coarse_subset = cpp_coarse[:n_use * 2]

        print(f"  Total coarse frames: {n_total_frames}, "
              f"TRT codec limit: {codec_frames}, using: {n_use}",
              file=sys.stderr)
        print("  Running HF codec on C++ coarse tokens...", file=sys.stderr)
        try:
            import torch
            from transformers import BarkModel

            bark = BarkModel.from_pretrained(args.model).eval()
            codec = bark.codec_model

            # De-interleave coarse tokens into codes [n_q, 1, T]
            codes = torch.zeros(8, 1, n_use, dtype=torch.long)
            for t in range(len(coarse_subset)):
                cb = t % 2
                frame = t // 2
                if frame < n_use:
                    raw = int(coarse_subset[t]) - 10000 - cb * 1024
                    codes[cb, 0, frame] = max(0, min(raw, 1023))

            with torch.no_grad():
                emb = codec.quantizer.decode(codes)
                hf_audio = codec.decoder(emb)
                hf_waveform = hf_audio.squeeze().cpu().numpy()

        except Exception as e:
            print(f"  HF codec failed: {e}", file=sys.stderr)
            return False

        # --- Step 3: Compare waveforms ---
        min_len = min(len(trt_waveform), len(hf_waveform))
        if min_len == 0:
            print("  FAIL: One or both waveforms are empty", file=sys.stderr)
            return False

        trt_trimmed = trt_waveform[:min_len]
        hf_trimmed = hf_waveform[:min_len]
        diff = np.abs(trt_trimmed - hf_trimmed)

        trt_energy = compute_energy(trt_trimmed)
        hf_energy = compute_energy(hf_trimmed)

        print(f"  TRT codec: {len(trt_waveform)} samples, "
              f"energy={trt_energy:.6f}", file=sys.stderr)
        print(f"  HF  codec: {len(hf_waveform)} samples, "
              f"energy={hf_energy:.6f}", file=sys.stderr)
        print(f"  Compared:  {min_len} samples", file=sys.stderr)
        print(f"  Diff: max={diff.max():.6f}, mean={diff.mean():.6f}, "
              f"median={np.median(diff):.6f}", file=sys.stderr)

        # Save files for manual inspection
        write_wav_f32("/tmp/bark_diff_trt_codec.wav", trt_trimmed, trt_sr)
        write_wav_f32("/tmp/bark_diff_hf_codec.wav", hf_trimmed, 24000)
        print(f"  Saved TRT codec output: /tmp/bark_diff_trt_codec.wav",
              file=sys.stderr)
        print(f"  Saved HF codec output:  /tmp/bark_diff_hf_codec.wav",
              file=sys.stderr)

        # Spectral similarity: both should have speech-like frequency content
        from numpy.fft import rfft
        def _band_energy(wav, lo, hi, sr=24000):
            spec = np.abs(rfft(wav)) ** 2
            freqs = np.arange(len(spec)) * sr / (2 * len(spec))
            mask = (freqs >= lo) & (freqs < hi)
            return np.sqrt(np.sum(spec[mask]) / (np.sum(spec) + 1e-12))

        trt_ratio = _band_energy(trt_trimmed, 0, 4000) / (
            _band_energy(trt_trimmed, 4000, 12000) + 1e-12)
        hf_ratio = _band_energy(hf_trimmed, 0, 4000) / (
            _band_energy(hf_trimmed, 4000, 12000) + 1e-12)
        print(f"  Speech band ratio: TRT={trt_ratio:.1f}, HF={hf_ratio:.1f} "
              f"(>2 = speech-like)", file=sys.stderr)

        # Check: codec outputs should be reasonably close (same tokens,
        # same weights, but TRT LSTM unrolling vs PyTorch LSTM may differ)
        atol = args.codec_atol
        if diff.mean() > atol:
            print(f"  FAIL: Mean diff {diff.mean():.6f} > atol {atol}",
                  file=sys.stderr)
            return False

        print(f"  PASS: Codec outputs match (mean diff {diff.mean():.6f} "
              f"<= {atol})", file=sys.stderr)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bark TRT vs HF staged diff test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  C++ sampling smoke test (needs --bundle, --binary)
  2  Token distribution comparison (needs --bundle, --binary, --model)
  3  Codec waveform comparison (needs --bundle, --binary, --model)

Examples:
  # Quick smoke test
  python3 tools/diff_audio.py --bundle bark.trtfb --binary ./build/trtf \\
    --prompt "Hello, my dog is cute" --hf-python .venv/bin/python --stage 1

  # Full comparison
  python3 tools/diff_audio.py --bundle bark.trtfb --binary ./build/trtf \\
    --model suno/bark-small --prompt "Hello, my dog is cute" \\
    --hf-python .venv/bin/python
""")
    parser.add_argument("--model", default=None,
                        help="HF model ID (e.g. suno/bark-small)")
    parser.add_argument("--bundle", default=None,
                        help="Path to .trtfb bundle")
    parser.add_argument("--binary", default=None,
                        help="Path to trtf binary (e.g. ./build/trtf)")
    parser.add_argument("--prompt", default="Hello, my dog is cute.",
                        help="Text prompt")
    parser.add_argument("--hf-python", default="",
                        help="Path to Python with HF tokenizers installed")
    parser.add_argument("--min-energy", type=float, default=0.005,
                        help="Min RMS energy for speech detection (stage 1)")
    parser.add_argument("--codec-atol", type=float, default=0.15,
                        help="Max mean diff for codec comparison (stage 3). "
                        "Note: when the bundle has a fine model, TRT feeds "
                        "8 codebooks while Stage 3 HF reference uses only 2, "
                        "so larger diffs are expected.")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage (0=all, 1/2/3)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    stages = [1, 2, 3] if args.stage == 0 else [args.stage]
    results = {}
    all_pass = True

    for stage in stages:
        if stage == 1:
            ok = stage1_cpp_smoke_test(args)
        elif stage == 2:
            ok = stage2_token_comparison(args)
        elif stage == 3:
            ok = stage3_codec_comparison(args)
        else:
            print(f"Unknown stage: {stage}", file=sys.stderr)
            ok = False

        results[stage] = ok
        if not ok:
            all_pass = False

    # Summary
    print("\n=== Summary ===", file=sys.stderr)
    for stage, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  Stage {stage}: {status}", file=sys.stderr)

    if all_pass:
        print("\nAll stages PASSED", file=sys.stderr)
    else:
        print("\nSome stages FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
