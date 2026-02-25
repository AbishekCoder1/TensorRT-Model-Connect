#!/usr/bin/env python3
"""Mimi neural codec helper for PersonaPlex speech-to-speech pipeline.

Called by the C++ speech backend via subprocess. Provides encode/decode using
the official Moshi Mimi path from `nvidia/personaplex-7b-v1`.

Usage:
    # Encode: audio waveform (binary float32 on stdin) -> codec tokens (stdout)
    python3 mimi_codec.py encode --num-samples N [--sample-rate 24000]

    # Decode: codec tokens (binary int32 on stdin) -> audio waveform (stdout)
    python3 mimi_codec.py decode --num-codebooks N --num-frames F [--sample-rate 24000]

Input/output format:
    encode input:  binary float32 array [num_samples] on stdin
    encode output: binary int32 array [num_codebooks * num_frames] on stdout
                   (row-major: codebook 0 frames, codebook 1 frames, ...)
                   First line on stderr: "FRAMES <num_frames> CODEBOOKS <num_codebooks>"

    decode input:  binary int32 array [num_codebooks * num_frames] on stdin
    decode output: binary float32 array [num_samples] on stdout
                   First line on stderr: "SAMPLES <num_samples>"
"""

import argparse
import os
import struct
import sys
from pathlib import Path

# Disable moshi/torch runtime compilation in subprocess mode.
# This avoids Triton JIT toolchain issues in containerized inference runs.
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("NO_CUDA_GRAPH", "1")


def _load_model():
    """Load official PersonaPlex Mimi model lazily (heavy import)."""
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mimi_weight = hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME)
    model = loaders.get_mimi(mimi_weight, device=device)
    return model


def cmd_encode(args):
    """Encode audio waveform to codec tokens (official streaming path)."""
    import torch
    import numpy as np
    from moshi.models.lm import (
        _iterate_audio as lm_iterate_audio,
        encode_from_sphn as lm_encode_from_sphn,
    )

    model = _load_model()
    device = next(model.parameters()).device

    # Read raw float32 samples from stdin
    raw = sys.stdin.buffer.read()
    num_samples = len(raw) // 4
    if num_samples <= 0:
        print("ERROR: no audio data on stdin", file=sys.stderr)
        sys.exit(1)

    samples = np.frombuffer(raw, dtype=np.float32)[:num_samples].astype(np.float32)
    # moshi lm helpers expect shape (C, T)
    sample_pcm = samples.reshape(1, -1)
    if args.sample_rate and int(args.sample_rate) != int(model.sample_rate):
        import sphn
        sample_pcm = sphn.resample(
            sample_pcm,
            src_sample_rate=int(args.sample_rate),
            dst_sample_rate=int(model.sample_rate),
        ).astype(np.float32)
    frame_size = int(model.sample_rate / model.frame_rate)

    model.streaming_forever(1)
    model.reset_streaming()

    encoded_chunks = []
    with torch.no_grad():
        for user_encoded in lm_encode_from_sphn(
            model,
            lm_iterate_audio(sample_pcm, sample_interval_size=frame_size, pad=True),
            max_batch=1,
        ):
            # user_encoded: [1, K, F] (typically F=1)
            encoded_chunks.append(user_encoded.detach().cpu().numpy())

    if encoded_chunks:
        codes_np = np.concatenate(encoded_chunks, axis=-1)[0].astype(np.int32)
    else:
        # Fall back to an empty 8-codebook tensor
        codes_np = np.zeros((8, 0), dtype=np.int32)

    num_codebooks, num_frames = codes_np.shape

    # Write metadata to stderr
    print(f"FRAMES {num_frames} CODEBOOKS {num_codebooks}", file=sys.stderr, flush=True)

    # Write binary int32 to stdout (row-major)
    sys.stdout.buffer.write(codes_np.tobytes())
    sys.stdout.buffer.flush()


def cmd_decode(args):
    """Decode codec tokens to audio waveform (official streaming path)."""
    import torch
    import numpy as np

    model = _load_model()
    device = next(model.parameters()).device

    num_codebooks = args.num_codebooks
    num_frames = args.num_frames

    # Read binary int32 tokens from stdin
    expected_bytes = num_codebooks * num_frames * 4
    raw = sys.stdin.buffer.read(expected_bytes)
    if len(raw) < expected_bytes:
        print(f"ERROR: expected {expected_bytes} bytes, got {len(raw)}", file=sys.stderr)
        sys.exit(1)

    tokens = np.frombuffer(raw, dtype=np.int32).reshape(
        num_codebooks, num_frames).copy()

    model.streaming_forever(1)
    model.reset_streaming()

    pcm_frames = []
    with torch.no_grad():
        for f in range(num_frames):
            step_codes = torch.from_numpy(tokens[:, f:f+1]).long().unsqueeze(0).to(device)
            pcm = model.decode(step_codes)  # [1, 1, frame_samples]
            pcm_np = pcm.detach().cpu().numpy().squeeze(0).squeeze(0).astype(np.float32)
            pcm_frames.append(pcm_np)

    if pcm_frames:
        audio_np = np.concatenate(pcm_frames, axis=0)
    else:
        audio_np = np.zeros((0,), dtype=np.float32)
    num_samples = int(audio_np.shape[0])

    # Write metadata to stderr
    print(f"SAMPLES {num_samples}", file=sys.stderr, flush=True)

    # Write binary float32 to stdout
    sys.stdout.buffer.write(audio_np.tobytes())
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser(description="Mimi neural codec helper")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode audio to codec tokens")
    enc.add_argument("--num-samples", type=int, required=True)
    enc.add_argument("--sample-rate", type=int, default=24000)

    dec = sub.add_parser("decode", help="Decode codec tokens to audio")
    dec.add_argument("--num-codebooks", type=int, required=True)
    dec.add_argument("--num-frames", type=int, required=True)
    dec.add_argument("--sample-rate", type=int, default=24000)

    args = parser.parse_args()

    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "decode":
        cmd_decode(args)


if __name__ == "__main__":
    main()
