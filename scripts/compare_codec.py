#!/usr/bin/env python3
"""Compare TRT codec vs HF codec on the same coarse tokens.

Takes a coarse token dump file (one token per line, as produced by
TRTF_BARK_DUMP) and optionally a TRT WAV file, runs the tokens through
the HF EnCodec decoder, and compares the waveforms.

Usage:
    # Compare C++ dump against HF codec
    python3 scripts/compare_codec.py \
      --tokens /tmp/bark_dump.coarse_tokens \
      --model suno/bark-small

    # Also compare against TRT WAV output
    python3 scripts/compare_codec.py \
      --tokens /tmp/bark_dump.coarse_tokens \
      --trt-wav /tmp/bark_trt.wav \
      --model suno/bark-small

    # Save HF codec output
    python3 scripts/compare_codec.py \
      --tokens /tmp/bark_dump.coarse_tokens \
      --model suno/bark-small \
      --output /tmp/hf_codec_output.wav
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np
import torch
from transformers import BarkModel


def read_tokens(path: str) -> np.ndarray:
    """Read newline-delimited token file."""
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(int(line))
    return np.array(tokens, dtype=np.int32)


def write_wav_f32(path: str, samples: np.ndarray, sr: int = 24000):
    """Write IEEE float32 WAV."""
    n = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 4))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * 4, 4, 32))
        f.write(b"data")
        f.write(struct.pack("<I", n * 4))
        f.write(samples.astype(np.float32).tobytes())


def read_wav_f32(path: str) -> np.ndarray:
    """Read float32 samples from a WAV file (skip 44-byte header)."""
    return np.fromfile(path, dtype=np.float32, offset=44)


def deinterleave_coarse(tokens: np.ndarray, n_codebooks: int = 2,
                        semantic_vocab: int = 10000,
                        codebook_size: int = 1024) -> torch.Tensor:
    """Convert interleaved coarse tokens to [n_q, 1, T] codes for EnCodec.

    Coarse tokens are interleaved: [CB0_f0, CB1_f0, CB0_f1, CB1_f1, ...]
    CB0 range: [semantic_vocab, semantic_vocab + codebook_size)
    CB1 range: [semantic_vocab + codebook_size, semantic_vocab + 2*codebook_size)
    """
    n_frames = len(tokens) // n_codebooks
    codes = torch.zeros(8, 1, n_frames, dtype=torch.long)

    for t in range(len(tokens)):
        cb = t % n_codebooks
        frame = t // n_codebooks
        if frame < n_frames:
            raw = int(tokens[t]) - semantic_vocab - cb * codebook_size
            codes[cb, 0, frame] = max(0, min(raw, codebook_size - 1))

    return codes


def main():
    parser = argparse.ArgumentParser(
        description="Compare TRT vs HF EnCodec on same coarse tokens")
    parser.add_argument("--tokens", required=True,
                        help="Coarse token dump file (one per line)")
    parser.add_argument("--model", default="suno/bark-small",
                        help="HF model ID for EnCodec weights")
    parser.add_argument("--trt-wav", default=None,
                        help="TRT codec WAV output for comparison")
    parser.add_argument("--output", default=None,
                        help="Save HF codec output WAV to this path")
    args = parser.parse_args()

    # Load tokens
    tokens = read_tokens(args.tokens)
    print(f"Coarse tokens: {len(tokens)}, "
          f"range [{tokens.min()}, {tokens.max()}]")

    cb0_count = np.sum((tokens >= 10000) & (tokens < 11024))
    cb1_count = np.sum((tokens >= 11024) & (tokens < 12048))
    print(f"CB0 tokens: {cb0_count}, CB1 tokens: {cb1_count}")

    # De-interleave
    codes = deinterleave_coarse(tokens)
    n_frames = codes.shape[2]

    cb0_nz = int((codes[0, 0] != 0).sum())
    cb1_nz = int((codes[1, 0] != 0).sum())
    print(f"Frames: {n_frames}, CB0 nonzero: {cb0_nz}, CB1 nonzero: {cb1_nz}")
    print(f"CB0[:10] = {codes[0, 0, :10].tolist()}")
    print(f"CB1[:10] = {codes[1, 0, :10].tolist()}")

    # Run HF codec
    print(f"\nLoading HF model: {args.model}")
    bark = BarkModel.from_pretrained(args.model).eval()
    codec = bark.codec_model

    with torch.no_grad():
        emb = codec.quantizer.decode(codes)
        print(f"Embeddings: {emb.shape}")
        audio_out = codec.decoder(emb)
        print(f"Decoder out: {audio_out.shape}")
        hf = audio_out.squeeze().cpu().numpy()

    print(f"HF codec: {len(hf)} samples, "
          f"range [{hf.min():.4f}, {hf.max():.4f}]")

    rms = float(np.sqrt(np.mean(hf ** 2)))
    print(f"HF RMS energy: {rms:.6f}")

    # Save output
    if args.output:
        write_wav_f32(args.output, hf)
        print(f"Saved: {args.output}")

    # Compare with TRT
    if args.trt_wav:
        trt = read_wav_f32(args.trt_wav)
        mn = min(len(trt), len(hf))
        if mn > 0:
            diff = np.abs(trt[:mn] - hf[:mn])
            print(f"\nTRT vs HF codec ({mn} samples):")
            print(f"  max diff:  {diff.max():.6f}")
            print(f"  mean diff: {diff.mean():.6f}")
            print(f"  TRT[:5] = {trt[:5].tolist()}")
            print(f"  HF[:5]  = {hf[:5].tolist()}")

            trt_rms = float(np.sqrt(np.mean(trt[:mn] ** 2)))
            print(f"  TRT RMS: {trt_rms:.6f}, HF RMS: {rms:.6f}")
        else:
            print("WARNING: No overlap between TRT and HF outputs")


if __name__ == "__main__":
    main()
