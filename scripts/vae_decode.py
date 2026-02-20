#!/usr/bin/env python3
"""VAE decode subprocess — loads diffusers AutoencoderKLWan and decodes latents.

Called by the C++ DiffusionBackend via subprocess. Reads binary float32
latents from a file, decodes with the VAE, and writes binary float32 video.

Usage:
    python scripts/vae_decode.py \
        --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --latents-file /tmp/latents.bin \
        --output-file /tmp/video.bin \
        --shape 1,16,2,16,16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="VAE decode subprocess")
    parser.add_argument("--model-id", required=True,
                        help="HF model ID or local path for the VAE")
    parser.add_argument("--latents-file", required=True,
                        help="Path to binary float32 latents")
    parser.add_argument("--output-file", required=True,
                        help="Path to write binary float32 video output")
    parser.add_argument("--shape", required=True,
                        help="Latent shape as comma-separated: B,C,T,H,W")
    args = parser.parse_args()

    # Parse shape
    shape = tuple(int(x) for x in args.shape.split(","))
    assert len(shape) == 5, f"Expected 5D shape, got {shape}"
    B, C, T, H, W = shape

    # Load latents
    latents_np = np.fromfile(args.latents_file, dtype=np.float32)
    expected_size = int(np.prod(shape))
    if latents_np.size < expected_size:
        print(f"Error: latents file too small: {latents_np.size} < {expected_size}",
              file=sys.stderr)
        sys.exit(1)
    latents_np = latents_np[:expected_size].reshape(shape)

    print(f"[vae_decode] Loading VAE from {args.model_id} ...", file=sys.stderr)

    # Try to load the Wan VAE
    try:
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            args.model_id, subfolder="vae",
            torch_dtype=torch.float32)
    except Exception:
        # Fallback: try loading from the model path directly
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32)

    vae.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)

    print(f"[vae_decode] VAE loaded on {device}, decoding {shape} ...",
          file=sys.stderr)

    # Decode
    latents_t = torch.from_numpy(latents_np).to(device)

    with torch.no_grad():
        video = vae.decode(latents_t).sample  # [B, 3, T_out, H_out, W_out]

    video_np = video.cpu().numpy().astype(np.float32)
    print(f"[vae_decode] Output shape: {video_np.shape}", file=sys.stderr)

    # Write output
    video_np.tofile(args.output_file)
    print(f"[vae_decode] Written to {args.output_file} "
          f"({video_np.nbytes / (1024*1024):.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
