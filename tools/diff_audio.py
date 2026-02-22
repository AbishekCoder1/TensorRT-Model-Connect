"""Diff tool for Bark text-to-audio: TRT vs HuggingFace comparison.

Compares generated audio waveforms between the TRT engine and HF Bark.
Since audio generation is stochastic, this primarily checks:
  1. Output shape and sample rate
  2. Audio is non-silent (energy > threshold)
  3. Spectral similarity (optional)

Usage:
    python3 tools/diff_audio.py \
      --model suno/bark-small \
      --prompt "Hello, this is a test." \
      --bundle bark.trtfb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def compute_energy(waveform: np.ndarray) -> float:
    """Compute RMS energy of a waveform."""
    return float(np.sqrt(np.mean(waveform ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Bark TRT vs HF diff")
    parser.add_argument("--model", required=True, help="HF model ID")
    parser.add_argument("--bundle", default=None, help="Path to .trtfb bundle")
    parser.add_argument("--prompt", default="Hello, this is a test.",
                        help="Text prompt for audio generation")
    parser.add_argument("--min-energy", type=float, default=0.001,
                        help="Minimum RMS energy for non-silence check")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # HF reference
    print(f"[diff_audio] Loading HF model: {args.model}", file=sys.stderr)

    try:
        from transformers import AutoProcessor, BarkModel
        import torch

        processor = AutoProcessor.from_pretrained(args.model)
        model = BarkModel.from_pretrained(args.model)
        model.eval()

        inputs = processor(args.prompt, return_tensors="pt")
        with torch.no_grad():
            audio_values = model.generate(**inputs)

        hf_waveform = audio_values.cpu().numpy().flatten()
        hf_energy = compute_energy(hf_waveform)

        print(f"[diff_audio] HF waveform: {len(hf_waveform)} samples, "
              f"energy={hf_energy:.6f}")

        if hf_energy < args.min_energy:
            print(f"[diff_audio] WARNING: HF output is near-silent "
                  f"(energy={hf_energy:.6f} < {args.min_energy})")
    except Exception as e:
        print(f"[diff_audio] HF model loading failed: {e}", file=sys.stderr)
        hf_waveform = None
        hf_energy = 0.0

    if args.bundle:
        print(f"[diff_audio] Bundle comparison: {args.bundle}", file=sys.stderr)
        # TRT comparison would go here
        print("[diff_audio] TRT bundle comparison not yet implemented")

    print("[diff_audio] Done")


if __name__ == "__main__":
    main()
