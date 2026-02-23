#!/usr/bin/env python3
"""Extract mel spectrogram from audio file for Whisper.

Writes binary output: [mel_bins(int32), mel_length(int32), float32 data].
Called by C++ runtime as a subprocess (same pattern as hf_tokenizer.py).
"""
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Extract mel spectrogram for Whisper")
    parser.add_argument("--audio", required=True, help="Path to input .wav file")
    parser.add_argument("--output", required=True, help="Path to output binary file")
    args = parser.parse_args()

    # Load audio at 16kHz
    try:
        import librosa
        audio, sr = librosa.load(args.audio, sr=16000)
    except ImportError:
        import soundfile as sf
        audio, sr = sf.read(args.audio)
        if sr != 16000:
            raise RuntimeError(
                f"Audio sample rate is {sr}Hz but Whisper requires 16kHz. "
                "Install librosa for automatic resampling: pip install librosa"
            )
        audio = audio.astype(np.float32)

    # Compute mel spectrogram using HuggingFace WhisperFeatureExtractor
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor()
    features = extractor(audio, sampling_rate=16000, return_tensors="np")
    mel = features["input_features"][0].astype(np.float32)  # [mel_bins, mel_length]

    # Write binary: shape header (2 x int32) + float32 data
    shape = np.array(mel.shape, dtype=np.int32)  # [mel_bins, mel_length]
    with open(args.output, "wb") as f:
        f.write(shape.tobytes())
        f.write(mel.tobytes())

    print(f"Mel spectrogram: {mel.shape[0]} bins x {mel.shape[1]} frames", file=sys.stderr)


if __name__ == "__main__":
    main()
