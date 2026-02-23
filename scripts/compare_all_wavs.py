#!/usr/bin/env python3
"""Compare all debug WAV files."""
import numpy as np
import struct
import os

def read_wav(path):
    with open(path, "rb") as f:
        f.read(12); sr = 24000; data = b""
        while True:
            cid = f.read(4)
            if len(cid) < 4: break
            csz = struct.unpack("<I", f.read(4))[0]
            if cid == b"fmt ": f.read(csz)
            elif cid == b"data": data = f.read(csz)
            else: f.read(csz)
    return np.frombuffer(data, dtype=np.float32), sr

from numpy.fft import rfft
def analyze(wav, sr=24000):
    rms = float(np.sqrt(np.mean(wav**2)))
    spec = np.abs(rfft(wav))**2
    freqs = np.arange(len(spec)) * sr / (2 * len(spec))
    lo = np.sqrt(np.sum(spec[(freqs >= 0) & (freqs < 4000)]) / (np.sum(spec) + 1e-12))
    hi = np.sqrt(np.sum(spec[(freqs >= 4000) & (freqs < 12000)]) / (np.sum(spec) + 1e-12))
    zcr = np.sum(np.abs(np.diff(np.sign(wav)))) / (2 * len(wav))
    return rms, lo/(hi+1e-10), zcr

files = [
    ("HF sampled (golden-like)",    "/tmp/bark_hf_sampled.wav"),
    ("HF greedy fine",              "/tmp/bark_hf_greedy_fine.wav"),
    ("HF greedy coarse-only",      "/tmp/bark_hf_greedy_coarse.wav"),
    ("HF codec, C++ coarse",       "/tmp/bark_hf_codec_cpp_coarse.wav"),
    ("HF fine+codec, C++ coarse",  "/tmp/bark_hf_fine_cpp_coarse.wav"),
    ("C++ sampled (latest)",        "/tmp/bark_sampled_fixed.wav"),
    ("C++ greedy",                  "/tmp/bark_greedy_test2.wav"),
    ("Golden",                      "/workspace/trt-transformers-cpp/bark_e2e_hf.wav"),
]

header = "%-30s %8s %6s %7s %7s %6s" % ("Name", "Samples", "Dur", "RMS", "SpR", "ZCR")
print(header)
print("-" * len(header))
for name, path in files:
    if os.path.exists(path):
        wav, _ = read_wav(path)
        rms, ratio, zcr = analyze(wav)
        dur = len(wav) / 24000
        print("%-30s %8d %5.2fs %7.4f %7.1f %6.4f" % (
            name, len(wav), dur, rms, ratio, zcr))
