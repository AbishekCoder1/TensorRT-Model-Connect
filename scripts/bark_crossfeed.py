#!/usr/bin/env python3
"""Cross-feed test: Take C++ semantic tokens and run them through HF coarse+fine+codec.

If this produces good speech, then C++ semantic is fine and the issue is in
C++ coarse/fine/codec. If this produces bad speech too, the C++ semantic
tokens themselves are problematic.
"""
import numpy as np
import struct
import sys
import torch
from transformers import AutoProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig,
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
)


def write_wav(path, samples, sr=24000):
    n = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 4))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * 4, 4, 32))
        f.write(b"data")
        f.write(struct.pack("<I", n * 4))
        f.write(np.array(samples, dtype=np.float32).tobytes())


# Load C++ semantic tokens (from sampling run, not greedy)
# We need the sampling dump. Let's use the dump from the latest run
import os
dump_prefix = "/tmp/bark_crossfeed_dump"

# First generate C++ tokens with sampling and dump
print("Step 1: Checking for C++ token dumps...", file=sys.stderr)

sem_file = dump_prefix + ".sem_tokens"
coarse_file = dump_prefix + ".coarse_tokens"

if not os.path.exists(sem_file):
    print(f"  Token dumps not found at {dump_prefix}.*", file=sys.stderr)
    print(f"  Run C++ first with: TRTF_BARK_DUMP={dump_prefix}", file=sys.stderr)
    sys.exit(1)

cpp_sem = np.array([int(l.strip()) for l in open(sem_file)])
cpp_coarse = np.array([int(l.strip()) for l in open(coarse_file)])
print(f"  C++ semantic: {len(cpp_sem)} tokens", file=sys.stderr)
print(f"  C++ coarse: {len(cpp_coarse)} tokens", file=sys.stderr)

# Load HF model
print("Step 2: Loading HF model...", file=sys.stderr)
model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_cfg = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
fine_cfg = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

# Construct the semantic output in the format HF coarse expects
# HF semantic.generate() output: just the semantic tokens + EOS
# (no text prefix for bark-small)
semantic_output = torch.tensor(
    np.concatenate([cpp_sem, [10000]]),  # append EOS=semantic_pad_token
    dtype=torch.long).unsqueeze(0)

# Replace pad token for coarse
semantic_for_coarse = semantic_output.clone()
semantic_for_coarse.masked_fill_(
    semantic_for_coarse == sem_cfg.semantic_pad_token,
    coarse_cfg.coarse_semantic_pad_token)

print(f"  Semantic output shape: {semantic_for_coarse.shape}", file=sys.stderr)

# Step 3: Run HF coarse on C++ semantic tokens (sampling)
print("Step 3: HF coarse on C++ semantic tokens (sampling)...", file=sys.stderr)
with torch.no_grad():
    hf_coarse_out = model.coarse_acoustics.generate(
        semantic_for_coarse,
        semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg,
    )

hf_coarse_all = hf_coarse_out[0].cpu().numpy()
hf_coarse_tokens = hf_coarse_all[(hf_coarse_all >= 10000) & (hf_coarse_all < 12048)]
print(f"  HF coarse: {len(hf_coarse_tokens)} tokens", file=sys.stderr)

# Step 4: Run HF fine on HF coarse output
print("Step 4: HF fine on HF coarse output...", file=sys.stderr)
with torch.no_grad():
    hf_fine_out = model.fine_acoustics.generate(
        hf_coarse_out,
        semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg,
        fine_generation_config=fine_cfg,
    )
print(f"  HF fine shape: {hf_fine_out.shape}", file=sys.stderr)

# Step 5: Run HF codec
print("Step 5: HF codec...", file=sys.stderr)
with torch.no_grad():
    audio = model.codec_decode(hf_fine_out)
wav = audio.squeeze().cpu().numpy()
rms = np.sqrt(np.mean(wav ** 2))
write_wav("/tmp/bark_crossfeed_hf_from_cpp_sem.wav", wav)
print(f"  -> /tmp/bark_crossfeed_hf_from_cpp_sem.wav: "
      f"{len(wav)} samples ({len(wav)/24000:.2f}s), RMS={rms:.4f}", file=sys.stderr)

# Step 6: Also run HF codec on C++ coarse tokens directly (no HF coarse)
print("Step 6: HF codec on C++ coarse tokens directly...", file=sys.stderr)
n_frames = len(cpp_coarse) // 2
n_use = min(n_frames, 256)
codes = torch.zeros(1, 8, n_use, dtype=torch.long)
for t in range(n_use * 2):
    cb = t % 2
    frame = t // 2
    raw = int(cpp_coarse[t]) - 10000 - cb * 1024
    codes[0, cb, frame] = max(0, min(raw, 1023))

with torch.no_grad():
    audio2 = model.codec_decode(codes)
wav2 = audio2.squeeze().cpu().numpy()
rms2 = np.sqrt(np.mean(wav2 ** 2))
write_wav("/tmp/bark_crossfeed_hf_codec_cpp_coarse.wav", wav2)
print(f"  -> /tmp/bark_crossfeed_hf_codec_cpp_coarse.wav: "
      f"{len(wav2)} samples, RMS={rms2:.4f}", file=sys.stderr)

print("\n=== Key comparison ===", file=sys.stderr)
print("  bark_crossfeed_hf_from_cpp_sem.wav = HF coarse+fine+codec on C++ semantic tokens", file=sys.stderr)
print("  bark_crossfeed_hf_codec_cpp_coarse.wav = HF codec on C++ coarse tokens (no fine)", file=sys.stderr)
print("  bark_sampled_fixed.wav = Full C++ pipeline", file=sys.stderr)
print("  bark_hf_sampled.wav = Full HF pipeline", file=sys.stderr)
print("If bark_crossfeed_hf_from_cpp_sem.wav sounds good -> C++ semantic OK, issue in C++ coarse/fine/codec", file=sys.stderr)
print("If bark_crossfeed_hf_from_cpp_sem.wav sounds bad  -> C++ semantic tokens are the problem", file=sys.stderr)
