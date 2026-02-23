#!/usr/bin/env python3
"""Deep debug: Run HF Bark E2E and save WAV at each stage.

Produces:
  /tmp/bark_hf_greedy.wav          - HF full pipeline, greedy
  /tmp/bark_hf_greedy_coarse.wav   - HF codec on coarse-only (2 CB), greedy
  /tmp/bark_hf_greedy_fine.wav     - HF codec on fine (8 CB), greedy
  /tmp/bark_hf_sampled.wav         - HF full pipeline, sampling (default)

Also feeds C++ coarse tokens through HF fine + codec to isolate TRT fine vs HF fine.
"""

import numpy as np
import struct
import sys
import os

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


def read_wav(path):
    with open(path, "rb") as f:
        f.read(12)
        sr = 24000
        data = b""
        while True:
            cid = f.read(4)
            if len(cid) < 4:
                break
            csz = struct.unpack("<I", f.read(4))[0]
            if cid == b"fmt ":
                fmt = f.read(csz)
                sr = struct.unpack("<I", fmt[4:8])[0]
            elif cid == b"data":
                data = f.read(csz)
            else:
                f.read(csz)
    return np.frombuffer(data, dtype=np.float32), sr


print("Loading HF model...", file=sys.stderr)
model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_cfg = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
fine_cfg = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

# ============================================================
# 1. HF full pipeline with SAMPLING (the golden reference approach)
# ============================================================
print("\n1. HF full pipeline (sampling)...", file=sys.stderr)
with torch.no_grad():
    audio = model.generate(**inputs)
hf_sampled = audio.cpu().numpy().flatten()
write_wav("/tmp/bark_hf_sampled.wav", hf_sampled)
print(f"   -> /tmp/bark_hf_sampled.wav: {len(hf_sampled)} samples, "
      f"RMS={np.sqrt(np.mean(hf_sampled**2)):.4f}", file=sys.stderr)

# ============================================================
# 2. HF step-by-step with GREEDY
# ============================================================
print("\n2. HF step-by-step (greedy)...", file=sys.stderr)
sem_cfg_greedy = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
sem_cfg_greedy.do_sample = False
sem_cfg_greedy.temperature = 1.0

coarse_cfg_greedy = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
coarse_cfg_greedy.do_sample = False
coarse_cfg_greedy.temperature = 1.0

fine_cfg_greedy = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)
fine_cfg_greedy.do_sample = False
fine_cfg_greedy.temperature = 1.0

with torch.no_grad():
    # Semantic
    hf_sem = model.semantic.generate(
        inputs["input_ids"], semantic_generation_config=sem_cfg_greedy)
    sem_tokens = hf_sem[0].cpu().numpy()
    sem_valid = sem_tokens[sem_tokens < 10000]
    print(f"   Semantic: {len(sem_valid)} tokens", file=sys.stderr)

    # Coarse
    hf_coarse = model.coarse_acoustics.generate(
        hf_sem,
        semantic_generation_config=sem_cfg_greedy,
        coarse_generation_config=coarse_cfg_greedy)
    coarse_tokens = hf_coarse[0].cpu().numpy()
    coarse_valid = coarse_tokens[(coarse_tokens >= 10000) & (coarse_tokens < 12048)]
    print(f"   Coarse: {len(coarse_valid)} tokens", file=sys.stderr)

    # Fine
    hf_fine = model.fine_acoustics.generate(
        hf_coarse,
        semantic_generation_config=sem_cfg_greedy,
        coarse_generation_config=coarse_cfg_greedy,
        fine_generation_config=fine_cfg_greedy)
    print(f"   Fine output shape: {hf_fine.shape}", file=sys.stderr)
    for cb in range(8):
        codes = hf_fine[0, cb].cpu().numpy()
        print(f"     CB{cb}: unique={len(np.unique(codes))}, "
              f"range=[{codes.min()}, {codes.max()}]", file=sys.stderr)

    # Codec with full fine codes
    audio_fine = model.codec_decode(hf_fine)
    hf_greedy_fine = audio_fine.squeeze().cpu().numpy()
    write_wav("/tmp/bark_hf_greedy_fine.wav", hf_greedy_fine)
    print(f"   -> /tmp/bark_hf_greedy_fine.wav: {len(hf_greedy_fine)} samples, "
          f"RMS={np.sqrt(np.mean(hf_greedy_fine**2)):.4f}", file=sys.stderr)

    # Codec with coarse-only (zero out fine codebooks)
    coarse_only = hf_fine.clone()
    coarse_only[:, 2:, :] = 0
    audio_coarse = model.codec_decode(coarse_only)
    hf_greedy_coarse = audio_coarse.squeeze().cpu().numpy()
    write_wav("/tmp/bark_hf_greedy_coarse.wav", hf_greedy_coarse)
    print(f"   -> /tmp/bark_hf_greedy_coarse.wav: {len(hf_greedy_coarse)} samples, "
          f"RMS={np.sqrt(np.mean(hf_greedy_coarse**2)):.4f}", file=sys.stderr)

# ============================================================
# 3. Feed C++ coarse tokens through HF fine + codec
# ============================================================
cpp_coarse_file = "/tmp/bark_greedy_dump2.coarse_tokens"
if os.path.exists(cpp_coarse_file):
    print("\n3. HF fine+codec on C++ coarse tokens (greedy)...", file=sys.stderr)
    cpp_coarse = np.array([int(l.strip()) for l in open(cpp_coarse_file)])
    n_frames = len(cpp_coarse) // 2

    # Build the format HF fine expects: the output of coarse.generate()
    # which is [batch, seq_len] with interleaved coarse tokens
    # preceded by the semantic context
    # Actually, the fine model expects the full coarse output which includes
    # semantic prefix + coarse tokens
    # Let's use codec_decode directly with de-interleaved codes

    # De-interleave to [1, 8, n_frames]
    n_use = min(n_frames, 256)
    codes_cpp = torch.zeros(1, 8, n_use, dtype=torch.long)
    for t in range(n_use * 2):
        cb = t % 2
        frame = t // 2
        raw = int(cpp_coarse[t]) - 10000 - cb * 1024
        codes_cpp[0, cb, frame] = max(0, min(raw, 1023))

    with torch.no_grad():
        # Coarse-only through HF codec
        audio = model.codec_decode(codes_cpp)
        wav = audio.squeeze().cpu().numpy()
        write_wav("/tmp/bark_hf_codec_cpp_coarse.wav", wav)
        print(f"   -> /tmp/bark_hf_codec_cpp_coarse.wav (coarse-only): "
              f"{len(wav)} samples, RMS={np.sqrt(np.mean(wav**2)):.4f}",
              file=sys.stderr)

    # Now run HF fine on C++ coarse tokens
    # The fine model needs the coarse output in the format [batch, n_coarse_codebooks, seq_len]
    # But the generate() expects the full output from coarse model (with semantic prefix)
    # Let's manually run the fine model instead
    print("   Running HF fine model on C++ coarse codes...", file=sys.stderr)
    with torch.no_grad():
        # The fine model forward takes:
        # input_ids: [batch, seq_len, n_codebooks] where each position has codes for each CB
        # codebook_idx: which codebook to predict
        fine_input = codes_cpp[0, :, :].T  # [n_use, 8] -> but need [1, n_use, 8]
        fine_input = fine_input.unsqueeze(0)  # [1, n_use, 8]

        for cb_idx in range(2, 8):
            logits = model.fine_acoustics.forward(
                codebook_idx=cb_idx,
                input_ids=fine_input,
            ).logits
            # logits: [1, n_use, codebook_size]
            predicted = logits[:, :, :1024].argmax(-1)  # [1, n_use]
            fine_input[0, :, cb_idx] = predicted[0]
            codes_cpp[0, cb_idx, :] = predicted[0]

        for cb in range(8):
            codes = codes_cpp[0, cb].cpu().numpy()
            print(f"     CB{cb}: unique={len(np.unique(codes))}, "
                  f"range=[{codes.min()}, {codes.max()}]", file=sys.stderr)

        # Decode with HF fine codes applied to C++ coarse
        audio_cpp_fine = model.codec_decode(codes_cpp)
        wav_cpp_fine = audio_cpp_fine.squeeze().cpu().numpy()
        write_wav("/tmp/bark_hf_fine_cpp_coarse.wav", wav_cpp_fine)
        print(f"   -> /tmp/bark_hf_fine_cpp_coarse.wav (HF fine on C++ coarse): "
              f"{len(wav_cpp_fine)} samples, RMS={np.sqrt(np.mean(wav_cpp_fine**2)):.4f}",
              file=sys.stderr)

# ============================================================
# 4. Compare C++ output directly
# ============================================================
cpp_wav_file = "/tmp/bark_sampled_fixed.wav"
cpp_greedy_file = "/tmp/bark_greedy_test2.wav"
for label, path in [("C++ sampled", cpp_wav_file), ("C++ greedy", cpp_greedy_file)]:
    if os.path.exists(path):
        wav, _ = read_wav(path)
        print(f"\n{label}: {len(wav)} samples, RMS={np.sqrt(np.mean(wav**2)):.4f}",
              file=sys.stderr)

print("\n=== Files for listening ===", file=sys.stderr)
print("  /tmp/bark_hf_sampled.wav         - HF sampling (golden-like)", file=sys.stderr)
print("  /tmp/bark_hf_greedy_fine.wav     - HF greedy, full fine", file=sys.stderr)
print("  /tmp/bark_hf_greedy_coarse.wav   - HF greedy, coarse-only", file=sys.stderr)
print("  /tmp/bark_hf_codec_cpp_coarse.wav - HF codec, C++ coarse tokens", file=sys.stderr)
print("  /tmp/bark_hf_fine_cpp_coarse.wav - HF fine+codec, C++ coarse tokens", file=sys.stderr)
print("  /tmp/bark_sampled_fixed.wav      - C++ TRT sampling (latest)", file=sys.stderr)
print("  /tmp/bark_greedy_test2.wav       - C++ TRT greedy", file=sys.stderr)
