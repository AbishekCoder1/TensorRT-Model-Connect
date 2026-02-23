#!/usr/bin/env python3
"""Stage-by-stage comparison: C++ TRT vs HF Bark (deterministic/greedy).

Runs both pipelines in greedy mode and compares at each stage:
  Stage 0: Tokenization (input_ids)
  Stage 1: Semantic model (text -> semantic tokens)
  Stage 2: Coarse model (semantic -> coarse codes)
  Stage 3: Codec (codes -> waveform)

Usage:
  # First run C++ in greedy mode with dump:
  TRTF_BARK_GREEDY=1 TRTF_BARK_DUMP=/tmp/bark_greedy_dump \
    ./build/trtf generate-audio bundle.trtfb --prompt "Hello, my dog is cute" \
    --output /tmp/bark_greedy_test.wav --hf-python .venv/bin/python

  # Then run this script:
  python3 scripts/bark_stage_compare.py \
    --cpp-dump /tmp/bark_greedy_dump \
    --cpp-wav /tmp/bark_greedy_test.wav \
    --model suno/bark-small \
    --prompt "Hello, my dog is cute"
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np


def read_token_file(path):
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(int(line))
    return np.array(tokens, dtype=np.int32)


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


def compare_tokens(hf_tokens, cpp_tokens, label):
    """Compare two token arrays and report first divergence."""
    min_len = min(len(hf_tokens), len(cpp_tokens))
    if min_len == 0:
        print(f"  Cannot compare: HF={len(hf_tokens)}, C++={len(cpp_tokens)}")
        return False

    match = int(np.sum(hf_tokens[:min_len] == cpp_tokens[:min_len]))
    pct = 100.0 * match / min_len
    print(f"  Token match: {match}/{min_len} ({pct:.1f}%)")

    if match == min_len and len(hf_tokens) == len(cpp_tokens):
        print(f"  IDENTICAL ({min_len} tokens)")
        return True

    # Find first divergence
    for i in range(min_len):
        if hf_tokens[i] != cpp_tokens[i]:
            # Show context around divergence
            start = max(0, i - 3)
            end = min(min_len, i + 5)
            print(f"  FIRST DIVERGENCE at position {i}:")
            print(f"    HF[{start}:{end}]  = {hf_tokens[start:end].tolist()}")
            print(f"    C++[{start}:{end}] = {cpp_tokens[start:end].tolist()}")
            # Count total mismatches
            mismatches = int(np.sum(hf_tokens[:min_len] != cpp_tokens[:min_len]))
            print(f"  Total mismatches: {mismatches}/{min_len}")
            return False

    if len(hf_tokens) != len(cpp_tokens):
        print(f"  Length mismatch: HF={len(hf_tokens)}, C++={len(cpp_tokens)}")
        print(f"  (first {min_len} tokens match)")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stage-by-stage Bark C++ vs HF comparison")
    parser.add_argument("--cpp-dump", required=True,
                        help="Path prefix for C++ token dumps (TRTF_BARK_DUMP value)")
    parser.add_argument("--cpp-wav", default=None,
                        help="Path to C++ output WAV")
    parser.add_argument("--model", default="suno/bark-small",
                        help="HF model ID")
    parser.add_argument("--prompt", default="Hello, my dog is cute",
                        help="Text prompt (must match what C++ used)")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, BarkModel
    from transformers.models.bark.generation_configuration_bark import (
        BarkSemanticGenerationConfig,
        BarkCoarseGenerationConfig,
        BarkFineGenerationConfig,
    )

    print("Loading HF model...", file=sys.stderr)
    model = BarkModel.from_pretrained(args.model).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    # ================================================================
    # Stage 0: Tokenization
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 0: Text Tokenization")
    print("=" * 60)

    inputs = processor(args.prompt, return_tensors="pt")
    hf_input_ids = inputs["input_ids"][0].cpu().numpy()
    print(f"  HF input_ids ({len(hf_input_ids)}): {hf_input_ids.tolist()}")
    print("  (C++ uses HfPythonTokenizer -- should produce identical IDs)")

    # ================================================================
    # Stage 1: Semantic model (GREEDY)
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Semantic Model (greedy)")
    print("=" * 60)

    sem_cfg = BarkSemanticGenerationConfig(
        **model.generation_config.semantic_config)
    sem_cfg.do_sample = False
    sem_cfg.temperature = 1.0

    with torch.no_grad():
        hf_sem_raw = model.semantic.generate(
            inputs["input_ids"],
            semantic_generation_config=sem_cfg,
        )

    # The output is the full sequence: [input_tokens, infer_token, generated_tokens]
    # We need to extract just the generated semantic tokens.
    # In Bark, semantic tokens are in [0, semantic_vocab_size=10000).
    # The raw output includes text tokens (offset by 10048) + pad tokens (10000).
    hf_sem_all = hf_sem_raw[0].cpu().numpy()
    hf_sem_tokens = hf_sem_all[hf_sem_all < 10000]

    print(f"  HF raw output: {len(hf_sem_all)} tokens total")
    print(f"  HF semantic (filtered <10000): {len(hf_sem_tokens)} tokens")
    if len(hf_sem_tokens) > 0:
        print(f"  HF range: [{hf_sem_tokens.min()}, {hf_sem_tokens.max()}]")
        print(f"  HF first 20: {hf_sem_tokens[:20].tolist()}")
        print(f"  HF last 5:  {hf_sem_tokens[-5:].tolist()}")

    # C++ semantic tokens
    cpp_sem_file = args.cpp_dump + ".sem_tokens"
    if os.path.exists(cpp_sem_file):
        cpp_sem = read_token_file(cpp_sem_file)
        print(f"  C++ semantic: {len(cpp_sem)} tokens")
        if len(cpp_sem) > 0:
            print(f"  C++ range: [{cpp_sem.min()}, {cpp_sem.max()}]")
            print(f"  C++ first 20: {cpp_sem[:20].tolist()}")
            print(f"  C++ last 5:  {cpp_sem[-5:].tolist()}")

        sem_ok = compare_tokens(hf_sem_tokens, cpp_sem, "semantic")
        if not sem_ok:
            print("\n  >>> DIVERGENCE FOUND IN SEMANTIC STAGE <<<")
            print("  The semantic tokens differ. This is the root cause.")
            print("  Debug: check embedding lookup, position encoding,")
            print("  attention mask, and logit suppression logic.")
    else:
        print(f"  C++ dump not found: {cpp_sem_file}")
        sem_ok = False

    # ================================================================
    # Stage 2: Coarse model (GREEDY, using HF semantic tokens)
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Coarse Model (greedy)")
    print("=" * 60)

    coarse_cfg = BarkCoarseGenerationConfig(
        **model.generation_config.coarse_acoustics_config)
    coarse_cfg.do_sample = False
    coarse_cfg.temperature = 1.0

    with torch.no_grad():
        hf_coarse_raw = model.coarse_acoustics.generate(
            hf_sem_raw,
            semantic_generation_config=sem_cfg,
            coarse_generation_config=coarse_cfg,
        )

    hf_coarse_all = hf_coarse_raw[0].cpu().numpy()
    # Coarse tokens are in [semantic_vocab_size, semantic_vocab_size + 2*codebook_size)
    # = [10000, 12048)
    mask = (hf_coarse_all >= 10000) & (hf_coarse_all < 12048)
    hf_coarse_tokens = hf_coarse_all[mask]

    print(f"  HF raw output: {len(hf_coarse_all)} tokens total")
    print(f"  HF coarse (filtered [10000,12048)): {len(hf_coarse_tokens)} tokens")
    if len(hf_coarse_tokens) > 0:
        print(f"  HF range: [{hf_coarse_tokens.min()}, {hf_coarse_tokens.max()}]")
        print(f"  HF first 20: {hf_coarse_tokens[:20].tolist()}")
        # Check CB0/CB1 interleaving
        cb0 = hf_coarse_tokens[0::2]
        cb1 = hf_coarse_tokens[1::2]
        print(f"  HF CB0 range: [{cb0.min()}, {cb0.max()}] (should be [10000,11024))")
        print(f"  HF CB1 range: [{cb1.min()}, {cb1.max()}] (should be [11024,12048))")

    cpp_coarse_file = args.cpp_dump + ".coarse_tokens"
    if os.path.exists(cpp_coarse_file):
        cpp_coarse = read_token_file(cpp_coarse_file)
        print(f"  C++ coarse: {len(cpp_coarse)} tokens")
        if len(cpp_coarse) > 0:
            print(f"  C++ range: [{cpp_coarse.min()}, {cpp_coarse.max()}]")
            print(f"  C++ first 20: {cpp_coarse[:20].tolist()}")
            cb0_cpp = cpp_coarse[0::2]
            cb1_cpp = cpp_coarse[1::2]
            print(f"  C++ CB0 range: [{cb0_cpp.min()}, {cb0_cpp.max()}]")
            print(f"  C++ CB1 range: [{cb1_cpp.min()}, {cb1_cpp.max()}]")

        coarse_ok = compare_tokens(hf_coarse_tokens, cpp_coarse, "coarse")
        if not coarse_ok and sem_ok:
            print("\n  >>> DIVERGENCE FOUND IN COARSE STAGE <<<")
            print("  Semantic tokens match but coarse tokens differ.")
            print("  Debug: check coarse embedding, sliding window,")
            print("  codebook masking, and logit suppression.")
    else:
        print(f"  C++ dump not found: {cpp_coarse_file}")
        coarse_ok = False

    # ================================================================
    # Stage 3: Codec (feed C++ coarse tokens through HF codec)
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Codec Comparison")
    print("=" * 60)

    if os.path.exists(cpp_coarse_file) and args.cpp_wav:
        n_frames = len(cpp_coarse) // 2
        n_use = min(n_frames, 256)

        # De-interleave C++ coarse tokens
        codes = torch.zeros(1, 8, n_use, dtype=torch.long)
        for t in range(n_use * 2):
            cb = t % 2
            frame = t // 2
            raw = int(cpp_coarse[t]) - 10000 - cb * 1024
            codes[0, cb, frame] = max(0, min(raw, 1023))

        with torch.no_grad():
            hf_codec_out = model.codec_decode(codes)
            hf_wav = hf_codec_out.squeeze().cpu().numpy()

        cpp_wav, _ = read_wav(args.cpp_wav)

        hf_rms = np.sqrt(np.mean(hf_wav ** 2))
        cpp_rms = np.sqrt(np.mean(cpp_wav ** 2))
        print(f"  HF codec ({n_use} frames, 2 CB): {len(hf_wav)} samples, "
              f"RMS={hf_rms:.4f}")
        print(f"  C++ codec: {len(cpp_wav)} samples, RMS={cpp_rms:.4f}")

        mn = min(len(hf_wav), len(cpp_wav))
        if mn > 0:
            diff = np.abs(hf_wav[:mn] - cpp_wav[:mn])
            print(f"  Diff (first {mn}): mean={diff.mean():.6f}, "
                  f"max={diff.max():.6f}")
    else:
        print("  Skipped (need --cpp-wav and coarse dump)")

    # ================================================================
    # Also try: feed C++ semantic tokens to HF coarse model
    # ================================================================
    if os.path.exists(cpp_sem_file) and not sem_ok:
        print("\n" + "=" * 60)
        print("CROSS-CHECK: Feed C++ semantic tokens to HF coarse")
        print("=" * 60)

        # Reconstruct the full semantic sequence that HF coarse expects
        # The coarse model expects: [semantic_tokens_with_padding, infer_token]
        # Format: the output of model.semantic.generate()
        cpp_sem = read_token_file(cpp_sem_file)

        # Build the format that HF coarse expects
        # HF semantic.generate() output: [text_tokens_padded_to_256, infer_token, semantic_tokens, eos]
        # We need to replicate this format with C++ semantic tokens
        kMaxTextLen = 256
        text_encoding_offset = 10048
        semantic_infer_token = 129599
        semantic_pad_token = 10000

        # Build: [padded_text (256 positions), infer_token, c++ semantic tokens, eos]
        hf_text_padded = np.zeros(kMaxTextLen, dtype=np.int64)
        hf_text_padded[:len(hf_input_ids)] = hf_input_ids
        full_seq = np.concatenate([
            hf_text_padded,
            [semantic_infer_token],
            cpp_sem.astype(np.int64),
            [semantic_pad_token],
        ])
        full_seq_t = torch.tensor(full_seq, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            hf_coarse_from_cpp_sem = model.coarse_acoustics.generate(
                full_seq_t,
                semantic_generation_config=sem_cfg,
                coarse_generation_config=coarse_cfg,
            )

        hf_cross = hf_coarse_from_cpp_sem[0].cpu().numpy()
        hf_cross_coarse = hf_cross[(hf_cross >= 10000) & (hf_cross < 12048)]
        print(f"  HF coarse from C++ sem: {len(hf_cross_coarse)} tokens")
        if len(hf_cross_coarse) > 0:
            print(f"  Range: [{hf_cross_coarse.min()}, {hf_cross_coarse.max()}]")
            print(f"  First 20: {hf_cross_coarse[:20].tolist()}")

        if os.path.exists(cpp_coarse_file):
            print("\n  Comparing HF(C++ sem tokens) vs C++ coarse:")
            compare_tokens(hf_cross_coarse, cpp_coarse, "cross-coarse")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if sem_ok:
        print("  Stage 1 (Semantic): MATCH")
    else:
        print("  Stage 1 (Semantic): DIVERGED")
    if sem_ok and coarse_ok:
        print("  Stage 2 (Coarse):   MATCH")
    elif sem_ok:
        print("  Stage 2 (Coarse):   DIVERGED (semantic was OK)")
    else:
        print("  Stage 2 (Coarse):   CANNOT ASSESS (depends on semantic)")


if __name__ == "__main__":
    main()
