#!/usr/bin/env python3
"""Isolation test: TRT semantic logits + HF sampling (torch.multinomial).

This script isolates whether Bark's audio quality issue comes from:
  (A) C++ sampling (std::mt19937 + CDF) vs PyTorch (torch.multinomial), or
  (B) TRT engine logits being subtly different from HF logits.

Strategy:
  1. Run the TRT semantic engine step-by-step via TrtRunner (from bundle)
  2. Use HF's exact sampling chain: suppress → temperature → top-k → softmax → multinomial
  3. Feed resulting semantic tokens through HF coarse + fine + codec
  4. Save WAV for listening

Result interpretation:
  - If output sounds like intelligible speech → Path A (C++ sampling is the issue)
  - If output sounds bad → Path B (TRT logits diverge from HF)

Usage:
  python3 scripts/bark_trt_hf_sampling.py --bundle /tmp/bark.trtfb
  python3 scripts/bark_trt_hf_sampling.py --bundle /tmp/bark.trtfb --seed 42
  python3 scripts/bark_trt_hf_sampling.py --bundle /tmp/bark.trtfb --prompt "The quick brown fox"
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Bundle reading
# ---------------------------------------------------------------------------

BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"


def read_bundle(path: str) -> tuple[dict, dict[str, bytes]]:
    """Read a .trtfb bundle, returning (header_dict, {section_name: bytes})."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != BUNDLE_MAGIC:
            raise ValueError(f"Invalid .trtfb bundle: {path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 16 + header_len
        sections = {}
        for name, meta in header.get("sections", {}).items():
            f.seek(data_start + meta["offset"])
            sections[name] = f.read(meta["size"])
    return header, sections


# ---------------------------------------------------------------------------
# WAV writer
# ---------------------------------------------------------------------------

def write_wav(path: str, samples: np.ndarray, sr: int = 24000) -> None:
    n = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 4))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * 4, 4, 32))
        f.write(b"data")
        f.write(struct.pack("<I", n * 4))
        f.write(np.asarray(samples, dtype=np.float32).tobytes())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TRT semantic + HF sampling isolation test")
    parser.add_argument("--bundle", required=True,
                        help="Path to Bark .trtfb bundle")
    parser.add_argument("--model", default="suno/bark-small",
                        help="HF model ID for coarse/fine/codec stages")
    parser.add_argument("--prompt", default="Hello, my dog is cute",
                        help="Text prompt")
    parser.add_argument("--output", default="/tmp/bark_trt_hf_sampling.wav",
                        help="Output WAV path")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for torch.multinomial")
    parser.add_argument("--max-semantic-tokens", type=int, default=768,
                        help="Max semantic tokens to generate")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy (argmax) instead of sampling for comparison")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, BarkModel
    from transformers.models.bark.generation_configuration_bark import (
        BarkSemanticGenerationConfig,
        BarkCoarseGenerationConfig,
        BarkFineGenerationConfig,
    )

    # NOTE: seed is applied AFTER all model loading, right before sampling,
    # to match HF's RNG state at the first torch.multinomial call.
    # torch.multinomial consumes 2*vocab_size RNG values per call, so the
    # probability tensor MUST be the same size as HF's (full output_vocab)
    # to keep the RNG synchronized.

    # ------------------------------------------------------------------
    # Step 1: Load TRT semantic engine from bundle
    # ------------------------------------------------------------------
    print("=" * 60, file=sys.stderr)
    print("Step 1: Load TRT semantic engine from bundle", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    t0 = time.time()
    header, sections = read_bundle(args.bundle)
    bundle_config = json.loads(sections["config.json"].decode("utf-8"))

    # Extract semantic engine plan
    if "engine_plan" not in sections:
        raise RuntimeError("Bundle has no engine_plan (semantic engine)")
    sem_plan = sections["engine_plan"]

    # Extract semantic embedding table
    if "semantic_embed" not in sections:
        raise RuntimeError("Bundle has no semantic_embed section")
    sem_embed_bytes = sections["semantic_embed"]

    # Parse config for Bark constants
    hidden_size = bundle_config["hidden_size"]
    num_layers = bundle_config.get("num_hidden_layers",
                                    bundle_config.get("num_layers", 12))
    max_cache = header.get("max_cache_length", 1024)
    semantic_pad_token = bundle_config.get("semantic_pad_token", 10000)
    semantic_infer_token = bundle_config.get("semantic_infer_token", 129599)
    text_encoding_offset = bundle_config.get("text_encoding_offset", 10048)
    semantic_vocab_size = bundle_config.get("semantic_vocab_size", 10000)
    temperature = 0.7
    top_k = 50

    # Reshape embedding table
    sem_embed_np = np.frombuffer(sem_embed_bytes, dtype=np.float32).copy()
    embed_vocab = len(sem_embed_np) // hidden_size
    sem_embed_np = sem_embed_np.reshape(embed_vocab, hidden_size)

    print(f"  Bundle: {args.bundle}", file=sys.stderr)
    print(f"  Hidden={hidden_size}, Layers={num_layers}, "
          f"MaxCache={max_cache}", file=sys.stderr)
    print(f"  Embed table: [{embed_vocab}, {hidden_size}]", file=sys.stderr)
    print(f"  Loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 2: Create TRT runner
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, file=sys.stderr)
    print("Step 2: Create TRT runner", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    from trtf_build.debug_runner import TrtRunner

    t0 = time.time()
    runner = TrtRunner(
        engine_plan=sem_plan,
        max_cache_length=max_cache,
        num_layers=num_layers,
    )
    print(f"  TrtRunner ready ({time.time() - t0:.1f}s)", file=sys.stderr)
    print(f"  has_embed_input={runner.has_embed_input}", file=sys.stderr)

    if not runner.has_embed_input:
        raise RuntimeError("Semantic engine missing embed_input support. "
                           "Was the bundle built with embed_input=True?")

    # ------------------------------------------------------------------
    # Step 3: Tokenize with HF processor
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, file=sys.stderr)
    print("Step 3: Tokenize", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    processor = AutoProcessor.from_pretrained(args.model)
    inputs = processor(args.prompt, return_tensors="pt")
    text_ids = inputs["input_ids"][0].numpy()
    print(f"  Prompt: {args.prompt!r}", file=sys.stderr)
    print(f"  Tokens ({len(text_ids)}): {text_ids.tolist()}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 4: Semantic prefill (matching C++ BarkBackend::run_semantic)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, file=sys.stderr)
    print("Step 4: Semantic prefill (256 text + 1 infer)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    MAX_TEXT_LEN = 256

    # Pad text_ids to 256 with zeros (matching HF Bark processor)
    padded_text = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
    copy_len = min(len(text_ids), MAX_TEXT_LEN)
    padded_text[:copy_len] = text_ids[:copy_len]

    t0 = time.time()
    for pos in range(MAX_TEXT_LEN):
        # embed = embed_table[text_tok + offset] + embed_table[semantic_pad_token]
        text_tok = int(padded_text[pos]) + text_encoding_offset
        hist_tok = semantic_pad_token
        embed = (sem_embed_np[text_tok] + sem_embed_np[hist_tok]).reshape(1, -1)
        runner.step(0, input_embed=embed, use_input_embed=1.0)

    # Feed the infer token
    infer_embed = sem_embed_np[semantic_infer_token].reshape(1, -1)
    result = runner.step(0, input_embed=infer_embed, use_input_embed=1.0)
    print(f"  Prefill done ({time.time() - t0:.1f}s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 5: Autoregressive semantic generation with HF sampling
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, file=sys.stderr)
    if args.greedy:
        print("Step 5: Semantic generation (GREEDY / argmax)", file=sys.stderr)
    else:
        print("Step 5: Semantic generation (HF sampling: "
              f"temp={temperature}, top_k={top_k})", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Seed RIGHT before sampling loop — after all model loading and prefill.
    # This ensures the RNG state at the first torch.multinomial matches HF's
    # generate() which seeds right before its sampling loop.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    t0 = time.time()
    semantic_tokens = []
    logits = result["logits"].flatten()

    for step in range(args.max_semantic_tokens):
        logits_t = torch.tensor(logits, dtype=torch.float32)

        # Token suppression: mask [semantic_pad_token+1, output_vocab)
        # (matching HF BarkSuppressTokensProcessor exactly)
        logits_t[semantic_pad_token + 1:] = float("-inf")

        if args.greedy:
            # Argmax over [0, semantic_pad_token+1)
            token = int(torch.argmax(logits_t[:semantic_pad_token + 1]).item())
        else:
            # HF's exact sampling chain:
            # 1. Temperature scaling
            logits_t = logits_t / temperature

            # 2. Top-k filtering (matching HF TopKLogitsWarper)
            k = min(top_k, semantic_pad_token + 1)
            topk_vals, topk_idx = torch.topk(logits_t[:semantic_pad_token + 1], k)
            filtered = torch.full_like(logits_t, float("-inf"))
            filtered.scatter_(0, topk_idx, topk_vals)

            # 3. Softmax + multinomial sampling
            probs = torch.softmax(filtered, dim=0)
            token = int(torch.multinomial(probs, 1).item())

        if token == semantic_pad_token:
            print(f"  EOS at step {step}", file=sys.stderr)
            break

        semantic_tokens.append(token)

        # Feed token back to TRT engine (token_id lookup, no embed override)
        # This matches C++ BarkBackend::run_semantic autoregressive path
        result = runner.step(token)
        logits = result["logits"].flatten()

    gen_time = time.time() - t0
    print(f"  Generated {len(semantic_tokens)} tokens in {gen_time:.1f}s "
          f"({len(semantic_tokens) / max(gen_time, 0.001):.1f} tok/s)",
          file=sys.stderr)

    # Token statistics
    if semantic_tokens:
        from collections import Counter
        c = Counter(semantic_tokens)
        max_consec = 0
        consec = 0
        for i in range(1, len(semantic_tokens)):
            if semantic_tokens[i] == semantic_tokens[i - 1]:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        print(f"  Unique tokens: {len(set(semantic_tokens))}", file=sys.stderr)
        print(f"  Max consecutive repeats: {max_consec}", file=sys.stderr)
        print(f"  Token range: [{min(semantic_tokens)}, {max(semantic_tokens)}]",
              file=sys.stderr)
        print(f"  First 20: {semantic_tokens[:20]}", file=sys.stderr)
        print(f"  Last 10:  {semantic_tokens[-10:]}", file=sys.stderr)

    if not semantic_tokens:
        print("ERROR: No semantic tokens generated!", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Feed through HF coarse + fine + codec
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, file=sys.stderr)
    print("Step 6: HF coarse + fine + codec", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    t0 = time.time()
    hf_model = BarkModel.from_pretrained(args.model).eval()
    print(f"  HF model loaded ({time.time() - t0:.1f}s)", file=sys.stderr)

    sem_cfg = BarkSemanticGenerationConfig(
        **hf_model.generation_config.semantic_config)
    coarse_cfg = BarkCoarseGenerationConfig(
        **hf_model.generation_config.coarse_acoustics_config)
    fine_cfg = BarkFineGenerationConfig(
        **hf_model.generation_config.fine_acoustics_config)

    # BarkSemanticModel.generate() returns ONLY the generated tokens
    # (it strips the 257-token prefill). So the coarse model expects:
    # [semantic_tokens, eos_token] — NOT the full text+infer+semantic sequence.
    sem_seq = np.array(semantic_tokens + [semantic_pad_token], dtype=np.int64)
    sem_seq_t = torch.tensor(sem_seq, dtype=torch.long).unsqueeze(0)

    print(f"  Semantic output for coarse: {len(sem_seq)} tokens", file=sys.stderr)

    with torch.no_grad():
        t0 = time.time()
        hf_coarse = hf_model.coarse_acoustics.generate(
            sem_seq_t,
            semantic_generation_config=sem_cfg,
            coarse_generation_config=coarse_cfg,
        )
        print(f"  Coarse: {hf_coarse.shape[1]} tokens ({time.time() - t0:.1f}s)",
              file=sys.stderr)

        t0 = time.time()
        hf_fine = hf_model.fine_acoustics.generate(
            hf_coarse,
            semantic_generation_config=sem_cfg,
            coarse_generation_config=coarse_cfg,
            fine_generation_config=fine_cfg,
        )
        print(f"  Fine: {hf_fine.shape} ({time.time() - t0:.1f}s)",
              file=sys.stderr)

        t0 = time.time()
        audio = hf_model.codec_decode(hf_fine)
        wav = audio.squeeze().cpu().numpy()
        print(f"  Codec: {len(wav)} samples ({time.time() - t0:.1f}s)",
              file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 7: Save WAV
    # ------------------------------------------------------------------
    write_wav(args.output, wav)
    duration = len(wav) / 24000
    rms = float(np.sqrt(np.mean(wav ** 2)))

    print("\n" + "=" * 60, file=sys.stderr)
    print("RESULT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)
    print(f"  Duration: {duration:.2f}s ({len(wav)} samples @ 24kHz)",
          file=sys.stderr)
    print(f"  RMS: {rms:.4f}", file=sys.stderr)
    print(f"  Semantic tokens: {len(semantic_tokens)}", file=sys.stderr)
    mode = "GREEDY" if args.greedy else "HF sampling (torch.multinomial)"
    print(f"  Sampling: {mode}", file=sys.stderr)
    print("", file=sys.stderr)
    print("LISTEN to the output WAV:", file=sys.stderr)
    print(f"  - If it sounds like '{args.prompt}' → C++ sampling is the issue "
          "(Path A)", file=sys.stderr)
    print("  - If it sounds bad → TRT logits diverge from HF (Path B)",
          file=sys.stderr)

    # Also dump semantic tokens for further analysis
    token_path = args.output.replace(".wav", ".tokens")
    with open(token_path, "w") as f:
        for t in semantic_tokens:
            f.write(f"{t}\n")
    print(f"\n  Semantic tokens saved: {token_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Bonus: Also run pure HF pipeline for A/B comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, file=sys.stderr)
    print("Bonus: Pure HF reference (same prompt, HF semantic + all stages)",
          file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    try:
        with torch.no_grad():
            if args.seed is not None:
                torch.manual_seed(args.seed)
            t0 = time.time()
            # BarkModel.generate() may return a dict or raw tensor depending
            # on the transformers version.  Use return_output_lengths to force
            # a consistent code path, and also request return_dict_in_generate.
            hf_full = hf_model.generate(
                inputs["input_ids"],
                semantic_generation_config=sem_cfg,
                coarse_generation_config=coarse_cfg,
                fine_generation_config=fine_cfg,
                return_dict_in_generate=True,
            )
            # Extract fine output — could be dict key or attribute
            if isinstance(hf_full, dict):
                fine_out = hf_full["fine_output"]
            elif hasattr(hf_full, "fine_output"):
                fine_out = hf_full.fine_output
            else:
                # Fallback: raw tensor is the fine output directly
                fine_out = hf_full
            if fine_out.dim() == 2:
                fine_out = fine_out.unsqueeze(0)
            hf_full_audio = hf_model.codec_decode(fine_out)
            hf_wav = hf_full_audio.squeeze().cpu().numpy()
            print(f"  Pure HF done ({time.time() - t0:.1f}s)", file=sys.stderr)

        hf_path = args.output.replace(".wav", "_hf_ref.wav")
        write_wav(hf_path, hf_wav)
        hf_rms = float(np.sqrt(np.mean(hf_wav ** 2)))
        print(f"  HF reference: {hf_path} ({len(hf_wav) / 24000:.2f}s, "
              f"RMS={hf_rms:.4f})", file=sys.stderr)
    except Exception as e:
        print(f"  Bonus HF reference failed: {e}", file=sys.stderr)
        print("  (Main test WAV was saved successfully above)", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
