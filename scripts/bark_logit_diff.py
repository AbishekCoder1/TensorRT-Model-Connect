#!/usr/bin/env python3
"""Compare TRT vs HF semantic model logits step-by-step during GREEDY generation.

Because generation is greedy (argmax), both models produce identical tokens at
each step.  This lets us compare logit vectors in lockstep and measure how
much the TRT engine diverges from the HF reference.

Usage:
  python3 scripts/bark_logit_diff.py --bundle /tmp/bark.trtfb
  python3 scripts/bark_logit_diff.py --bundle /tmp/bark.trtfb --max-tokens 200
  python3 scripts/bark_logit_diff.py --bundle /tmp/bark.trtfb --prompt "The quick brown fox"
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
# Statistics helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare TRT vs HF semantic logits step-by-step (greedy)")
    parser.add_argument("--bundle", required=True,
                        help="Path to Bark .trtfb bundle")
    parser.add_argument("--model", default="suno/bark-small",
                        help="HF model ID for BarkModel")
    parser.add_argument("--prompt", default="Hello, my dog is cute",
                        help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max autoregressive tokens to generate")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, BarkModel
    from transformers.models.bark.generation_configuration_bark import (
        BarkSemanticGenerationConfig,
    )

    # ------------------------------------------------------------------
    # Step 1: Load TRT semantic engine from bundle
    # ------------------------------------------------------------------
    print("=" * 70, file=sys.stderr)
    print("Step 1: Load TRT semantic engine from bundle", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    t0 = time.time()
    header, sections = read_bundle(args.bundle)
    bundle_config = json.loads(sections["config.json"].decode("utf-8"))

    if "engine_plan" not in sections:
        raise RuntimeError("Bundle has no engine_plan (semantic engine)")
    sem_plan = sections["engine_plan"]

    if "semantic_embed" not in sections:
        raise RuntimeError("Bundle has no semantic_embed section")
    sem_embed_bytes = sections["semantic_embed"]

    hidden_size = bundle_config["hidden_size"]
    num_layers = bundle_config.get("num_hidden_layers",
                                    bundle_config.get("num_layers", 12))
    max_cache = header.get("max_cache_length", 1024)
    semantic_pad_token = bundle_config.get("semantic_pad_token", 10000)
    semantic_infer_token = bundle_config.get("semantic_infer_token", 129599)
    text_encoding_offset = bundle_config.get("text_encoding_offset", 10048)
    semantic_vocab_size = bundle_config.get("semantic_vocab_size", 10000)

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
    print("\n" + "=" * 70, file=sys.stderr)
    print("Step 2: Create TRT runner", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

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
    # Step 3: Load HF model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, file=sys.stderr)
    print("Step 3: Load HF BarkModel", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    t0 = time.time()
    hf_model = BarkModel.from_pretrained(args.model).eval()
    hf_semantic = hf_model.semantic
    hf_sem_cfg = BarkSemanticGenerationConfig(
        **hf_model.generation_config.semantic_config)
    print(f"  HF model loaded ({time.time() - t0:.1f}s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 4: Tokenize
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, file=sys.stderr)
    print("Step 4: Tokenize", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    processor = AutoProcessor.from_pretrained(args.model)
    inputs = processor(args.prompt, return_tensors="pt")
    text_ids = inputs["input_ids"][0].numpy()
    print(f"  Prompt: {args.prompt!r}", file=sys.stderr)
    print(f"  Tokens ({len(text_ids)}): {text_ids[:20].tolist()}{'...' if len(text_ids) > 20 else ''}",
          file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 5: Prefill both models (256 text positions + 1 infer token)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, file=sys.stderr)
    print("Step 5: Prefill (256 text + 1 infer token)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    MAX_TEXT_LEN = 256

    # --- TRT prefill ---
    print("  TRT prefill...", file=sys.stderr)
    t0 = time.time()
    padded_text = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
    copy_len = min(len(text_ids), MAX_TEXT_LEN)
    padded_text[:copy_len] = text_ids[:copy_len]

    for pos in range(MAX_TEXT_LEN):
        text_tok = int(padded_text[pos]) + text_encoding_offset
        hist_tok = semantic_pad_token
        embed = (sem_embed_np[text_tok] + sem_embed_np[hist_tok]).reshape(1, -1)
        runner.step(0, input_embed=embed, use_input_embed=1.0)

    # Feed the infer token
    infer_embed = sem_embed_np[semantic_infer_token].reshape(1, -1)
    trt_result = runner.step(0, input_embed=infer_embed, use_input_embed=1.0)
    trt_logits = trt_result["logits"].flatten()
    print(f"  TRT prefill done ({time.time() - t0:.1f}s)", file=sys.stderr)

    # --- HF prefill ---
    # Replicate HF's semantic.generate() prefill: builds inputs_embeds as
    #   embed(text_ids + offset) + embed(history)
    # Then appends embed(semantic_infer_token) and runs a single forward pass.
    print("  HF prefill...", file=sys.stderr)
    t0 = time.time()

    with torch.no_grad():
        # Build the same input_ids that HF uses
        hf_input_ids = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0)
        hf_input_ids = hf_input_ids + hf_sem_cfg.text_encoding_offset

        # History = all semantic_pad_token
        semantic_history = torch.full(
            (MAX_TEXT_LEN,),
            hf_sem_cfg.semantic_pad_token,
            dtype=torch.int,
        ).unsqueeze(0)

        infer_array = torch.tensor(
            [[hf_sem_cfg.semantic_infer_token]], dtype=torch.int)

        # Build inputs_embeds: embed(text+offset) + embed(history), then infer
        inputs_embeds = torch.cat([
            hf_semantic.input_embeds_layer(
                hf_input_ids[:, :MAX_TEXT_LEN])
            + hf_semantic.input_embeds_layer(
                semantic_history[:, :MAX_TEXT_LEN]),
            hf_semantic.input_embeds_layer(infer_array),
        ], dim=1)

        hf_outputs = hf_semantic(
            inputs_embeds=inputs_embeds, use_cache=True)
        hf_logits = hf_outputs.logits[0, -1].cpu().numpy()
        hf_past_kv = hf_outputs.past_key_values

    print(f"  HF prefill done ({time.time() - t0:.1f}s)", file=sys.stderr)

    # Quick sanity check on prefill logits
    prefill_max_diff = float(np.max(np.abs(trt_logits - hf_logits)))
    prefill_cos = cosine_similarity(trt_logits, hf_logits)
    print(f"  Prefill logit comparison: max_diff={prefill_max_diff:.6f}, "
          f"cos_sim={prefill_cos:.8f}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 6: Autoregressive greedy generation with step-by-step comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"Step 6: Autoregressive greedy generation ({args.max_tokens} steps)",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Tracking variables
    step_stats = []
    worst_max_diff = 0.0
    worst_max_diff_step = -1
    total_cos_sim = 0.0
    agreements = 0
    disagreements = 0

    # Use the post-prefill logits for the first greedy token
    current_trt_logits = trt_logits
    current_hf_logits = hf_logits

    t0 = time.time()

    for step in range(args.max_tokens):
        # --- Compare logits at this step ---
        # Only compare over the valid semantic vocab range [0, semantic_pad_token+1)
        # (tokens beyond semantic_pad_token are suppressed anyway)
        trt_valid = current_trt_logits[:semantic_pad_token + 1]
        hf_valid = current_hf_logits[:semantic_pad_token + 1]

        max_diff = float(np.max(np.abs(trt_valid - hf_valid)))
        mean_diff = float(np.mean(np.abs(trt_valid - hf_valid)))
        cos_sim = cosine_similarity(trt_valid, hf_valid)

        # Full-range comparison too (for reference)
        full_max_diff = float(np.max(np.abs(current_trt_logits - current_hf_logits)))

        # Greedy: argmax over the valid range
        trt_token = int(np.argmax(trt_valid))
        hf_token = int(np.argmax(hf_valid))
        agree = trt_token == hf_token

        if agree:
            agreements += 1
            token = trt_token
            agree_str = "AGREE"
        else:
            disagreements += 1
            token = trt_token  # Use TRT token to keep TRT side in sync
            agree_str = f"DISAGREE (trt={trt_token}, hf={hf_token})"

        # Track worst step
        if max_diff > worst_max_diff:
            worst_max_diff = max_diff
            worst_max_diff_step = step
        total_cos_sim += cos_sim

        step_stats.append({
            "step": step,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "cos_sim": cos_sim,
            "full_max_diff": full_max_diff,
            "trt_token": trt_token,
            "hf_token": hf_token,
            "agree": agree,
        })

        # Print per-step output
        print(f"Step {step:3d}: max_diff={max_diff:.6f}, "
              f"mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f}, "
              f"token={token:5d} ({agree_str})")

        # Check for EOS
        if token == semantic_pad_token:
            print(f"  [EOS at step {step}]", file=sys.stderr)
            break

        # --- Feed the greedy token to both models ---

        # TRT: autoregressive step (token_id lookup, no embed override)
        trt_result = runner.step(token)
        current_trt_logits = trt_result["logits"].flatten()

        # HF: autoregressive step with past_key_values
        with torch.no_grad():
            token_t = torch.tensor([[token]], dtype=torch.long)
            hf_outputs = hf_semantic(
                input_ids=token_t,
                past_key_values=hf_past_kv,
                use_cache=True,
            )
            current_hf_logits = hf_outputs.logits[0, -1].cpu().numpy()
            hf_past_kv = hf_outputs.past_key_values

    gen_time = time.time() - t0
    num_steps = len(step_stats)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Prompt: {args.prompt!r}")
    print(f"  Steps compared: {num_steps}")
    print(f"  Generation time: {gen_time:.1f}s "
          f"({num_steps / max(gen_time, 0.001):.1f} steps/s)")
    print()
    print(f"  Prefill logit max_diff:  {prefill_max_diff:.6f}")
    print(f"  Prefill cosine sim:      {prefill_cos:.8f}")
    print()

    if num_steps > 0:
        all_max_diffs = [s["max_diff"] for s in step_stats]
        all_mean_diffs = [s["mean_diff"] for s in step_stats]
        all_cos_sims = [s["cos_sim"] for s in step_stats]

        print(f"  Autoregressive stats (over valid vocab [0, {semantic_pad_token}]):")
        print(f"    Max absolute diff across all steps: "
              f"{worst_max_diff:.6f} (at step {worst_max_diff_step})")
        print(f"    Mean of per-step max diffs:         "
              f"{np.mean(all_max_diffs):.6f}")
        print(f"    Mean of per-step mean diffs:        "
              f"{np.mean(all_mean_diffs):.6f}")
        print(f"    Mean cosine similarity:             "
              f"{np.mean(all_cos_sims):.8f}")
        print(f"    Min cosine similarity:              "
              f"{np.min(all_cos_sims):.8f} (at step "
              f"{int(np.argmin(all_cos_sims))})")
        print()
        print(f"  Token agreement: {agreements}/{num_steps} "
              f"({100.0 * agreements / num_steps:.1f}%)")
        if disagreements > 0:
            disagree_steps = [s["step"] for s in step_stats if not s["agree"]]
            print(f"  Disagreement steps: {disagree_steps}")
        print()

        # Percentile breakdown of max_diff
        pcts = [50, 90, 95, 99, 100]
        print("  Max diff percentiles:")
        for p in pcts:
            val = float(np.percentile(all_max_diffs, p))
            print(f"    p{p:3d}: {val:.6f}")
    else:
        print("  No autoregressive steps were performed.")

    print()
    print("  Generated tokens (first 30):", file=sys.stderr)
    tokens_generated = [s["trt_token"] for s in step_stats
                        if s["trt_token"] != semantic_pad_token]
    print(f"  {tokens_generated[:30]}", file=sys.stderr)


if __name__ == "__main__":
    main()
