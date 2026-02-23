#!/usr/bin/env python3
"""Trace the exact token feeding pattern of HF vs C++ semantic model.

In HF, the semantic model's input during generation is:
  input_ids (from processor) + text_encoding_offset -> prefill
  Then autoregressive: each new token is fed directly

In C++, the semantic model's input during generation is:
  For each prefill position: embed_table[text + offset] + embed_table[pad_token]
  Then autoregressive: embed_table[semantic_token]

This script checks if there's a mismatch in the prefill feeding pattern.
"""
import numpy as np
import sys
import torch
from transformers import AutoProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig


print("Loading model...", file=sys.stderr)
model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
semantic = model.semantic

# Check what HF's semantic.generate() actually feeds as input
input_ids_raw = inputs["input_ids"][0].numpy()  # [256] from processor
print(f"Processor input_ids (first 10): {input_ids_raw[:10].tolist()}")
print(f"Processor input_ids (non-zero): {np.sum(input_ids_raw != 0)}")

# In HF's semantic.generate(), it does: input_ids = input_ids + text_encoding_offset
# Then feeds ALL of input_ids to the model at once for prefill
offset_ids = input_ids_raw + sem_cfg.text_encoding_offset
print(f"\nAfter offset (first 10): {offset_ids[:10].tolist()}")
print(f"Padding value: {sem_cfg.text_encoding_offset} (0 + offset)")

# In C++, the feeding pattern is:
# For pos in range(256):
#   text_tok = padded_text[pos] + text_encoding_offset
#   hist_tok = semantic_pad_token (10000)
#   embed = embed_table[text_tok] + embed_table[hist_tok]
#
# This SUMS two embeddings. But HF just does embed_table[text_tok + offset].
# Are these equivalent?

# Check embed table values
embed_table = semantic.input_embeds_layer.weight.data  # [vocab_size, hidden]
print(f"\nEmbedding table shape: {embed_table.shape}")

# Check if embed[semantic_pad_token=10000] is near zero
pad_embed = embed_table[10000].cpu().numpy()
pad_norm = np.linalg.norm(pad_embed)
print(f"embed[10000] (semantic_pad_token) norm: {pad_norm:.6f}")
print(f"embed[10000] first 5: {pad_embed[:5].tolist()}")
print(f"embed[10000] max abs: {np.abs(pad_embed).max():.6f}")

if pad_norm > 0.01:
    print("\n*** WARNING: embed[10000] is NOT zero! ***")
    print("*** C++ adds this to every prefill position, HF does not. ***")
    print("*** This means C++ prefill embeddings differ from HF! ***")

    # Quantify the difference
    # C++ embed for pos 0: embed[41226] + embed[10000]
    # HF  embed for pos 0: embed[41226]
    tok0 = int(offset_ids[0])
    cpp_embed0 = (embed_table[tok0] + embed_table[10000]).cpu().numpy()
    hf_embed0 = embed_table[tok0].cpu().numpy()
    diff_embed = np.abs(cpp_embed0 - hf_embed0)
    print(f"\n  Position 0 embed diff: mean={diff_embed.mean():.6f}, max={diff_embed.max():.6f}")
    print(f"  This is the embed[10000] vector being added to every position")

    # But greedy tokens match 100%! How?
    # Maybe the engine handles this internally?
    # Let's check: is embed_input=True? If so, C++ feeds pre-computed embeddings.
    # The engine has NO embedding table - it just sees the float vector.
    print("\n  But greedy tokens match 100%...")
    print("  If the engine is embed_input=True, C++ controls what embedding goes in.")
    print("  The bug: C++ adds embed[10000] to each prefill position.")
    print("  This changes the hidden state, but if the model is robust,")
    print("  greedy argmax might still produce the same token (just barely).")
    print("  With sampling, this perturbation shifts the distribution enough")
    print("  to cause different/worse sampling outcomes.")

# Also check: does HF's semantic model use two embedding layers?
print(f"\nSemantic model embedding layers:")
for name, param in semantic.named_parameters():
    if "embed" in name.lower():
        print(f"  {name}: {param.shape}")
