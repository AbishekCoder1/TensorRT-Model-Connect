#!/usr/bin/env python3
"""Compare step-0 logits between HF (correct prefill) and C++ (TRT).

HF's semantic.generate() uses inputs_embeds = embed(text+offset) + embed(history)
for prefill, NOT a simple token ID forward pass. This script replicates that
exact prefill and compares the resulting logits with C++ output.
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
semantic = model.semantic

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)

# Replicate EXACT HF prefill from semantic.generate() lines 64-71
input_ids = inputs["input_ids"].clone()
input_ids = input_ids + sem_cfg.text_encoding_offset

max_input_semantic_length = sem_cfg.max_input_semantic_length  # 256

# No voice preset -> history is all semantic_pad_token
semantic_history = torch.full(
    (max_input_semantic_length,),
    sem_cfg.semantic_pad_token,
    dtype=torch.int,
).unsqueeze(0)

infer_array = torch.tensor([[sem_cfg.semantic_infer_token]], dtype=torch.int)

# Build inputs_embeds exactly as HF does
inputs_embeds = torch.cat([
    semantic.input_embeds_layer(input_ids[:, :max_input_semantic_length])
    + semantic.input_embeds_layer(semantic_history[:, :max_input_semantic_length + 1]),
    semantic.input_embeds_layer(infer_array),
], dim=1)

print(f"inputs_embeds shape: {inputs_embeds.shape}")  # [1, 257, 768]

# Run model with inputs_embeds (matching HF's generate path)
with torch.no_grad():
    outputs = semantic(inputs_embeds=inputs_embeds, use_cache=True)
    logits_hf = outputs.logits[0, -1].cpu().numpy()  # step 0 logits
    past_kv = outputs.past_key_values

print(f"HF step 0 logits shape: {logits_hf.shape}")
print(f"HF step 0 top-5 indices: {np.argsort(logits_hf)[::-1][:5].tolist()}")
print(f"HF step 0 top-5 values: {np.sort(logits_hf)[::-1][:5].tolist()}")

# Apply suppression (matching HF's SuppressTokensLogitsProcessor)
suppressed_hf = logits_hf.copy()
# range(semantic_vocab_size, semantic_pad_token) = range(10000, 10000) = empty
# range(semantic_pad_token+1, output_vocab_size) = range(10001, 10048)
for i in range(sem_cfg.semantic_pad_token + 1, len(suppressed_hf)):
    suppressed_hf[i] = float('-inf')

# After suppression, top-k with temperature
temp = sem_cfg.temperature  # 0.7
top_k = sem_cfg.top_k  # 50

topk_idx = np.argsort(suppressed_hf)[::-1][:top_k]
topk_logits = suppressed_hf[topk_idx]
topk_scaled = (topk_logits - topk_logits[0]) / temp
topk_probs = np.exp(topk_scaled)
topk_probs /= topk_probs.sum()

print(f"\nAfter suppression + temp={temp} + top_k={top_k}:")
print(f"  top-5 tokens: {topk_idx[:5].tolist()}")
print(f"  top-5 probs:  {topk_probs[:5].tolist()}")
print(f"  top-1 prob:   {topk_probs[0]:.4f}")

# Generate a few tokens using HF's actual generate to see what it picks
print("\nHF generate (5 tokens):")
with torch.no_grad():
    for step in range(5):
        # Apply suppression
        logits_t = outputs.logits[0, -1].clone()
        logits_t[sem_cfg.semantic_pad_token + 1:] = float('-inf')
        # Temperature + top-k + sample
        logits_t = logits_t / temp
        topk_vals, topk_indices = torch.topk(logits_t, top_k)
        filtered = torch.full_like(logits_t, float('-inf'))
        filtered.scatter_(0, topk_indices, topk_vals)
        probs = torch.softmax(filtered, dim=0)
        token = torch.multinomial(probs, 1).item()
        print(f"  Step {step}: token={token}")

        # Feed back
        token_t = torch.tensor([[token]], dtype=torch.long)
        outputs = semantic(input_ids=token_t, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values

# Now compare: also do the prefill WITHOUT the history sum
# (what the old C++ bug was doing)
print("\n=== COMPARISON: with vs without history embedding ===")
# Without history (single lookup for each text position)
inputs_embeds_nohistory = torch.cat([
    semantic.input_embeds_layer(input_ids[:, :max_input_semantic_length]),
    semantic.input_embeds_layer(infer_array),
], dim=1)

with torch.no_grad():
    outputs_nh = semantic(inputs_embeds=inputs_embeds_nohistory)
    logits_nh = outputs_nh.logits[0, -1].cpu().numpy()

print(f"With history    - top-5: {np.argsort(logits_hf)[::-1][:5].tolist()}, "
      f"values: {np.sort(logits_hf)[::-1][:5].tolist()}")
print(f"Without history - top-5: {np.argsort(logits_nh)[::-1][:5].tolist()}, "
      f"values: {np.sort(logits_nh)[::-1][:5].tolist()}")

diff = np.abs(logits_hf - logits_nh)
print(f"Logit diff: mean={diff.mean():.4f}, max={diff.max():.4f}")
