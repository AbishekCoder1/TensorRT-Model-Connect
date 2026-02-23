#!/usr/bin/env python3
"""Compare C++ vs HF sampling step-by-step for the semantic model.

Runs HF semantic model step-by-step, logging logits and sampled tokens,
to compare with C++ behavior.
"""
import numpy as np
import sys
import torch
from transformers import AutoProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig,
)


print("Loading model...", file=sys.stderr)
model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)

# Run HF semantic with sampling and capture per-step tokens
# We need to manually step through the generation
semantic = model.semantic

# Prepare input: text tokens + text_encoding_offset, padded to 256
input_ids = inputs["input_ids"][0].numpy()  # [256]
text_encoding_offset = sem_cfg.text_encoding_offset  # 10048
semantic_pad_token = sem_cfg.semantic_pad_token  # 10000
semantic_infer_token = sem_cfg.semantic_infer_token  # 129599
semantic_vocab_size = sem_cfg.semantic_vocab_size  # 10000
min_eos_p = 0.2
temperature = sem_cfg.temperature  # 0.7
top_k = sem_cfg.top_k  # 50

print(f"Config: temp={temperature}, top_k={top_k}, min_eos_p={min_eos_p}")
print(f"  text_encoding_offset={text_encoding_offset}")
print(f"  semantic_pad_token={semantic_pad_token}")
print(f"  semantic_infer_token={semantic_infer_token}")

# Build input sequence matching C++:
# [text_tokens + offset, padded_zeros + offset] for 256 positions
input_tokens = (input_ids + text_encoding_offset).astype(np.int64)
print(f"Input tokens first 10: {input_tokens[:10].tolist()}")

# Prefill all 256 text positions + infer token
full_input = np.concatenate([input_tokens, [semantic_infer_token]])
full_input_t = torch.tensor(full_input, dtype=torch.long).unsqueeze(0)

print(f"\nPrefilling {len(full_input)} tokens...", file=sys.stderr)

with torch.no_grad():
    # Run the model on the full prefix to get logits for the first generated token
    outputs = semantic(full_input_t)
    logits_step0 = outputs.logits[0, -1].cpu().numpy()  # last position's logits

print(f"Step 0 logits shape: {logits_step0.shape}")
print(f"Step 0 logits[0:5]: {logits_step0[:5].tolist()}")
print(f"Step 0 logits[9995:10005]: {logits_step0[9995:10005].tolist()}")

# Apply suppression (same as C++)
suppressed = logits_step0.copy()
suppressed[semantic_vocab_size:semantic_pad_token] = -1e9
suppressed[semantic_pad_token + 1:] = -1e9

# Compute EOS probability
valid_logits = suppressed[:semantic_pad_token + 1].copy()
max_val = valid_logits.max()
exp_vals = np.exp(valid_logits - max_val)
exp_vals[valid_logits < -1e8] = 0  # suppress -inf
eos_p = exp_vals[semantic_pad_token] / exp_vals.sum()
print(f"Step 0 EOS probability: {eos_p:.6f} (threshold={min_eos_p})")

# Top-k analysis
top_k_indices = np.argsort(suppressed)[::-1][:top_k]
top_k_logits = suppressed[top_k_indices]
print(f"Step 0 top-5 indices: {top_k_indices[:5].tolist()}")
print(f"Step 0 top-5 logits: {top_k_logits[:5].tolist()}")

# Apply temperature and softmax over top-k
top_k_scaled = (top_k_logits - top_k_logits[0]) / temperature
top_k_probs = np.exp(top_k_scaled)
top_k_probs /= top_k_probs.sum()
print(f"Step 0 top-5 probs: {top_k_probs[:5].tolist()}")
print(f"Step 0 top-1 prob: {top_k_probs[0]:.4f} (token {top_k_indices[0]})")

# HF sampling (torch multinomial)
logits_t = torch.tensor(suppressed[:semantic_pad_token + 1], dtype=torch.float32)
logits_t = logits_t / temperature
# Top-k filter
topk_vals, topk_idx = torch.topk(logits_t, top_k)
filter_t = torch.full_like(logits_t, float('-inf'))
filter_t.scatter_(0, topk_idx, topk_vals)
probs_t = torch.softmax(filter_t, dim=0)

# Sample 100 times to see distribution
samples = torch.multinomial(probs_t, 100, replacement=True).numpy()
from collections import Counter
sample_counts = Counter(samples.tolist())
top_sampled = sample_counts.most_common(10)
print(f"\nHF multinomial sampling (100 draws):")
for tok, cnt in top_sampled:
    print(f"  token {tok}: {cnt} times ({100*cnt/100:.0f}%)")

# C++ mt19937 sampling simulation
print(f"\nC++ mt19937 sampling simulation (100 draws, seed=42):")
import random
rng = random.Random()
# Use numpy's mt19937 to match C++ exactly
cpp_rng = np.random.RandomState(42)
cpp_samples = []
for _ in range(100):
    r = cpp_rng.random()  # uniform [0, 1)
    cumulative = 0.0
    selected = top_k_indices[top_k - 1]
    for i in range(top_k):
        cumulative += top_k_probs[i]
        if r < cumulative:
            selected = top_k_indices[i]
            break
    cpp_samples.append(selected)
cpp_counts = Counter(cpp_samples)
top_cpp = cpp_counts.most_common(10)
for tok, cnt in top_cpp:
    print(f"  token {tok}: {cnt} times ({100*cnt/100:.0f}%)")

# Now simulate full C++ generation for first 50 steps
print(f"\n=== Full C++ simulation vs actual C++ tokens (first 50 steps) ===")
cpp_sem_actual = np.array([int(l.strip()) for l in open("/tmp/bark_crossfeed_dump.sem_tokens")])

# We need to step through the model one token at a time
generated = []
cpp_rng2 = np.random.RandomState(42)
past_kv = None
current_input = full_input_t.clone()

with torch.no_grad():
    for step in range(min(50, len(cpp_sem_actual))):
        if past_kv is None:
            outputs = semantic(current_input, use_cache=True)
            past_kv = outputs.past_key_values
        else:
            token_input = torch.tensor([[generated[-1]]], dtype=torch.long)
            outputs = semantic(token_input, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values

        logits = outputs.logits[0, -1].cpu().numpy()

        # Suppress
        logits[semantic_vocab_size:semantic_pad_token] = -1e9
        logits[semantic_pad_token + 1:] = -1e9

        # EOS check
        valid = logits[:semantic_pad_token + 1].copy()
        mx = valid.max()
        ev = np.exp(valid - mx)
        ev[valid < -1e8] = 0
        eos_p = ev[semantic_pad_token] / ev.sum()

        if eos_p > min_eos_p:
            print(f"  Step {step}: EOS triggered (p={eos_p:.4f})")
            break

        # Top-k sampling (matching C++ exactly)
        topk_idx = np.argsort(logits[:semantic_pad_token + 1])[::-1][:top_k]
        topk_logits = logits[topk_idx]
        topk_scaled = (topk_logits - topk_logits[0]) / temperature
        topk_probs = np.exp(topk_scaled)
        topk_probs /= topk_probs.sum()

        r = cpp_rng2.random()
        cumul = 0.0
        token = topk_idx[top_k - 1]
        for i in range(top_k):
            cumul += topk_probs[i]
            if r < cumul:
                token = topk_idx[i]
                break

        if token == semantic_pad_token:
            print(f"  Step {step}: Sampled EOS")
            break

        generated.append(int(token))
        actual = int(cpp_sem_actual[step]) if step < len(cpp_sem_actual) else -1
        match = "OK" if token == actual else "DIFF"
        if step < 20 or match == "DIFF":
            print(f"  Step {step:3d}: simulated={token:5d}, actual={actual:5d}  {match}"
                  f"  (r={r:.4f}, top1_p={topk_probs[0]:.3f}, top1={topk_idx[0]})")

print(f"\nFirst 50 simulated: {generated[:50]}")
print(f"First 50 actual:    {cpp_sem_actual[:50].tolist()}")
sim_match = sum(1 for a, b in zip(generated, cpp_sem_actual[:len(generated)]) if a == b)
print(f"Match: {sim_match}/{len(generated)}")
