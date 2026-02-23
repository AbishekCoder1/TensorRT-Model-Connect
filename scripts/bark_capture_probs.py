#!/usr/bin/env python3
"""Capture exact probability tensors at first multinomial in HF generate vs manual."""
import torch
import numpy as np
import sys
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

model = BarkModel.from_pretrained("suno/bark-small").eval()
sem = model.semantic
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
sp = 10000; si = 129599; to = 10048

# Capture probs from HF generate
_orig_multi = torch.multinomial
captured_probs = [None]
call_count = [0]

def capture_multinomial(input, num_samples, **kwargs):
    call_count[0] += 1
    if call_count[0] == 1:
        captured_probs[0] = input.clone().detach()
    return _orig_multi(input, num_samples, **kwargs)

torch.multinomial = capture_multinomial
torch.manual_seed(42)
with torch.no_grad():
    out = sem.generate(inputs["input_ids"], semantic_generation_config=sem_cfg)
hf_gen_toks = out[0, 257:].numpy()
hf_gen_toks = hf_gen_toks[hf_gen_toks < sp]
torch.multinomial = _orig_multi

hf_probs = captured_probs[0]
print(f"HF generate first multinomial probs shape: {hf_probs.shape}")
print(f"HF generate first 5 tokens: {hf_gen_toks[:5].tolist()}")
hf_p1d = hf_probs.squeeze()

# Manual prefill + process
text_ids = inputs["input_ids"][0]
padded = torch.zeros(256, dtype=torch.long)
padded[:len(text_ids)] = text_ids
input_off = padded.unsqueeze(0) + to
hist = torch.full((1, 256), sp, dtype=torch.long)
infer = torch.tensor([[si]], dtype=torch.long)

with torch.no_grad():
    emb = torch.cat([
        sem.input_embeds_layer(input_off) + sem.input_embeds_layer(hist),
        sem.input_embeds_layer(infer)
    ], dim=1)
    fwd = sem(inputs_embeds=emb, use_cache=True)
manual_logits = fwd.logits[0, -1]

# Apply HF processing chain manually
scores = manual_logits.clone()
scores[sp + 1:] = float("-inf")
scores = torch.nn.functional.log_softmax(scores.float(), dim=-1)
scores = scores / 0.7
top_k = min(50, scores.size(-1))
indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
scores = scores.masked_fill(indices_to_remove, float("-inf"))
manual_probs = torch.softmax(scores, dim=-1)

# Compare
diff = (hf_p1d - manual_probs).abs()
print(f"\nProb comparison:")
print(f"  Shape: HF={hf_p1d.shape}, Manual={manual_probs.shape}")
print(f"  Max diff: {diff.max():.10f}")
print(f"  Sum HF probs: {hf_p1d.sum():.10f}")
print(f"  Sum manual probs: {manual_probs.sum():.10f}")

# Top tokens from each
hf_top5 = hf_p1d.topk(5)
man_top5 = manual_probs.topk(5)
print(f"\n  HF top-5 probs:    {hf_top5.values.tolist()}")
print(f"  HF top-5 indices:  {hf_top5.indices.tolist()}")
print(f"  Man top-5 probs:   {man_top5.values.tolist()}")
print(f"  Man top-5 indices: {man_top5.indices.tolist()}")

# What token would be sampled from each with seed 42?
torch.manual_seed(42)
tok_hf = _orig_multi(hf_p1d.unsqueeze(0), 1).item()
torch.manual_seed(42)
tok_man = _orig_multi(manual_probs.unsqueeze(0), 1).item()
print(f"\n  From HF probs, seed 42: token {tok_hf}")
print(f"  From manual probs, seed 42: token {tok_man}")

# Check: do they differ?
if tok_hf != tok_man:
    print(f"\n  DIFFERENT! Investigating...")
    print(f"  P({tok_hf}) in HF: {hf_p1d[tok_hf]:.10f}")
    print(f"  P({tok_hf}) in manual: {manual_probs[tok_hf]:.10f}")
    print(f"  P({tok_man}) in HF: {hf_p1d[tok_man]:.10f}")
    print(f"  P({tok_man}) in manual: {manual_probs[tok_man]:.10f}")
