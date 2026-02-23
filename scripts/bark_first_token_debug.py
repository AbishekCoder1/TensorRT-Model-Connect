#!/usr/bin/env python3
"""Find why HF generate() and manual loop produce different tokens."""
import torch
import numpy as np
import sys
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

model = BarkModel.from_pretrained("suno/bark-small").eval()
sem = model.semantic
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sem_cfg_dict = model.generation_config.semantic_config
sem_cfg = BarkSemanticGenerationConfig(**sem_cfg_dict)
sp = 10000; si = 129599; to = 10048

# Method A: Manual loop (matching what the isolation script does)
text_ids = inputs["input_ids"][0]
padded = torch.zeros(256, dtype=torch.long)
padded[:len(text_ids)] = text_ids
input_off = padded.unsqueeze(0) + to
hist = torch.full((1, 256), sp, dtype=torch.long)
infer = torch.tensor([[si]], dtype=torch.long)

with torch.no_grad():
    emb = torch.cat([
        sem.input_embeds_layer(input_off)
        + sem.input_embeds_layer(hist),
        sem.input_embeds_layer(infer)
    ], dim=1)
    out = sem(inputs_embeds=emb, use_cache=True)
manual_logits = out.logits[0, -1].clone()  # Last position logits

# Method B: Replicate HF generate's prefill exactly
# HF generate passes input_ids = ones(1, 257) + inputs_embeds
# and uses super().generate() which does prefill via _prefill()
# Let's trace what _prefill() does
dummy_ids = torch.ones((1, 257), dtype=torch.int, device=sem.device)
with torch.no_grad():
    # This is what HF's generate does internally for prefill
    hf_out = sem(input_ids=None, inputs_embeds=emb, use_cache=True)
hf_logits = hf_out.logits[0, -1].clone()

# Compare logits
diff = (manual_logits - hf_logits).abs()
print(f"Manual vs HF forward logits:")
print(f"  Max diff: {diff.max():.10f}")
print(f"  Mean diff: {diff.mean():.10f}")
print(f"  Identical: {torch.allclose(manual_logits, hf_logits)}")

# Now check: with same RNG, do both produce same first token?
# Apply the exact HF processing chain
def hf_process_logits(logits_1d):
    """Apply HF's exact logits processor + warper chain."""
    # 1. SuppressTokensLogitsProcessor: bark suppresses semantic_vocab_size..semantic_pad_token
    # and semantic_pad_token+1..output_vocab_size
    # For bark-small: semantic_vocab_size=10000, semantic_pad_token=10000
    # So range(10000, 10000) is empty, and range(10001, output_vocab) is suppressed
    scores = logits_1d.clone()
    scores[sp + 1:] = float("-inf")
    # 2. BarkEosPrioritizerLogitsProcessor: min_eos_p=None, no-op
    # 3. renormalize_logits=True: log_softmax
    scores = torch.nn.functional.log_softmax(scores.float(), dim=-1)
    # 4. TemperatureLogitsWarper: divide by 0.7
    scores = scores / 0.7
    # 5. TopKLogitsWarper: keep top 50
    top_k = min(50, scores.size(-1))
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, float("-inf"))
    return scores

manual_scores = hf_process_logits(manual_logits)
manual_probs = torch.softmax(manual_scores, dim=-1)

# Compare with simple processing (what bark_trt_hf_sampling.py does)
def simple_process_logits(logits_1d):
    scores = logits_1d.clone()
    scores[sp + 1:] = float("-inf")
    scores = scores / 0.7
    tv, ti = torch.topk(scores[:sp + 1], min(50, sp + 1))
    filt = torch.full_like(scores, float("-inf"))
    filt.scatter_(0, ti, tv)
    return filt

simple_scores = simple_process_logits(manual_logits)
simple_probs = torch.softmax(simple_scores, dim=-1)

# Compare probability distributions
tvd = 0.5 * (manual_probs - simple_probs).abs().sum()
print(f"\nHF processing vs simple processing:")
print(f"  TVD: {tvd:.10f}")
print(f"  Top-5 probs (HF):     {manual_probs.topk(5).values.tolist()}")
print(f"  Top-5 probs (simple): {simple_probs.topk(5).values.tolist()}")
print(f"  Top-5 idx (HF):       {manual_probs.topk(5).indices.tolist()}")
print(f"  Top-5 idx (simple):   {simple_probs.topk(5).indices.tolist()}")

# Now sample with same seed
for seed in [42]:
    torch.manual_seed(seed)
    tok_hf = int(torch.multinomial(manual_probs.unsqueeze(0), 1).item())  # 2D

    torch.manual_seed(seed)
    tok_simple = int(torch.multinomial(simple_probs, 1).item())  # 1D

    torch.manual_seed(seed)
    tok_hf_1d = int(torch.multinomial(manual_probs, 1).item())  # 1D with HF logits

    print(f"\nSeed={seed}:")
    print(f"  HF logits, 2D multinomial: {tok_hf}")
    print(f"  HF logits, 1D multinomial: {tok_hf_1d}")
    print(f"  Simple logits, 1D:         {tok_simple}")

# Test: does 1D vs 2D multinomial consume different RNG?
print("\n1D vs 2D multinomial RNG consumption:")
for dims in ["1D", "2D"]:
    torch.manual_seed(42)
    p = manual_probs if dims == "1D" else manual_probs.unsqueeze(0)
    tok = torch.multinomial(p, 1)
    # Check next value
    next_val = torch.empty(1).uniform_().item()
    print(f"  {dims}: token={tok.item()}, next_uniform={next_val:.10f}")
