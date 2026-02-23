#!/usr/bin/env python3
"""Capture exact multinomial inputs/outputs inside HF generate."""
import torch
import sys
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)

_orig = torch.multinomial
call_data = []

def hook(input_tensor, num_samples, **kwargs):
    state_before = torch.random.get_rng_state().clone()
    result = _orig(input_tensor, num_samples, **kwargs)
    state_after = torch.random.get_rng_state().clone()

    if len(call_data) < 5:
        call_data.append({
            "probs_shape": tuple(input_tensor.shape),
            "probs_sum": float(input_tensor.sum()),
            "probs_max_idx": int(input_tensor.argmax()),
            "probs_nonzero": int((input_tensor > 0).sum()),
            "result": result.clone(),
            "kwargs": dict(kwargs),
            "state_changed": not torch.equal(state_before, state_after),
        })
    return result

torch.multinomial = hook

torch.manual_seed(42)
with torch.no_grad():
    out = model.semantic.generate(inputs["input_ids"],
                                   semantic_generation_config=sem_cfg)
gen_toks = out[0, 257:].numpy()
gen_toks = gen_toks[gen_toks < 10000]

torch.multinomial = _orig

print(f"HF generate: {len(gen_toks)} tokens, first 5: {gen_toks[:5].tolist()}")
print(f"\nTotal multinomial calls captured: {len(call_data)}")

for i, d in enumerate(call_data):
    print(f"\nCall #{i+1}:")
    print(f"  Shape: {d['probs_shape']}")
    print(f"  Sum: {d['probs_sum']:.6f}")
    print(f"  Non-zero: {d['probs_nonzero']}")
    print(f"  Argmax idx: {d['probs_max_idx']}")
    print(f"  Result: {d['result'].tolist()}")
    print(f"  Kwargs: {d['kwargs']}")
    print(f"  State changed: {d['state_changed']}")
