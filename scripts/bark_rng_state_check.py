#!/usr/bin/env python3
"""Save RNG state at exact moment of first multinomial in HF generate."""
import torch
import sys
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)

# Capture RNG state at first multinomial
_orig = torch.multinomial
captured = {}

def hook_multi(input, num_samples, **kwargs):
    if "state" not in captured:
        captured["state"] = torch.random.get_rng_state().clone()
        captured["probs"] = input.clone()
    return _orig(input, num_samples, **kwargs)

torch.multinomial = hook_multi

torch.manual_seed(42)
initial_state = torch.random.get_rng_state().clone()

with torch.no_grad():
    out = model.semantic.generate(inputs["input_ids"],
                                   semantic_generation_config=sem_cfg)
gen_toks = out[0, 257:].numpy()
gen_toks = gen_toks[gen_toks < 10000]
print(f"HF generate: {len(gen_toks)} tokens, first 5: {gen_toks[:5].tolist()}")

torch.multinomial = _orig

# Compare states
multi_state = captured["state"]
match = torch.equal(initial_state, multi_state)
print(f"\nRNG state at seed(42) == state at first multinomial: {match}")

if not match:
    # Find how many values were consumed
    # Try drawing values from initial state and seeing when we match multi_state
    torch.random.set_rng_state(initial_state)
    for i in range(100):
        torch.empty(1).uniform_()
        if torch.equal(torch.random.get_rng_state(), multi_state):
            print(f"  RNG advanced by {i+1} uniform draws")
            break
    else:
        # Try with multinomial draws
        torch.random.set_rng_state(initial_state)
        dummy_p = torch.ones(2) / 2
        for i in range(100):
            torch.multinomial(dummy_p, 1)
            if torch.equal(torch.random.get_rng_state(), multi_state):
                print(f"  RNG advanced by {i+1} multinomial draws")
                break
        else:
            print("  Could not determine RNG advance count")
            # Check the actual bytes difference
            diff_count = (initial_state != multi_state).sum().item()
            print(f"  State bytes that differ: {diff_count}/{len(initial_state)}")

# Now sample from captured probs with captured state
torch.random.set_rng_state(captured["state"])
tok = _orig(captured["probs"], 1).item()
print(f"\nSampling from captured probs with captured state: token {tok}")
print(f"HF's actual first token: {gen_toks[0]}")
print(f"Match: {tok == gen_toks[0]}")

# And with initial state
torch.random.set_rng_state(initial_state)
tok2 = _orig(captured["probs"], 1).item()
print(f"Sampling with initial state: token {tok2}")
