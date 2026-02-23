#!/usr/bin/env python3
"""Debug: what consumes torch global RNG between seed and first multinomial?"""
import torch
import numpy as np

# Test 1: are global seed and Generator.seed equivalent?
torch.manual_seed(42)
gen = torch.Generator().manual_seed(42)

# Get first uniform from each
v_global = torch.empty(1).uniform_().item()
v_gen = torch.empty(1).uniform_(generator=gen).item()
print(f"First uniform - global: {v_global:.10f}")
print(f"First uniform - gen:    {v_gen:.10f}")
print(f"Match: {abs(v_global - v_gen) < 1e-10}")

# Test 2: does model loading consume RNG?
torch.manual_seed(42)
v_before_load = torch.empty(1).uniform_().item()

torch.manual_seed(42)
from transformers import BarkModel
model = BarkModel.from_pretrained("suno/bark-small")
v_after_load = torch.empty(1).uniform_().item()
print(f"\nFirst uniform after model load: {v_after_load:.10f}")
print(f"Model load consumed RNG: {abs(v_before_load - v_after_load) > 1e-10}")

# Test 3: trace what consumes RNG inside generate()
# Override multinomial to see what values it gets
orig_multinomial = torch.multinomial
call_count = [0]

def debug_multinomial(probs, num_samples, **kwargs):
    call_count[0] += 1
    if call_count[0] <= 3:
        # Check RNG state
        r = torch.empty(1).uniform_().item()
        print(f"  multinomial call #{call_count[0]}: next_uniform_would_be={r:.10f}")
    return orig_multinomial(probs, num_samples, **kwargs)

torch.multinomial = debug_multinomial

from transformers import AutoProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)

# What's the RNG state right before generate?
torch.manual_seed(42)
print(f"\nAfter manual_seed(42):")
test_v = torch.empty(1).uniform_().item()
print(f"  First uniform: {test_v:.10f}")

torch.manual_seed(42)
print(f"\nCalling semantic.generate()...")
call_count[0] = 0
with torch.no_grad():
    out = model.semantic.generate(inputs["input_ids"], semantic_generation_config=sem_cfg)
gen_toks = out[0, 257:].numpy()
gen_toks = gen_toks[gen_toks < 10000]
print(f"  Total multinomial calls: {call_count[0]}")
print(f"  Generated {len(gen_toks)} tokens")

torch.multinomial = orig_multinomial

# Test 4: check if there's RNG consumption between seed and first multinomial
# by comparing the state just before generate vs after generate setup
print(f"\nTest: RNG consumption during generate setup")
torch.manual_seed(42)
state_before = torch.random.get_rng_state().clone()
# Simulate what generate does before first multinomial:
# 1. input_ids + offset
input_ids = inputs["input_ids"].clone() + 10048
# 2. Create semantic_history
semantic_history = torch.full((256,), 10000, dtype=torch.int)
# 3. Create infer array
infer_array = torch.tensor([[129599]], dtype=torch.int)
# 4. Embed
with torch.no_grad():
    embs = torch.cat([
        model.semantic.input_embeds_layer(input_ids[:, :256])
        + model.semantic.input_embeds_layer(semantic_history.unsqueeze(0)[:, :257]),
        model.semantic.input_embeds_layer(infer_array)
    ], dim=1)
    # 5. Forward pass
    out = model.semantic(inputs_embeds=embs, use_cache=True)

state_after = torch.random.get_rng_state().clone()
consumed = not torch.equal(state_before, state_after)
print(f"  RNG consumed during prefill: {consumed}")

if consumed:
    # How many values were consumed?
    torch.manual_seed(42)
    vals_before = [torch.empty(1).uniform_().item() for _ in range(10)]
    torch.random.set_rng_state(state_after)
    vals_after = [torch.empty(1).uniform_().item() for _ in range(10)]
    # Find which value from "before" matches first value of "after"
    for i, v in enumerate(vals_before):
        if abs(v - vals_after[0]) < 1e-10:
            print(f"  RNG consumed {i} values during prefill")
            break
    else:
        print(f"  Before[0:10]: {vals_before}")
        print(f"  After[0:10]:  {vals_after}")
