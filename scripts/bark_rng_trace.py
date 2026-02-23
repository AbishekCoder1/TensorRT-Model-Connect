#!/usr/bin/env python3
"""Trace exactly how many RNG values HF's generate() consumes before first multinomial."""
import torch
import sys
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

model = BarkModel.from_pretrained("suno/bark-small").eval()
sem = model.semantic
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)

# Hook into torch's RNG to count consumptions
class RNGTracer:
    def __init__(self):
        self.count = 0
        self.log = []
        self.enabled = True

    def trace_uniform(self, original_fn):
        tracer = self
        def wrapped(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            if tracer.enabled:
                tracer.count += result.numel()
                import traceback
                if tracer.count <= 20:  # Log first 20
                    # Get caller info
                    stack = traceback.extract_stack()
                    caller = None
                    for frame in reversed(stack):
                        if 'bark_rng_trace' not in frame.filename and 'torch' not in frame.filename:
                            caller = f"{frame.filename}:{frame.lineno} {frame.name}"
                            break
                    if caller is None:
                        caller = f"{stack[-3].filename}:{stack[-3].lineno}"
                    tracer.log.append((tracer.count, caller))
            return result
        return wrapped

tracer = RNGTracer()

# Monkey-patch key RNG-consuming functions
import torch
_orig_uniform = torch.Tensor.uniform_
_orig_normal = torch.Tensor.normal_
_orig_random = torch.Tensor.random_
_orig_bernoulli = torch.Tensor.bernoulli_
_orig_multinomial = torch.multinomial

torch.Tensor.uniform_ = tracer.trace_uniform(_orig_uniform)
torch.Tensor.normal_ = tracer.trace_uniform(_orig_normal)
torch.Tensor.random_ = tracer.trace_uniform(_orig_random)
torch.Tensor.bernoulli_ = tracer.trace_uniform(_orig_bernoulli)

multi_calls = [0]
def traced_multinomial(input, num_samples, **kwargs):
    multi_calls[0] += 1
    if multi_calls[0] == 1:
        print(f"\n=== FIRST multinomial call ===")
        print(f"  RNG values consumed before this: {tracer.count}")
        for c, caller in tracer.log:
            print(f"  #{c}: {caller}")
    tracer.enabled = False  # Stop tracing after first multinomial
    return _orig_multinomial(input, num_samples, **kwargs)

torch.multinomial = traced_multinomial

# Now run HF generate
torch.manual_seed(42)
print(f"After manual_seed(42), running generate...")
with torch.no_grad():
    out = sem.generate(inputs["input_ids"], semantic_generation_config=sem_cfg)
gen_toks = out[0, 257:].numpy()
gen_toks = gen_toks[gen_toks < 10000]
print(f"\nGenerated {len(gen_toks)} tokens")
print(f"First 5: {gen_toks[:5].tolist()}")

# Restore
torch.Tensor.uniform_ = _orig_uniform
torch.Tensor.normal_ = _orig_normal
torch.Tensor.random_ = _orig_random
torch.Tensor.bernoulli_ = _orig_bernoulli
torch.multinomial = _orig_multinomial
