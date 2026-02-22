#!/usr/bin/env python3
"""Debug coarse: capture HF's generated tokens, replay exact same input to TRT, compare."""
import sys, struct, torch, numpy as np
torch.set_grad_enabled(False)
device = "cuda"

from transformers import BarkModel, AutoProcessor, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bark.generation_configuration_bark import *

model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model.eval()

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_cfg = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)

inputs = processor("Hello, my dog is cute", voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

# Step 1: Get HF semantic output (deterministic)
torch.manual_seed(42)
with torch.no_grad():
    hf_sem = model.semantic.generate(inputs["input_ids"],
        history_prompt=inputs.get("history_prompt"),
        semantic_generation_config=sem_cfg)
print(f"Semantic: {hf_sem.shape}", file=sys.stderr)

# Step 2: Hook into HF coarse forward to capture input_ids at each call
_orig_fwd = model.coarse_acoustics.forward
_hf_calls = []

def _spy(input_ids=None, **kw):
    toks = input_ids[0].cpu().tolist() if input_ids is not None else []
    out = _orig_fwd(input_ids=input_ids, **kw)
    logits = out.logits[0, -1, :].cpu().numpy()
    _hf_calls.append({"input_ids": toks, "logits": logits, "argmax": int(np.argmax(logits))})
    return out

model.coarse_acoustics.forward = _spy

torch.manual_seed(42)
with torch.no_grad():
    hf_coarse = model.coarse_acoustics.generate(
        hf_sem, history_prompt=inputs.get("history_prompt"),
        semantic_generation_config=sem_cfg, coarse_generation_config=coarse_cfg)

print(f"HF coarse: {len(_hf_calls)} calls, output shape {hf_coarse.shape}", file=sys.stderr)
model.coarse_acoustics.forward = _orig_fwd  # restore

# Step 3: Replay exact same input_ids to TRT and compare logits
from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
from trtf_build.debug_runner import TrtRunner
import glob

model_dir = glob.glob("/root/.cache/huggingface/hub/models--suno--bark-small/snapshots/*")[0]
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin("bark")
weights = plugin.load_weights(model_dir, config)
extra = plugin.build_extra_engines(config, weights, 1024)
coarse_plan = extra["coarse_engine"]
ce = model.coarse_acoustics.input_embeds_layer.weight.detach().cpu().numpy()

print(f"\n=== Replaying {len(_hf_calls)} HF calls through TRT ===", file=sys.stderr)

runner = None
mismatches = 0
max_diffs = []

for i, call in enumerate(_hf_calls):
    toks = call["input_ids"]
    hf_logits = call["logits"]
    hf_argmax = call["argmax"]

    if len(toks) > 1:
        # Prefill: new runner
        runner = TrtRunner(coarse_plan, max_cache_length=1024, num_layers=12)
        for t in toks[:-1]:
            runner.step(0, input_embed=ce[t:t+1].astype(np.float32), use_input_embed=1.0)
        r = runner.step(0, input_embed=ce[toks[-1]:toks[-1]+1].astype(np.float32), use_input_embed=1.0)
    else:
        # Decode
        r = runner.step(0, input_embed=ce[toks[-1]:toks[-1]+1].astype(np.float32), use_input_embed=1.0)

    trt_logits = r["logits"].flatten()
    trt_argmax = int(np.argmax(trt_logits))

    m = min(len(trt_logits), len(hf_logits))
    diff = np.max(np.abs(trt_logits[:m] - hf_logits[:m]))
    max_diffs.append(diff)

    if trt_argmax != hf_argmax:
        mismatches += 1
        if mismatches <= 10:
            print(f"  MISMATCH at call {i}: HF={hf_argmax}, TRT={trt_argmax}, "
                  f"max_diff={diff:.4f}, seq_len={len(toks)}")

    if i < 3 or (i < len(_hf_calls) - 1 and i % 20 == 0):
        print(f"  Call {i}: max_diff={diff:.4f}, argmax_match={trt_argmax==hf_argmax}, "
              f"seq_len={len(toks)}")

print(f"\n=== RESULTS ===")
print(f"Total calls: {len(_hf_calls)}")
print(f"Argmax mismatches: {mismatches}/{len(_hf_calls)} ({mismatches*100/max(len(_hf_calls),1):.1f}%)")
print(f"Max logit diff: max={max(max_diffs):.4f}, mean={np.mean(max_diffs):.4f}")
