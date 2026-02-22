#!/usr/bin/env python3
"""Check coarse TRT vs HF logits step by step."""
import torch, numpy as np
torch.set_grad_enabled(False)
from transformers import BarkModel
from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
from trtf_build.debug_runner import TrtRunner
import glob

device = "cuda"
model = BarkModel.from_pretrained("suno/bark-small").to(device)
model.eval()

model_dir = glob.glob("/root/.cache/huggingface/hub/models--suno--bark-small/snapshots/*")[0]
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin("bark")
weights = plugin.load_weights(model_dir, config)
extra = plugin.build_extra_engines(config, weights, 256)
coarse_plan = extra["coarse_engine"]
runner = TrtRunner(coarse_plan, max_cache_length=256, num_layers=12)
ce = model.coarse_acoustics.input_embeds_layer.weight.detach().cpu().numpy()

toks = [12048, 12048, 12048, 12048, 12050]

# TRT prefill
for t in toks[:-1]:
    runner.step(0, input_embed=ce[t:t+1].astype(np.float32), use_input_embed=1.0)
r = runner.step(0, input_embed=ce[toks[-1]:toks[-1]+1].astype(np.float32), use_input_embed=1.0)
tl = r["logits"].flatten()

# HF prefill
with torch.no_grad():
    ho = model.coarse_acoustics(input_ids=torch.tensor([toks], device=device))
hl = ho.logits[0, -1, :].cpu().numpy()

m = min(len(tl), len(hl))
print(f"Prefill: max_diff={np.max(np.abs(tl[:m]-hl[:m])):.6f}, match={np.argmax(tl)==np.argmax(hl)}")

# 10 decode steps feeding argmax
prev_hf = ho
for step in range(10):
    nt = int(np.argmax(hl))
    r2 = runner.step(0, input_embed=ce[nt:nt+1].astype(np.float32), use_input_embed=1.0)
    with torch.no_grad():
        ho2 = model.coarse_acoustics(input_ids=torch.tensor([[nt]], device=device),
                                      past_key_values=prev_hf.past_key_values)
    hl2 = ho2.logits[0, -1, :].cpu().numpy()
    tl2 = r2["logits"].flatten()
    d = np.max(np.abs(tl2[:m] - hl2[:m]))
    print(f"Step {step}: max_diff={d:.6f}, match={np.argmax(tl2)==np.argmax(hl2)}")
    hl = hl2
    prev_hf = ho2
