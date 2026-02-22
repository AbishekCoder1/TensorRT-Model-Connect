#!/usr/bin/env python3
"""Compare HF vs TRT coarse tokens."""
import torch, numpy as np
torch.set_grad_enabled(False)
from transformers import BarkModel, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bark.generation_configuration_bark import *
from trtf_build.debug_runner import TrtRunner
from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
import glob

device = "cuda"
model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model.eval()
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_cfg = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
inputs = processor("Hello, my dog is cute", voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

# HF coarse tokens
torch.manual_seed(42)
hf_sem = model.semantic.generate(inputs["input_ids"], history_prompt=inputs.get("history_prompt"),
    semantic_generation_config=sem_cfg)
torch.manual_seed(42)
hf_coarse = model.coarse_acoustics.generate(hf_sem, history_prompt=inputs.get("history_prompt"),
    semantic_generation_config=sem_cfg, coarse_generation_config=coarse_cfg)
hf_toks = hf_coarse[0].cpu().tolist()

# TRT coarse
model_dir = glob.glob("/root/.cache/huggingface/hub/models--suno--bark-small/snapshots/*")[0]
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin("bark")
weights = plugin.load_weights(model_dir, config)
extra = plugin.build_extra_engines(config, weights, 1024)
coarse_plan = extra["coarse_engine"]
ce = model.coarse_acoustics.input_embeds_layer.weight.detach().cpu().numpy()

class _TC:
    def __init__(self, s=0): self._s = s
    def get_seq_length(self, l=0): return self._s
    def get_max_cache_shape(self): return None
    def update(self, k, v, l, **kw):
        if l == 0: self._s += k.shape[2]
        return (k, v)
    def __getitem__(self, i): return (torch.empty(0,device=device), torch.empty(0,device=device))
    def __len__(self): return 12
    def __iter__(self): return iter([(torch.empty(0,device=device), torch.empty(0,device=device))]*12)

_r = [None]; _c = [None]
def _fwd(input_ids=None, **kw):
    past = kw.get("past_key_values")
    use_cache = kw.get("use_cache", True)
    if past is None or (input_ids is not None and input_ids.shape[1] > 1):
        _r[0] = TrtRunner(coarse_plan, max_cache_length=1024, num_layers=12)
        toks = input_ids[0].cpu().tolist()
        for t in toks[:-1]:
            _r[0].step(0, input_embed=ce[t:t+1].astype(np.float32), use_input_embed=1.0)
        res = _r[0].step(0, input_embed=ce[toks[-1]:toks[-1]+1].astype(np.float32), use_input_embed=1.0)
        _c[0] = _TC(len(toks))
    else:
        t = input_ids[0,-1].item()
        res = _r[0].step(0, input_embed=ce[t:t+1].astype(np.float32), use_input_embed=1.0)
        _c[0]._s += 1
    return CausalLMOutputWithPast(logits=torch.tensor(res["logits"],device=device).unsqueeze(0),
                                   past_key_values=_c[0] if use_cache else None)

model.coarse_acoustics.forward = _fwd
torch.manual_seed(42)
trt_coarse = model.coarse_acoustics.generate(hf_sem, history_prompt=inputs.get("history_prompt"),
    semantic_generation_config=sem_cfg, coarse_generation_config=coarse_cfg)
trt_toks = trt_coarse[0].cpu().tolist()

match = sum(1 for a,b in zip(hf_toks, trt_toks) if a==b)
total = min(len(hf_toks), len(trt_toks))
print(f"HF tokens: {len(hf_toks)}, TRT tokens: {len(trt_toks)}")
print(f"Token match: {match}/{total} ({match*100/max(total,1):.1f}%)")

for i,(a,b) in enumerate(zip(hf_toks, trt_toks)):
    if a != b:
        print(f"First divergence at position {i}: HF={a}, TRT={b}")
        break
else:
    print("PERFECT: All tokens match!")
