#!/usr/bin/env python3
"""Bisect which stage causes divergence by patching one stage at a time."""
import sys, struct, torch, numpy as np
torch.set_grad_enabled(False)

def write_wav(path, a, sr=24000):
    n=len(a); ds=n*4
    with open(path,"wb") as f:
        f.write(b"RIFF"+struct.pack("<I",36+ds)+b"WAVEfmt "+struct.pack("<IHHIIHH",16,3,1,sr,sr*4,4,32)+b"data"+struct.pack("<I",ds)+a.astype(np.float32).tobytes())

device = "cuda"
print("Loading...", file=sys.stderr)
from transformers import BarkModel, AutoProcessor, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model.eval()

from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
from trtf_build.debug_runner import TrtRunner
import glob

model_dir = glob.glob("/root/.cache/huggingface/hub/models--suno--bark-small/snapshots/*")[0]
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin("bark")
weights = plugin.load_weights(model_dir, config)

prompt = "Hello, my dog is cute"
inputs = processor(prompt, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

sem_embed_np = model.semantic.input_embeds_layer.weight.detach().cpu().numpy()
coarse_embed_np = model.coarse_acoustics.input_embeds_layer.weight.detach().cpu().numpy()

# Build engines
print("Building engines...", file=sys.stderr)
sem_plan = plugin.build_engine(config, weights, 512)
extra = plugin.build_extra_engines(config, weights, 1024)
coarse_plan = extra["coarse_engine"]

# ============================================================
# Reference: pure HF
# ============================================================
print("\n=== Test 0: Pure HF ===", file=sys.stderr)
torch.manual_seed(42)
ref_audio = model.generate(**inputs)
ref_np = ref_audio.cpu().numpy().squeeze()
write_wav("/tmp/bisect_0_hf.wav", ref_np)
print(f"  HF: {len(ref_np)/24000:.2f}s, RMS={np.sqrt(np.mean(ref_np**2)):.4f}")

# ============================================================
# Test 1: TRT semantic ONLY (coarse/fine/codec = HF)
# This should be bit-identical (we proved it)
# ============================================================
print("\n=== Test 1: TRT semantic only ===", file=sys.stderr)
_sem_runner = TrtRunner(sem_plan, max_cache_length=512, num_layers=12)

_orig_sem_fwd = model.semantic.forward
def _trt_sem(input_ids=None, inputs_embeds=None, **kw):
    if inputs_embeds is not None:
        e = inputs_embeds[0].cpu().numpy()
        for p in range(e.shape[0]-1):
            _sem_runner.step(0, input_embed=e[p:p+1].astype(np.float32), use_input_embed=1.0)
        r = _sem_runner.step(0, input_embed=e[-1:].astype(np.float32), use_input_embed=1.0)
    elif input_ids is not None:
        t = input_ids[0,-1].item()
        r = _sem_runner.step(0, input_embed=sem_embed_np[t:t+1].astype(np.float32), use_input_embed=1.0)
    else:
        raise ValueError()
    return CausalLMOutputWithPast(logits=torch.tensor(r["logits"],device=device).unsqueeze(0))

model.semantic.forward = _trt_sem
torch.manual_seed(42)
t1_audio = model.generate(**inputs)
t1_np = t1_audio.cpu().numpy().squeeze()
write_wav("/tmp/bisect_1_sem.wav", t1_np)
T = min(len(t1_np), len(ref_np))
corr1 = np.corrcoef(t1_np[:T], ref_np[:T])[0,1] if T > 0 else 0
print(f"  TRT sem: {len(t1_np)/24000:.2f}s, corr={corr1:.6f}")
model.semantic.forward = _orig_sem_fwd  # restore

# ============================================================
# Test 2: TRT coarse ONLY (semantic/fine/codec = HF)
# This isolates whether coarse TRT diverges
# ============================================================
print("\n=== Test 2: TRT coarse only ===", file=sys.stderr)
_orig_coarse_fwd = model.coarse_acoustics.forward
_cr = {"runner": TrtRunner(coarse_plan, max_cache_length=1024, num_layers=12), "cache": None}

def _trt_coarse(input_ids=None, **kw):
    past = kw.get("past_key_values")
    use_cache = kw.get("use_cache", True)
    r = _cr["runner"]
    if input_ids is not None and input_ids.shape[1] > 1:
        r = TrtRunner(coarse_plan, max_cache_length=1024, num_layers=12)
        _cr["runner"] = r
        toks = input_ids[0].cpu().tolist()
        for t in toks[:-1]:
            r.step(0, input_embed=coarse_embed_np[t:t+1].astype(np.float32), use_input_embed=1.0)
        res = r.step(0, input_embed=coarse_embed_np[toks[-1]:toks[-1]+1].astype(np.float32), use_input_embed=1.0)
        cache = DynamicCache()
        for l in range(12):
            cache.update(torch.zeros(1,12,len(toks),64,device=device),
                        torch.zeros(1,12,len(toks),64,device=device), l)
        _cr["cache"] = cache
    elif input_ids is not None:
        t = input_ids[0,-1].item()
        res = r.step(0, input_embed=coarse_embed_np[t:t+1].astype(np.float32), use_input_embed=1.0)
        if _cr["cache"] is not None:
            for l in range(12):
                _cr["cache"].update(torch.zeros(1,12,1,64,device=device),
                                   torch.zeros(1,12,1,64,device=device), l)
    else:
        raise ValueError()
    return CausalLMOutputWithPast(
        logits=torch.tensor(res["logits"],device=device).unsqueeze(0),
        past_key_values=_cr["cache"] if use_cache else None)

model.coarse_acoustics.forward = _trt_coarse
torch.manual_seed(42)
t2_audio = model.generate(**inputs)
t2_np = t2_audio.cpu().numpy().squeeze()
write_wav("/tmp/bisect_2_coarse.wav", t2_np)
T = min(len(t2_np), len(ref_np))
corr2 = np.corrcoef(t2_np[:T], ref_np[:T])[0,1] if T > 0 else 0
print(f"  TRT coarse: {len(t2_np)/24000:.2f}s, corr={corr2:.6f}")
model.coarse_acoustics.forward = _orig_coarse_fwd  # restore

# ============================================================
# Test 3: TRT semantic + TRT coarse (fine/codec = HF)
# ============================================================
print("\n=== Test 3: TRT semantic + TRT coarse ===", file=sys.stderr)
_sem_runner2 = TrtRunner(sem_plan, max_cache_length=512, num_layers=12)
def _trt_sem2(input_ids=None, inputs_embeds=None, **kw):
    if inputs_embeds is not None:
        e = inputs_embeds[0].cpu().numpy()
        for p in range(e.shape[0]-1):
            _sem_runner2.step(0, input_embed=e[p:p+1].astype(np.float32), use_input_embed=1.0)
        r = _sem_runner2.step(0, input_embed=e[-1:].astype(np.float32), use_input_embed=1.0)
    elif input_ids is not None:
        t = input_ids[0,-1].item()
        r = _sem_runner2.step(0, input_embed=sem_embed_np[t:t+1].astype(np.float32), use_input_embed=1.0)
    else:
        raise ValueError()
    return CausalLMOutputWithPast(logits=torch.tensor(r["logits"],device=device).unsqueeze(0))

_cr2 = {"runner": TrtRunner(coarse_plan, max_cache_length=1024, num_layers=12), "cache": None}
def _trt_coarse2(input_ids=None, **kw):
    past = kw.get("past_key_values")
    use_cache = kw.get("use_cache", True)
    r = _cr2["runner"]
    if input_ids is not None and input_ids.shape[1] > 1:
        r = TrtRunner(coarse_plan, max_cache_length=1024, num_layers=12)
        _cr2["runner"] = r
        toks = input_ids[0].cpu().tolist()
        for t in toks[:-1]:
            r.step(0, input_embed=coarse_embed_np[t:t+1].astype(np.float32), use_input_embed=1.0)
        res = r.step(0, input_embed=coarse_embed_np[toks[-1]:toks[-1]+1].astype(np.float32), use_input_embed=1.0)
        cache = DynamicCache()
        for l in range(12):
            cache.update(torch.zeros(1,12,len(toks),64,device=device),
                        torch.zeros(1,12,len(toks),64,device=device), l)
        _cr2["cache"] = cache
    elif input_ids is not None:
        t = input_ids[0,-1].item()
        res = r.step(0, input_embed=coarse_embed_np[t:t+1].astype(np.float32), use_input_embed=1.0)
        if _cr2["cache"] is not None:
            for l in range(12):
                _cr2["cache"].update(torch.zeros(1,12,1,64,device=device),
                                    torch.zeros(1,12,1,64,device=device), l)
    else:
        raise ValueError()
    return CausalLMOutputWithPast(
        logits=torch.tensor(res["logits"],device=device).unsqueeze(0),
        past_key_values=_cr2["cache"] if use_cache else None)

model.semantic.forward = _trt_sem2
model.coarse_acoustics.forward = _trt_coarse2
torch.manual_seed(42)
t3_audio = model.generate(**inputs)
t3_np = t3_audio.cpu().numpy().squeeze()
write_wav("/tmp/bisect_3_both.wav", t3_np)
T = min(len(t3_np), len(ref_np))
corr3 = np.corrcoef(t3_np[:T], ref_np[:T])[0,1] if T > 0 else 0
print(f"  TRT sem+coarse: {len(t3_np)/24000:.2f}s, corr={corr3:.6f}")

# ============================================================
# Summary
# ============================================================
print(f"\n=== SUMMARY ===")
print(f"Test 0 (HF baseline):       {len(ref_np)/24000:.2f}s")
print(f"Test 1 (TRT sem only):      corr={corr1:.6f} {'PASS' if corr1 > 0.99 else 'FAIL'}")
print(f"Test 2 (TRT coarse only):   corr={corr2:.6f} {'PASS' if corr2 > 0.99 else 'FAIL'}")
print(f"Test 3 (TRT sem+coarse):    corr={corr3:.6f} {'PASS' if corr3 > 0.99 else 'FAIL'}")
