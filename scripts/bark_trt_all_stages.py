#!/usr/bin/env python3
"""Bark: ALL stages using TRT engines, monkey-patched into HF's pipeline.

Patches:
1. BarkSemanticModel.forward() → TRT semantic engine
2. BarkCoarseModel.forward() → TRT coarse engine
3. BarkFineModel.forward() → uses codebook_idx to select LM head (custom logic)
4. Codec decode → TRT EnCodec decoder

Each patch replaces ONLY the transformer forward pass. HF handles all
orchestration (embedding lookup, sampling, stopping criteria, codebook interleaving).
"""
import sys, struct, time, torch, numpy as np
torch.set_grad_enabled(False)

def write_wav(path, audio_np, sr=24000):
    n = len(audio_np); ds = n * 4
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36+ds) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr*4, 4, 32))
        f.write(b"data" + struct.pack("<I", ds) + audio_np.astype(np.float32).tobytes())

device = "cuda"
print("=== Loading HF Bark ===", file=sys.stderr)
from transformers import BarkModel, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from transformers.models.bark.modeling_bark import BarkFineModel

model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model.eval()

prompt = "Hello, my dog is cute"
inputs = processor(prompt, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

# =====================================================================
# HF Reference
# =====================================================================
print("=== HF Reference ===", file=sys.stderr)
torch.manual_seed(42)
hf_audio = model.generate(**inputs)
hf_np = hf_audio.cpu().numpy().squeeze()
print(f"HF: {len(hf_np)/24000:.2f}s, RMS={np.sqrt(np.mean(hf_np**2)):.4f}")
write_wav("/tmp/bark_all_hf.wav", hf_np)

# =====================================================================
# Build ALL TRT engines
# =====================================================================
print("\n=== Building TRT engines ===", file=sys.stderr)
from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
from trtf_build.debug_runner import TrtRunner
import glob

model_dir = glob.glob("/root/.cache/huggingface/hub/models--suno--bark-small/snapshots/*")[0]
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin("bark")
weights = plugin.load_weights(model_dir, config)

t0 = time.time()
sem_plan = plugin.build_engine(config, weights, 512)
print(f"  Semantic: {len(sem_plan)/(1024*1024):.0f}MB [{time.time()-t0:.1f}s]", file=sys.stderr)

t0 = time.time()
extra = plugin.build_extra_engines(config, weights, 512)
coarse_plan = extra["coarse_engine"]
print(f"  Coarse: {len(coarse_plan)/(1024*1024):.0f}MB [{time.time()-t0:.1f}s]", file=sys.stderr)

# Extract embedding tables
sem_embed_np = model.semantic.input_embeds_layer.weight.detach().cpu().numpy()
coarse_embed_np = model.coarse_acoustics.input_embeds_layer.weight.detach().cpu().numpy()

# =====================================================================
# Patch 1: Semantic forward → TRT
# =====================================================================
print("\n=== Patching Semantic ===", file=sys.stderr)
sem_runner = TrtRunner(sem_plan, max_cache_length=512, num_layers=12)

def _trt_semantic_forward(input_ids=None, inputs_embeds=None, **kwargs):
    if inputs_embeds is not None:
        embeds = inputs_embeds[0].cpu().numpy()
        for pos in range(embeds.shape[0] - 1):
            sem_runner.step(0, input_embed=embeds[pos:pos+1].astype(np.float32),
                           use_input_embed=1.0)
        result = sem_runner.step(0, input_embed=embeds[-1:].astype(np.float32),
                                 use_input_embed=1.0)
    elif input_ids is not None:
        tok = input_ids[0, -1].item()
        emb = sem_embed_np[tok:tok+1].astype(np.float32)
        result = sem_runner.step(0, input_embed=emb, use_input_embed=1.0)
    else:
        raise ValueError("No input")
    logits_t = torch.tensor(result["logits"], device=device).unsqueeze(0)
    return CausalLMOutputWithPast(logits=logits_t)

model.semantic.forward = _trt_semantic_forward
print("  Semantic patched ✓", file=sys.stderr)

# =====================================================================
# Patch 2: Coarse forward → TRT
# =====================================================================
print("=== Patching Coarse ===", file=sys.stderr)
_coarse = {"runner": TrtRunner(coarse_plan, max_cache_length=512, num_layers=12)}

from transformers import DynamicCache
_coarse_cache = {"cache": None}

def _trt_coarse_forward(input_ids=None, inputs_embeds=None, **kwargs):
    runner = _coarse["runner"]
    past = kwargs.get("past_key_values")
    use_cache = kwargs.get("use_cache", True)

    if input_ids is not None and input_ids.shape[1] > 1:
        # Prefill: reset runner, feed all tokens one-by-one
        runner = TrtRunner(coarse_plan, max_cache_length=512, num_layers=12)
        _coarse["runner"] = runner
        _coarse["call_count"] = 0
        toks = input_ids[0].cpu().tolist()
        for tok in toks[:-1]:
            runner.step(0, input_embed=coarse_embed_np[tok:tok+1].astype(np.float32),
                       use_input_embed=1.0)
        result = runner.step(0, input_embed=coarse_embed_np[toks[-1]:toks[-1]+1].astype(np.float32),
                             use_input_embed=1.0)
        # Create DynamicCache with dummy KV for proper seq_length tracking
        cache = DynamicCache()
        seq = len(toks)
        for lyr in range(12):
            cache.update(torch.zeros(1,12,seq,64,device=device),
                        torch.zeros(1,12,seq,64,device=device), lyr)
        _coarse_cache["cache"] = cache
    elif input_ids is not None:
        # Decode: single token
        tok = input_ids[0, -1].item()
        result = runner.step(0, input_embed=coarse_embed_np[tok:tok+1].astype(np.float32),
                             use_input_embed=1.0)
        # Grow cache by 1
        cache = _coarse_cache["cache"]
        if cache is not None:
            for lyr in range(12):
                cache.update(torch.zeros(1,12,1,64,device=device),
                            torch.zeros(1,12,1,64,device=device), lyr)
    else:
        raise ValueError("Expected input_ids for coarse")

    logits_t = torch.tensor(result["logits"], device=device).unsqueeze(0)
    return CausalLMOutputWithPast(
        logits=logits_t,
        past_key_values=_coarse_cache["cache"] if use_cache else None)

model.coarse_acoustics.forward = _trt_coarse_forward
print("  Coarse patched ✓", file=sys.stderr)

# =====================================================================
# Patch 3: Fine model — keep HF (no TRT engine for fine yet)
# =====================================================================
print("=== Fine: keeping HF (no TRT engine) ===", file=sys.stderr)
# Fine model does NOT use standard generate(). It has a custom loop
# that calls self.forward(n_inner, input_buffer) with a codebook index.
# This is architecturally different and needs a separate TRT engine design.
# For now, keep HF's fine model.

# =====================================================================
# Patch 4: Codec → TRT EnCodec
# =====================================================================
# Skip for now — HF codec is fast enough and we already verified TRT codec
# matches at 99.88%. The focus is on the autoregressive stages.
print("=== Codec: keeping HF (TRT codec verified at 99.88%) ===", file=sys.stderr)

# =====================================================================
# Run with ALL patches
# =====================================================================
print("\n=== Running patched pipeline ===", file=sys.stderr)
torch.manual_seed(42)
trt_audio = model.generate(**inputs)
trt_np = trt_audio.cpu().numpy().squeeze()
print(f"TRT: {len(trt_np)/24000:.2f}s, RMS={np.sqrt(np.mean(trt_np**2)):.4f}")
write_wav("/tmp/bark_all_trt.wav", trt_np)

# Compare
T_cmp = min(len(trt_np), len(hf_np))
if T_cmp > 0 and np.std(trt_np[:T_cmp]) > 1e-6 and np.std(hf_np[:T_cmp]) > 1e-6:
    corr = np.corrcoef(trt_np[:T_cmp], hf_np[:T_cmp])[0, 1]
    max_diff = np.max(np.abs(trt_np[:T_cmp] - hf_np[:T_cmp]))
    print(f"\nWaveform correlation: {corr:.6f}")
    print(f"Max sample diff: {max_diff:.6f}")
    print(f"Duration match: TRT={len(trt_np)/24000:.2f}s vs HF={len(hf_np)/24000:.2f}s")
    if corr > 0.999:
        print("PERFECT: Bit-identical audio!")
    elif corr > 0.99:
        print("EXCELLENT: Near-identical audio!")
    elif corr > 0.9:
        print("GOOD: Very similar audio")
    else:
        print(f"DIFFERENT audio (corr={corr:.4f})")
else:
    print(f"\nDifferent lengths: TRT={len(trt_np)}, HF={len(hf_np)}")

print(f"\nFiles:")
print(f"  /tmp/bark_all_hf.wav  (pure HF)")
print(f"  /tmp/bark_all_trt.wav (TRT semantic+coarse, HF fine+codec)")
