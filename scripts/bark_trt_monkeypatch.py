#!/usr/bin/env python3
"""Bark: produce EXACT same audio as HF by monkey-patching HF to use TRT forward passes.

Strategy: Patch HF's BarkSemanticModel.forward() to use TRT engine for logits.
HF's own generate() handles all the sampling, logit processing, stopping criteria.
The TRT engine produces identical logits → HF's sampler picks identical tokens → identical audio.
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
print("Loading HF Bark...", file=sys.stderr)
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.modeling_bark import BarkSemanticModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig, BarkCoarseGenerationConfig, BarkFineGenerationConfig)

model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model.eval()

prompt = "Hello, my dog is cute"
inputs = processor(prompt, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

# =====================================================================
# HF reference
# =====================================================================
print("HF reference...", file=sys.stderr)
torch.manual_seed(42)
hf_audio = model.generate(**inputs)
hf_np = hf_audio.cpu().numpy().squeeze()
print(f"HF: {len(hf_np)/24000:.2f}s, RMS={np.sqrt(np.mean(hf_np**2)):.4f}")
write_wav("/tmp/bark_mp_hf.wav", hf_np)

# =====================================================================
# Build TRT semantic engine
# =====================================================================
print("\nBuilding TRT semantic...", file=sys.stderr)
from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
from trtf_build.debug_runner import TrtRunner
import glob

model_dir = glob.glob("/root/.cache/huggingface/hub/models--suno--bark-small/snapshots/*")[0]
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin("bark")
weights = plugin.load_weights(model_dir, config)
sem_plan = plugin.build_engine(config, weights, 512)
sem_runner = TrtRunner(sem_plan, max_cache_length=512, num_layers=12)
sem_embed_np = model.semantic.input_embeds_layer.weight.detach().cpu().numpy()

# =====================================================================
# Monkey-patch semantic forward to use TRT
# =====================================================================
print("Patching semantic forward to use TRT...", file=sys.stderr)

_original_forward = model.semantic.forward
_sem_runner = sem_runner
_sem_embed = sem_embed_np
_step_count = [0]

def _trt_semantic_forward(
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,
):
    """Replace HF's semantic forward with TRT engine."""
    # HF calls forward with either input_ids or inputs_embeds
    if inputs_embeds is not None:
        # Prefill: inputs_embeds is [batch, seq, hidden]
        embeds = inputs_embeds[0].cpu().numpy()  # [seq, hidden]
        for pos in range(embeds.shape[0] - 1):
            _sem_runner.step(0, input_embed=embeds[pos:pos+1].astype(np.float32),
                           use_input_embed=1.0)
        result = _sem_runner.step(0, input_embed=embeds[-1:].astype(np.float32),
                                 use_input_embed=1.0)
    elif input_ids is not None:
        # Decode: input_ids is [batch, 1] — single token
        tok = input_ids[0, -1].item()
        emb = _sem_embed[tok:tok+1].astype(np.float32)
        result = _sem_runner.step(0, input_embed=emb, use_input_embed=1.0)
    else:
        raise ValueError("Neither input_ids nor inputs_embeds provided")

    _step_count[0] += 1
    logits_np = result["logits"]  # [1, vocab]
    logits_t = torch.tensor(logits_np, device=device).unsqueeze(0)  # [1, 1, vocab]

    return CausalLMOutputWithPast(logits=logits_t)

model.semantic.forward = _trt_semantic_forward

# =====================================================================
# Run HF pipeline with TRT semantic forward
# =====================================================================
print("Running HF pipeline with TRT semantic...", file=sys.stderr)
torch.manual_seed(42)
_step_count[0] = 0
trt_audio = model.generate(**inputs)
trt_np = trt_audio.cpu().numpy().squeeze()
print(f"TRT: {len(trt_np)/24000:.2f}s, RMS={np.sqrt(np.mean(trt_np**2)):.4f}")
print(f"TRT semantic steps: {_step_count[0]}")
write_wav("/tmp/bark_mp_trt.wav", trt_np)

# Compare
T_cmp = min(len(trt_np), len(hf_np))
if T_cmp > 0 and np.std(trt_np[:T_cmp]) > 1e-6 and np.std(hf_np[:T_cmp]) > 1e-6:
    corr = np.corrcoef(trt_np[:T_cmp], hf_np[:T_cmp])[0, 1]
    max_diff = np.max(np.abs(trt_np[:T_cmp] - hf_np[:T_cmp]))
    print(f"\nWaveform correlation: {corr:.6f}")
    print(f"Max sample diff: {max_diff:.6f}")
    if corr > 0.99:
        print("EXCELLENT: Near-identical audio!")
    elif corr > 0.9:
        print("GOOD: Very similar audio")
    else:
        print(f"Different audio (correlation={corr:.4f})")
else:
    print(f"Different lengths: TRT={len(trt_np)}, HF={len(hf_np)}")

print(f"\nFiles:")
print(f"  /tmp/bark_mp_hf.wav  (HF reference)")
print(f"  /tmp/bark_mp_trt.wav (TRT semantic)")
