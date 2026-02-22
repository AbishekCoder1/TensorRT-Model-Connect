#!/usr/bin/env python3
"""Bark E2E: TRT semantic stage + HF coarse/fine + HF codec → compare with HF full pipeline."""
import sys, struct, torch, numpy as np

def write_wav(path, audio_np, sr=24000):
    n = len(audio_np); ds = n * 4
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36+ds) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr*4, 4, 32))
        f.write(b"data" + struct.pack("<I", ds) + audio_np.astype(np.float32).tobytes())

def top_k_sample(logits, top_k=50, temperature=0.7):
    logits = logits.astype(np.float64) / max(temperature, 1e-8)
    if top_k > 0 and top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        vals = logits[idx]
    else:
        idx = np.arange(len(logits)); vals = logits
    vals = vals - vals.max()
    probs = np.exp(vals); probs /= probs.sum()
    return int(idx[np.random.choice(len(idx), p=probs)])

device = "cuda"
TEXT_ENCODING_OFFSET = 10048
SEMANTIC_PAD_TOKEN = 10000
SEMANTIC_INFER_TOKEN = 129599
SEMANTIC_VOCAB_SIZE = 10000
MAX_INPUT_SEMANTIC_LENGTH = 256
COARSE_SEMANTIC_PAD_TOKEN = 12048

print("Loading HF model...", file=sys.stderr)
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig, BarkCoarseGenerationConfig, BarkFineGenerationConfig)

model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model.eval()

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_cfg = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
fine_cfg = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

prompt = "Hello, my dog is cute"
inputs = processor(prompt, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

# HF reference
captured = [None]
_orig = model.codec_decode
def _hook(f, *a, **k): captured[0] = f.clone(); return _orig(f, *a, **k)
model.codec_decode = _hook
torch.manual_seed(42)
with torch.no_grad():
    hf_audio = model.generate(**inputs)
hf_np = hf_audio.cpu().numpy().squeeze()
print(f"HF: {len(hf_np)/24000:.2f}s, RMS={np.sqrt(np.mean(hf_np**2)):.4f}", file=sys.stderr)
write_wav("/tmp/bark_e2e_hf.wav", hf_np)

# Extract embedding table
sem_embed = model.semantic.input_embeds_layer.weight.detach().cpu().numpy()

# === TRT Semantic ===
print("Building TRT semantic...", file=sys.stderr)
from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
import glob
model_dir = glob.glob("/root/.cache/huggingface/hub/models--suno--bark-small/snapshots/*")[0]
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin("bark")
weights = plugin.load_weights(model_dir, config)
sem_plan = plugin.build_engine(config, weights, 512)

from trtf_build.debug_runner import TrtRunner
sem_runner = TrtRunner(sem_plan, max_cache_length=512, num_layers=12)

input_ids = inputs["input_ids"][0].cpu().tolist()
history_prompt = inputs.get("history_prompt")

# Get semantic history
if history_prompt is not None:
    sem_history = history_prompt["semantic_prompt"].cpu().numpy().flatten()[-MAX_INPUT_SEMANTIC_LENGTH:]
    if len(sem_history) < MAX_INPUT_SEMANTIC_LENGTH:
        sem_history = np.concatenate([sem_history, np.full(MAX_INPUT_SEMANTIC_LENGTH - len(sem_history), SEMANTIC_PAD_TOKEN)])
else:
    sem_history = np.full(MAX_INPUT_SEMANTIC_LENGTH, SEMANTIC_PAD_TOKEN)

text_ids_offset = [tid + TEXT_ENCODING_OFFSET for tid in input_ids[:MAX_INPUT_SEMANTIC_LENGTH]]
np.random.seed(42)

# Prefill: sum text_embed + history_embed
seq_len = min(len(text_ids_offset), MAX_INPUT_SEMANTIC_LENGTH)
for pos in range(seq_len):
    text_emb = sem_embed[text_ids_offset[pos]]
    hist_emb = sem_embed[int(sem_history[pos])]
    combined = (text_emb + hist_emb).reshape(1, -1).astype(np.float32)
    sem_runner.step(0, input_embed=combined, use_input_embed=1.0)

# Infer token
infer_emb = sem_embed[SEMANTIC_INFER_TOKEN].reshape(1, -1).astype(np.float32)
result = sem_runner.step(0, input_embed=infer_emb, use_input_embed=1.0)

# Generate
semantic_tokens = []
for _ in range(768):
    logits = result["logits"].flatten()
    suppressed = logits.copy()
    suppressed[SEMANTIC_VOCAB_SIZE:SEMANTIC_PAD_TOKEN] = -1e9
    suppressed[SEMANTIC_PAD_TOKEN+1:] = -1e9
    tok = top_k_sample(suppressed, top_k=50, temperature=0.7)
    semantic_tokens.append(tok)
    if tok == SEMANTIC_PAD_TOKEN:
        break
    tok_emb = sem_embed[tok].reshape(1, -1).astype(np.float32)
    result = sem_runner.step(0, input_embed=tok_emb, use_input_embed=1.0)

print(f"TRT semantic: {len(semantic_tokens)} tokens", file=sys.stderr)

# === HF Coarse + Fine + Codec (using TRT semantic tokens) ===
sem_tensor = torch.tensor([semantic_tokens], device=device)
sem_tensor[sem_tensor == SEMANTIC_PAD_TOKEN] = COARSE_SEMANTIC_PAD_TOKEN

with torch.no_grad():
    coarse_out = model.coarse_acoustics.generate(
        sem_tensor, history_prompt=history_prompt,
        semantic_generation_config=sem_cfg, coarse_generation_config=coarse_cfg)
    fine_out = model.fine_acoustics.generate(
        coarse_out, semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg, fine_generation_config=fine_cfg)
    trt_audio = model.codec_decode(fine_out)

trt_np = trt_audio.cpu().numpy().squeeze()
print(f"TRT E2E: {len(trt_np)/24000:.2f}s, RMS={np.sqrt(np.mean(trt_np**2)):.4f}", file=sys.stderr)
write_wav("/tmp/bark_e2e_trt.wav", trt_np)

print(f"\nResults:")
print(f"  HF:  /tmp/bark_e2e_hf.wav  ({len(hf_np)/24000:.2f}s)")
print(f"  TRT: /tmp/bark_e2e_trt.wav ({len(trt_np)/24000:.2f}s)")
print(f"  Both should be intelligible speech saying 'Hello, my dog is cute'")
