#!/usr/bin/env python3
"""Run TRT semantic model with HF's sampling to isolate RNG vs model issues.

If TRT model + HF sampling = good speech -> C++ sampling is the problem
If TRT model + HF sampling = bad speech -> TRT model is the problem
"""
import numpy as np
import struct
import sys
import torch

print("Loading TRT engine from bundle...", file=sys.stderr)

from trtf_build.engine_builder import _resolve_model
from trtf_build.config import ModelConfig
from trtf_build.families import find_plugin
from trtf_build.debug_runner import TrtRunner

# Build TRT semantic engine
model_dir = _resolve_model("suno/bark-small")
config = ModelConfig.from_dir(model_dir)
plugin = find_plugin(config.model_type)
weights = plugin.load_weights(model_dir, config)
sem_cfg = weights["_semantic_cfg"]

# Build semantic engine
engine_plan = plugin.build_engine(config, weights, 1024, verbose=False)

# Create TRT runner
runner = TrtRunner(
    engine_plan=engine_plan,
    max_cache_length=1024,
    num_layers=sem_cfg["num_layers"],
)

print(f"TRT engine ready. Layers={sem_cfg['num_layers']}, hidden={sem_cfg['hidden_size']}", file=sys.stderr)

# Load HF model for embedding table
from transformers import AutoProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

hf_model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")

hf_sem_cfg = BarkSemanticGenerationConfig(**hf_model.generation_config.semantic_config)
embed_table = hf_model.semantic.input_embeds_layer.weight.data.cpu().numpy().astype(np.float32)

# Prefill (matching HF's semantic.generate exactly)
text_ids = inputs["input_ids"][0].numpy() + hf_sem_cfg.text_encoding_offset
hist_tok = hf_sem_cfg.semantic_pad_token  # 10000
hidden_size = sem_cfg["hidden_size"]

print("Prefilling 257 tokens...", file=sys.stderr)
for pos in range(256):
    # embed = embed_table[text_tok] + embed_table[hist_tok] (matching HF)
    embed = embed_table[text_ids[pos]] + embed_table[hist_tok]
    logits = runner.step_embed(embed)

# Infer token
infer_embed = embed_table[hf_sem_cfg.semantic_infer_token]
logits = runner.step_embed(infer_embed)

# Now do autoregressive with HF's torch.multinomial sampling
print("Generating with HF sampling (torch.multinomial)...", file=sys.stderr)
temp = hf_sem_cfg.temperature  # 0.7
top_k = hf_sem_cfg.top_k  # 50
semantic_pad_token = hf_sem_cfg.semantic_pad_token  # 10000

generated = []
for step in range(768):
    logits_t = torch.tensor(logits, dtype=torch.float32)

    # Suppress [semantic_pad_token+1, output_vocab)
    logits_t[semantic_pad_token + 1:] = float('-inf')

    # Temperature + top-k + sample (HF's exact sampling path)
    logits_t = logits_t / temp
    topk_vals, topk_idx = torch.topk(logits_t, min(top_k, len(logits_t)))
    filtered = torch.full_like(logits_t, float('-inf'))
    filtered.scatter_(0, topk_idx, topk_vals)
    probs = torch.softmax(filtered, dim=0)
    token = torch.multinomial(probs, 1).item()

    if token == semantic_pad_token:
        break

    generated.append(token)
    # Feed back single embedding (matching HF's autoregressive step)
    embed = embed_table[token]
    logits = runner.step_embed(embed)

print(f"Generated {len(generated)} semantic tokens", file=sys.stderr)

from collections import Counter
c = Counter(generated)
mx = 0; consec = 0
for i in range(1, len(generated)):
    if generated[i] == generated[i-1]: consec += 1
    else: consec = 0
    if consec > mx: mx = consec
print(f"  unique={len(set(generated))}, max_consec={mx}", file=sys.stderr)

# Feed through HF coarse+fine+codec
from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig, BarkFineGenerationConfig)

coarse_cfg = BarkCoarseGenerationConfig(**hf_model.generation_config.coarse_acoustics_config)
fine_cfg = BarkFineGenerationConfig(**hf_model.generation_config.fine_acoustics_config)
sem_cfg_gen = hf_sem_cfg

sem_out = torch.tensor(generated + [10000], dtype=torch.long).unsqueeze(0)
sem_for_coarse = sem_out.clone()
sem_for_coarse.masked_fill_(sem_for_coarse == 10000, 12048)

with torch.no_grad():
    hf_coarse = hf_model.coarse_acoustics.generate(
        sem_for_coarse, semantic_generation_config=sem_cfg_gen, coarse_generation_config=coarse_cfg)
    hf_fine = hf_model.fine_acoustics.generate(
        hf_coarse, semantic_generation_config=sem_cfg_gen,
        coarse_generation_config=coarse_cfg, fine_generation_config=fine_cfg)
    audio = hf_model.codec_decode(hf_fine)
    wav = audio.squeeze().cpu().numpy()

def write_wav(path, samples, sr=24000):
    n = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + n*4))
        f.write(b"WAVEfmt "); f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr*4, 4, 32))
        f.write(b"data"); f.write(struct.pack("<I", n*4))
        f.write(np.array(samples, dtype=np.float32).tobytes())

write_wav("/tmp/bark_trt_hf_sampling.wav", wav)
rms = np.sqrt(np.mean(wav**2))
print(f"Output: {len(wav)} samples ({len(wav)/24000:.2f}s), RMS={rms:.4f}", file=sys.stderr)
print(f"Saved: /tmp/bark_trt_hf_sampling.wav", file=sys.stderr)
