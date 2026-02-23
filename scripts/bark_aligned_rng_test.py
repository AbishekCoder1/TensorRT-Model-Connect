#!/usr/bin/env python3
"""Test TRT semantic with properly aligned RNG.

Uses an explicit torch.Generator seeded right before each multinomial
call, ensuring TRT gets the same random numbers as HF would.
Then feeds tokens through HF coarse+fine+codec.
"""
import torch, numpy as np, sys, json, struct, time
from trtf_build.debug_runner import TrtRunner
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig, BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
)

# Read bundle
with open("/tmp/bark.trtfb", "rb") as f:
    magic = f.read(8); hl = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hl).decode()); ds = 16 + hl; sections = {}
    for name, meta in header.get("sections", {}).items():
        f.seek(ds + meta["offset"]); sections[name] = f.read(meta["size"])

cfg = json.loads(sections["config.json"])
hidden = cfg["hidden_size"]; num_layers = cfg.get("num_hidden_layers", 12)
max_cache = header.get("max_cache_length", 1024)
sem_pad = 10000; sem_infer = 129599; text_off = 10048
embed_np = np.frombuffer(sections["semantic_embed"], dtype=np.float32).copy().reshape(-1, hidden)

# Setup
runner = TrtRunner(sections["engine_plan"], max_cache, num_layers)
model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
text_ids = inputs["input_ids"][0].numpy()
padded = np.zeros(256, dtype=np.int32)
padded[:min(len(text_ids), 256)] = text_ids[:256]

sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_cfg = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
fine_cfg = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

# TRT prefill
print("Prefilling...", file=sys.stderr)
for pos in range(256):
    e = (embed_np[int(padded[pos]) + text_off] + embed_np[sem_pad]).reshape(1, -1)
    runner.step(0, input_embed=e, use_input_embed=1.0)
result = runner.step(0, input_embed=embed_np[sem_infer].reshape(1, -1), use_input_embed=1.0)

# Also do HF semantic for comparison
print("Running HF semantic for reference...", file=sys.stderr)
torch.manual_seed(42)
with torch.no_grad():
    hf_sem = model.semantic.generate(inputs["input_ids"],
                                      semantic_generation_config=sem_cfg)
hf_sem_tokens = hf_sem[0, 257:].cpu().numpy()
hf_sem_tokens = hf_sem_tokens[hf_sem_tokens < 10000]
print(f"  HF semantic: {len(hf_sem_tokens)} tokens", file=sys.stderr)

# TRT sampling with SAME per-step RNG as HF would use
# HF uses torch.manual_seed(42) then multinomial at each step.
# We replicate by creating a generator with seed 42.
print("Generating TRT tokens with aligned RNG...", file=sys.stderr)
gen = torch.Generator().manual_seed(42)
semantic_tokens = []
logits = result["logits"].flatten()

for step in range(768):
    lt = torch.tensor(logits[:sem_pad + 1], dtype=torch.float32)
    lt = lt / 0.7  # temperature
    k = min(50, len(lt))
    tv, ti = torch.topk(lt, k)
    filt = torch.full_like(lt, float("-inf"))
    filt.scatter_(0, ti, tv)
    probs = torch.softmax(filt, dim=0)
    tok = int(torch.multinomial(probs, 1, generator=gen).item())

    if tok == sem_pad:
        print(f"  EOS at step {step}", file=sys.stderr)
        break
    semantic_tokens.append(tok)
    result = runner.step(tok)
    logits = result["logits"].flatten()

print(f"  TRT (aligned RNG): {len(semantic_tokens)} tokens", file=sys.stderr)
print(f"  First 20: {semantic_tokens[:20]}", file=sys.stderr)

# Compare with HF
n = min(len(semantic_tokens), len(hf_sem_tokens))
matches = sum(1 for i in range(n) if semantic_tokens[i] == hf_sem_tokens[i])
print(f"  Token match: {matches}/{n} ({100*matches/max(n,1):.1f}%)", file=sys.stderr)

# Feed TRT tokens through HF coarse+fine+codec
print("\nFeeding TRT tokens through HF coarse+fine+codec...", file=sys.stderr)
sem_seq = np.array(semantic_tokens + [sem_pad], dtype=np.int64)
sem_seq_t = torch.tensor(sem_seq, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    hf_coarse = model.coarse_acoustics.generate(
        sem_seq_t, semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg)
    print(f"  Coarse: {hf_coarse.shape[1]} tokens", file=sys.stderr)

    hf_fine = model.fine_acoustics.generate(
        hf_coarse, semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg,
        fine_generation_config=fine_cfg)
    print(f"  Fine: {hf_fine.shape}", file=sys.stderr)

    audio = model.codec_decode(hf_fine)
    wav = audio.squeeze().cpu().numpy()
    print(f"  Codec: {len(wav)} samples ({len(wav)/24000:.2f}s)", file=sys.stderr)

# Also feed HF tokens through same pipeline for reference
print("\nFeeding HF tokens through same pipeline...", file=sys.stderr)
hf_seq = np.array(list(hf_sem_tokens) + [sem_pad], dtype=np.int64)
hf_seq_t = torch.tensor(hf_seq, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    hf_c2 = model.coarse_acoustics.generate(
        hf_seq_t, semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg)
    hf_f2 = model.fine_acoustics.generate(
        hf_c2, semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg,
        fine_generation_config=fine_cfg)
    hf_audio2 = model.codec_decode(hf_f2)
    hf_wav = hf_audio2.squeeze().cpu().numpy()
    print(f"  HF codec: {len(hf_wav)} samples ({len(hf_wav)/24000:.2f}s)", file=sys.stderr)

# Save both
def write_wav(path, samples, sr=24000):
    n = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + n*4))
        f.write(b"WAVEfmt "); f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr*4, 4, 32))
        f.write(b"data"); f.write(struct.pack("<I", n*4))
        f.write(np.asarray(samples, dtype=np.float32).tobytes())

write_wav("/tmp/bark_trt_aligned_rng.wav", wav)
write_wav("/tmp/bark_hf_same_pipeline.wav", hf_wav)

trt_rms = np.sqrt(np.mean(wav**2))
hf_rms = np.sqrt(np.mean(hf_wav**2))
print(f"\nTRT (aligned RNG): /tmp/bark_trt_aligned_rng.wav "
      f"({len(wav)/24000:.2f}s, RMS={trt_rms:.4f})", file=sys.stderr)
print(f"HF reference:      /tmp/bark_hf_same_pipeline.wav "
      f"({len(hf_wav)/24000:.2f}s, RMS={hf_rms:.4f})", file=sys.stderr)
