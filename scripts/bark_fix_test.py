#!/usr/bin/env python3
"""Test the text_pad_token fix: use embed[129595] for pad positions instead of embed[10048]."""
import torch, numpy as np, struct, sys, json
from trtf_build.debug_runner import TrtRunner
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig, BarkCoarseGenerationConfig,
    BarkFineGenerationConfig)

# Bundle
with open("/tmp/bark.trtfb", "rb") as f:
    magic = f.read(8); hl = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hl).decode()); ds = 16 + hl; sections = {}
    for name, meta in header.get("sections", {}).items():
        f.seek(ds + meta["offset"]); sections[name] = f.read(meta["size"])
cfg = json.loads(sections["config.json"])
hidden = cfg["hidden_size"]; nl = cfg.get("num_hidden_layers", 12)
mc = header.get("max_cache_length", 1024)
sp = 10000; si = 129599; to = 10048
TEXT_PAD_TOKEN = 129595  # THE FIX

enp = np.frombuffer(sections["semantic_embed"], dtype=np.float32).copy().reshape(-1, hidden)

model = BarkModel.from_pretrained("suno/bark-small").eval()
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sc = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
cc = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
fc = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

text_ids = inputs["input_ids"][0].numpy()

def write_wav(path, samples, sr=24000):
    n = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", 36+n*4))
        f.write(b"WAVEfmt "); f.write(struct.pack("<IHHIIHH", 16, 3, 1, sr, sr*4, 4, 32))
        f.write(b"data"); f.write(struct.pack("<I", n*4))
        f.write(np.asarray(samples, dtype=np.float32).tobytes())

# TRT semantic with FIXED prefill (text_pad_token for pad positions)
print("TRT semantic with text_pad_token fix...", file=sys.stderr)
torch.manual_seed(42)
runner = TrtRunner(sections["engine_plan"], mc, nl)
n_real = int((text_ids > 0).sum())
for pos in range(256):
    if pos < n_real:
        text_tok = int(text_ids[pos]) + to
    else:
        text_tok = TEXT_PAD_TOKEN  # THE FIX: use 129595, not 0+10048
    hist_tok = sp
    embed = (enp[text_tok] + enp[hist_tok]).reshape(1, -1)
    runner.step(0, input_embed=embed, use_input_embed=1.0)
result = runner.step(0, input_embed=enp[si].reshape(1, -1), use_input_embed=1.0)

# Sample (seed right before loop)
torch.manual_seed(42)
trt_tokens = []
logits = result["logits"].flatten()
for step in range(768):
    lt = torch.tensor(logits, dtype=torch.float32)
    lt[sp + 1:] = float("-inf")
    lt = lt / 0.7
    top_k = min(50, lt.size(-1))
    rem = lt < torch.topk(lt, top_k)[0][-1]
    lt = lt.masked_fill(rem, float("-inf"))
    probs = torch.softmax(lt, dim=-1)
    tok = int(torch.multinomial(probs.unsqueeze(0), 1).item())
    if tok == sp: break
    trt_tokens.append(tok)
    result = runner.step(tok)
    logits = result["logits"].flatten()
del runner
print(f"  TRT: {len(trt_tokens)} tokens, first 10: {trt_tokens[:10]}", file=sys.stderr)

# HF reference (uses attention_mask correctly via model.generate)
print("HF reference via BarkModel.generate...", file=sys.stderr)
torch.manual_seed(42)
with torch.no_grad():
    hf_audio = model.generate(**inputs)
hf_wav = hf_audio.squeeze().cpu().numpy()
write_wav("/tmp/bark_fix_hf_ref.wav", hf_wav)
print(f"  HF: {len(hf_wav)} samples ({len(hf_wav)/24000:.2f}s)", file=sys.stderr)

# Feed TRT tokens through HF downstream
print("TRT tokens -> HF coarse+fine+codec...", file=sys.stderr)
seq = np.array(trt_tokens + [sp], dtype=np.int64)
seq_t = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    co = model.coarse_acoustics.generate(seq_t, semantic_generation_config=sc,
                                          coarse_generation_config=cc)
    fi = model.fine_acoustics.generate(co, semantic_generation_config=sc,
                                        coarse_generation_config=cc,
                                        fine_generation_config=fc)
    audio = model.codec_decode(fi)
    trt_wav = audio.squeeze().cpu().numpy()
write_wav("/tmp/bark_fix_trt.wav", trt_wav)
print(f"  TRT: {len(trt_wav)} samples ({len(trt_wav)/24000:.2f}s)", file=sys.stderr)

# Compare
print(f"\nResults:", file=sys.stderr)
print(f"  HF ref:  /tmp/bark_fix_hf_ref.wav ({len(hf_wav)/24000:.2f}s, "
      f"RMS={np.sqrt(np.mean(hf_wav**2)):.4f})", file=sys.stderr)
print(f"  TRT fix: /tmp/bark_fix_trt.wav ({len(trt_wav)/24000:.2f}s, "
      f"RMS={np.sqrt(np.mean(trt_wav**2)):.4f})", file=sys.stderr)
