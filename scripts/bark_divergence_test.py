#!/usr/bin/env python3
"""Find the exact step where TRT and HF semantic sampling diverge
when given identical RNG at each step."""
import torch, numpy as np, sys, json, struct
from trtf_build.debug_runner import TrtRunner
from transformers import BarkModel, AutoProcessor

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

model = BarkModel.from_pretrained("suno/bark-small").eval()
sem_model = model.semantic
proc = AutoProcessor.from_pretrained("suno/bark-small")
text_ids = proc("Hello, my dog is cute", return_tensors="pt")["input_ids"][0].numpy()
padded = np.zeros(256, dtype=np.int32)
padded[:min(len(text_ids), 256)] = text_ids[:256]

# TRT prefill
runner = TrtRunner(sections["engine_plan"], max_cache, num_layers)
for pos in range(256):
    e = (embed_np[int(padded[pos]) + text_off] + embed_np[sem_pad]).reshape(1, -1)
    runner.step(0, input_embed=e, use_input_embed=1.0)
trt_r = runner.step(0, input_embed=embed_np[sem_infer].reshape(1, -1), use_input_embed=1.0)

# HF prefill
input_ids_off = torch.tensor(padded, dtype=torch.long).unsqueeze(0) + text_off
sem_hist = torch.full((1, 256), sem_pad, dtype=torch.long)
infer_a = torch.tensor([[sem_infer]], dtype=torch.long)
with torch.no_grad():
    hf_emb = torch.cat([sem_model.input_embeds_layer(input_ids_off) + sem_model.input_embeds_layer(sem_hist),
                         sem_model.input_embeds_layer(infer_a)], dim=1)
    hf_out = sem_model(inputs_embeds=hf_emb, use_cache=True)
    past_kv = hf_out.past_key_values

trt_logits = trt_r["logits"].flatten()
hf_logits = hf_out.logits[0, -1].cpu().numpy()

# Shared RNG
base_gen = torch.Generator().manual_seed(42)
first_div = -1
trt_eos = -1
hf_eos = -1

for step in range(768):
    state = base_gen.get_state()

    # Sample from TRT logits
    lt = torch.tensor(trt_logits[:sem_pad + 1], dtype=torch.float32) / 0.7
    tv, ti = torch.topk(lt, min(50, len(lt)))
    filt = torch.full_like(lt, float("-inf"))
    filt.scatter_(0, ti, tv)
    g1 = torch.Generator().set_state(state)
    trt_tok = int(torch.multinomial(torch.softmax(filt, dim=0), 1, generator=g1).item())

    # Sample from HF logits (same RNG state)
    lt2 = torch.tensor(hf_logits[:sem_pad + 1], dtype=torch.float32) / 0.7
    tv2, ti2 = torch.topk(lt2, min(50, len(lt2)))
    filt2 = torch.full_like(lt2, float("-inf"))
    filt2.scatter_(0, ti2, tv2)
    g2 = torch.Generator().set_state(state)
    hf_tok = int(torch.multinomial(torch.softmax(filt2, dim=0), 1, generator=g2).item())

    # Advance shared generator
    _ = torch.multinomial(torch.ones(2), 1, generator=base_gen)

    diverged = (trt_tok != hf_tok)
    if diverged and first_div < 0:
        first_div = step
        # Show probability context
        trt_p = torch.softmax(filt, dim=0)
        hf_p = torch.softmax(filt2, dim=0)
        print(f"FIRST DIVERGENCE at step {step}: TRT={trt_tok}, HF={hf_tok}")
        print(f"  P(TRT tok={trt_tok}): TRT={float(trt_p[trt_tok]):.6f}, HF={float(hf_p[trt_tok]):.6f}")
        print(f"  P(HF  tok={hf_tok}):  TRT={float(trt_p[hf_tok]):.6f}, HF={float(hf_p[hf_tok]):.6f}")

    if step < 10 or (step < 100 and step % 10 == 0) or step % 50 == 0:
        m = "=" if not diverged else "X"
        print(f"Step {step:3d}: TRT={trt_tok:5d} HF={hf_tok:5d} [{m}]", end="")
        if first_div >= 0:
            print(f"  (diverged at {first_div})", end="")
        print()

    # Track EOS
    if trt_tok == sem_pad and trt_eos < 0:
        trt_eos = step
        print(f"  TRT EOS at step {step}")
    if hf_tok == sem_pad and hf_eos < 0:
        hf_eos = step
        print(f"  HF EOS at step {step}")
    if trt_eos >= 0 and hf_eos >= 0:
        break

    # Advance models (each feeds its own token)
    if trt_tok != sem_pad:
        trt_r = runner.step(trt_tok)
        trt_logits = trt_r["logits"].flatten()

    if hf_tok != sem_pad:
        with torch.no_grad():
            hf_out = sem_model(input_ids=torch.tensor([[hf_tok]]),
                               past_key_values=past_kv, use_cache=True)
            hf_logits = hf_out.logits[0, -1].cpu().numpy()
            past_kv = hf_out.past_key_values

print(f"\nFirst divergence: step {first_div}")
print(f"TRT EOS: step {trt_eos}")
print(f"HF EOS: step {hf_eos}")
