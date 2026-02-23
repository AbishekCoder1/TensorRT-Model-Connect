#!/usr/bin/env python3
"""Compare torch.multinomial vs manual CDF with torch RNG vs numpy MT19937."""
import torch, numpy as np, sys, json, struct
from trtf_build.debug_runner import TrtRunner
from transformers import AutoProcessor

with open("/tmp/bark.trtfb", "rb") as f:
    magic = f.read(8); hl = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hl).decode()); ds = 16 + hl; sections = {}
    for name, meta in header.get("sections", {}).items():
        f.seek(ds + meta["offset"]); sections[name] = f.read(meta["size"])
cfg = json.loads(sections["config.json"])
hidden = cfg["hidden_size"]; nl = cfg.get("num_hidden_layers", 12)
mc = header.get("max_cache_length", 1024)
sp = 10000; si = 129599; to = 10048
enp = np.frombuffer(sections["semantic_embed"], dtype=np.float32).copy().reshape(-1, hidden)
proc = AutoProcessor.from_pretrained("suno/bark-small")
text_ids = proc("Hello, my dog is cute", return_tensors="pt")["input_ids"][0].numpy()
pad_t = np.zeros(256, dtype=np.int32)
pad_t[:min(len(text_ids), 256)] = text_ids[:256]

def prefill(runner):
    for pos in range(256):
        e = (enp[int(pad_t[pos]) + to] + enp[sp]).reshape(1, -1)
        runner.step(0, input_embed=e, use_input_embed=1.0)
    return runner.step(0, input_embed=enp[si].reshape(1, -1), use_input_embed=1.0)

def run_torch_multinomial(seed):
    gen = torch.Generator().manual_seed(seed)
    runner = TrtRunner(sections["engine_plan"], mc, nl)
    result = prefill(runner)
    count = 0
    logits = result["logits"].flatten()
    for step in range(768):
        lt = torch.tensor(logits[:sp+1], dtype=torch.float32) / 0.7
        tv, ti = torch.topk(lt, min(50, len(lt)))
        filt = torch.full_like(lt, float("-inf"))
        filt.scatter_(0, ti, tv)
        probs = torch.softmax(filt, dim=0)
        tok = int(torch.multinomial(probs, 1, generator=gen).item())
        if tok == sp: break
        count += 1
        result = runner.step(tok)
        logits = result["logits"].flatten()
    del runner
    return count

def run_torch_cdf(seed):
    gen = torch.Generator().manual_seed(seed)
    runner = TrtRunner(sections["engine_plan"], mc, nl)
    result = prefill(runner)
    count = 0
    logits = result["logits"].flatten()
    for step in range(768):
        lt = torch.tensor(logits[:sp+1], dtype=torch.float32) / 0.7
        tv, ti = torch.topk(lt, min(50, len(lt)))
        filt = torch.full_like(lt, float("-inf"))
        filt.scatter_(0, ti, tv)
        probs = torch.softmax(filt, dim=0)
        # Manual CDF using torch RNG for the uniform value
        r = float(torch.empty(1).uniform_(generator=gen).item())
        cumul = 0.0
        tok = int(ti[0].item())
        sorted_p, sorted_i = torch.sort(probs[ti], descending=True)
        for j in range(len(sorted_p)):
            cumul += float(sorted_p[j])
            if r < cumul:
                tok = int(ti[sorted_i[j]].item())
                break
        if tok == sp: break
        count += 1
        result = runner.step(tok)
        logits = result["logits"].flatten()
    del runner
    return count

def run_numpy_cdf(seed):
    rng = np.random.RandomState(seed)
    runner = TrtRunner(sections["engine_plan"], mc, nl)
    result = prefill(runner)
    count = 0
    logits = result["logits"].flatten()
    for step in range(768):
        valid = logits[:sp+1].copy() / 0.7
        topk_idx = np.argsort(valid)[-50:]
        mx = valid[topk_idx].max()
        p = np.exp(valid[topk_idx] - mx)
        p /= p.sum()
        r = rng.uniform(0.0, 1.0)
        cumul = 0.0
        order = np.argsort(-p)
        tok = int(topk_idx[order[0]])
        for j in order:
            cumul += p[j]
            if r < cumul:
                tok = int(topk_idx[j])
                break
        if tok == sp: break
        count += 1
        result = runner.step(tok)
        logits = result["logits"].flatten()
    del runner
    return count

print("seed | torch_multi | torch_cdf | numpy_cdf")
print("-----|-------------|-----------|----------")
for seed in [0, 1, 2, 42, 100]:
    n1 = run_torch_multinomial(seed)
    n2 = run_torch_cdf(seed)
    n3 = run_numpy_cdf(seed)
    print(f"{seed:4d} | {n1:11d} | {n2:9d} | {n3:9d}")
    sys.stdout.flush()
