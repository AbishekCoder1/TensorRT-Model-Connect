#!/usr/bin/env python3
"""Compare: manual loop with global RNG vs per-step Generator vs HF generate."""
import torch
import numpy as np
import sys
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig

model = BarkModel.from_pretrained("suno/bark-small").eval()
sem = model.semantic
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
sp = 10000; si = 129599; to = 10048

text_ids = inputs["input_ids"][0]
padded = torch.zeros(256, dtype=torch.long)
padded[:len(text_ids)] = text_ids
input_off = padded.unsqueeze(0) + to
hist = torch.full((1, 256), sp, dtype=torch.long)
infer = torch.tensor([[si]], dtype=torch.long)

def do_prefill():
    with torch.no_grad():
        emb = torch.cat([
            sem.input_embeds_layer(input_off) + sem.input_embeds_layer(hist),
            sem.input_embeds_layer(infer)
        ], dim=1)
        out = sem(inputs_embeds=emb, use_cache=True)
    return out.logits[0, -1].clone(), out.past_key_values

def sample_token_hf_style(logits):
    """Apply HF's exact processing chain and sample."""
    scores = logits.clone()
    scores[sp + 1:] = float("-inf")
    scores = torch.nn.functional.log_softmax(scores.float(), dim=-1)
    scores = scores / 0.7
    top_k = min(50, scores.size(-1))
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs

def run_manual_global_rng(seed, first_n_tokens=None):
    """Manual loop using GLOBAL RNG (same as HF generate)."""
    torch.manual_seed(seed)
    logits, pkv = do_prefill()
    tokens = []
    for step in range(768):
        probs = sample_token_hf_style(logits)
        tok = int(torch.multinomial(probs.unsqueeze(0), 1).item())  # 2D like HF
        if tok == sp: break
        tokens.append(tok)
        if first_n_tokens and step < first_n_tokens:
            pass  # continue
        with torch.no_grad():
            out = sem(input_ids=torch.tensor([[tok]]), past_key_values=pkv, use_cache=True)
            logits = out.logits[0, -1]
            pkv = out.past_key_values
    return tokens

def run_manual_gen_rng(seed):
    """Manual loop using per-step Generator."""
    gen = torch.Generator().manual_seed(seed)
    logits, pkv = do_prefill()
    tokens = []
    for step in range(768):
        probs = sample_token_hf_style(logits)
        tok = int(torch.multinomial(probs, 1, generator=gen).item())
        if tok == sp: break
        tokens.append(tok)
        with torch.no_grad():
            out = sem(input_ids=torch.tensor([[tok]]), past_key_values=pkv, use_cache=True)
            logits = out.logits[0, -1]
            pkv = out.past_key_values
    return tokens

def run_hf_generate(seed):
    """HF's own generate method."""
    torch.manual_seed(seed)
    with torch.no_grad():
        out = sem.generate(inputs["input_ids"], semantic_generation_config=sem_cfg)
    gen_toks = out[0, 257:].numpy()
    return list(gen_toks[gen_toks < sp])

print("seed | global_rng | gen_rng | hf_generate | first_tok_match")
print("-----|-----------|---------|-------------|----------------")
for seed in [0, 1, 42]:
    t_global = run_manual_global_rng(seed)
    t_gen = run_manual_gen_rng(seed)
    t_hf = run_hf_generate(seed)

    match_g = "Y" if t_global and t_hf and t_global[0] == t_hf[0] else "N"
    match_gg = "Y" if t_global and t_gen and t_global[0] == t_gen[0] else "N"

    print(f"{seed:4d} | {len(t_global):9d} | {len(t_gen):7d} | {len(t_hf):11d} | global-hf:{match_g} global-gen:{match_gg}")

    # Show first 5 tokens from each
    print(f"       global: {t_global[:5]}")
    print(f"       gen:    {t_gen[:5]}")
    print(f"       hf:     {t_hf[:5]}")
    sys.stdout.flush()
