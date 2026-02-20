#!/usr/bin/env python3
"""Full pipeline diff: Python TRT path (replicating C++) vs HF reference.

Runs 30 steps with CFG using TRT T5 + TRT DiT, matching C++ logic exactly.
"""
import torch, numpy as np, sys, struct, json, os, glob
from safetensors import safe_open
from PIL import Image

BUNDLE = "/mnt/storage/trt-transformers/engines/wan21-t2v-1.3b.trtfb"
MODEL_DIR = "/root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"

def silu(x): return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))
def gelu_tanh(x): return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

# Load preprocessor weights
with open(BUNDLE, "rb") as f:
    f.read(8); jl = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(jl)); ds = 16 + jl
sec = hdr["sections"]["preprocessor_weights"]
with open(BUNDLE, "rb") as f:
    f.seek(ds + sec["offset"]); ppd = f.read(sec["size"])
il = struct.unpack("<I", ppd[:4])[0]; ppx = json.loads(ppd[4:4+il]); blob = ppd[4+il:]
def load(k):
    i = ppx[k]; c = int(np.prod(i["shape"]))
    return np.frombuffer(blob, np.float32, c, i["offset"]).reshape(i["shape"])
pp = {k: load(k) for k in ppx}

from trtf_build.diffusion_runner import DiffusionRunner
from trtf_build.schedulers.flow_match_euler import FlowMatchEulerScheduler
from transformers import AutoTokenizer

runner = DiffusionRunner(BUNDLE)
tok = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/tokenizer")

# T5 encode prompt + null
tokens = tok("A cat walking in the garden", return_tensors="np",
             padding="max_length", max_length=226, truncation=True)
trt_text = runner.encode_text(tokens["input_ids"].astype(np.int32))

neg_tokens = tok("", return_tensors="np", padding="max_length",
                 max_length=226, truncation=True)
trt_neg = runner.encode_text(neg_tokens["input_ids"].astype(np.int32))

print(f"Prompt T5: {trt_text.shape}, std={trt_text.std():.6f}", file=sys.stderr)
print(f"Neg T5: {trt_neg.shape}, std={trt_neg.std():.6f}", file=sys.stderr)

# Project text
def project(text_np):
    flat = text_np.reshape(226, 4096)
    out = flat @ pp["condition_embedder.text_embedding.weight"] + pp["condition_embedder.text_embedding.bias"]
    out = gelu_tanh(out)
    out = out @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]
    return out

text_proj = project(trt_text)
null_proj = project(trt_neg)

# Scheduler
sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=3.0)
sched.set_timesteps(30)

# Dimensions
t_lat, h_lat, w_lat = 5, 60, 104
pt, ph, pw = 1, 2, 2
nt, nh, nw = t_lat//pt, h_lat//ph, w_lat//pw

# RoPE
rope_cos, rope_sin = runner._compute_3d_rope(nt, nh, nw, 128)

# Initial noise (numpy, matching C++ seed structure)
rng = np.random.default_rng(42)
latents = rng.standard_normal((1, 16, t_lat, h_lat, w_lat)).astype(np.float32)

# Denoising loop with CFG
for step in range(30):
    t_val = float(sched.timesteps[step])

    # Timestep embedding
    freqs = np.exp(-np.log(10000.0) * np.arange(128, dtype=np.float64) / 128)
    embed = np.concatenate([np.cos(t_val * freqs), np.sin(t_val * freqs)]).astype(np.float32).reshape(1, 256)
    h = silu(embed @ pp["condition_embedder.time_embedding.0.weight"] + pp["condition_embedder.time_embedding.0.bias"])
    te = h @ pp["condition_embedder.time_embedding.2.weight"] + pp["condition_embedder.time_embedding.2.bias"]
    temb = silu(te.copy()) @ pp["condition_embedder.time_proj.weight"] + pp["condition_embedder.time_proj.bias"]

    # Patchify + embed
    patches = runner._patchify(latents, [pt, ph, pw])
    hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

    # Conditional
    cond = runner._run_engine("denoiser", {
        "hidden_states": hidden, "timestep_embedding": temb.reshape(1, -1),
        "time_embed": te.reshape(1, -1), "encoder_hidden_states": text_proj,
        "rotary_cos": rope_cos, "rotary_sin": rope_sin,
    })["output"]

    # Unconditional (T5 of empty string, NOT zeros)
    uncond = runner._run_engine("denoiser", {
        "hidden_states": hidden, "timestep_embedding": temb.reshape(1, -1),
        "time_embed": te.reshape(1, -1), "encoder_hidden_states": null_proj,
        "rotary_cos": rope_cos, "rotary_sin": rope_sin,
    })["output"]

    # CFG
    noise_pred = uncond + 5.0 * (cond - uncond)

    # Unpatchify
    noise_spatial = runner._unpatchify(noise_pred, [pt, ph, pw], 16, t_lat, h_lat, w_lat)

    # Scheduler step
    latents = sched.step(noise_spatial, t_val, latents, step)

    if step % 10 == 0:
        print(f"  Step {step+1}/30: latents std={latents.std():.4f}", file=sys.stderr)

# Denormalize
sec = hdr["sections"]["config.json"]
with open(BUNDLE, "rb") as f:
    f.seek(ds + sec["offset"]); cfg = json.loads(f.read(sec["size"]))
lm = np.array(cfg["latents_mean"], dtype=np.float32).reshape(1, 16, 1, 1, 1)
ls = np.array(cfg["latents_std"], dtype=np.float32).reshape(1, 16, 1, 1, 1)
latents = latents * ls + lm

# VAE decode
import subprocess
lat_file = "/tmp/trtf_python_lat.bin"
out_file = "/tmp/trtf_python_out.bin"
with open(lat_file, "wb") as f:
    f.write(latents.astype(np.float32).tobytes())

subprocess.run([
    "/workspace/trt-transformers-cpp/.venv/bin/python",
    "/workspace/trt-transformers-cpp/scripts/vae_decode.py",
    "--model-id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "--latents-file", lat_file, "--output-file", out_file,
    "--shape", f"1,16,{t_lat},{h_lat},{w_lat}",
], check=True)

raw = np.fromfile(out_file, dtype=np.float32).reshape(3, 17, 480, 832)
video = np.clip((raw + 1) / 2, 0, 1)

os.makedirs("/tmp/trtf_python_full", exist_ok=True)
for t in range(17):
    frame = (video[:, t] * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(frame).save(f"/tmp/trtf_python_full/frame_{t:04d}.png")
print(f"Saved 17 frames to /tmp/trtf_python_full/", file=sys.stderr)
