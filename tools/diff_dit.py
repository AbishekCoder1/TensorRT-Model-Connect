#!/usr/bin/env python3
"""Diff test: TRT DiT engine vs HF WanTransformer3DModel.

Feeds identical inputs to both and compares outputs at bundle resolution.
"""
import torch, numpy as np, sys, struct, json, os, glob
from safetensors import safe_open

BUNDLE = "/mnt/storage/trt-transformers/engines/wan21-t2v-1.3b.trtfb"
MODEL_DIR = "/root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"

def silu(x): return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))
def gelu_tanh(x): return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

# Load bundle config + preprocessor weights
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

sec = hdr["sections"]["config.json"]
with open(BUNDLE, "rb") as f:
    f.seek(ds + sec["offset"]); cfg = json.loads(f.read(sec["size"]))

# Load HF pipeline (for DiT + T5)
from diffusers import WanPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.float32)
for f in sorted(glob.glob(os.path.join(f"{MODEL_DIR}/text_encoder", "*.safetensors"))):
    with safe_open(f, framework="pt") as sf:
        if "shared.weight" in sf.keys():
            pipe.text_encoder.encoder.embed_tokens.weight.data = sf.get_tensor("shared.weight")
            break

# Load TRT runner
from trtf_build.diffusion_runner import DiffusionRunner
from transformers import AutoTokenizer
runner = DiffusionRunner(BUNDLE)

# Get T5 text (use HF T5 so both sides get identical text)
print("=== Encoding text ===", file=sys.stderr)
hf_embeds, _ = pipe.encode_prompt("A cat walking in the garden")
text_torch = hf_embeds.detach()  # [1, 226, 4096]
print(f"Text: {text_torch.shape}", file=sys.stderr)

# Dimensions
t_lat, h_lat, w_lat = 5, 60, 104
pt, ph, pw = 1, 2, 2
nt, nh, nw = t_lat // pt, h_lat // ph, w_lat // pw

# Test at multiple timesteps
for t_val in [1000.0, 500.0, 100.0]:
    print(f"\n=== DiT step at t={t_val} ===")
    torch.manual_seed(42)
    latent = torch.randn(1, 16, t_lat, h_lat, w_lat)
    timestep = torch.tensor([t_val])

    # --- HF forward ---
    with torch.no_grad():
        hf_out = pipe.transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=text_torch,
        ).sample
    hf_np = hf_out.numpy()

    # --- TRT forward (preprocess + engine) ---
    patches = runner._patchify(latent.numpy(), [pt, ph, pw])
    hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

    half = 128
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    embed = np.concatenate([np.cos(t_val * freqs), np.sin(t_val * freqs)]).astype(np.float32).reshape(1, 256)
    h = silu(embed @ pp["condition_embedder.time_embedding.0.weight"] + pp["condition_embedder.time_embedding.0.bias"])
    te = h @ pp["condition_embedder.time_embedding.2.weight"] + pp["condition_embedder.time_embedding.2.bias"]
    temb = silu(te.copy()) @ pp["condition_embedder.time_proj.weight"] + pp["condition_embedder.time_proj.bias"]

    text_np = text_torch.numpy().reshape(226, 4096)
    text_proj = gelu_tanh(text_np @ pp["condition_embedder.text_embedding.weight"] + pp["condition_embedder.text_embedding.bias"])
    text_proj = text_proj @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]

    rope_cos, rope_sin = runner._compute_3d_rope(nt, nh, nw, 128)

    trt_raw = runner._run_engine("denoiser", {
        "hidden_states": hidden,
        "timestep_embedding": temb.reshape(1, -1),
        "time_embed": te.reshape(1, -1),
        "encoder_hidden_states": text_proj,
        "rotary_cos": rope_cos, "rotary_sin": rope_sin,
    })["output"]
    trt_spatial = runner._unpatchify(trt_raw, [pt, ph, pw], 16, t_lat, h_lat, w_lat)

    # Compare
    diff = np.abs(trt_spatial.flatten() - hf_np.flatten())
    cs = np.dot(trt_spatial.flatten().astype(np.float64), hf_np.flatten().astype(np.float64)) / (
        np.linalg.norm(trt_spatial.flatten().astype(np.float64)) * np.linalg.norm(hf_np.flatten().astype(np.float64)))
    print(f"  HF:  range=[{hf_np.min():.4f}, {hf_np.max():.4f}], std={hf_np.std():.4f}")
    print(f"  TRT: range=[{trt_spatial.min():.4f}, {trt_spatial.max():.4f}], std={trt_spatial.std():.4f}")
    print(f"  cosine={cs:.6f}, max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

    # Also compare at patch level (before unpatchify)
    hf_patched = runner._patchify(hf_np, [pt, ph, pw])
    patch_diff = np.abs(trt_raw.flatten() - hf_patched.flatten())
    patch_cs = np.dot(trt_raw.flatten().astype(np.float64), hf_patched.flatten().astype(np.float64)) / (
        np.linalg.norm(trt_raw.flatten().astype(np.float64)) * np.linalg.norm(hf_patched.flatten().astype(np.float64)))
    print(f"  patch-level: cosine={patch_cs:.6f}, max_diff={patch_diff.max():.6f}")

    # Check sorted values
    sorted_diff = np.abs(np.sort(trt_spatial.flatten()) - np.sort(hf_np.flatten()))
    print(f"  sorted: max_diff={sorted_diff.max():.6f}")

# Multi-step test: 5 steps with identical inputs
print("\n=== Multi-step (5 steps) ===")
pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
pipe.scheduler.set_timesteps(5)
from trtf_build.schedulers.flow_match_euler import FlowMatchEulerScheduler
our_sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=3.0)
our_sched.set_timesteps(5)

torch.manual_seed(42)
latent_torch = torch.randn(1, 16, t_lat, h_lat, w_lat)
latent_np = latent_torch.numpy().copy()

for step in range(5):
    t_hf = pipe.scheduler.timesteps[step]
    t_val = float(our_sched.timesteps[step])

    # HF
    with torch.no_grad():
        hf_noise = pipe.transformer(hidden_states=latent_torch, timestep=t_hf.unsqueeze(0),
                                     encoder_hidden_states=text_torch).sample
        latent_torch = pipe.scheduler.step(hf_noise, t_hf, latent_torch, return_dict=False)[0]

    # TRT
    patches = runner._patchify(latent_np.reshape(1,16,t_lat,h_lat,w_lat), [pt,ph,pw])
    hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]
    freqs = np.exp(-np.log(10000.0) * np.arange(128, dtype=np.float64) / 128)
    embed = np.concatenate([np.cos(t_val*freqs), np.sin(t_val*freqs)]).astype(np.float32).reshape(1,256)
    h = silu(embed @ pp["condition_embedder.time_embedding.0.weight"] + pp["condition_embedder.time_embedding.0.bias"])
    te = h @ pp["condition_embedder.time_embedding.2.weight"] + pp["condition_embedder.time_embedding.2.bias"]
    temb = silu(te.copy()) @ pp["condition_embedder.time_proj.weight"] + pp["condition_embedder.time_proj.bias"]

    trt_raw = runner._run_engine("denoiser", {
        "hidden_states": hidden, "timestep_embedding": temb.reshape(1,-1),
        "time_embed": te.reshape(1,-1), "encoder_hidden_states": text_proj,
        "rotary_cos": rope_cos, "rotary_sin": rope_sin,
    })["output"]
    noise_spatial = runner._unpatchify(trt_raw, [pt,ph,pw], 16, t_lat, h_lat, w_lat)
    latent_np = our_sched.step(noise_spatial, t_val, latent_np.reshape(1,16,t_lat,h_lat,w_lat), step)

    cs = np.dot(latent_np.flatten().astype(np.float64), latent_torch.numpy().flatten().astype(np.float64)) / (
        np.linalg.norm(latent_np.flatten().astype(np.float64)) * np.linalg.norm(latent_torch.numpy().flatten().astype(np.float64)))
    mx = np.abs(latent_np.flatten() - latent_torch.numpy().flatten()).max()
    print(f"  Step {step+1}: cosine={cs:.6f}, max_diff={mx:.4f}")
