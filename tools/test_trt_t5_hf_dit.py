#!/usr/bin/env python3
"""Test: inject TRT T5 embeddings into HF pipe() for identical noise/loop."""
import torch, numpy as np, os, sys, glob
from PIL import Image
from safetensors import safe_open

MODEL_DIR = "/root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"
BUNDLE = "/mnt/storage/trt-transformers/engines/wan21-t2v-1.3b.trtfb"

# --- TRT T5 ---
from transformers import AutoTokenizer
from trtf_build.diffusion_runner import DiffusionRunner

tok = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/tokenizer")
tokens = tok("A cat walking in the garden", return_tensors="np",
             padding="max_length", max_length=226, truncation=True)
runner = DiffusionRunner(BUNDLE)
trt_text = runner.encode_text(tokens["input_ids"].astype(np.int32))
print(f"TRT T5: {trt_text.shape}, std={trt_text.std():.6f}", file=sys.stderr)

# --- HF pipeline ---
from diffusers import WanPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.float32)
# Fix T5 embed_tokens
for f in sorted(glob.glob(os.path.join(f"{MODEL_DIR}/text_encoder", "*.safetensors"))):
    with safe_open(f, framework="pt") as sf:
        if "shared.weight" in sf.keys():
            pipe.text_encoder.encoder.embed_tokens.weight.data = sf.get_tensor("shared.weight")
            break
pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
pipe = pipe.to("cuda")

# --- Compare T5 ---
hf_embeds, hf_neg = pipe.encode_prompt("A cat walking in the garden")
t1 = trt_text.flatten().astype(np.float64)
t2 = hf_embeds.detach().cpu().numpy().flatten().astype(np.float64)
cos = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))
print(f"T5 cosine: {cos:.6f}, max_diff: {np.abs(t1-t2).max():.6f}", file=sys.stderr)

# --- Generate: HF reference (pure HF T5) ---
generator = torch.Generator("cuda").manual_seed(42)
with torch.no_grad():
    hf_output = pipe(
        prompt_embeds=hf_embeds,
        negative_prompt_embeds=hf_neg,
        num_frames=17, height=480, width=832,
        num_inference_steps=30, guidance_scale=5.0,
        generator=generator,
    )
os.makedirs("/tmp/hf_ref", exist_ok=True)
for i, frame in enumerate(hf_output.frames[0]):
    arr = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(f"/tmp/hf_ref/frame_{i:04d}.png")
print(f"HF ref: {len(hf_output.frames[0])} frames", file=sys.stderr)

# --- Generate: TRT T5 injected into same pipe() ---
trt_embeds_torch = torch.from_numpy(trt_text).to("cuda")

# Encode empty string for negative (NOT zeros — HF uses T5(""))
neg_tokens = tok("", return_tensors="np", padding="max_length", max_length=226, truncation=True)
trt_neg_np = runner.encode_text(neg_tokens["input_ids"].astype(np.int32))
trt_neg = torch.from_numpy(trt_neg_np).to("cuda")
print(f"Neg embeds: HF std={hf_neg.std():.6f}, TRT std={trt_neg.std():.6f}", file=sys.stderr)

generator = torch.Generator("cuda").manual_seed(42)  # same seed
with torch.no_grad():
    trt_output = pipe(
        prompt_embeds=trt_embeds_torch,
        negative_prompt_embeds=trt_neg,
        num_frames=17, height=480, width=832,
        num_inference_steps=30, guidance_scale=5.0,
        generator=generator,
    )
os.makedirs("/tmp/trt_t5_ref", exist_ok=True)
for i, frame in enumerate(trt_output.frames[0]):
    arr = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(f"/tmp/trt_t5_ref/frame_{i:04d}.png")
print(f"TRT T5 ref: {len(trt_output.frames[0])} frames", file=sys.stderr)

# --- Compare frames ---
import hashlib
for i in range(min(3, len(hf_output.frames[0]))):
    h1 = hashlib.md5(open(f"/tmp/hf_ref/frame_{i:04d}.png", "rb").read()).hexdigest()
    h2 = hashlib.md5(open(f"/tmp/trt_t5_ref/frame_{i:04d}.png", "rb").read()).hexdigest()
    match = "MATCH" if h1 == h2 else "DIFFER"
    print(f"Frame {i}: {match}", file=sys.stderr)
