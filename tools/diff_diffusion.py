#!/usr/bin/env python3
"""Diff test for the C++ diffusion pipeline.

Compares preprocessor computations, T5 output, and denoiser output
between the bundle's weights/engines and HF reference.
"""
import struct
import json
import sys

import numpy as np

BUNDLE_PATH = "/mnt/storage/trt-transformers/engines/wan21-t2v-1.3b.trtfb"
MODEL_DIR = "/root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"


def load_bundle_pp_weights(bundle_path):
    """Load preprocessor weights from bundle."""
    with open(bundle_path, "rb") as f:
        f.read(8)  # magic
        json_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(json_len))
        data_start = 16 + json_len

    sec = header["sections"]["preprocessor_weights"]
    with open(bundle_path, "rb") as f:
        f.seek(data_start + sec["offset"])
        pp_data = f.read(sec["size"])

    idx_len = struct.unpack("<I", pp_data[:4])[0]
    pp_index = json.loads(pp_data[4 : 4 + idx_len])
    blob = pp_data[4 + idx_len :]

    def load(key):
        info = pp_index[key]
        count = int(np.prod(info["shape"]))
        return np.frombuffer(
            blob, dtype=np.float32, count=count, offset=info["offset"]
        ).reshape(info["shape"])

    return {k: load(k) for k in pp_index}


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def main():
    import torch
    from safetensors import safe_open
    import glob

    pp = load_bundle_pp_weights(BUNDLE_PATH)
    for k, v in pp.items():
        print(f"  {k}: {v.shape}")

    # Load HF reference weights
    transformer_dir = f"{MODEL_DIR}/transformer"
    files = sorted(glob.glob(f"{transformer_dir}/*.safetensors"))
    readers = [safe_open(f, framework="pt") for f in files]

    def get_tensor(key):
        for r in readers:
            if key in r.keys():
                return r.get_tensor(key)
        return None

    # ---- 1. Timestep embedding ----
    print("\n=== 1. Timestep embedding (t=1000) ===")
    from diffusers.models.embeddings import Timesteps, TimestepEmbedding

    time_proj_hf = Timesteps(256, True, 0)
    time_emb_hf = TimestepEmbedding(256, 1536)
    time_emb_hf.linear_1.weight.data = get_tensor(
        "condition_embedder.time_embedder.linear_1.weight"
    )
    time_emb_hf.linear_1.bias.data = get_tensor(
        "condition_embedder.time_embedder.linear_1.bias"
    )
    time_emb_hf.linear_2.weight.data = get_tensor(
        "condition_embedder.time_embedder.linear_2.weight"
    )
    time_emb_hf.linear_2.bias.data = get_tensor(
        "condition_embedder.time_embedder.linear_2.bias"
    )

    t_tensor = torch.tensor([1000.0])
    sinusoidal_hf = time_proj_hf(t_tensor)
    time_embed_hf = time_emb_hf(sinusoidal_hf).detach().numpy()

    # Our computation
    freq_dim = 256
    half = freq_dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    args = 1000.0 * freqs
    sinusoidal = np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32).reshape(1, freq_dim)

    hidden1 = sinusoidal @ pp["condition_embedder.time_embedding.0.weight"] + pp["condition_embedder.time_embedding.0.bias"]
    hidden1 = silu(hidden1)
    time_embed_ours = hidden1 @ pp["condition_embedder.time_embedding.2.weight"] + pp["condition_embedder.time_embedding.2.bias"]

    diff = np.abs(time_embed_ours.flatten() - time_embed_hf.flatten())
    print(f"  time_embed: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    print(f"  PASS" if diff.max() < 0.001 else f"  FAIL")

    # ---- 2. Text projection ----
    print("\n=== 2. Text projection ===")
    from diffusers.models.embeddings import PixArtAlphaTextProjection

    text_proj_hf = PixArtAlphaTextProjection(4096, 1536)
    text_proj_hf.linear_1.weight.data = get_tensor("condition_embedder.text_embedder.linear_1.weight")
    text_proj_hf.linear_1.bias.data = get_tensor("condition_embedder.text_embedder.linear_1.bias")
    text_proj_hf.linear_2.weight.data = get_tensor("condition_embedder.text_embedder.linear_2.weight")
    text_proj_hf.linear_2.bias.data = get_tensor("condition_embedder.text_embedder.linear_2.bias")

    torch.manual_seed(42)
    test_text = torch.randn(1, 512, 4096)
    hf_out = text_proj_hf(test_text).detach().numpy().reshape(512, 1536)

    test_np = test_text.numpy().reshape(512, 4096)
    our_out = test_np @ pp["condition_embedder.text_embedding.weight"] + pp["condition_embedder.text_embedding.bias"]
    our_out = silu(our_out)
    our_out = our_out @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]

    diff = np.abs(our_out.flatten() - hf_out.flatten())
    print(f"  text_proj: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    print(f"  PASS" if diff.max() < 0.01 else f"  FAIL")

    # ---- 3. Patch embedding ----
    print("\n=== 3. Patch embedding ===")
    conv_w = get_tensor("patch_embedding.weight")
    conv_b = get_tensor("patch_embedding.bias")
    conv = torch.nn.Conv3d(16, 1536, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=True)
    conv.weight.data = conv_w
    conv.bias.data = conv_b

    torch.manual_seed(42)
    test_latent = torch.randn(1, 16, 2, 16, 16)
    hf_patch_out = conv(test_latent)  # [1, 1536, 2, 8, 8]
    hf_flat = hf_patch_out.permute(0, 2, 3, 4, 1).reshape(-1, 1536).detach().numpy()

    # Our: patchify then matmul
    lat_np = test_latent.numpy().reshape(16, 2, 16, 16)
    c, t, h, w = 16, 2, 16, 16
    pt, ph, pw = 1, 2, 2
    patches = []
    for ti in range(t // pt):
        for hi in range(h // ph):
            for wi in range(w // pw):
                patch = []
                for ci in range(c):
                    for pti in range(pt):
                        for phi_ in range(ph):
                            for pwi in range(pw):
                                patch.append(lat_np[ci, ti * pt + pti, hi * ph + phi_, wi * pw + pwi])
                patches.append(patch)
    patches_np = np.array(patches, dtype=np.float32)
    our_flat = patches_np @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

    diff = np.abs(our_flat.flatten() - hf_flat.flatten())
    print(f"  patch_embed: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    print(f"  PASS" if diff.max() < 0.01 else f"  FAIL")

    # ---- 4. T5 + denoiser step via Python DiffusionRunner ----
    print("\n=== 4. Full denoiser step (Python TRT runner) ===")
    sys.path.insert(0, ".")
    from trtf_build.diffusion_runner import DiffusionRunner

    runner = DiffusionRunner(BUNDLE_PATH)

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/tokenizer")
    tokens = tokenizer(
        "A cat walking in the garden",
        return_tensors="np",
        padding="max_length",
        max_length=512,
        truncation=True,
    )
    input_ids = tokens["input_ids"].astype(np.int32)
    print(f"  Tokens (first 10): {input_ids[0, :10]}")

    # T5 encode
    text_emb = runner.encode_text(input_ids)
    print(f"  T5 output: shape={text_emb.shape}, mean={text_emb.mean():.6f}, std={text_emb.std():.6f}")

    # Prepare denoiser inputs using our preprocessor
    text_proj_np = text_emb.reshape(512, 4096) @ pp["condition_embedder.text_embedding.weight"] + pp["condition_embedder.text_embedding.bias"]
    text_proj_np = silu(text_proj_np)
    text_proj_np = text_proj_np @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]
    print(f"  Text projected: shape={text_proj_np.shape}, mean={text_proj_np.mean():.6f}")

    # Timestep embedding
    te_silu = silu(time_embed_ours)
    temb_6d = te_silu @ pp["condition_embedder.time_proj.weight"] + pp["condition_embedder.time_proj.bias"]
    print(f"  temb_6d: shape={temb_6d.shape}, mean={temb_6d.mean():.6f}")

    # Random latents (same seed as C++)
    rng = np.random.default_rng(42)
    latents = rng.standard_normal((1, 16, 2, 16, 16)).astype(np.float32)

    # Patchify
    patches = runner._patchify(latents, [1, 2, 2])
    hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]
    print(f"  Hidden: shape={hidden.shape}, mean={hidden.mean():.6f}")

    # 3D RoPE
    rope_cos, rope_sin = runner._compute_3d_rope(2, 8, 8, 128)

    # Run denoiser
    output = runner._run_engine(
        "denoiser",
        {
            "hidden_states": hidden,
            "timestep_embedding": temb_6d.reshape(1, 9216),
            "time_embed": time_embed_ours.reshape(1, 1536),
            "encoder_hidden_states": text_proj_np,
            "rotary_cos": rope_cos,
            "rotary_sin": rope_sin,
        },
    )
    denoiser_out = output["output"]
    print(f"  Denoiser output: shape={denoiser_out.shape}")
    print(f"    mean={denoiser_out.mean():.6f}, std={denoiser_out.std():.6f}")
    print(f"    min={denoiser_out.min():.6f}, max={denoiser_out.max():.6f}")
    print(f"    all zeros? {np.allclose(denoiser_out, 0, atol=1e-6)}")
    print(f"    sample: {denoiser_out[0, :5]}")

    # ---- 5. Full 30-step denoise + decode ----
    print("\n=== 5. Full 30-step pipeline (Python) ===")
    video = runner.generate(
        input_ids,
        num_inference_steps=30,
        guidance_scale=5.0,
        video_num_frames=5,
        video_height=128,
        video_width=128,
        seed=42,
    )
    print(f"  Video: shape={video.shape}, mean={video.mean():.4f}, std={video.std():.4f}")
    print(f"  min={video.min():.4f}, max={video.max():.4f}")

    # Save Python-generated frames for comparison
    from PIL import Image
    for t_idx in range(video.shape[2]):
        frame = video[0, :, t_idx]  # [3, H, W]
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frame = frame.transpose(1, 2, 0)  # [H, W, 3]
        Image.fromarray(frame).save(f"/tmp/trtf_frames_python/frame_{t_idx:04d}.png")

    print(f"  Saved Python frames to /tmp/trtf_frames_python/")


if __name__ == "__main__":
    main()
