#!/usr/bin/env python3
"""Component-by-component comparison: HF diffusers vs TRT bundle.

Runs one denoising step with identical inputs and compares each component.
"""
import struct, json, sys, os
import numpy as np
import torch

BUNDLE = "/mnt/storage/trt-transformers/engines/wan21-t2v-1.3b.trtfb"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = "A cat walking in the garden"


def load_pp_weights(bundle_path):
    with open(bundle_path, "rb") as f:
        f.read(8)
        jl = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(jl))
        ds = 16 + jl
    sec = hdr["sections"]["preprocessor_weights"]
    with open(bundle_path, "rb") as f:
        f.seek(ds + sec["offset"])
        ppd = f.read(sec["size"])
    il = struct.unpack("<I", ppd[:4])[0]
    ppx = json.loads(ppd[4 : 4 + il])
    blob = ppd[4 + il :]
    def load(k):
        i = ppx[k]
        c = int(np.prod(i["shape"]))
        return np.frombuffer(blob, np.float32, c, i["offset"]).reshape(i["shape"])
    return {k: load(k) for k in ppx}


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def main():
    from diffusers import WanPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    pp = load_pp_weights(BUNDLE)
    print("Loaded preprocessor weights from bundle")

    # Load HF pipeline
    print("Loading HF pipeline...", file=sys.stderr)
    pipe = WanPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    # Load TRT runner
    from trtf_build.diffusion_runner import DiffusionRunner
    runner = DiffusionRunner(BUNDLE)

    # === 1. Tokenize ===
    tokens = pipe.tokenizer(
        PROMPT, return_tensors="pt", padding="max_length",
        max_length=512, truncation=True,
    )
    input_ids = tokens.input_ids  # [1, 512]
    attn_mask = tokens.attention_mask
    print(f"\n1. Tokens: {input_ids[0, :8].tolist()}")

    # === 2. T5 text encoding ===
    with torch.no_grad():
        hf_text = pipe.text_encoder(input_ids, attention_mask=attn_mask)[0]
    trt_text = runner.encode_text(input_ids.numpy().astype(np.int32))

    diff = np.abs(trt_text.flatten() - hf_text.numpy().flatten())
    print(f"\n2. T5 encoding:")
    print(f"   HF:  mean={hf_text.mean():.6f}, std={hf_text.std():.6f}")
    print(f"   TRT: mean={trt_text.mean():.6f}, std={trt_text.std():.6f}")
    print(f"   diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # === 3. Text projection ===
    with torch.no_grad():
        hf_text_proj = pipe.transformer.condition_embedder.text_embedder(hf_text)

    # Our text projection
    te = trt_text.reshape(512, 4096)
    our_tp = te @ pp["condition_embedder.text_embedding.weight"] + pp["condition_embedder.text_embedding.bias"]
    our_tp = gelu_tanh(our_tp)
    our_tp = our_tp @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]

    diff = np.abs(our_tp.flatten() - hf_text_proj.numpy().reshape(-1))
    print(f"\n3. Text projection:")
    print(f"   HF:  mean={hf_text_proj.mean():.6f}")
    print(f"   Ours: mean={our_tp.mean():.6f}")
    print(f"   diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # === 4. Timestep embedding ===
    with torch.no_grad():
        hf_sinusoidal = pipe.transformer.condition_embedder.timesteps_proj(torch.tensor([1000.0]))
        hf_time_embed = pipe.transformer.condition_embedder.time_embedder(hf_sinusoidal)
        hf_temb = pipe.transformer.condition_embedder.time_proj(
            pipe.transformer.condition_embedder.act_fn(hf_time_embed)
        )

    # Our computation
    half = 128
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    our_sin = np.concatenate([np.cos(1000.0 * freqs), np.sin(1000.0 * freqs)]).astype(np.float32).reshape(1, 256)
    h = silu(our_sin @ pp["condition_embedder.time_embedding.0.weight"] + pp["condition_embedder.time_embedding.0.bias"])
    our_te = h @ pp["condition_embedder.time_embedding.2.weight"] + pp["condition_embedder.time_embedding.2.bias"]
    our_temb = silu(our_te) @ pp["condition_embedder.time_proj.weight"] + pp["condition_embedder.time_proj.bias"]

    diff_te = np.abs(our_te.flatten() - hf_time_embed.numpy().flatten())
    diff_temb = np.abs(our_temb.flatten() - hf_temb.numpy().flatten())
    print(f"\n4. Timestep embedding (t=1000):")
    print(f"   time_embed: HF mean={hf_time_embed.mean():.6f}, Ours={our_te.mean():.6f}, max_diff={diff_te.max():.6f}")
    print(f"   temb_6d:    HF mean={hf_temb.mean():.6f}, Ours={our_temb.mean():.6f}, max_diff={diff_temb.max():.6f}")

    # Check: does HF use temb as [1, 6*dim] or differently?
    print(f"   HF temb shape: {hf_temb.shape}")
    print(f"   HF time_embed shape: {hf_time_embed.shape}")

    # === 5. Run one denoiser step: HF vs TRT ===
    print(f"\n5. Single denoiser step:")

    # Prepare identical inputs
    rng = np.random.default_rng(42)
    latents = rng.standard_normal((1, 16, 2, 16, 16)).astype(np.float32)
    patches = runner._patchify(latents, [1, 2, 2])
    hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]
    rope_cos, rope_sin = runner._compute_3d_rope(2, 8, 8, 128)

    # TRT denoiser
    trt_out = runner._run_engine("denoiser", {
        "hidden_states": hidden,
        "timestep_embedding": our_temb.reshape(1, 9216),
        "time_embed": our_te.reshape(1, 1536),
        "encoder_hidden_states": our_tp,
        "rotary_cos": rope_cos,
        "rotary_sin": rope_sin,
    })["output"]
    print(f"   TRT denoiser: shape={trt_out.shape}, mean={trt_out.mean():.6f}, std={trt_out.std():.6f}")

    # HF denoiser - need to call transformer forward manually
    hidden_t = torch.from_numpy(hidden).unsqueeze(0)  # [1, num_patches, dim]
    temb_t = hf_temb.unsqueeze(0) if hf_temb.dim() == 1 else hf_temb  # [1, 6*dim]
    te_t = hf_time_embed  # [1, dim]
    text_t = hf_text_proj  # [1, 512, dim]
    cos_t = torch.from_numpy(rope_cos).unsqueeze(0).unsqueeze(2)  # [1, patches, 1, head_dim]
    sin_t = torch.from_numpy(rope_sin).unsqueeze(0).unsqueeze(2)

    print(f"   HF inputs: hidden={hidden_t.shape}, temb={temb_t.shape}, time_embed={te_t.shape}")
    print(f"              text={text_t.shape}, cos={cos_t.shape}")

    # The HF transformer.forward expects different input format
    # Let's check what forward() expects
    import inspect
    sig = inspect.signature(pipe.transformer.forward)
    print(f"   HF forward params: {list(sig.parameters.keys())}")

    # Run HF transformer forward
    with torch.no_grad():
        try:
            hf_out = pipe.transformer(
                hidden_states=hidden_t,
                timestep=temb_t,
                encoder_hidden_states=text_t,
                rotary_emb=(cos_t, sin_t),
            )
            if hasattr(hf_out, 'sample'):
                hf_denoiser = hf_out.sample.numpy()
            else:
                hf_denoiser = hf_out[0].numpy() if isinstance(hf_out, tuple) else hf_out.numpy()
            print(f"   HF denoiser: shape={hf_denoiser.shape}, mean={hf_denoiser.mean():.6f}, std={hf_denoiser.std():.6f}")

            diff = np.abs(trt_out.flatten() - hf_denoiser.flatten()[:trt_out.size])
            print(f"   diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
        except Exception as e:
            print(f"   HF forward failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
