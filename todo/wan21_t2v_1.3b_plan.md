# Plan: Wan2.1 Text-to-Video (T2V) 1.3B Support

## Context

Add support for the [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) text-to-video diffusion model. This is a fundamentally new modality — the first non-autoregressive pipeline in the codebase. Instead of generating tokens one-at-a-time, it runs an iterative denoising loop (50 steps) across three neural networks to produce video frames.

**User requirements**: TRT-ify all 3 components, implement both Python and C++ paths, support CFG with batch=2.

**Reference backend**: `diffusers.WanPipeline` for diff testing.

## Wan2.1 Architecture Summary

| Component | Class | Size | Runs | Purpose |
|-----------|-------|------|------|---------|
| T5 Text Encoder | `UMT5EncoderModel` | 22.7 GB | 1x | `token_ids [1, 512]` → `text_embeds [1, 512, 4096]` |
| DiT Transformer | `WanTransformer3DModel` | 5.7 GB | 50x | `(noisy_latent, timestep, text_embeds)` → `noise_pred` |
| VAE Decoder | `AutoencoderKLWan` | 508 MB | 1x | `latent [1, 16, T/4, H/8, W/8]` → `video [1, 3, T, H, W]` |

**Pipeline loop**: T5 encode → initialize noise → 50× DiT denoise (with CFG batch=2) → VAE decode → video frames

**Scheduler**: UniPCMultistepScheduler with flow matching (`prediction_type="flow_prediction"`, `flow_shift=3.0`)

---

## Phase 0: Graph Ops Foundation

Add new atomic TRT operations to `graph_ops.py`:

**DiT ops:**

| Op | Purpose | Notes |
|----|---------|-------|
| `add_3d_rope_dit()` | DiT 3D positional encoding (T/H/W) | Different from existing 1D/2D RoPE |
| `add_adaptive_layernorm()` | AdaLN: `gate * (scale * norm(x) + shift)` | Core of DiT block |
| `add_timestep_embedding()` | Sinusoidal → MLP → 6 modulation params/layer | DiT conditioning |
| `add_cross_attention()` | Q from video, KV from text (no RoPE on KV) | DiT text conditioning |
| `add_3d_patch_embed()` | Conv3d(16, 1536, [1,2,2]) + flatten to sequence | DiT input |
| `add_3d_unpatchify()` | Reshape patches back to 3D volume | DiT output |

**VAE ops:**

| Op | Purpose | Notes |
|----|---------|-------|
| `add_group_norm()` | GroupNorm (32 groups) | VAE uses GroupNorm throughout |
| `add_causal_conv3d()` | 3D conv with causal temporal padding + cache I/O | Core VAE primitive, see Phase 3 |
| `add_temporal_upsample()` | Nearest-neighbor upsample along temporal dim | VAE temporal expansion |
| `add_spatial_upsample()` | Nearest-neighbor upsample along H/W | VAE spatial expansion |

**New blocks in `graph_blocks.py`:**

| Block | Purpose | Notes |
|-------|---------|-------|
| `add_dit_block()` | AdaLN self-attn + cross-attn + AdaLN FFN | One DiT transformer layer |
| `add_gated_gelu_mlp()` | `gate * gelu(up) * down` | T5 encoder + DiT FFN |
| `add_vae_resblock()` | conv1 + conv2 + shortcut with cache I/O | VAE ResidualBlock3D |
| `add_vae_upblock()` | N ResBlocks + optional temporal/spatial upsample | VAE UpBlock3D |

**Modify**: `trtf_build/trtf_build/graph_ops.py`, `trtf_build/trtf_build/graph_blocks.py`

**Test**: Unit tests for each op in `tests/builder/test_graph_ops.py` (CPU shapes + `@pytest.mark.trt` GPU)

---

## Phase 1: T5 Encoder Engine Builder

Build TRT engine for `UMT5EncoderModel` (encoder-only, no KV cache, single forward pass).

- Config: `d_model=4096, num_heads=64, d_kv=64, d_ff=10240, num_layers=24, vocab=256384`
- Uses: RMSNorm (T5LayerNorm), gated-GELU MLP, T5 relative position bias
- Engine I/O: `input_ids [1, 512]` int32 → `text_embeddings [1, 512, 4096]` fp16

**Create**: `trtf_build/trtf_build/wan_t5_encoder_builder.py`
**New block**: `add_gated_gelu_mlp()` in `graph_blocks.py` (like SwiGLU but `gate * gelu(up)`)
**Test**: `tools/diff_t5.py` — compare vs `transformers.UMT5EncoderModel`

---

## Phase 2: DiT Engine Builder

Build TRT engine for `WanTransformer3DModel`.

- Config: `dim=1536, heads=12, head_dim=128, layers=30, ffn_dim=8960, text_dim=4096, patch=[1,2,2]`
- Block: AdaLN self-attn (3D RoPE, qk_norm) → cross-attn (text) → AdaLN FFN (GELU)
- Engine I/O (CFG batch=2):
  - In: `noisy_latent [2, 16, T/4, H/2, W/2]`, `timestep [2]`, `text_embeds [2, 512, 4096]`
  - Out: `noise_pred [2, 16, T/4, H/2, W/2]`
- Patch embedding flattens 3D→2D sequence, then standard multi-head attention

**Create**: `trtf_build/trtf_build/wan_dit_builder.py`
**Test**: `tools/diff_dit.py` — single denoising step vs `diffusers.WanTransformer3DModel`

---

## Phase 3: VAE Decoder Engine Builder

Build TRT engine for `AutoencoderKLWan` decoder using manual graph_ops (no ONNX).

- Config: `base_dim=96, dim_mult=[1,2,4,4], z_dim=16, temporal_downsample=[F,T,T]`
- Channel progression: `z_dim=16` → `base_dim * dim_mult[-1]=384` → ... → `base_dim=96` → `3` (RGB)
- Spatial 8× upsample, temporal 4× upsample
- No attention layers (`attn_scales=[]`)

### Frame-by-frame with causal cache (like KV cache pattern)

The VAE decoder uses `WanCausalConv3d` throughout — 3D convolutions with causal temporal padding. The HF reference (`_decode()`) always processes **one temporal frame at a time**, maintaining `feat_cache` between frames. We mirror this exactly.

**Engine I/O** (single-frame step):
- Inputs:
  - `latent_frame [1, 16, 1, H/8, W/8]` — one temporal frame of the latent
  - `cache_0..N [1, C_i, CACHE_T, H_i, W_i]` — causal conv cache per layer (CACHE_T=2)
  - `is_first_frame [1]` — flag: first frame uses zero-padded cache
- Outputs:
  - `video_frames [1, 3, T_out, H, W]` — decoded frames for this temporal step (T_out=1 for most frames, T_out=4 for first due to temporal upsample factor)
  - `present_cache_0..N [1, C_i, CACHE_T, H_i, W_i]` — updated cache

**Run T_latent times** (21 times for 81-frame video), feeding each step's `present_cache` as the next step's `cache` input. Identical to the autoregressive decoder's KV cache loop.

### Cache tensor inventory (~30 pairs)

Each `WanCausalConv3d` with kernel_size > 1 needs a cache pair. Kernel=1 (shortcut convs) needs no cache.

| Component | ResBlocks | Cached convs/block | Subtotal |
|-----------|-----------|---------------------|----------|
| `conv_in` (k=3) | — | 1 | 1 |
| `mid_block` (2 ResBlocks) | 2 | 2 | 4 |
| `up_block_0` (3 ResBlocks + upsample) | 3 | 2 + 1 time_conv | 7 |
| `up_block_1` (3 ResBlocks + upsample) | 3 | 2 + 1 time_conv | 7 |
| `up_block_2` (3 ResBlocks + upsample) | 3 | 2 + 1 time_conv | 7 |
| `up_block_3` (3 ResBlocks, no upsample) | 3 | 2 | 6 |
| `conv_out` (k=3) | — | 1 | 1 |
| **Total** | | | **~33** |

Exact count to be verified from HF source during implementation.

### VAE graph_blocks composition

```
add_vae_resblock(network, x, weights, prefix, in_dim, out_dim, cache_in, cache_out):
    # norm1 → silu → causal_conv3d(k=3, cache) → norm2 → silu → causal_conv3d(k=3, cache)
    # + shortcut (conv1x1 if in_dim != out_dim, else identity)

add_vae_upblock(network, x, weights, prefix, in_dim, out_dim, num_res_blocks,
                upsample_mode, caches_in, caches_out):
    # (num_res_blocks+1) × add_vae_resblock
    # + optional temporal_upsample / spatial_upsample via IResizeLayer
```

### C++ runtime: VAE decode loop

The C++ `DiffusionT2VBackend` runs the VAE frame-by-frame, reusing the same `DeviceResources`-style persistent buffers:

```
run_vae_decode(latents):
  caches = allocate_zero_caches(33 pairs)
  video_frames = []
  for t in range(T_latent):  // 21 iterations
    frame = latents[:, :, t:t+1, :, :]
    output, caches = run_vae_engine(frame, caches, is_first=(t==0))
    video_frames.append(output)
  return concat(video_frames, dim=temporal)
```

**Create**: `trtf_build/trtf_build/wan_vae_builder.py`
**Test**: `tools/diff_vae.py` — compare frame-by-frame TRT decode vs `diffusers.AutoencoderKLWan.decode()` (must match to atol=0.01)

---

## Phase 4: Plugin + Bundle + Python Runner

### Family plugin
**Create**: `trtf_build/trtf_build/families/wan_t2v.py`
- `runtime_strategy = "diffusion_t2v"`
- Detects from `model_index.json` (`_class_name == "WanPipeline"`)
- Loads weights from 3 subdirs: `text_encoder/`, `transformer/`, `vae/`
- Builds 3 engines via Phase 1–3 builders
- New plugin methods: `build_text_encoder_engine()`, `build_vae_decoder_engine()`

### Bundle format (3 engines)
**Modify**: `trtf_build/trtf_build/engine_builder.py`
- Call `plugin.build_text_encoder_engine()` and `plugin.build_vae_decoder_engine()`
- New bundle sections: `text_encoder_plan`, `dit_plan` (replaces `engine_plan`), `vae_decoder_plan`
- Inject diffusion config: scheduler params, latent norm constants, video resolution

**Modify**: `trtf_build/trtf_build/families/base.py` — add optional protocol methods

### UniPC scheduler (Python)
**Create**: `trtf_build/trtf_build/unipc_scheduler.py`
- Flow matching with `flow_shift=3.0`, `solver_order=2`
- Port from `diffusers.UniPCMultistepScheduler`

### Python debug runner
**Create**: `trtf_build/trtf_build/diffusion_runner.py`
- `DiffusionT2VRunner` class: deserializes 3 TRT engines, runs full denoising loop
- CFG: stack batch=2 inputs, split after DiT, apply guidance_scale
- Returns video frames as numpy array

**Test**: `tools/diff_wan.py` — full pipeline vs `diffusers.WanPipeline` output

---

## Phase 5: C++ Runtime

### New files

| File | Purpose |
|------|---------|
| `src/runtime/trt/diffusion_common.h` | `DiffusionConfig`, `VideoResult` structs |
| `src/runtime/trt/diffusion_engine.h` | `TextEncoderEngine`, `DiTEngine`, `VaeDecoderEngine` structs (VAE includes ~33 cache tensor name pairs, analogous to KV cache names in `DecoderStepEngine`) |
| `src/runtime/trt/diffusion_backend.h/cpp` | `DiffusionT2VBackend` — main denoising loop |
| `src/runtime/trt/unipc_scheduler.h/cpp` | C++ UniPC scheduler (CPU math, runs between GPU steps) |

### Modified files

| File | Change |
|------|--------|
| `src/cabi/trtf_c.cpp` | New `create_diffusion_pipeline()` factory + dispatch for `"diffusion_t2v"` |
| `src/cabi/bundle_helpers.h/cpp` | Add `text_encoder_plan_data`, `dit_plan_data`, `vae_decoder_plan_data` to `BundleSections` |
| `src/cabi/fast_path_config.h/cpp` | Diffusion fields: `num_inference_steps`, `guidance_scale`, `flow_shift`, `video_dims`, `latents_mean/std` |
| `include/trtf/pipeline.h` | Add `VideoResult` struct, `generate_video()` method, `supports_video()` |
| `include/trtf/backend.h` | Add `generate_video()` to `IGenerationBackend` |
| `examples/trtf_cli.cpp` | Add `generate-video` subcommand |
| `CMakeLists.txt` | Add new source files |

### Memory management strategy

Sequential engine lifecycle for GPUs with <24GB VRAM:
1. Deserialize T5, run, **destroy** T5 engine → peak ~11.4GB (FP16)
2. Deserialize DiT, run 50 steps, **destroy** → peak ~3.2GB + text_embeds
3. Deserialize VAE, run, **destroy** → peak ~0.5GB + latents

For 24GB+ GPUs: hold all 3 engines simultaneously (~14.6GB FP16 total).

Pre-allocate persistent device buffers in `DiffusionT2VBackend`:
- `mTextEmbeddings` [1, 512, 4096] — 8MB, persists across DiT loop
- `mLatents` [2, 16, T/4, H/8, W/8] — ~17MB for CFG, reused each step
- `mNoisePred` — same shape as latents
- `mVaeCaches` — ~33 cache buffer pairs, device-resident across VAE frame loop
- `mVideoFrames` [1, 3, T, H, W] — ~75MB

### C++ full pipeline pseudocode
```
generate_video(prompt):
  // Phase 1: Text encoding (T5, 1x)
  token_ids = tokenizer.encode(prompt)
  text_embeds = run_t5(token_ids)
  null_embeds = run_t5(empty_ids)                    // for CFG

  // Phase 2: Denoising loop (DiT, 50x)
  latents = random_noise(shape)
  scheduler.set_timesteps(50)
  for t in scheduler.timesteps:
    batched_latents = stack(latents, latents)         // CFG batch=2
    batched_embeds = stack(text_embeds, null_embeds)
    noise_pred = run_dit(batched_latents, t, batched_embeds)
    cond, uncond = split(noise_pred)
    guided = uncond + guidance_scale * (cond - uncond)
    latents = scheduler.step(guided, t, latents)

  // Phase 3: VAE decode (frame-by-frame with cache, 21x)
  latents = denormalize(latents, mean, std)
  zero_fill(vae_caches)
  video_frames = []
  for t in range(T_latent):                          // 21 iterations
    frame = latents[:, :, t:t+1, :, :]
    decoded, vae_caches = run_vae(frame, vae_caches, is_first=(t==0))
    video_frames.append(decoded)
  return concat(video_frames)
```

---

## Phase 6: Diff Testing + E2E

**Create**:
- `tools/diff_wan.py` — full pipeline: TRT vs diffusers WanPipeline
- `tools/diff_dit.py` — per-step DiT comparison
- `tools/diff_t5.py` — T5 encoder comparison
- `tools/diff_vae.py` — VAE decoder comparison
- `tests/e2e/models/wan21-t2v-1.3b.json` — E2E manifest (reduced resolution: 240x416, 17 frames, 10 steps for fast CI)
- `tests/builder/test_wan_t2v.py` — builder unit tests

**Modify**: `tests/e2e/test_full_pipeline.py` — handle `diffusion_t2v` strategy (video output comparison)

---

## Phase 7: Video Output

- C++ frame writing via stb_image_write (PNG sequence, already in `third_party/stb`)
- Optional MP4 encoding via ffmpeg subprocess
- Post-processing: denormalize latents → VAE decode → clamp [0,1] → uint8 → write

---

## Dependency Graph

```
Phase 0 (graph ops)
  ├─→ Phase 1 (T5 builder)  ─┐
  ├─→ Phase 2 (DiT builder)  ├─→ Phase 4 (plugin + bundle + Python runner) → Phase 5 (C++ runtime) → Phase 6 (testing) → Phase 7 (video output)
  └─→ Phase 3 (VAE builder) ─┘
```

Phases 1, 2, 3 can run **in parallel** after Phase 0.

## Verification

After each phase:
1. **Phase 0**: `pytest tests/builder/test_graph_ops.py -v -m trt` — new ops pass
2. **Phase 1**: `python tools/diff_t5.py --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --atol 1e-3`
3. **Phase 2**: `python tools/diff_dit.py --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --atol 1e-3`
4. **Phase 3**: `python tools/diff_vae.py --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --atol 0.01`
5. **Phase 4**: `python tools/diff_wan.py --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --num-steps 10` (Python-only E2E)
6. **Phase 5**: `./build/trtf generate-video <bundle> --prompt "A cat on a beach" --output /tmp/test.mp4`
7. **Phase 6**: `pytest tests/e2e/ -v -k wan21 --engine-dir /mnt/storage/trt-transformers/engines`

Full regression: existing decoder/mamba/VL tests must still pass (no regressions).
