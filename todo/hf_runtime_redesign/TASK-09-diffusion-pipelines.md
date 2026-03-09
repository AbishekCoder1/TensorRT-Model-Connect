# TASK-09: FluxPipeline + WanPipeline + ZImagePipeline

## Status: blocked (needs TASK-01, TASK-03)
## Phase: 3 (Diffusion)
## Risk: medium — diffusion has family-specific preprocessing; must match HF output

## Goal

Implement one pipeline per diffusion model family, each composing TrtModule
instances with a shared IScheduler and family-specific preprocessor. This
replaces the current 3 monolithic backends (~140KB of code) with clean
compositions.

## Architecture

```
Shared across all diffusion:
  TrtModule (text encoder, denoiser, VAE — each a separate module)
  IScheduler (FlowMatchEuler)
  DeviceTensor (latents stay on GPU across denoising steps)

Family-specific (one struct per family):
  FluxPreprocessor   — 2D patchify, guidance embed, BN denorm, 4D RoPE
  WanPreprocessor    — 3D patchify, text projection MLP, causal VAE, 3D RoPE, CFG
  ZImagePreprocessor — 2D patchify, text projection, simpler variant
```

## FluxPipeline

```cpp
class FluxPipeline final : public IPipeline {
    std::vector<std::unique_ptr<TrtModule>> text_encoders_;  // 1-2
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    std::unique_ptr<IScheduler> scheduler_;
    FluxPreprocessor preprocessor_;
    FluxConfig config_;

    MediaResult generate_media(const std::string& prompt,
                               const MediaConfig& cfg) override {
        // 1. Tokenize + text encode (TrtModule forward)
        // 2. Init noise as DeviceTensor (stays on GPU)
        // 3. Pre-compute RoPE (constant, computed once)
        // 4. Denoising loop:
        //    - compute temb from timestep + guidance (preprocessor, on GPU)
        //    - patchify latents (preprocessor, on GPU)
        //    - denoiser_.forward_device({patches, text, temb, rope})
        //    - unpatchify velocity (preprocessor, on GPU)
        //    - scheduler_.step(latents, velocity) (on GPU, no D2H)
        // 5. VAE decode (TrtModule forward)
        // 6. Return image/frames
    }
};
```

## WanPipeline

Same structure but:
- 3D latents [B, C, T, H, W] instead of 2D
- Text projection MLP (CPU side via preprocessor weights)
- CFG via dual denoiser pass (conditional + unconditional)
- Causal VAE with frame-by-frame decode + conv cache

```cpp
class WanPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> t5_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;  // causal 3D VAE
    std::unique_ptr<IScheduler> scheduler_;
    WanPreprocessor preprocessor_;
    WanConfig config_;

    MediaResult generate_media(...) override {
        // Same overall flow, but:
        // - patchify is 3D: [B,C,T,H,W] → [num_patches, patch_dim]
        // - CFG: run denoiser twice (cond + uncond), combine
        // - VAE: frame-by-frame with conv cache management
    }
};
```

## Preprocessor structs

```cpp
struct FluxPreprocessor {
    // Weights loaded from bundle's preprocessor_weights section
    DeviceTensor patch_embed_w;  // [128, 6144] (or baked into denoiser)
    DeviceTensor context_embed_w;
    std::vector<float> time_emb_weights;   // timestep MLP
    std::vector<float> guidance_emb_weights;
    std::vector<float> vae_bn_mean, vae_bn_var;

    DeviceTensor compute_temb(float t, float guidance, cudaStream_t s);
    DeviceTensor patchify(const DeviceTensor& latents, cudaStream_t s);
    DeviceTensor unpatchify(const DeviceTensor& velocity, cudaStream_t s);
    DeviceTensor prepare_for_vae(const DeviceTensor& latents, cudaStream_t s);
    std::pair<DeviceTensor, DeviceTensor> compute_rope(int h, int w);
};

struct WanPreprocessor {
    std::vector<float> text_proj_weights;  // text projection MLP
    std::vector<float> time_emb_weights;

    DeviceTensor compute_temb(float t, cudaStream_t s);
    DeviceTensor project_text(const DeviceTensor& text_emb, cudaStream_t s);
    DeviceTensor patchify_3d(const DeviceTensor& latents, cudaStream_t s);
    DeviceTensor unpatchify_3d(const DeviceTensor& velocity, cudaStream_t s);
    std::pair<DeviceTensor, DeviceTensor> compute_3d_rope(int nt, int nh, int nw);
    // Causal VAE: frame-by-frame decode with conv cache
    MediaResult decode_causal_vae(TrtModule& vae, const DeviceTensor& latents,
                                   cudaStream_t stream);
};
```

## What it replaces

- `DiffusionBackendBase` (diffusion_backend_base.cpp — 15KB shared CPU helpers)
- `FluxDiffusionBackend` (flux_diffusion_backend.cpp — 52KB)
- `WanDiffusionBackend` (wan_diffusion_backend.cpp — 49KB)
- `ZImageDiffusionBackend` (z_image_diffusion_backend.cpp — 39KB)
- `DiffusionConfig`, `PreprocessorWeights`, `DiffusionEngine` structs
- Diffusion strategy builder
- Diffusion adapter/port/service wrappers

Total replaced: ~155KB → estimated ~20KB (preprocessor structs + pipeline classes)

## Files to create

- `src/runtime/pipelines/flux_pipeline.h` + `.cpp`
- `src/runtime/pipelines/wan_pipeline.h` + `.cpp`
- `src/runtime/pipelines/z_image_pipeline.h` + `.cpp`
- `src/runtime/pipelines/diffusion_preprocessor.h` — shared types
- `tests/cpp/test_flux_pipeline.cpp`
- `tests/cpp/test_wan_pipeline.cpp`

## Acceptance criteria

- [ ] FLUX.2-dev generates a cat image matching FP32 reference
- [ ] Wan2.1-T2V generates video frames
- [ ] E2E test `test_e2e[flux-2-dev]` passes
- [ ] Denoiser-only benchmark matches or beats current 179ms/step

## Dependencies

TASK-01 (TrtModule, DeviceTensor), TASK-03 (IScheduler), TASK-04 (Factory)
