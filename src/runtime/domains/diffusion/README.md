# Diffusion Backends

Diffusion model runtime pipelines and shared base utilities.

Key files:
- `diffusion_backend.*`: abstract diffusion backend interface + factory entry.
- `diffusion_backend_base.cpp`: shared text/conditioning/denoiser plumbing.
- `wan_diffusion_backend.*`: Wan video diffusion implementation.
- `flux_diffusion_backend.*`: FLUX diffusion implementation.
- `z_image_diffusion_backend.*`: Z-Image diffusion implementation.

How to understand:
1. Start at each backend `generate_video` method.
2. Follow conditioning + scheduler/denoise loop helpers.
3. Inspect VAE decode/patchify helpers for output-frame reconstruction details.
