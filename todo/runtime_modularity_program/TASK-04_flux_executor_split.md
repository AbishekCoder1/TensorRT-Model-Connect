# TASK-04: FLUX Executor Split

## Objective

Make `FluxDiffusionBackend` coverage-friendly by separating planning, tensor mapping, executor shells, and result assembly.

## Own These Files

- `src/runtime/trt/diffusion/flux_diffusion_backend.cpp`
- `src/runtime/trt/diffusion/flux_diffusion_backend.h`
- `src/runtime/trt/diffusion/diffusion_generation_plan.h`
- `src/runtime/trt/diffusion/diffusion_denoising_step_seam.h`
- new FLUX-specific seam headers if needed
- `tests/cpp/test_diffusion_generation_plan.cpp`
- `tests/cpp/test_diffusion_denoising_step_seam.cpp`
- any new FLUX-specific harness tests

## Do Not Edit

- `wan_*`
- `z_image_*`
- shared coverage tooling

## Deliverables

1. Pull timestep embedding, rope, patchify/unpatchify, and output assembly out of long executor methods.
2. Introduce a thin denoiser/VAE executor shell abstraction where failure injection is possible.
3. Add unit or harness tests for scheduler-step behavior and executor error paths.

## Acceptance Criteria

- `generate_video()` reads as plan -> execute -> assemble
- remaining heavy TensorRT code is isolated in small executor shells
- direct tests cover success and failure branches

## Required Validation

- targeted diffusion `ctest`
- full `ctest --output-on-failure`
- CCM gate
