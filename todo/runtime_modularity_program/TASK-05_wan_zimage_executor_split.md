# TASK-05: Wan And Z-Image Executor Split

## Objective

Finish the executor-layer decomposition for Wan and Z-Image diffusion runtime code.

## Own These Files

- `src/runtime/trt/diffusion/wan_diffusion_backend.cpp`
- `src/runtime/trt/diffusion/wan_diffusion_backend.h`
- `src/runtime/trt/diffusion/wan_generation_conditioning.h`
- `src/runtime/trt/diffusion/z_image_diffusion_backend.cpp`
- `src/runtime/trt/diffusion/z_image_diffusion_backend.h`
- any new Wan/Z-Image seam headers
- `tests/cpp/test_wan_generation_conditioning.cpp`
- any new Wan/Z-Image harness tests

## Do Not Edit

- `flux_*`
- shared coverage tooling

## Deliverables

1. Separate conditioning, scheduler updates, latent initialization, and result assembly from engine-bound execution.
2. Add fakeable executor seams for denoiser and VAE work where failure-path coverage is currently impractical.
3. Add tests for both nominal and defensive branches.

## Acceptance Criteria

- Wan and Z-Image backends are mostly orchestration
- conditioning and scheduler logic is directly unit-tested
- executor failure paths are reachable in tests

## Required Validation

- targeted diffusion `ctest`
- full `ctest --output-on-failure`
- CCM gate
