#pragma once

#include "runtime/trt/diffusion/diffusion_backend.h"

namespace trtf {

#if TRTF_HAS_TRT

/// Wan2.1 diffusion backend: 3D RoPE, flow-match Euler, causal 3D VAE.
class WanDiffusionBackend final : public DiffusionBackendBase {
public:
    WanDiffusionBackend(
        std::vector<DiffusionEngine> text_encoders,
        DiffusionEngine denoiser,
        DiffusionEngine vae_decoder,
        DiffusionConfig config);

    ~WanDiffusionBackend() override = default;

    VideoResult generate_video(
        const std::vector<int32_t>& input_ids,
        int32_t num_inference_steps = -1,
        float guidance_scale = -1.0F) override;

private:
    void compute_timestep_embedding(float timestep,
                                    std::vector<float>& temb_6d,
                                    std::vector<float>& time_embed) const;

    void project_text(const std::vector<float>& in, int32_t seq_len,
                      std::vector<float>& out) const;

    void patchify(const std::vector<float>& latents,
                  int32_t c, int32_t t, int32_t h, int32_t w,
                  std::vector<float>& patches) const;

    void unpatchify(const std::vector<float>& patches,
                    int32_t c, int32_t t, int32_t h, int32_t w,
                    std::vector<float>& output) const;

    void compute_3d_rope(int32_t nt, int32_t nh, int32_t nw,
                         std::vector<float>& cos_out,
                         std::vector<float>& sin_out) const;

    bool decode_vae_native(const std::vector<float>& latents,
                           int32_t c, int32_t t, int32_t h, int32_t w,
                           VideoResult& result, std::string& error);

    void init_vae_buffers();

    // Native VAE decode state
    CudaBuffer mD_VaeInput;
    CudaBuffer mD_VaeOutput;
    std::vector<CudaBuffer> mD_VaeCacheIn;
    std::vector<CudaBuffer> mD_VaeCacheOut;
    std::vector<std::size_t> mVaeCacheSizes;
    int32_t mVaeOutputT{1};
};

#endif // TRTF_HAS_TRT

} // namespace trtf
