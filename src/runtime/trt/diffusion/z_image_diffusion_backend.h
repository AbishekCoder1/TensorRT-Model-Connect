#pragma once

#include "runtime/trt/diffusion/diffusion_backend.h"

namespace trtf {

#if TRTF_HAS_TRT

/// Z-Image preprocessor weights (different key layout from Wan/FLUX).
struct ZImagePreprocessorWeights {
    std::vector<float> t_emb_0_weight, t_emb_0_bias;
    std::vector<float> t_emb_2_weight, t_emb_2_bias;
    std::vector<float> cap_norm_weight;
    std::vector<float> cap_proj_weight, cap_proj_bias;
    std::vector<float> x_embed_weight, x_embed_bias;
    std::vector<float> final_adaln_weight, final_adaln_bias;
    std::vector<float> final_linear_weight, final_linear_bias;
    std::vector<float> cap_pad_token, x_pad_token;
    int32_t patch_dim{0};
    bool valid{false};
};

/// Z-Image-Turbo diffusion backend: pure TRT inference for all components.
///
/// Runs three TRT engines:
///   1. Qwen3 text encoder (non-autoregressive, returns hidden_states[-2])
///   2. Z-Image DiT denoiser (unified attention, AdaLN, SwiGLU, 3-axis RoPE)
///   3. AutoencoderKL VAE decoder (2D CNN)
///
/// NO Python subprocess for inference. Only the HF tokenizer uses Python.
class ZImageDiffusionBackend final : public DiffusionBackendBase {
public:
    ZImageDiffusionBackend(
        std::vector<DiffusionEngine> text_encoders,
        DiffusionEngine denoiser,
        DiffusionEngine vae_decoder,
        DiffusionConfig config);

    ~ZImageDiffusionBackend() override = default;

    VideoResult generate_video(
        const std::vector<int32_t>& input_ids,
        int32_t num_inference_steps = -1,
        float guidance_scale = -1.0F) override;

    void set_prompt(std::string prompt) override { mPrompt = std::move(prompt); }

    bool supports_video() const override { return true; }

    /// Apply Qwen3 chat template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    std::string prepare_prompt(const std::string& prompt) const override;

    /// Override: parse Z-Image-specific preprocessor weights from raw bytes.
    void set_preprocessor_weights(PreprocessorWeights weights) override;

    /// Load Z-Image preprocessor weights from raw section data.
    void load_z_image_preprocessor_weights(const std::vector<char>& raw_data);

private:
    /// Project text embeddings from Qwen3 dim (2560) to DiT dim (3840).
    void project_caption(
        const std::vector<float>& text_embeddings,
        int32_t seq_len,
        std::vector<float>& projected) const;

    /// Compute timestep embedding: sinusoidal(256) -> MLP -> [1, 256].
    void compute_timestep_embedding(
        float timestep,
        std::vector<float>& temb) const;

    /// Compute 3-axis RoPE for unified sequence (noise + caption).
    /// @param cap_padded_len  Actual caption length padded to SEQ_MULTI_OF (32).
    ///                        Used for noise position start offset and caption positions.
    void compute_3d_rope(
        int32_t nh, int32_t nw, int32_t text_seq_len,
        int32_t cap_padded_len,
        std::vector<float>& cos_out,
        std::vector<float>& sin_out) const;

    /// Patchify latents: [C, H, W] -> [num_patches, patch_dim].
    void patchify_2d(
        const std::vector<float>& latents,
        int32_t c, int32_t h, int32_t w,
        std::vector<float>& patches) const;

    /// Unpatchify: [num_patches, patch_dim] -> [C, H, W].
    void unpatchify_2d(
        const std::vector<float>& patches,
        int32_t c, int32_t h, int32_t w,
        std::vector<float>& output) const;

    /// Run VAE decoder TRT engine.
    bool decode_vae_native(
        const std::vector<float>& latents,
        int32_t c, int32_t h_lat, int32_t w_lat,
        VideoResult& result, std::string& error);

    void init_vae_buffers();

    // Z-Image-specific buffers
    CudaBuffer mD_DiTHidden;
    CudaBuffer mD_DiTEncoder;
    CudaBuffer mD_DiTTemb;
    CudaBuffer mD_DiTCos;
    CudaBuffer mD_DiTSin;
    CudaBuffer mD_DiTOutput;

    CudaBuffer mD_VaeInput;
    CudaBuffer mD_VaeOutput;

    std::string mPrompt;

    ZImagePreprocessorWeights mZWeights;
};

#endif // TRTF_HAS_TRT

} // namespace trtf
