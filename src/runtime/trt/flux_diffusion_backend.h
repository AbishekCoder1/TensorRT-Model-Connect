#pragma once

#include "runtime/trt/diffusion_backend.h"

namespace trtf {

#if TRTF_HAS_TRT

/// FLUX diffusion backend: 2D RoPE, flow-match Euler, AutoencoderKL VAE.
class FluxDiffusionBackend final : public DiffusionBackendBase {
public:
    FluxDiffusionBackend(
        std::vector<DiffusionEngine> text_encoders,
        DiffusionEngine denoiser,
        DiffusionEngine vae_decoder,
        DiffusionConfig config);

    ~FluxDiffusionBackend() override = default;

    void set_preprocessor_weights(PreprocessorWeights weights) override;
    void set_clip_tokenizer(std::unique_ptr<ITokenizer> tok) override;
    void set_prompt(std::string prompt) override;

    VideoResult generate_video(
        const std::vector<int32_t>& input_ids,
        int32_t num_inference_steps = -1,
        float guidance_scale = -1.0F) override;

private:
    void compute_flux_timestep_embedding(
        float timestep, float guidance,
        const std::vector<float>& pooled_text,
        std::vector<float>& temb) const;

    void compute_flux_rope(
        int32_t h_patches, int32_t w_patches, int32_t text_seq_len,
        std::vector<float>& cos_out,
        std::vector<float>& sin_out) const;

    bool run_clip_encoder(
        const std::vector<int32_t>& input_ids,
        std::vector<float>& pooled_output,
        std::string& error);

    bool run_flux_denoiser(
        const std::vector<float>& hidden,
        const std::vector<float>& encoder_hidden,
        const std::vector<float>& temb,
        const std::vector<float>& cos_vals,
        const std::vector<float>& sin_vals,
        std::vector<float>& output,
        std::string& error);

    /// Run T5 encoder at the given text_encoder index.
    bool run_t5_encoder_at(
        int32_t encoder_idx,
        const std::vector<int32_t>& input_ids,
        std::vector<float>& text_embeddings,
        std::string& error);

    /// Parse FLUX-specific preprocessor weights (x_embedder, context_embedder, etc.)
    void parse_flux_preprocessor_weights(const PreprocessorWeights& base_weights);

    // FLUX-specific buffers
    CudaBuffer mD_FluxHidden;
    CudaBuffer mD_FluxEncoder;
    CudaBuffer mD_FluxTemb;
    CudaBuffer mD_FluxCos;
    CudaBuffer mD_FluxSin;
    CudaBuffer mD_FluxOutput;

    // CLIP encoder buffers
    CudaBuffer mD_ClipInputIds;
    CudaBuffer mD_ClipTextEmb;
    CudaBuffer mD_ClipPooled;

    // FLUX preprocessor weights
    std::vector<float> mFluxXEmbedW;    // x_embedder.weight [in_ch, dit_dim]
    std::vector<float> mFluxXEmbedB;    // x_embedder.bias [dit_dim]
    std::vector<float> mFluxCtxEmbedW;  // context_embedder.weight [t5_dim, dit_dim]
    std::vector<float> mFluxCtxEmbedB;  // context_embedder.bias [dit_dim]
    bool mFluxWeightsLoaded{false};

    int32_t mNumImgTokens{0};
    int32_t mHLatent{0};
    int32_t mWLatent{0};

    // Dual tokenizer support: CLIP (BPE) for CLIP encoder, T5 (sentencepiece) for T5 encoder
    std::unique_ptr<ITokenizer> mClipTokenizer;
    int32_t mClipEosTokenId{-1};
    int32_t mClipPadTokenId{0};
    std::string mRawPrompt;
};

#endif // TRTF_HAS_TRT

} // namespace trtf
