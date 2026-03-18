#pragma once
// Diffusion pipelines: FluxPipeline, WanPipeline, ZImagePipeline.
// Each owns TrtModules for text encoding, denoising, and VAE decode.
// Replaces old IDiffusionBackend delegation with direct TrtModule::forward().

#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/device_tensor.h"
#include "runtime/trt/diffusion/diffusion_types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

namespace trtf {

// ---------------------------------------------------------------------------
// FluxPipeline: TrtModule(T5) + TrtModule(CLIP) + TrtModule(denoiser) + TrtModule(VAE)
// ---------------------------------------------------------------------------

class FluxPipeline final : public IPipeline {
public:
    FluxPipeline(
        std::vector<std::unique_ptr<TrtModule>> text_encoders,
        std::unique_ptr<TrtModule> denoiser,
        std::unique_ptr<TrtModule> vae,
        DiffusionConfig config,
        PreprocessorWeights weights,
        std::shared_ptr<ITokenizer> tokenizer,
        std::unique_ptr<ITokenizer> clip_tokenizer,
        std::string model_id_str);

    ~FluxPipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "FluxPipeline"; }

private:
    bool run_clip_encoder(const std::vector<int32_t>& input_ids,
                          std::vector<float>& pooled_output);
    bool run_t5_encoder(int32_t encoder_idx,
                        const std::vector<int32_t>& input_ids,
                        std::vector<float>& text_embeddings);
    bool run_flux_denoiser(const std::vector<float>& hidden,
                           const std::vector<float>& encoder_hidden,
                           const std::vector<float>& temb,
                           const std::vector<float>& cos_vals,
                           const std::vector<float>& sin_vals,
                           std::vector<float>& output);

    void compute_flux_timestep_embedding(
        float timestep, float guidance,
        const std::vector<float>& pooled_text,
        std::vector<float>& temb) const;
    void compute_flux_rope(
        int32_t h_patches, int32_t w_patches, int32_t text_seq_len,
        std::vector<float>& cos_out, std::vector<float>& sin_out) const;

    std::vector<std::unique_ptr<TrtModule>> text_encoders_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    DiffusionConfig config_;
    PreprocessorWeights weights_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITokenizer> clip_tokenizer_;
    std::string model_id_;
    std::string raw_prompt_;

    // FLUX-specific layout
    int32_t h_latent_{0};
    int32_t w_latent_{0};
    int32_t num_img_tokens_{0};
};

// ---------------------------------------------------------------------------
// WanPipeline: TrtModule(T5) + TrtModule(denoiser) + TrtModule(VAE)
// ---------------------------------------------------------------------------

class WanPipeline final : public IPipeline {
public:
    WanPipeline(
        std::unique_ptr<TrtModule> text_encoder,
        std::unique_ptr<TrtModule> denoiser,
        std::unique_ptr<TrtModule> vae,
        DiffusionConfig config,
        PreprocessorWeights weights,
        std::shared_ptr<ITokenizer> tokenizer,
        std::string model_id_str);

    ~WanPipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "WanPipeline"; }

private:
    bool run_t5_encoder(const std::vector<int32_t>& input_ids,
                        std::vector<float>& text_embeddings);
    bool run_denoiser(const std::vector<float>& hidden,
                      const std::vector<float>& temb_6d,
                      const std::vector<float>& time_embed,
                      const std::vector<float>& encoder_hidden,
                      const std::vector<float>& cos_vals,
                      const std::vector<float>& sin_vals,
                      std::vector<float>& output,
                      const std::vector<float>& encoder_attn_mask = {});

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
    bool decode_vae_2d(const std::vector<float>& latents,
                       int32_t c, int32_t h, int32_t w,
                       VideoResult& result);
    bool decode_vae_3d(const std::vector<float>& latents,
                       int32_t c, int32_t t, int32_t h, int32_t w,
                       VideoResult& result);

    // decode_vae_3d helpers (extracted for cyclomatic complexity)
    int32_t query_vae_output_temporal_dim() const;
    void init_vae_caches();
    void zero_vae_caches();
    void decode_vae_single_frame(
        const std::vector<float>& latents,
        int32_t c, int32_t t_lat, int32_t h_lat, int32_t w_lat,
        int32_t t, std::size_t out_frame_floats,
        std::vector<float>& all_raw_frames);

    // generate_image helpers (extracted for cyclomatic complexity)
    bool run_wan_text_conditioning(
        const std::vector<int32_t>& input_ids,
        int32_t seq_len,
        std::vector<float>& text_projected,
        std::vector<float>& null_text,
        std::string& error);
    bool run_wan_vae_decode(
        int32_t z_dim, int32_t t_lat, int32_t h_lat, int32_t w_lat,
        std::vector<float>& latents,
        VideoResult& result);

    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    DiffusionConfig config_;
    PreprocessorWeights weights_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    // VAE 3D cache state (DeviceTensors for D2D swap)
    std::vector<DeviceTensor> vae_cache_in_;
    std::vector<DeviceTensor> vae_cache_out_;
    bool vae_caches_initialized_{false};
};

// ---------------------------------------------------------------------------
// ZImagePipeline: TrtModule(Qwen3) + TrtModule(denoiser) + TrtModule(VAE)
// ---------------------------------------------------------------------------

/// Z-Image-specific preprocessor weights (separate from standard PreprocessorWeights)
struct ZImagePreprocessorWeights {
    std::vector<float> t_embedder_mlp_0_weight;
    std::vector<float> t_embedder_mlp_0_bias;
    std::vector<float> t_embedder_mlp_2_weight;
    std::vector<float> t_embedder_mlp_2_bias;
    std::vector<float> cap_proj_weight;
    std::vector<float> cap_proj_bias;
    std::vector<float> cap_norm_weight;
    std::vector<float> cap_pad_token;
    std::vector<float> x_embed_weight;
    std::vector<float> x_embed_bias;
    int32_t cap_dim{0};
    int32_t dit_dim{0};
    int32_t freq_dim{0};
    bool valid{false};
};

class ZImagePipeline final : public IPipeline {
public:
    ZImagePipeline(
        std::unique_ptr<TrtModule> text_encoder,
        std::unique_ptr<TrtModule> denoiser,
        std::unique_ptr<TrtModule> vae,
        DiffusionConfig config,
        PreprocessorWeights weights,
        ZImagePreprocessorWeights z_weights,
        std::shared_ptr<ITokenizer> tokenizer,
        std::string model_id_str,
        std::string bundle_path);

    ~ZImagePipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "ZImagePipeline"; }

private:
    bool run_text_encoder(const std::vector<int32_t>& input_ids,
                          std::vector<float>& text_embeddings);
    bool run_denoiser(const std::vector<float>& hidden,
                      const std::vector<float>& encoder_hidden,
                      const std::vector<float>& temb,
                      const std::vector<float>& cos_vals,
                      const std::vector<float>& sin_vals,
                      std::vector<float>& output);

    void project_caption(const std::vector<float>& text_emb,
                         int32_t actual_len, int32_t padded_len,
                         std::vector<float>& projected) const;
    void compute_3d_rope(int32_t cap_padded_len, int32_t num_patches,
                         int32_t nh, int32_t nw,
                         std::vector<float>& cos_out,
                         std::vector<float>& sin_out) const;
    void patchify_2d(const std::vector<float>& latents,
                     int32_t c, int32_t h, int32_t w,
                     std::vector<float>& patches) const;
    void unpatchify_2d(const std::vector<float>& patches,
                       int32_t c, int32_t h, int32_t w,
                       std::vector<float>& output) const;

    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    DiffusionConfig config_;
    PreprocessorWeights weights_;
    ZImagePreprocessorWeights z_weights_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::string bundle_path_;
};

} // namespace trtf

#endif // TRTF_HAS_TRT
