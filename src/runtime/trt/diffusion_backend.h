#pragma once

#include "trtf/backend.h"
#include "trtf/tokenizer.h"
#include "cabi/fast_path_config.h"
#include "runtime/trt/trt_common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

/// Configuration for the diffusion pipeline.
struct DiffusionConfig {
    std::string scheduler{"flow_match_euler"};
    int32_t num_inference_steps{50};
    float guidance_scale{5.0F};
    float flow_shift{1.0F};

    int32_t video_height{480};
    int32_t video_width{832};
    int32_t video_num_frames{81};

    int32_t z_dim{16};
    int32_t scale_factor_temporal{4};
    int32_t scale_factor_spatial{8};
    int32_t dit_dim{1536};
    int32_t dit_num_heads{12};
    int32_t freq_dim{256};
    int32_t text_seq_len{512};
    int32_t text_encoder_dim{4096};

    int32_t num_vae_caches{0};
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
    std::vector<int32_t> patch_size;  // [pt, ph, pw]
    std::string vae_model_id;

    bool guidance_embeds{false};

    std::string diffusion_backend_type{"wan_3d"};
};

/// Preprocessor weights for the DiT (external to the TRT engine graph).
struct PreprocessorWeights {
    // Patch embedding (Conv3D weights, used as matmul)
    std::vector<float> patch_embed_weight;  // [patch_dim, dit_dim] where patch_dim = C*pt*ph*pw
    std::vector<float> patch_embed_bias;    // [dit_dim]
    int32_t patch_dim{0};

    // TimestepEmbedding MLP: sinusoidal(freq_dim) -> Linear(freq_dim,dim) -> SiLU -> Linear(dim,dim)
    std::vector<float> time_emb_0_weight;   // [freq_dim, dim] (already transposed by Python)
    std::vector<float> time_emb_0_bias;     // [dim]
    std::vector<float> time_emb_2_weight;   // [dim, dim]
    std::vector<float> time_emb_2_bias;     // [dim]

    // time_proj: SiLU(time_embed) -> Linear(dim, 6*dim)
    std::vector<float> time_proj_weight;    // [dim, 6*dim]
    std::vector<float> time_proj_bias;      // [6*dim]

    // Text projection MLP: Linear(text_dim, dim) -> SiLU -> Linear(dim, dim)
    std::vector<float> text_proj_weight;    // [text_dim, dim]
    std::vector<float> text_proj_bias;      // [dim]
    std::vector<float> text_proj_2_weight;  // [dim, dim] (optional second layer)
    std::vector<float> text_proj_2_bias;    // [dim]

    // Context embedder (FLUX): Linear(text_encoder_dim, dit_dim)
    std::vector<float> context_embed_weight;  // [text_encoder_dim, dit_dim]
    std::vector<float> context_embed_bias;    // [dit_dim]

    // Guidance embedding MLP (FLUX): sinusoidal(freq_dim) -> Linear -> SiLU -> Linear
    std::vector<float> guidance_emb_0_weight;  // [freq_dim, dim]
    std::vector<float> guidance_emb_0_bias;    // [dim]
    std::vector<float> guidance_emb_2_weight;  // [dim, dim]
    std::vector<float> guidance_emb_2_bias;    // [dim]

    bool valid{false};
};

/// A generic TRT engine wrapper for diffusion components.
struct DiffusionEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;
    std::string name;
};

/// Result of video generation.
struct VideoResult {
    std::vector<float> frames;  // [T, H, W, 3] float in [0,1]
    int32_t num_frames{0};
    int32_t height{0};
    int32_t width{0};
};

/// Interface for diffusion backends. Each model family implements this.
class IDiffusionBackend : public IGenerationBackend {
public:
    ~IDiffusionBackend() override = default;

    /// Generate a video from text tokens.
    virtual VideoResult generate_video(
        const std::vector<int32_t>& input_ids,
        int32_t num_inference_steps = -1,
        float guidance_scale = -1.0F) = 0;

    virtual bool supports_video() const { return true; }

    /// Set preprocessor weights (patch embed, timestep MLP, text proj).
    virtual void set_preprocessor_weights(PreprocessorWeights weights) = 0;

    /// Set paths needed for VAE subprocess decode.
    virtual void set_hf_python(std::string path) = 0;
    virtual void set_bundle_path(std::string path) = 0;

    /// Set a secondary (CLIP) tokenizer for dual-tokenizer models like FLUX.
    virtual void set_clip_tokenizer(std::unique_ptr<ITokenizer> tok) { (void)tok; }

    /// Set raw prompt text (for subprocess-based backends like Z-Image).
    virtual void set_prompt(std::string prompt) { (void)prompt; }

    /// Prepare prompt text before tokenization (e.g. apply chat template).
    /// Default: return prompt unchanged. Override for models that need wrapping.
    virtual std::string prepare_prompt(const std::string& prompt) const { return prompt; }
};

/// Base class with shared utilities for diffusion backends.
class DiffusionBackendBase : public IDiffusionBackend {
public:
    DiffusionBackendBase(
        std::vector<DiffusionEngine> text_encoders,
        DiffusionEngine denoiser,
        DiffusionEngine vae_decoder,
        DiffusionConfig config);

    ~DiffusionBackendBase() override = default;

    bool is_available() const override;
    const char* name() const override;

    /// For diffusion, generate() returns empty. Use generate_video() instead.
    std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) override;

    void set_preprocessor_weights(PreprocessorWeights weights) override;
    void set_hf_python(std::string path) override { mHfPython = std::move(path); }
    void set_bundle_path(std::string path) override { mBundlePath = std::move(path); }

protected:
    // --- Shared CPU math helpers ---
    static void cpu_matmul_bias(const float* A, const float* B, const float* bias,
                                float* out, int32_t M, int32_t K, int32_t N);
    static void cpu_silu_inplace(float* data, std::size_t count);
    static void cpu_gelu_tanh_inplace(float* data, std::size_t count);

    // --- Shared engine execution ---
    bool run_t5_encoder(const std::vector<int32_t>& input_ids,
                        std::vector<float>& text_embeddings,
                        std::string& error);

    bool run_denoiser(const std::vector<float>& hidden,
                      const std::vector<float>& temb_6d,
                      const std::vector<float>& time_embed,
                      const std::vector<float>& encoder_hidden,
                      const std::vector<float>& cos_vals,
                      const std::vector<float>& sin_vals,
                      std::vector<float>& output,
                      std::string& error);

    bool decode_vae_subprocess(const std::vector<float>& latents,
                               int32_t c, int32_t t, int32_t h, int32_t w,
                               VideoResult& result, std::string& error);

    std::vector<DiffusionEngine> mTextEncoders;
    DiffusionEngine mDenoiser;
    DiffusionEngine mVaeDecoder;
    DiffusionConfig mConfig;
    PreprocessorWeights mWeights;

    CudaStream mStream;

    // Persistent device buffers for T5 encoder
    CudaBuffer mD_InputIds;
    CudaBuffer mD_AttentionMask;
    CudaBuffer mD_TextEmbeddings;

    // Persistent device buffers for DiT denoiser
    CudaBuffer mD_Hidden;
    CudaBuffer mD_Temb;
    CudaBuffer mD_TimeEmbed;
    CudaBuffer mD_EncoderHidden;
    CudaBuffer mD_RotaryCos;
    CudaBuffer mD_RotarySin;
    CudaBuffer mD_Output;

    std::string mHfPython;
    std::string mBundlePath;

    bool mOk{false};
};

/// Parse preprocessor weights from bundle section bytes.
PreprocessorWeights parse_preprocessor_weights(
    const std::vector<char>& data);

/// Factory: create the appropriate IDiffusionBackend based on config.
std::unique_ptr<IDiffusionBackend> CreateDiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    const FastPathModelConfig& fp_cfg);

#endif // TRTF_HAS_TRT

} // namespace trtf
