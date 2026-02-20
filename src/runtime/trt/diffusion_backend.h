#pragma once

#include "trtf/backend.h"
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

/// Generic diffusion backend: N text encoders + denoiser + VAE decoder.
///
/// Handles Wan (1 T5 + DiT + 3D VAE), FLUX (CLIP + T5 + MMDiT + 2D VAE),
/// etc. without per-model C++ code. The scheduler and component count are
/// determined by the bundle config.
class DiffusionBackend final : public IGenerationBackend {
public:
    DiffusionBackend(
        std::vector<DiffusionEngine> text_encoders,
        DiffusionEngine denoiser,
        DiffusionEngine vae_decoder,
        DiffusionConfig config);

    ~DiffusionBackend() override = default;

    bool is_available() const override;
    const char* name() const override;

    /// For diffusion, generate() is not text generation — it returns an
    /// empty vector. Use generate_video() instead.
    std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) override;

    /// Generate a video from text tokens.
    /// Returns frames as float array [T * H * W * 3].
    VideoResult generate_video(
        const std::vector<int32_t>& input_ids,
        int32_t num_inference_steps = -1,
        float guidance_scale = -1.0F);

    bool supports_video() const { return true; }

    /// Set preprocessor weights (patch embed, timestep MLP, text proj).
    void set_preprocessor_weights(PreprocessorWeights weights);

    /// Set paths needed for VAE subprocess decode.
    void set_hf_python(std::string path) { mHfPython = std::move(path); }
    void set_bundle_path(std::string path) { mBundlePath = std::move(path); }

private:
    std::vector<DiffusionEngine> mTextEncoders;
    DiffusionEngine mDenoiser;
    DiffusionEngine mVaeDecoder;
    DiffusionConfig mConfig;
    PreprocessorWeights mWeights;

    CudaStream mStream;

    // Persistent device buffers for T5 encoder
    CudaBuffer mD_InputIds;         // [1, text_seq_len] int32
    CudaBuffer mD_AttentionMask;    // [1, text_seq_len] float32
    CudaBuffer mD_TextEmbeddings;   // [1, text_seq_len, text_encoder_dim] float32

    // Persistent device buffers for DiT denoiser
    CudaBuffer mD_Hidden;           // [num_patches, dit_dim] float32
    CudaBuffer mD_Temb;             // [1, 6*dit_dim] float32
    CudaBuffer mD_TimeEmbed;        // [1, dit_dim] float32
    CudaBuffer mD_EncoderHidden;    // [text_seq_len, dit_dim] float32
    CudaBuffer mD_RotaryCos;        // [num_patches, head_dim] float32
    CudaBuffer mD_RotarySin;        // [num_patches, head_dim] float32
    CudaBuffer mD_Output;           // [num_patches, out_dim] float32

    // For VAE subprocess fallback
    std::string mHfPython;
    std::string mBundlePath;

    // Native VAE decode state
    CudaBuffer mD_VaeInput;         // [1, z_dim, 1, h_lat, w_lat]
    CudaBuffer mD_VaeOutput;        // [1, 3, T_out, h_out, w_out]
    std::vector<CudaBuffer> mD_VaeCacheIn;   // 32 cache input buffers
    std::vector<CudaBuffer> mD_VaeCacheOut;  // 32 cache output buffers
    std::vector<std::size_t> mVaeCacheSizes; // byte size per cache
    int32_t mVaeOutputT{1};         // temporal dim of VAE output per frame

    bool mOk{false};

    // --- Private helpers ---

    bool run_t5_encoder(const std::vector<int32_t>& input_ids,
                        std::vector<float>& text_embeddings,
                        std::string& error);

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

    bool run_denoiser(const std::vector<float>& hidden,
                      const std::vector<float>& temb_6d,
                      const std::vector<float>& time_embed,
                      const std::vector<float>& encoder_hidden,
                      const std::vector<float>& cos_vals,
                      const std::vector<float>& sin_vals,
                      std::vector<float>& output,
                      std::string& error);

    bool decode_vae_native(const std::vector<float>& latents,
                           int32_t c, int32_t t, int32_t h, int32_t w,
                           VideoResult& result, std::string& error);

    bool decode_vae_subprocess(const std::vector<float>& latents,
                               int32_t c, int32_t t, int32_t h, int32_t w,
                               VideoResult& result, std::string& error);

    void init_vae_buffers();
};

/// Parse preprocessor weights from bundle section bytes.
PreprocessorWeights parse_preprocessor_weights(
    const std::vector<char>& data);

/// Factory: create DiffusionBackend from deserialized engines + config.
std::unique_ptr<DiffusionBackend> CreateDiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    const FastPathModelConfig& fp_cfg);

#endif // TRTF_HAS_TRT

} // namespace trtf
