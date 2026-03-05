#pragma once

#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/core/device_kv_cache.h"
#include "cabi/config/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtf {

struct AudioResult {
    std::vector<float> waveform;  // [num_samples] float32
    int32_t sample_rate{24000};
    int32_t num_samples{0};
};

struct BarkConfig {
    int32_t sample_rate{24000};
    int32_t hidden_size{768};

    // Semantic model constants
    int32_t semantic_input_vocab{129600};
    int32_t semantic_output_vocab{10048};
    int32_t text_encoding_offset{10048};
    int32_t semantic_pad_token{10000};
    int32_t semantic_infer_token{129599};
    int32_t semantic_vocab_size{10000};  // valid semantic token range [0, 10000)
    int32_t text_pad_token{129595};     // HF text_pad_token for masked positions

    // Coarse model constants
    int32_t coarse_input_vocab{12096};
    int32_t coarse_semantic_pad_token{12048};
    int32_t coarse_infer_token{12050};
    int32_t n_coarse_codebooks{2};
    int32_t codebook_size{1024};
    int32_t coarse_rate_hz{75};
    float semantic_rate_hz{49.9F};
    int32_t max_coarse_history{630};
    int32_t max_coarse_input_length{256};
    int32_t sliding_window_len{60};

    // Codec (EnCodec) engine config
    int32_t codec_seq_length{0};       // max frames the codec engine was compiled for
    int32_t codec_upsample_factor{320}; // total upsample ratio (8*5*4*2)
    int32_t codec_n_codebooks{8};      // number of codebooks in codec engine input

    // Fine model config
    int32_t fine_hidden_size{768};
    int32_t fine_n_lm_heads{7};
    int32_t fine_codebook_size{1056};  // vocab per codebook (fine model)
    int32_t fine_seq_length{0};        // 0 = no fine engine

    // Sampling parameters
    float semantic_temperature{0.7F};
    float coarse_temperature{0.7F};
    int32_t top_k{50};
    float min_eos_p{0.0F};  // 0 = disabled (matching HF bark-small default)
    bool greedy{false};  // if true, use argmax instead of sampling
};

class BarkBackend {
public:
    BarkBackend(
        std::unique_ptr<DecoderStepEngine> semantic_engine,
        std::unique_ptr<DecoderStepEngine> coarse_engine,
        std::vector<float> semantic_embed,
        std::vector<float> coarse_embed,
        BarkConfig config);

    ~BarkBackend();

    bool is_available() const;

    AudioResult generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_semantic_tokens = 768);

    const BarkConfig& config() const { return mConfig; }

    // Set optional codec engine for waveform synthesis.
    void set_codec_engine(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);

    // Set optional fine engine for codebook prediction.
    void set_fine_engine(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);

    // Set fine embedding tables (8 codebook embeds + position embed).
    void set_fine_embeddings(std::vector<float> embed,
                             std::vector<float> pos_embed);

private:
    // Stage 1: text tokens -> semantic audio tokens
    std::vector<int32_t> run_semantic(const std::vector<int32_t>& text_ids,
                                       int32_t max_tokens);

    // Stage 2: semantic tokens -> coarse acoustic codes (2 codebooks)
    std::vector<int32_t> run_coarse(const std::vector<int32_t>& semantic_tokens);

    // Stage 2.5: Coarse codes -> Fine codes (8 codebooks)
    // Takes interleaved coarse tokens and predicts codebooks 2..7.
    // Returns codes [8 * n_frames] with layout codes[cb * n_frames + frame].
    std::vector<int32_t> run_fine(const std::vector<int32_t>& coarse_tokens);

    // Stage 3: coarse codes -> waveform via codec (EnCodec)
    std::vector<float> run_codec(const std::vector<int32_t>& coarse_tokens);

    // Stage 3 overload: pre-computed codes [8 * n_frames] -> waveform via codec
    std::vector<float> run_codec(const std::vector<int32_t>& codes_flat,
                                  int32_t n_frames);

    // Top-k sampling with temperature
    int32_t sample_top_k(const float* logits, int32_t vocab_size,
                         float temperature, int32_t top_k);

    // Embedding lookup: copies embed_table[token_id] into out
    void lookup_embed(const float* table, int32_t token_id,
                      float* out) const;

    // Element-wise sum: out[i] = a[i] + b[i] for hidden_size elements
    void sum_embeds(const float* a, const float* b, float* out) const;

    std::unique_ptr<DecoderStepEngine> mSemanticEngine;
    std::unique_ptr<DecoderStepEngine> mCoarseEngine;
    TrtUniquePtr<nvinfer1::ICudaEngine> mCodecEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mCodecCtx;
    TrtUniquePtr<nvinfer1::ICudaEngine> mFineEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mFineCtx;
    std::vector<float> mSemanticEmbed;  // [semantic_input_vocab * hidden_size]
    std::vector<float> mCoarseEmbed;    // [coarse_input_vocab * hidden_size]
    std::vector<float> mFineEmbed;      // all 8 embed tables concatenated [8 * fine_codebook_size * fine_hidden_size]
    std::vector<float> mFinePositionEmbed;  // [max_position * fine_hidden_size]
    BarkConfig mConfig;
    std::mt19937 mRng{std::random_device{}()};
};

// Write a WAV file with 44-byte RIFF header + IEEE float32 PCM.
bool write_wav(const std::string& path, const float* samples,
               int32_t num_samples, int32_t sample_rate);

// Create from engines + embedding tables + config.
std::unique_ptr<BarkBackend> CreateBarkBackend(
    std::unique_ptr<DecoderStepEngine> semantic_engine,
    std::unique_ptr<DecoderStepEngine> coarse_engine,
    std::vector<float> semantic_embed,
    std::vector<float> coarse_embed,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
