#pragma once

#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/device_kv_cache.h"
#include "runtime/trt/bark_backend.h"
#include "cabi/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

/// Configuration for the Qwen3-Omni multimodal pipeline.
struct OmniConfig {
    int32_t sample_rate{24000};

    // Thinker MoE config (text decoder)
    int32_t thinker_hidden_size{0};
    int32_t thinker_num_layers{0};
    int32_t thinker_num_heads{0};
    int32_t num_experts{8};
    int32_t num_experts_per_tok{2};

    // Audio encoder config
    int32_t audio_embed_dim{1280};
    int32_t audio_num_mel{128};
    int32_t audio_num_layers{0};
    int32_t audio_num_frames{1500};

    // Talker config
    int32_t talker_hidden_size{0};
    int32_t talker_num_layers{0};
    int32_t talker_n_codebooks{8};
    int32_t talker_codebook_size{2048};

    // Code2Wav config
    int32_t code2wav_upsample_factor{320};
    int32_t code2wav_max_frames{256};

    // Generation parameters
    bool greedy{false};
    float temperature{0.7F};
    int32_t top_k{50};
};

/// Qwen3-Omni multimodal backend: Thinker-Talker-Code2Wav pipeline.
///
/// Stage 0: Thinker (MoE decoder) generates text tokens from multimodal input.
///          Vision features from VL vision encoder, audio features from audio encoder.
/// Stage 1: Talker converts Thinker hidden states + text to RVQ codec tokens.
/// Stage 2: Code2Wav synthesizes waveform from RVQ tokens.
class OmniBackend {
public:
    OmniBackend(
        std::unique_ptr<DecoderStepEngine> thinker_engine,
        OmniConfig config);

    ~OmniBackend();

    bool is_available() const;

    /// Set the audio encoder TRT engine for processing audio input.
    void set_audio_encoder(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);

    /// Set the Talker decoder engine for RVQ code prediction.
    void set_talker_engine(
        std::unique_ptr<DecoderStepEngine> engine);

    /// Set the Code2Wav engine for waveform synthesis.
    void set_code2wav_engine(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);

    /// Full text-to-text generation using Thinker MoE decoder.
    /// Returns generated token IDs (text output).
    std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        int32_t max_new_tokens);

    /// Full text-to-audio generation: Thinker -> Talker -> Code2Wav.
    /// Returns AudioResult with waveform.
    AudioResult generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_semantic_tokens = 768);

    /// Run audio encoder on mel spectrogram features.
    /// Returns audio feature embeddings for Thinker input.
    std::vector<float> encode_audio(
        const float* mel_features,
        int32_t num_mel_bins,
        int32_t num_frames);

    const OmniConfig& config() const { return mConfig; }

private:
    /// Stage 0: Run Thinker MoE decoder (text generation).
    /// Returns generated token IDs and optionally saves hidden states
    /// for Talker input.
    std::vector<int32_t> run_thinker(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens,
        std::vector<float>* hidden_states_out = nullptr);

    /// Stage 1: Run Talker to produce RVQ codec tokens from hidden states.
    std::vector<int32_t> run_talker(
        const std::vector<float>& hidden_states,
        int32_t num_tokens);

    /// Stage 2: Run Code2Wav to synthesize waveform from RVQ tokens.
    std::vector<float> run_code2wav(
        const std::vector<int32_t>& codec_tokens,
        int32_t n_codebooks,
        int32_t n_frames);

    std::unique_ptr<DecoderStepEngine> mThinkerEngine;
    TrtUniquePtr<nvinfer1::ICudaEngine> mAudioEncoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mAudioEncoderCtx;
    std::unique_ptr<DecoderStepEngine> mTalkerEngine;
    TrtUniquePtr<nvinfer1::ICudaEngine> mCode2WavEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mCode2WavCtx;
    OmniConfig mConfig;
};

/// Create OmniBackend from Thinker engine + config.
std::unique_ptr<OmniBackend> CreateOmniBackend(
    std::unique_ptr<DecoderStepEngine> thinker_engine,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
