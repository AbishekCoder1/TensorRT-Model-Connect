#pragma once

#include "trtf/backend.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/core/device_kv_cache.h"
#include "cabi/config/fast_path_config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct TranscriptionResult {
    std::string text;
    std::vector<int32_t> output_ids;
    int32_t num_tokens{0};
};

struct WhisperConfig {
    int32_t num_mel_bins{80};
    int32_t max_source_positions{1500};
    int32_t max_target_positions{448};
    int32_t encoder_layers{0};
    int32_t decoder_layers{0};
    int32_t decoder_attention_heads{0};
    int32_t decoder_start_token_id{50258};  // <|startoftranscript|>
    int32_t language_token_id{50259};      // <|en|>
    int32_t translate_token_id{50358};
    int32_t transcribe_token_id{50359};
    int32_t notimestamps_token_id{50363};
    int32_t eot_token_id{50257};  // <|endoftext|>
    std::string language{"en"};

    int32_t mel_length{0};  // expected mel input length; 0 = auto (max_source_positions * 2)

    // Custom decoder start sequence (overrides individual token fields above).
    // When non-empty, transcribe() uses this instead of building from individual fields.
    // Set from bundle config.json "decoder_start_token_ids" for non-Whisper models (e.g. Canary).
    std::vector<int32_t> decoder_start_token_ids;
};

#if TRTF_HAS_TRT

class WhisperBackend : public IGenerationBackend {
public:
    WhisperBackend(
        std::unique_ptr<DecoderStepEngine> decoder_engine,
        TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
        WhisperConfig config,
        const FastPathModelConfig& fp_cfg);

    bool is_available() const override;
    const char* name() const override { return "whisper"; }

    // IGenerationBackend: autoregressive decode (returns token IDs)
    std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) override;

    // Transcription API: mel spectrogram -> text
    TranscriptionResult transcribe(
        const float* mel_data, int32_t mel_bins, int32_t mel_length,
        int32_t max_new_tokens = 224);

    bool supports_transcription() const { return true; }

private:
    // Run encoder: mel -> encoder_output [max_source_positions, hidden]
    void run_encoder(const float* mel_data, int32_t mel_bins, int32_t mel_length);

    // Compute cross-attention K/V from encoder output (once per utterance)
    void compute_cross_kv();

    // Bind cross_k/cross_v addresses on decoder context (persists across steps)
    void bind_cross_kv();

    // Autoregressive decoder loop
    std::vector<int32_t> run_decoder(
        const std::vector<int32_t>& initial_tokens, int32_t max_new_tokens);

    std::unique_ptr<DecoderStepEngine> mDecoderEngine;
    TrtUniquePtr<nvinfer1::ICudaEngine> mEncoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mEncoderContext;

    // Device resources for decoder (KV cache + I/O buffers)
    std::unique_ptr<DeviceKvCache> mCache;
    std::unique_ptr<DeviceResources> mResources;

    // Device buffers for encoder output + cross-attention K/V
    CudaBuffer mEncoderOutput;      // [max_source_positions * hidden]
    std::vector<CudaBuffer> mCrossK; // per-layer [max_source_positions * hidden]
    std::vector<CudaBuffer> mCrossV; // per-layer [max_source_positions * hidden]

    WhisperConfig mWhisperConfig;
    FastPathModelConfig mFpCfg;
    int32_t mActualEncSeqLen{0};  // valid encoder output length (0 = full)
};

std::unique_ptr<WhisperBackend> CreateWhisperBackend(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
    WhisperConfig config,
    const FastPathModelConfig& fp_cfg);

#endif // TRTF_HAS_TRT
} // namespace trtf
