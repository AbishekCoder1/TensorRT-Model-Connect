#pragma once

#include "runtime/trt/audio/audio_configs.h"
#include "runtime/trt/core/generation_backend.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/core/device_kv_cache.h"
#include "runtime/trt/audio/bark_backend.h"  // for LegacyAudioResult, write_wav
#include "cabi/config/fast_path_config.h"
#include "trtf/runtime/trt/audio/subprocess_runner.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

class SpeechToSpeechBackend {
public:
    SpeechToSpeechBackend(
        std::unique_ptr<DecoderStepEngine> temporal_engine,
        SpeechConfig config,
        std::shared_ptr<ISubprocessRunner> subprocess_runner
            = CreateDefaultSubprocessRunner());

    ~SpeechToSpeechBackend();

    bool is_available() const;

    /// Full speech-to-speech pipeline: audio in -> audio out.
    /// Reads WAV from audio_in, processes through all stages, writes WAV to
    /// audio_out. Returns LegacyAudioResult with the generated waveform.
    LegacyAudioResult process_audio(
        const float* input_samples, int32_t num_input_samples,
        int32_t max_output_frames = 375,
        int32_t input_sample_rate = 0,
        int32_t tail_frames = 0);

    const SpeechConfig& config() const { return mConfig; }

    /// Set a single depth engine (backward compatibility, used as cb0).
    void set_depth_engine(
        std::unique_ptr<DecoderStepEngine> engine);

    /// Set a per-codebook depth engine. The codebook index determines which
    /// engine is used at each depth step.
    void set_depth_engine(
        int32_t codebook_idx,
        std::unique_ptr<DecoderStepEngine> engine);

    /// Set the Mimi encoder TRT engine + context.
    void set_mimi_encoder(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);

    /// Set the Mimi decoder TRT engine + context.
    void set_mimi_decoder(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);

private:
    /// Stage 1: Encode audio waveform into codec tokens via Mimi encoder.
    /// Returns codec tokens [num_codebooks, num_frames].
    std::vector<int32_t> run_mimi_encode(
        const float* samples, int32_t num_samples,
        int32_t input_sample_rate);

    /// Stage 2: Process codec tokens through the Temporal Transformer.
    /// Returns hidden states for the Depth Transformer.
    std::vector<float> run_temporal(
        const std::vector<int32_t>& codec_tokens, int32_t num_frames,
        int32_t num_codebooks);

    /// Stage 3: Generate output codec tokens via the Depth Transformer.
    /// Takes temporal hidden state for one timestep, generates num_codebooks
    /// tokens autoregressively. text_token is the sampled text token from
    /// temporal logits, used as input at depth position 0.
    std::vector<int32_t> run_depth(
        const float* temporal_hidden, int32_t hidden_dim,
        int32_t text_token = 0,
        const int32_t* forced_audio_tokens = nullptr,
        const uint8_t* forced_audio_provided = nullptr);

    /// Stage 4: Decode codec tokens to waveform via Mimi decoder.
    std::vector<float> run_mimi_decode(
        const std::vector<int32_t>& codec_tokens, int32_t num_frames);

    /// Inject text prompt tokens into temporal transformer KV cache.
    /// Tokenizes the system prompt via Python subprocess and runs each token
    /// through temporal with silence audio, priming the KV cache.
    void run_text_prompt(
        DeviceKvCache& cache, DeviceResources& resources,
        CudaBuffer& d_hidden_state);

    // Temporal Transformer (main decoder)
    std::unique_ptr<DecoderStepEngine> mTemporalEngine;

    // Per-codebook Depth Transformer engines (index = codebook)
    std::vector<std::unique_ptr<DecoderStepEngine>> mDepthEngines;
    // Fallback single depth engine (backward compat)
    std::unique_ptr<DecoderStepEngine> mDepthEngine;

    // Mimi codec engines
    TrtUniquePtr<nvinfer1::ICudaEngine> mMimiEncoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mMimiEncoderCtx;
    TrtUniquePtr<nvinfer1::ICudaEngine> mMimiDecoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mMimiDecoderCtx;

    SpeechConfig mConfig;
    std::shared_ptr<ISubprocessRunner> mSubprocessRunner;

    // RNG state for sampling (xorshift64, seeded from time at construction)
    uint64_t mRngState{0};

    // Last Mimi encode output shape (set by run_mimi_encode).
    int32_t mLastEncodeFrames{0};
    int32_t mLastEncodeCodebooks{0};

    // Debug: counts run_depth calls within one process_audio invocation.
    int32_t mDepthDebugCallCount{0};
};

/// Factory: create from engines + config.
std::unique_ptr<SpeechToSpeechBackend> CreateSpeechBackend(
    std::unique_ptr<DecoderStepEngine> temporal_engine,
    const FastPathModelConfig& cfg,
    const std::string& hf_python = "");

} // namespace trtf

#endif // TRTF_HAS_TRT
