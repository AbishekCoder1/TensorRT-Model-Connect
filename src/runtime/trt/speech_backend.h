#pragma once

#include "trtf/backend.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/device_kv_cache.h"
#include "runtime/trt/bark_backend.h"  // for AudioResult, write_wav
#include "cabi/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct SpeechConfig {
    int32_t sample_rate{24000};
    int32_t num_codebooks{8};
    int32_t codebook_size{2048};
    float frame_rate{12.5F};

    // Temporal transformer
    int32_t temporal_hidden_size{0};
    int32_t temporal_num_layers{0};

    // Depth transformer
    int32_t depth_hidden_size{0};
    int32_t depth_num_layers{6};
    int32_t depth_num_heads{0};
    int32_t depth_num_kv_heads{0};
    int32_t depth_max_cache_length{16};

    // Temporal-to-depth projection matrix [depth_hidden, temporal_hidden]
    // Loaded from bundle 'depth_projection' section.
    std::vector<float> depth_projection;
    int32_t temporal_hidden_for_proj{0};  // temporal_hidden (cols of projection)

    // Per-codebook audio embedding tables for temporal transformer input.
    // Layout: [num_codebooks, audio_vocab_size, temporal_hidden_size] as float32.
    // The temporal input for each frame is the SUM of all codebook embeddings.
    std::vector<float> audio_embeddings;
    int32_t audio_vocab_size{2049};  // per-codebook vocab (Mimi: 2049)

    // Temporal text embedding table [text_vocab, temporal_hidden] as float32.
    // The official Moshi code sums text_emb(text_token) + audio embeddings.
    // During generation, the text token is the PAD token (text_padding_id).
    std::vector<float> temporal_text_embedding;
    int32_t temporal_text_vocab{0};
    int32_t text_padding_id{3};  // PAD token for text during generation

    // Depth decoder text embedding table [depth_text_vocab, depth_hidden] as float32.
    // Used at depth position 0 (text token step) before audio codebook generation.
    std::vector<float> depth_text_embedding;
    int32_t depth_text_vocab{0};

    // Number of output codebooks to send to Mimi decoder (default: 8).
    // The depth transformer generates num_codebooks (16) tokens, but
    // only the first mimi_decode_codebooks (moshi stream) are decoded.
    int32_t mimi_decode_codebooks{8};

    // Depth decoder per-codebook audio embedding tables.
    // Layout: [num_depformer_emb, audio_vocab_size, depth_hidden] as float32.
    // depformer_emb.{i} is used at depth position i+1.
    std::vector<float> depth_audio_embeddings;
    int32_t num_depformer_emb{0};

    // Delay pattern for temporal transformer input alignment.
    // Official Moshi: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    // Length = num_codebooks + 1 (text + 16 audio).
    // delay[k]=0: token from current frame; delay[k]=1: token from previous frame.
    std::vector<int32_t> delays;
    int32_t text_initial_token_id{32000};  // BOS for text stream at first step
    int32_t audio_initial_token_id{2048};  // BOS for audio streams at first step

    // Depth decoder sampling parameters (greedy if temperature <= 0)
    float depth_temperature{0.0F};  // 0 = greedy, 0.8 = official PersonaPlex
    int32_t depth_top_k{0};         // 0 = greedy, 250 = official PersonaPlex

    // System prompt for text prompt injection (primes temporal KV cache)
    std::string system_prompt;
    // Pre-tokenized system prompt IDs (avoids runtime tokenization)
    std::vector<int32_t> text_prompt_ids;

    // Python codec bridge (for Mimi encode/decode via HF MimiModel)
    std::string hf_python;    // path to Python interpreter with transformers
    std::string scripts_dir;  // path to scripts/ directory containing mimi_codec.py
};

class SpeechToSpeechBackend {
public:
    SpeechToSpeechBackend(
        std::unique_ptr<DecoderStepEngine> temporal_engine,
        SpeechConfig config);

    ~SpeechToSpeechBackend();

    bool is_available() const;

    /// Full speech-to-speech pipeline: audio in -> audio out.
    /// Reads WAV from audio_in, processes through all stages, writes WAV to
    /// audio_out. Returns AudioResult with the generated waveform.
    AudioResult process_audio(
        const float* input_samples, int32_t num_input_samples,
        int32_t max_output_frames = 375,
        int32_t input_sample_rate = 0);

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

    /// Python-based Mimi encode (fallback when TRT encoder not available).
    std::vector<int32_t> run_mimi_encode_python(
        const float* samples, int32_t num_samples,
        int32_t input_sample_rate,
        int32_t& out_num_frames, int32_t& out_num_codebooks);

    /// Python-based Mimi decode (fallback when TRT decoder not available).
    std::vector<float> run_mimi_decode_python(
        const std::vector<int32_t>& codec_tokens,
        int32_t num_codebooks, int32_t num_frames);

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
    const std::string& hf_python = "",
    const std::string& scripts_dir = "");

} // namespace trtf

#endif // TRTF_HAS_TRT
