#pragma once

#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/core/device_kv_cache.h"
#include "runtime/trt/audio/bark_backend.h"  // for AudioResult, write_wav
#include "cabi/config/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtf {

struct MagpieTTSConfig {
    int32_t sample_rate{22050};
    int32_t hidden_size{0};
    int32_t num_codebooks{8};
    int32_t codebook_size{2024};
    float frames_per_second{21.5F};
    int32_t num_speakers{5};
    int32_t encoder_layers{6};
    int32_t decoder_layers{12};
    int32_t text_vocab_size{0};
    int32_t max_source_positions{2048};
    int32_t xa_n_heads{1};
    int32_t xa_d_head{128};
    // Sampling
    float temperature{0.8F};
    int32_t top_k{80};
    bool greedy{false};
};

class MagpieTTSBackend {
public:
    MagpieTTSBackend(
        std::unique_ptr<DecoderStepEngine> decoder_engine,
        TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
        std::vector<float> audio_embed,     // [8 * codebook_size * hidden]
        std::vector<float> text_embed,      // [text_vocab * hidden]
        std::vector<float> context_embed,   // [num_speakers * context_frames * hidden]
        std::vector<int32_t> context_lengths, // [num_speakers]
        MagpieTTSConfig config);

    ~MagpieTTSBackend();

    bool is_available() const;

    AudioResult generate_audio(
        const std::vector<int32_t>& text_ids,
        int32_t max_frames = 500);

    const MagpieTTSConfig& config() const { return mConfig; }

    // Set codec engine for waveform synthesis
    void set_codec_engine(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);


private:
    // Pipeline stages (Whisper pattern)
    void run_encoder(const std::vector<int32_t>& text_ids,
                     int32_t speaker_id, int32_t language_id);
    void compute_cross_kv();
    void bind_cross_kv();

    // Decoder loop (multi-codebook autoregressive)
    std::vector<int32_t> run_decoder(int32_t max_frames);

    // Codec (Bark pattern)
    std::vector<float> run_codec(const std::vector<int32_t>& codes, int32_t num_frames);

    // Helpers (Bark pattern)
    void lookup_embed(const float* table, int32_t token_id,
                      float* out) const;
    void sum_embeds(const float* a, const float* b, float* out) const;
    int32_t sample_top_k(const float* logits, int32_t vocab_size,
                         float temperature, int32_t top_k);

    // Engines
    std::unique_ptr<DecoderStepEngine> mDecoderEngine;
    TrtUniquePtr<nvinfer1::ICudaEngine> mEncoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mEncoderContext;
    TrtUniquePtr<nvinfer1::ICudaEngine> mCodecEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mCodecCtx;

    // Decoder KV cache (Whisper pattern)
    std::unique_ptr<DeviceKvCache> mCache;
    std::unique_ptr<DeviceResources> mResources;

    // Encoder output + cross-attention (Whisper pattern)
    CudaBuffer mEncoderOutput;
    std::vector<CudaBuffer> mCrossK;
    std::vector<CudaBuffer> mCrossV;

    // Embedding tables (Bark pattern)
    std::vector<float> mAudioEmbed;      // [8 * codebook_size * hidden]
    std::vector<float> mTextEmbed;       // [text_vocab * hidden]
    std::vector<float> mContextEmbed;    // [num_speakers * max_context_frames * hidden]
    std::vector<int32_t> mContextLengths; // [num_speakers]

    MagpieTTSConfig mConfig;
    std::mt19937 mRng{std::random_device{}()};
};

std::unique_ptr<MagpieTTSBackend> CreateMagpieTTSBackend(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
    std::vector<float> audio_embed,
    std::vector<float> text_embed,
    std::vector<float> context_embed,
    std::vector<int32_t> context_lengths,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
