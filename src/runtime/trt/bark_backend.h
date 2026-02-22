#pragma once

#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/device_kv_cache.h"
#include "cabi/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
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
    int32_t semantic_vocab_size{10000};
    int32_t coarse_vocab_size{1024};
    int32_t fine_vocab_size{1024};
    int32_t n_coarse_codebooks{2};
    int32_t n_fine_codebooks{8};
    int32_t semantic_pad_token{10000};
    int32_t semantic_infer_token{10001};
    int32_t coarse_semantic_pad_token{12048};
    int32_t coarse_infer_token{12050};
};

class BarkBackend {
public:
    BarkBackend(
        std::unique_ptr<DecoderStepEngine> semantic_engine,
        BarkConfig config);

    ~BarkBackend();

    bool is_available() const;

    // Generate audio from text tokens.
    // Returns waveform samples at sample_rate.
    AudioResult generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_semantic_tokens = 768);

    const BarkConfig& config() const { return mConfig; }

    // Set optional extra engines (coarse, fine, codec).
    void set_coarse_engine(std::unique_ptr<DecoderStepEngine> engine);
    void set_fine_engine(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);
    void set_codec_engine(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);

private:
    std::unique_ptr<DecoderStepEngine> mSemanticEngine;
    std::unique_ptr<DecoderStepEngine> mCoarseEngine;
    TrtUniquePtr<nvinfer1::ICudaEngine> mFineEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mFineCtx;
    TrtUniquePtr<nvinfer1::ICudaEngine> mCodecEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mCodecCtx;
    BarkConfig mConfig;
};

// Write a WAV file with 44-byte RIFF header + IEEE float32 PCM.
bool write_wav(const std::string& path, const float* samples,
               int32_t num_samples, int32_t sample_rate);

// Create from semantic engine + config.
std::unique_ptr<BarkBackend> CreateBarkBackend(
    std::unique_ptr<DecoderStepEngine> semantic_engine,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
