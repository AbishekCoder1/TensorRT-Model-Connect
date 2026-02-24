#pragma once

#include "runtime/trt/trt_common.h"
#include "cabi/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct EncoderResult {
    std::vector<float> hidden_states;  // [seq_len * hidden_size]
    int32_t seq_length{0};
    int32_t hidden_size{0};
};

struct EncoderConfig {
    int32_t max_seq_length{512};
    int32_t hidden_size{768};
    int32_t type_vocab_size{2};
};

class EncoderBackend {
public:
    EncoderBackend(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context,
        EncoderConfig config);

    ~EncoderBackend();

    bool is_available() const;

    // Encode a sequence of token IDs. Returns hidden states for all positions.
    EncoderResult encode(const std::vector<int32_t>& input_ids,
                         const std::vector<int32_t>& token_type_ids);

    // Encode with default token_type_ids (all zeros).
    EncoderResult encode(const std::vector<int32_t>& input_ids);

    const EncoderConfig& config() const { return mConfig; }

private:
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext;
    EncoderConfig mConfig;
    CudaStream mStream;
};

// Create from engine + config
std::unique_ptr<EncoderBackend> CreateEncoderBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
