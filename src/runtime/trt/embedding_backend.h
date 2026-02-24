#pragma once

#include "runtime/trt/trt_common.h"
#include "cabi/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct EmbeddingResult {
    std::vector<float> embedding;  // [embedding_dim] L2-normalized vector
    int32_t embedding_dim{0};
};

struct EmbeddingConfig {
    int32_t max_seq_length{512};
    int32_t hidden_size{2048};
    int32_t embedding_dim{2048};  // output dimension (may differ from hidden)
};

class EmbeddingBackend {
public:
    EmbeddingBackend(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context,
        EmbeddingConfig config);

    ~EmbeddingBackend();

    bool is_available() const;

    // Encode token IDs into a normalized embedding vector.
    // Mean pools over non-padding positions and L2-normalizes.
    EmbeddingResult embed(const std::vector<int32_t>& input_ids);

    const EmbeddingConfig& config() const { return mConfig; }

private:
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext;
    EmbeddingConfig mConfig;
    CudaStream mStream;
};

// Create from engine + fast path config.
std::unique_ptr<EmbeddingBackend> CreateEmbeddingBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
