#pragma once

#include "runtime/trt/core/trt_common.h"
#include "cabi/config/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct RerankingResult {
    float score{0.0F};  // relevance score
};

struct RerankingConfig {
    int32_t max_seq_length{512};
    int32_t hidden_size{2048};
};

class RerankingBackend {
public:
    RerankingBackend(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context,
        RerankingConfig config);

    ~RerankingBackend();

    bool is_available() const;

    // Cross-encode query + document and return a relevance score.
    // input_ids should be the concatenated query + document tokens.
    RerankingResult rerank(const std::vector<int32_t>& input_ids);

    const RerankingConfig& config() const { return mConfig; }

private:
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext;
    RerankingConfig mConfig;
    CudaStream mStream;
};

// Create from engine + fast path config.
std::unique_ptr<RerankingBackend> CreateRerankingBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
