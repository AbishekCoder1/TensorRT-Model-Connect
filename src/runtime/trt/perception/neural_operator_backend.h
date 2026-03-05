#pragma once
// Stub: neural operator backend (not yet implemented)
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "runtime/trt/core/trt_common.h"

#if TRTF_HAS_TRT
#include <NvInferRuntime.h>
#endif

namespace trtf {

struct NeuralOperatorResult {
    std::vector<float> output;
    int32_t output_dim{0};
    int32_t out_channels{0};
    int32_t height{0};
    int32_t width{0};
};

class NeuralOperatorBackend {
public:
    virtual ~NeuralOperatorBackend() = default;
    virtual bool is_available() const { return false; }
    virtual NeuralOperatorResult solve(
        const float* /*branch*/, int32_t /*branch_len*/,
        const float* /*trunk*/, int32_t /*trunk_len*/) { return {}; }
    virtual NeuralOperatorResult solve_field(
        const float* /*field*/, int32_t /*size*/) { return {}; }
};

#if TRTF_HAS_TRT
struct FastPathModelConfig;
inline std::unique_ptr<NeuralOperatorBackend> CreateNeuralOperatorBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> /*engine*/,
    TrtUniquePtr<nvinfer1::IExecutionContext> /*ctx*/,
    const FastPathModelConfig& /*cfg*/) { return nullptr; }
#endif

} // namespace trtf
