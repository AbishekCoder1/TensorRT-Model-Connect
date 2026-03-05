#pragma once
// Stub: detection backend (not yet implemented)
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "runtime/trt/core/trt_common.h"

#if TRTF_HAS_TRT
#include <NvInferRuntime.h>
#endif

namespace trtf {

struct Detection {
    int32_t class_id{0};
    float confidence{0.0f};
    float x1{0}, y1{0}, x2{0}, y2{0};
};

struct DetectionResult {
    bool ok{false};
    std::vector<Detection> detections;
};

class DetectionBackend {
public:
    virtual ~DetectionBackend() = default;
    virtual bool is_available() const { return false; }
    virtual DetectionResult detect_image(const std::string& /*path*/) { return {}; }
};

#if TRTF_HAS_TRT
struct FastPathModelConfig;
inline std::unique_ptr<DetectionBackend> CreateDetectionBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> /*engine*/,
    TrtUniquePtr<nvinfer1::IExecutionContext> /*ctx*/,
    const FastPathModelConfig& /*cfg*/) { return nullptr; }
#endif

} // namespace trtf
