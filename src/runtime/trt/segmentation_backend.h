#pragma once

#include "runtime/trt/trt_common.h"
#include "cabi/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct SegmentationResult {
    std::vector<int32_t> class_map;  // [H, W] class indices
    int32_t height{0};
    int32_t width{0};
    int32_t num_classes{0};
};

struct SegmentationConfig {
    int32_t num_classes{150};
    int32_t input_image_h{512};
    int32_t input_image_w{512};
    int32_t output_h{128};
    int32_t output_w{128};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

class SegmentationBackend {
public:
    SegmentationBackend(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context,
        SegmentationConfig config);

    ~SegmentationBackend();

    bool is_available() const;

    // Segment an image file. Returns per-pixel class indices.
    SegmentationResult segment_image(const std::string& image_path);

    const SegmentationConfig& config() const { return mConfig; }

private:
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext;
    SegmentationConfig mConfig;
    CudaStream mStream;

    // Device buffers
    CudaBuffer mInputBuffer;   // pixel_values [1, 3, H, W]
    CudaBuffer mOutputBuffer;  // logits [1, num_classes, H/4, W/4]
};

// Create from engine + config
std::unique_ptr<SegmentationBackend> CreateSegmentationBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
