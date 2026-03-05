#pragma once

#include "runtime/trt/core/trt_common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <NvInferRuntime.h>
#endif

namespace trtf {

#if TRTF_HAS_TRT

struct VisionStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    std::string pixel_input_name{"pixel_values"};
    std::string features_output_name{"image_features"};

    int32_t num_output_features{0};  // e.g. 256 (num merged tokens)
    int32_t feature_dim{0};          // e.g. 2048 (text hidden size)
};

// Run the vision encoder on preprocessed pixel values.
// pixel_values: [T*C, H, W] float32 (from load_and_preprocess_image)
// image_features: output [num_output_features, feature_dim] float32
bool run_vision_encoder(
    const VisionStepEngine& engine,
    const float* pixel_values, std::size_t pixel_bytes,
    std::vector<float>& image_features,
    std::string& error);

// Extended: also extract deepstack_features_0..N outputs (if present in engine).
bool run_vision_encoder_with_deepstack(
    const VisionStepEngine& engine,
    const float* pixel_values, std::size_t pixel_bytes,
    std::vector<float>& image_features,
    std::vector<std::vector<float>>& deepstack_features,
    std::string& error);

#endif // TRTF_HAS_TRT

} // namespace trtf
