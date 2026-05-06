#pragma once

#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct VisionStepEngine {
    TrtModule* module{nullptr};

    std::string pixel_input_name{"pixel_values"};
    std::string features_output_name{"image_features"};

    int32_t num_output_features{0}; // e.g. 256 (num merged tokens)
    int32_t feature_dim{0};         // e.g. 2048 (text hidden size)
};

// Run the vision encoder on preprocessed pixel values.
// pixel_values: [T*C, H, W] float32 (from load_and_preprocess_image)
// image_features: output [num_output_features, feature_dim] float32
bool run_vision_encoder(const VisionStepEngine& engine, const float* pixel_values,
                        std::size_t pixel_bytes, std::vector<float>& image_features,
                        std::string& error);

// Extended: also extract deepstack_features_0..N outputs (if present in engine).
bool run_vision_encoder_with_deepstack(const VisionStepEngine& engine, const float* pixel_values,
                                       std::size_t pixel_bytes, std::vector<float>& image_features,
                                       std::vector<std::vector<float>>& deepstack_features,
                                       std::string& error);

} // namespace trtmc
