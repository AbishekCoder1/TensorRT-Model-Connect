#pragma once

#include "runtime/core/decoded_image.h"
#include "runtime/domains/perception/perception_types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

inline std::vector<float>
preprocess_segmentation_image(const runtime::adapters::io::DecodedImage& image,
                              const SegmentationConfig& config) {
    if (image.empty()) {
        throw std::runtime_error("Failed to preprocess segmentation image: decoded image missing");
    }

    const int32_t input_h = config.input_image_h;
    const int32_t input_w = config.input_image_w;
    std::vector<float> pixel_values(static_cast<std::size_t>(3) * input_h * input_w);

    for (int32_t y = 0; y < input_h; ++y) {
        for (int32_t x = 0; x < input_w; ++x) {
            const float src_y = static_cast<float>(y) * static_cast<float>(image.height) /
                                static_cast<float>(input_h);
            const float src_x = static_cast<float>(x) * static_cast<float>(image.width) /
                                static_cast<float>(input_w);
            const int32_t y0 = std::min(static_cast<int32_t>(src_y), image.height - 1);
            const int32_t x0 = std::min(static_cast<int32_t>(src_x), image.width - 1);
            const auto src_idx = static_cast<std::size_t>((y0 * image.width + x0) * image.channels);
            for (int32_t c = 0; c < 3; ++c) {
                float value = static_cast<float>(image.pixels[src_idx + c]) / 255.0F;
                value = (value - config.image_mean[c]) / config.image_std[c];
                pixel_values[static_cast<std::size_t>(c) * input_h * input_w +
                             static_cast<std::size_t>(y) * input_w + x] = value;
            }
        }
    }

    return pixel_values;
}

} // namespace trtmc
