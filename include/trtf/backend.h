#pragma once

#include "trtf/generation.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

class IGenerationBackend {
public:
    virtual ~IGenerationBackend() = default;

    virtual bool is_available() const = 0;
    virtual const char* name() const = 0;

    virtual const char* unavailable_reason() const
    {
        return "";
    }

    virtual std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) = 0;

    // VL generation: text with interleaved image features.
    // Default implementation ignores image features and falls back to text-only.
    virtual std::vector<int32_t> generate_vl(
        const std::vector<int32_t>& input_ids,
        const float* image_features, int32_t num_features, int32_t feature_dim,
        const std::vector<int32_t>& image_positions,
        const GenerationConfig& config)
    {
        (void) image_features;
        (void) num_features;
        (void) feature_dim;
        (void) image_positions;
        return generate(input_ids, config);
    }

    // Returns true if this backend supports VL generation.
    virtual bool supports_vision() const { return false; }
};

} // namespace trtf
