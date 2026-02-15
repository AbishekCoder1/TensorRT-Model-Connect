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
};

} // namespace trtf
