#pragma once

#include <cstddef>
#include <string>

namespace trtf {

struct GenerationConfig {
    std::size_t max_new_tokens{20};
    bool do_sample{false};
    float temperature{1.0F};
};

struct GenerationResult {
    std::string generated_text;
};

} // namespace trtf
