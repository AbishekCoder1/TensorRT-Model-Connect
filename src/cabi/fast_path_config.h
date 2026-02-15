#pragma once

#include <cstdint>
#include <string>

namespace trtf {

struct FastPathModelConfig {
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{1};
    int32_t num_heads{1};
    int32_t num_kv_heads{1};
    int32_t head_dim{0};
    int32_t attention_size{0};
    int32_t max_cache_length{32};
    int32_t id_bos{-1};
    int32_t id_eos{-1};
};

// Parse model configuration from config.json text for the fast path.
// max_cache_length_override: value from TRTF_MAX_CACHE_LENGTH env, or -1 to use config.
FastPathModelConfig parse_fast_path_config(const std::string& config_text, int32_t max_cache_length_override);

} // namespace trtf
