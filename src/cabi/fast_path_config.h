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

    // Runtime strategy determines which backend and state management to use.
    // "decoder_kv_cache" (default): standard attention-based decoder with KV cache
    // "decoder_moe":  MoE decoder (same KV cache, routing in TRT graph)
    // "ssm_recurrent": Mamba/SSM (conv_state + ssm_state, no KV cache)
    // "vision_language": two-engine (vision encoder + text decoder)
    std::string runtime_strategy{"decoder_kv_cache"};
};

// Parse model configuration from config.json text for the fast path.
// max_cache_length_override: value from TRTF_MAX_CACHE_LENGTH env, or -1 to use config.
FastPathModelConfig parse_fast_path_config(const std::string& config_text, int32_t max_cache_length_override);

} // namespace trtf
