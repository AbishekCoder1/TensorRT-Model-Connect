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

    // Mamba/SSM-specific fields (used when runtime_strategy == "ssm_recurrent")
    int32_t d_inner{0};      // intermediate_size / d_inner
    int32_t state_size{16};  // SSM state dimension
    int32_t conv_kernel{4};  // causal conv1d kernel size

    // Vision-Language fields (used when runtime_strategy == "vision_language")
    bool has_vision_engine{false};  // bundle contains vision_engine_plan
    bool embed_input{false};        // text decoder uses input_embed mode
    int32_t image_token_id{-1};     // special token ID for <|image_pad|>
    int32_t vision_output_dim{0};   // vision encoder output feature dim (= text hidden)
    int32_t fixed_image_size{448};  // image size the vision engine was compiled for
    int32_t num_image_pad_tokens{0}; // number of image pad tokens per image
    std::string vl_prompt_template; // prompt template with {image_pads} and {prompt}
    std::string image_token_str;    // string for one image pad token (e.g. "<|image_pad|>")
};

// Parse model configuration from config.json text for the fast path.
// max_cache_length_override: value from TRTF_MAX_CACHE_LENGTH env, or -1 to use config.
FastPathModelConfig parse_fast_path_config(const std::string& config_text, int32_t max_cache_length_override);

} // namespace trtf
