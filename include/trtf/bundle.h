#pragma once

#include <cstdint>
#include <string>

namespace trtf {

struct BundleInfo {
    std::string model_id;
    std::string model_type;
    std::string family;
    std::string precision;
    std::string trt_version;
    std::string trt_abi;
    std::string gpu_name;
    std::string created_at;
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{0};
    int32_t num_attention_heads{1};
    int32_t num_key_value_heads{1};
    int32_t max_cache_length{32};
    std::string runtime_strategy; // e.g. "decoder_kv_cache", "diffusion", etc.
};

// Read metadata without loading the engine.
BundleInfo InspectBundle(const std::string& bundle_path);

// Check if path is a .trtfb file (valid magic bytes).
bool IsBundle(const std::string& path);

} // namespace trtf
