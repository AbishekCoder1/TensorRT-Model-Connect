#pragma once

#include <cstdint>
#include <string>

namespace trtf {

struct BundleInfo {
    std::string model_id;
    std::string model_type;
    std::string family;
    std::string trt_version;
    std::string gpu_name;
    std::string created_at;
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{0};
    int32_t num_attention_heads{1};
    int32_t num_key_value_heads{1};
    int32_t max_cache_length{32};
};

// Compile model -> save as .trtfb bundle.
void BuildBundle(const std::string& model_dir,
    const std::string& output_path,
    int max_cache_length = -1,
    const std::string& hf_python = "");

// Read metadata without loading the engine.
BundleInfo InspectBundle(const std::string& bundle_path);

// Check if path is a .trtfb file (valid magic bytes).
bool IsBundle(const std::string& path);

} // namespace trtf
