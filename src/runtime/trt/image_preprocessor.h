#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

// VL preprocessing config parsed from bundle's config.json + preprocessor_config.json.
struct VLPreprocessConfig {
    int32_t fixed_image_size{448};
    int32_t patch_size{14};
    int32_t merge_size{2};
    int32_t temporal_patch_size{2};
    int32_t in_channels{3};
    float image_mean[3]{0.48145466F, 0.4578275F, 0.40821073F};
    float image_std[3]{0.26862954F, 0.26130258F, 0.27577711F};
    int32_t num_image_pad_tokens{256};
    int32_t image_token_id{-1};
    int32_t vision_output_dim{0};
    std::string vl_prompt_template;
    std::string image_token_str;
};

// Preprocessed pixel values ready for the vision TRT engine.
struct PreprocessedImage {
    std::vector<float> pixel_values;  // [C*T, H, W] with merge-group permutation
    int32_t channels{0};              // C*T (e.g. 6 for temporal_patch_size=2)
    int32_t height{0};
    int32_t width{0};
    bool ok{false};
};

// Load and preprocess an image for the vision encoder.
// Returns pixel_values in [C*T, H, W] layout with merge-group patch
// permutation and channel order [R_t0, R_t1, G_t0, G_t1, B_t0, B_t1]
// matching HF's Qwen2VLImageProcessor._preprocess exactly.
PreprocessedImage load_and_preprocess_image(
    const std::string& image_path,
    const VLPreprocessConfig& config);

// Format a VL prompt with image pad tokens from the template.
// Replaces {image_pads} and {prompt} in vl_prompt_template.
std::string format_vl_prompt(
    const std::string& user_prompt,
    const VLPreprocessConfig& config);

// Parse VLPreprocessConfig from config.json text.
VLPreprocessConfig parse_vl_preprocess_config(
    const std::string& config_text,
    const std::string& preprocessor_config_text);

} // namespace trtf
