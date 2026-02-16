#include "runtime/trt/image_preprocessor.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

#include "stb_image.h"
#include "stb_image_resize2.h"

namespace trtf {

PreprocessedImage load_and_preprocess_image(
    const std::string& image_path,
    const VLPreprocessConfig& config)
{
    PreprocessedImage result;

    // 1. Load image with stb_image (always request 3 channels = RGB)
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
    {
        std::cerr << "[trtf] Failed to load image: " << image_path
                  << " (" << stbi_failure_reason() << ")" << std::endl;
        return result;
    }

    const int target_size = config.fixed_image_size;

    // 2. Resize to fixed_image_size x fixed_image_size using Catmull-Rom (≈ bicubic)
    std::vector<unsigned char> resized(
        static_cast<std::size_t>(target_size) * target_size * 3);

    unsigned char* resize_result = stbir_resize_uint8_linear(
        raw, width, height, width * 3,
        resized.data(), target_size, target_size, target_size * 3,
        STBIR_RGB);

    stbi_image_free(raw);

    if (resize_result == nullptr)
    {
        std::cerr << "[trtf] Failed to resize image" << std::endl;
        return result;
    }

    // 3. Convert uint8 -> float32, normalize per channel: (pixel/255 - mean) / std
    //    Store as [C, H, W] (CHW) for patch extraction.
    const int T = config.temporal_patch_size;
    const int C = config.in_channels;
    const int total_channels = C * T;
    const std::size_t pixel_count = static_cast<std::size_t>(target_size) * target_size;

    // Intermediate: normalized image in [C, H, W] layout
    std::vector<float> img_chw(static_cast<std::size_t>(C) * pixel_count);
    for (int c = 0; c < C; ++c)
    {
        const float mean = config.image_mean[c];
        const float inv_std =
            (config.image_std[c] > 1e-8F) ? (1.0F / config.image_std[c]) : 1.0F;

        for (int y = 0; y < target_size; ++y)
        {
            for (int x = 0; x < target_size; ++x)
            {
                const std::size_t src_idx =
                    (static_cast<std::size_t>(y) * target_size + x) * 3 + c;
                const float pixel = static_cast<float>(resized[src_idx]) / 255.0F;

                const std::size_t dst_idx =
                    static_cast<std::size_t>(c) * pixel_count
                    + static_cast<std::size_t>(y) * target_size + x;
                img_chw[dst_idx] = (pixel - mean) * inv_std;
            }
        }
    }

    // 4. Merge-group patch permutation (matches HF Qwen2VLImageProcessor._preprocess).
    //
    // The TRT vision encoder's Conv2D produces patches in raster order.
    // HF's processor rearranges pixels so patches come out in merge-group order:
    //   for (mh, mw, dh, dw): original patch = (mh*merge+dh, mw*merge+dw)
    // We replicate this by placing each merge-group-ordered patch at the
    // corresponding raster position in a pseudo-image.
    const int pH = config.patch_size;
    const int pW = config.patch_size;
    const int grid_h = target_size / pH;
    const int grid_w = target_size / pW;
    const int merge = config.merge_size;
    const int merge_h = grid_h / merge;
    const int merge_w = grid_w / merge;

    // Build merge-group → original-patch mapping and write the pseudo-image
    // directly in [C*T, H, W] layout with channel order [C, T]:
    //   out_channel = c * T + t  →  [R_t0, R_t1, G_t0, G_t1, B_t0, B_t1]
    result.pixel_values.resize(static_cast<std::size_t>(total_channels) * pixel_count);
    result.channels = total_channels;
    result.height = target_size;
    result.width = target_size;

    int dst_patch_idx = 0;
    for (int mh = 0; mh < merge_h; ++mh)
    {
        for (int mw = 0; mw < merge_w; ++mw)
        {
            for (int dh = 0; dh < merge; ++dh)
            {
                for (int dw = 0; dw < merge; ++dw)
                {
                    // Source: original patch at (orig_h, orig_w) in the grid
                    const int orig_h = mh * merge + dh;
                    const int orig_w = mw * merge + dw;

                    // Destination: raster position dst_patch_idx in pseudo-image
                    const int dst_h = dst_patch_idx / grid_w;
                    const int dst_w = dst_patch_idx % grid_w;

                    // Copy patch pixels for each (c, t) channel
                    for (int c = 0; c < C; ++c)
                    {
                        for (int t = 0; t < T; ++t)
                        {
                            const int out_ch = c * T + t;
                            for (int py = 0; py < pH; ++py)
                            {
                                for (int px = 0; px < pW; ++px)
                                {
                                    // Source pixel from original image [C, H, W]
                                    const std::size_t src =
                                        static_cast<std::size_t>(c) * pixel_count
                                        + static_cast<std::size_t>(orig_h * pH + py) * target_size
                                        + (orig_w * pW + px);

                                    // Destination in pseudo-image [C*T, H, W]
                                    const std::size_t dst =
                                        static_cast<std::size_t>(out_ch) * pixel_count
                                        + static_cast<std::size_t>(dst_h * pH + py) * target_size
                                        + (dst_w * pW + px);

                                    result.pixel_values[dst] = img_chw[src];
                                }
                            }
                        }
                    }

                    ++dst_patch_idx;
                }
            }
        }
    }

    result.ok = true;
    return result;
}

std::string format_vl_prompt(
    const std::string& user_prompt,
    const VLPreprocessConfig& config)
{
    // Build image_pads string: repeat image_token_str num_image_pad_tokens times
    std::string image_pads;
    image_pads.reserve(
        static_cast<std::size_t>(config.num_image_pad_tokens)
        * config.image_token_str.size());
    for (int32_t i = 0; i < config.num_image_pad_tokens; ++i)
    {
        image_pads += config.image_token_str;
    }

    // Replace {image_pads} and {prompt} in the template
    std::string result = config.vl_prompt_template;

    const std::string pads_placeholder = "{image_pads}";
    const std::size_t pads_pos = result.find(pads_placeholder);
    if (pads_pos != std::string::npos)
    {
        result.replace(pads_pos, pads_placeholder.size(), image_pads);
    }

    const std::string prompt_placeholder = "{prompt}";
    const std::size_t prompt_pos = result.find(prompt_placeholder);
    if (prompt_pos != std::string::npos)
    {
        result.replace(prompt_pos, prompt_placeholder.size(), user_prompt);
    }

    return result;
}

VLPreprocessConfig parse_vl_preprocess_config(
    const std::string& config_text,
    const std::string& preprocessor_config_text)
{
    VLPreprocessConfig cfg;

    // From config.json (injected by engine_builder.py)
    cfg.image_token_id = extract_json_int(config_text, "image_token_id", -1);
    cfg.fixed_image_size = extract_json_int(config_text, "fixed_image_size", 448);
    cfg.num_image_pad_tokens = extract_json_int(config_text, "num_image_pad_tokens", 256);
    cfg.vision_output_dim = extract_json_int(config_text, "vision_output_dim", 0);
    cfg.vl_prompt_template = extract_json_string(config_text, "vl_prompt_template", "");
    cfg.image_token_str = extract_json_string(config_text, "image_token_str", "");

    // Unescape \\n -> \n in template string (JSON escaping)
    std::string& tpl = cfg.vl_prompt_template;
    std::size_t pos = 0;
    while ((pos = tpl.find("\\n", pos)) != std::string::npos)
    {
        tpl.replace(pos, 2, "\n");
        ++pos;
    }

    // From preprocessor_config.json (original HF file)
    if (!preprocessor_config_text.empty())
    {
        cfg.patch_size = extract_json_int(preprocessor_config_text, "patch_size", 14);
        cfg.merge_size = extract_json_int(preprocessor_config_text, "merge_size", 2);
        cfg.temporal_patch_size = extract_json_int(preprocessor_config_text, "temporal_patch_size", 2);

        auto mean_vals = extract_json_float_array(preprocessor_config_text, "image_mean", 3);
        if (mean_vals.size() >= 3)
        {
            cfg.image_mean[0] = mean_vals[0];
            cfg.image_mean[1] = mean_vals[1];
            cfg.image_mean[2] = mean_vals[2];
        }

        auto std_vals = extract_json_float_array(preprocessor_config_text, "image_std", 3);
        if (std_vals.size() >= 3)
        {
            cfg.image_std[0] = std_vals[0];
            cfg.image_std[1] = std_vals[1];
            cfg.image_std[2] = std_vals[2];
        }
    }

    return cfg;
}

} // namespace trtf
