#include "runtime/trt/multimodal/image_preprocessor.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

#include "stb_image.h"
#include "stb_image_resize2.h"

namespace trtf {

// ---------------------------------------------------------------------------
// Interpolation filter resolution
// ---------------------------------------------------------------------------

static stbir_filter resolve_stbir_filter(const std::string& interpolation)
{
    if (interpolation == "bilinear")
        return STBIR_FILTER_TRIANGLE;
    if (interpolation == "nearest")
        return STBIR_FILTER_POINT_SAMPLE;
    // "bicubic" or anything else -> Catmull-Rom (matches PIL BICUBIC)
    return STBIR_FILTER_CATMULLROM;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct LoadedImage {
    std::vector<float> img_chw;  // [C, H, W] normalized
    int target_size{0};
    int channels{0};
    bool ok{false};
};

// Resize raw uint8 RGB buffer to target_size x target_size using the given filter.
static std::vector<unsigned char> resize_raw(
    const unsigned char* raw, int width, int height,
    int target_size, stbir_filter filter)
{
    std::vector<unsigned char> resized(
        static_cast<std::size_t>(target_size) * target_size * 3);

    void* result = stbir_resize(
        raw, width, height, width * 3,
        resized.data(), target_size, target_size, target_size * 3,
        STBIR_RGB, STBIR_TYPE_UINT8,
        STBIR_EDGE_CLAMP, filter);

    if (result == nullptr)
    {
        return {};
    }
    return resized;
}

// Convert resized uint8 HWC buffer to float32 CHW, normalizing per channel.
static void normalize_to_chw(
    const std::vector<unsigned char>& resized,
    int target_size, const VLPreprocessConfig& config,
    std::vector<float>& out_chw)
{
    const int C = config.in_channels;
    const std::size_t pixel_count = static_cast<std::size_t>(target_size) * target_size;
    out_chw.resize(static_cast<std::size_t>(C) * pixel_count);

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
                out_chw[dst_idx] = (pixel - mean) * inv_std;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Load strategies
// ---------------------------------------------------------------------------

static LoadedImage load_resize_normalize(
    const std::string& image_path,
    const VLPreprocessConfig& config)
{
    LoadedImage loaded;

    // 1. Load image with stb_image (always request 3 channels = RGB)
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
    {
        std::cerr << "[trtf] Failed to load image: " << image_path
                  << " (" << stbi_failure_reason() << ")" << std::endl;
        return loaded;
    }

    const int target_size = config.fixed_image_size;

    // 2. Resize to fixed_image_size x fixed_image_size
    auto resized = resize_raw(raw, width, height, target_size,
                              resolve_stbir_filter(config.interpolation));
    stbi_image_free(raw);

    if (resized.empty())
    {
        std::cerr << "[trtf] Failed to resize image" << std::endl;
        return loaded;
    }

    // 3. Normalize to [C, H, W]
    normalize_to_chw(resized, target_size, config, loaded.img_chw);
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Center-crop to square, then resize + normalize.
static LoadedImage load_crop_resize_normalize(
    const std::string& image_path,
    const VLPreprocessConfig& config)
{
    LoadedImage loaded;

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
    {
        std::cerr << "[trtf] Failed to load image: " << image_path
                  << " (" << stbi_failure_reason() << ")" << std::endl;
        return loaded;
    }

    // Center-crop to square
    const int crop_size = std::min(width, height);
    const int x_off = (width - crop_size) / 2;
    const int y_off = (height - crop_size) / 2;

    std::vector<unsigned char> cropped(
        static_cast<std::size_t>(crop_size) * crop_size * 3);
    for (int y = 0; y < crop_size; ++y)
    {
        const unsigned char* src_row = raw + (static_cast<std::size_t>(y + y_off) * width + x_off) * 3;
        unsigned char* dst_row = cropped.data() + static_cast<std::size_t>(y) * crop_size * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(crop_size) * 3);
    }
    stbi_image_free(raw);

    const int target_size = config.fixed_image_size;
    auto resized = resize_raw(cropped.data(), crop_size, crop_size, target_size,
                              resolve_stbir_filter(config.interpolation));
    if (resized.empty())
    {
        std::cerr << "[trtf] Failed to resize cropped image" << std::endl;
        return loaded;
    }

    normalize_to_chw(resized, target_size, config, loaded.img_chw);
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Aspect-ratio-preserving resize + zero-pad to square, then normalize.
static LoadedImage load_aspect_preserve_resize_normalize(
    const std::string& image_path,
    const VLPreprocessConfig& config)
{
    LoadedImage loaded;

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
    {
        std::cerr << "[trtf] Failed to load image: " << image_path
                  << " (" << stbi_failure_reason() << ")" << std::endl;
        return loaded;
    }

    const int target_size = config.fixed_image_size;
    const stbir_filter filter = resolve_stbir_filter(config.interpolation);

    // Compute scaled dimensions that fit inside target_size x target_size
    const float scale = static_cast<float>(target_size) /
                        static_cast<float>(std::max(width, height));
    const int new_w = std::max(1, static_cast<int>(width * scale));
    const int new_h = std::max(1, static_cast<int>(height * scale));

    // Resize preserving aspect ratio
    std::vector<unsigned char> resized_small(
        static_cast<std::size_t>(new_w) * new_h * 3);

    void* resize_result = stbir_resize(
        raw, width, height, width * 3,
        resized_small.data(), new_w, new_h, new_w * 3,
        STBIR_RGB, STBIR_TYPE_UINT8,
        STBIR_EDGE_CLAMP, filter);

    stbi_image_free(raw);

    if (resize_result == nullptr)
    {
        std::cerr << "[trtf] Failed to resize image (aspect-preserve)" << std::endl;
        return loaded;
    }

    // Zero-pad to target_size x target_size (top-left aligned)
    std::vector<unsigned char> padded(
        static_cast<std::size_t>(target_size) * target_size * 3, 0);
    for (int y = 0; y < new_h; ++y)
    {
        const unsigned char* src_row = resized_small.data() + static_cast<std::size_t>(y) * new_w * 3;
        unsigned char* dst_row = padded.data() + static_cast<std::size_t>(y) * target_size * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(new_w) * 3);
    }

    normalize_to_chw(padded, target_size, config, loaded.img_chw);
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Aspect-ratio-preserving resize + center-pad with mean color, then normalize.
// Matches PIL ImageOps.pad(image, (size, size), color=mean*255).
static LoadedImage load_pad_center_resize_normalize(
    const std::string& image_path,
    const VLPreprocessConfig& config)
{
    LoadedImage loaded;

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
    {
        std::cerr << "[trtf] Failed to load image: " << image_path
                  << " (" << stbi_failure_reason() << ")" << std::endl;
        return loaded;
    }

    const int target_size = config.fixed_image_size;
    const stbir_filter filter = resolve_stbir_filter(config.interpolation);

    // Compute scaled dimensions that fit inside target_size x target_size
    // This matches PIL ImageOps.pad behavior: scale to fit, then center.
    const float scale_w = static_cast<float>(target_size) / static_cast<float>(width);
    const float scale_h = static_cast<float>(target_size) / static_cast<float>(height);
    const float scale = std::min(scale_w, scale_h);
    const int new_w = std::max(1, static_cast<int>(width * scale));
    const int new_h = std::max(1, static_cast<int>(height * scale));

    // Resize preserving aspect ratio
    std::vector<unsigned char> resized_small(
        static_cast<std::size_t>(new_w) * new_h * 3);

    void* resize_result = stbir_resize(
        raw, width, height, width * 3,
        resized_small.data(), new_w, new_h, new_w * 3,
        STBIR_RGB, STBIR_TYPE_UINT8,
        STBIR_EDGE_CLAMP, filter);

    stbi_image_free(raw);

    if (resize_result == nullptr)
    {
        std::cerr << "[trtf] Failed to resize image (pad-center)" << std::endl;
        return loaded;
    }

    // Fill pad with mean color (mean * 255), matching ImageOps.pad color arg
    const unsigned char pad_r = static_cast<unsigned char>(config.image_mean[0] * 255.0F);
    const unsigned char pad_g = static_cast<unsigned char>(config.image_mean[1] * 255.0F);
    const unsigned char pad_b = static_cast<unsigned char>(config.image_mean[2] * 255.0F);

    std::vector<unsigned char> padded(
        static_cast<std::size_t>(target_size) * target_size * 3);
    for (std::size_t i = 0; i < padded.size(); i += 3)
    {
        padded[i + 0] = pad_r;
        padded[i + 1] = pad_g;
        padded[i + 2] = pad_b;
    }

    // Center the resized image in the padded canvas
    const int x_off = (target_size - new_w) / 2;
    const int y_off = (target_size - new_h) / 2;
    for (int y = 0; y < new_h; ++y)
    {
        const unsigned char* src_row = resized_small.data()
            + static_cast<std::size_t>(y) * new_w * 3;
        unsigned char* dst_row = padded.data()
            + (static_cast<std::size_t>(y + y_off) * target_size + x_off) * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(new_w) * 3);
    }

    normalize_to_chw(padded, target_size, config, loaded.img_chw);
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// ---------------------------------------------------------------------------
// Strategy: qwen_merge_group
// ---------------------------------------------------------------------------

static PreprocessedImage preprocess_qwen_merge_group(
    const LoadedImage& loaded,
    const VLPreprocessConfig& config)
{
    PreprocessedImage result;

    const int target_size = loaded.target_size;
    const int C = loaded.channels;
    const int T = config.temporal_patch_size;
    const int total_channels = C * T;
    const std::size_t pixel_count = static_cast<std::size_t>(target_size) * target_size;

    // Merge-group patch permutation (matches HF Qwen2VLImageProcessor._preprocess).
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

    // Build merge-group -> original-patch mapping and write the pseudo-image
    // directly in [C*T, H, W] layout with channel order [C, T]:
    //   out_channel = c * T + t  ->  [R_t0, R_t1, G_t0, G_t1, B_t0, B_t1]
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

                                    result.pixel_values[dst] = loaded.img_chw[src];
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

// ---------------------------------------------------------------------------
// Strategy: simple_chw
// ---------------------------------------------------------------------------

static PreprocessedImage preprocess_simple_chw(
    const LoadedImage& loaded,
    const VLPreprocessConfig& config)
{
    PreprocessedImage result;

    // Standard [C, H, W] — already produced by load_resize_normalize.
    // No patch permutation, no temporal duplication.
    result.pixel_values = loaded.img_chw;
    result.channels = loaded.channels;
    result.height = loaded.target_size;
    result.width = loaded.target_size;
    result.ok = true;

    (void)config;  // only image_mean/std/fixed_image_size used (already in loaded)
    return result;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

using LoadImageFn = LoadedImage (*)(
    const std::string& image_path,
    const VLPreprocessConfig& config);

using PreprocessImageFn = PreprocessedImage (*)(
    const LoadedImage& loaded,
    const VLPreprocessConfig& config);

struct PreprocessDispatch {
    LoadImageFn load_fn;
    PreprocessImageFn preprocess_fn;
    bool warn_unknown_type{false};
};

static PreprocessDispatch resolve_preprocess_dispatch(const std::string& preprocessor_type)
{
    if (preprocessor_type == "center_crop_chw")
        return {load_crop_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "aspect_preserve_chw")
        return {load_aspect_preserve_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "pad_center_chw")
        return {load_pad_center_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "simple_chw")
        return {load_resize_normalize, preprocess_simple_chw, false};

    const bool warn_unknown = (preprocessor_type != "qwen_merge_group");
    return {load_resize_normalize, preprocess_qwen_merge_group, warn_unknown};
}

static PreprocessedImage run_preprocess_dispatch(
    const std::string& image_path,
    const VLPreprocessConfig& config,
    const PreprocessDispatch& dispatch)
{
    LoadedImage loaded = dispatch.load_fn(image_path, config);
    if (!loaded.ok)
    {
        return PreprocessedImage{};
    }
    return dispatch.preprocess_fn(loaded, config);
}

PreprocessedImage load_and_preprocess_image(
    const std::string& image_path,
    const VLPreprocessConfig& config)
{
    const auto dispatch = resolve_preprocess_dispatch(config.preprocessor_type);
    if (dispatch.warn_unknown_type)
    {
        std::cerr << "[trtf] WARNING: Unknown preprocessor_type \""
                  << config.preprocessor_type << "\", falling back to qwen_merge_group"
                  << std::endl;
    }

    return run_preprocess_dispatch(image_path, config, dispatch);
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

static void assign_image_norm_triplet(const std::vector<float>& values, float (&target)[3])
{
    if (values.size() < 3)
    {
        return;
    }
    target[0] = values[0];
    target[1] = values[1];
    target[2] = values[2];
}

static void unescape_newline_sequences(std::string& text)
{
    std::size_t pos = 0;
    while ((pos = text.find("\\n", pos)) != std::string::npos)
    {
        text.replace(pos, 2, "\n");
        ++pos;
    }
}

static void parse_base_vl_config(
    const std::string& config_text,
    VLPreprocessConfig& cfg)
{
    cfg.preprocessor_type = extract_json_string(config_text, "preprocessor_type", "qwen_merge_group");
    cfg.image_token_id = extract_json_int(config_text, "image_token_id", -1);
    cfg.fixed_image_size = extract_json_int(config_text, "fixed_image_size", 448);
    cfg.num_image_pad_tokens = extract_json_int(config_text, "num_image_pad_tokens", 256);
    cfg.vision_output_dim = extract_json_int(config_text, "vision_output_dim", 0);
    cfg.vl_prompt_template = extract_json_string(config_text, "vl_prompt_template", "");
    cfg.image_token_str = extract_json_string(config_text, "image_token_str", "");
    cfg.interpolation = extract_json_string(config_text, "interpolation", "bicubic");
}

static void maybe_apply_resample_fallback(
    const std::string& config_text,
    const std::string& preprocessor_config_text,
    VLPreprocessConfig& cfg)
{
    if (!extract_json_string(config_text, "interpolation", "").empty())
    {
        return;
    }

    const int resample = extract_json_int(preprocessor_config_text, "resample", -1);
    if (resample == 0)
        cfg.interpolation = "nearest";
    else if (resample == 2)
        cfg.interpolation = "bilinear";
    else if (resample == 3)
        cfg.interpolation = "bicubic";
}

static void apply_preprocessor_config_overrides(
    const std::string& config_text,
    const std::string& preprocessor_config_text,
    VLPreprocessConfig& cfg)
{
    if (preprocessor_config_text.empty())
    {
        return;
    }

    cfg.patch_size = extract_json_int(preprocessor_config_text, "patch_size", 14);
    cfg.merge_size = extract_json_int(preprocessor_config_text, "merge_size", 2);
    cfg.temporal_patch_size = extract_json_int(preprocessor_config_text, "temporal_patch_size", 2);

    auto mean_vals = extract_json_float_array(preprocessor_config_text, "image_mean", 3);
    assign_image_norm_triplet(mean_vals, cfg.image_mean);

    auto std_vals = extract_json_float_array(preprocessor_config_text, "image_std", 3);
    assign_image_norm_triplet(std_vals, cfg.image_std);

    maybe_apply_resample_fallback(config_text, preprocessor_config_text, cfg);
}

static void apply_config_image_norm_overrides(
    const std::string& config_text,
    VLPreprocessConfig& cfg)
{
    auto mean_vals = extract_json_float_array(config_text, "image_mean", 3);
    assign_image_norm_triplet(mean_vals, cfg.image_mean);

    auto std_vals = extract_json_float_array(config_text, "image_std", 3);
    assign_image_norm_triplet(std_vals, cfg.image_std);
}

VLPreprocessConfig parse_vl_preprocess_config(
    const std::string& config_text,
    const std::string& preprocessor_config_text)
{
    VLPreprocessConfig cfg;
    parse_base_vl_config(config_text, cfg);
    unescape_newline_sequences(cfg.vl_prompt_template);
    apply_preprocessor_config_overrides(config_text, preprocessor_config_text, cfg);
    apply_config_image_norm_overrides(config_text, cfg);

    return cfg;
}

} // namespace trtf
