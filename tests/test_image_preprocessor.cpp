// =============================================================================
// Test suite: VL image preprocessing (stb_image-based)
// =============================================================================
//
// Tests load_and_preprocess_image(), format_vl_prompt(), and
// parse_vl_preprocess_config() from image_preprocessor.h.
//
// These tests are CPU-only, no GPU/TRT required. Image loading tests use
// a small in-memory PPM image written to a temp file.
// =============================================================================

#include "runtime/trt/image_preprocessor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Write a tiny 4x4 PPM image (binary format) to a file.
static std::string write_test_ppm(const std::string& dir)
{
    const std::string path = dir + "/test_image.ppm";
    std::ofstream out(path, std::ios::binary);
    out << "P6\n4 4\n255\n";
    // 4x4 pixels, each RGB
    for (int i = 0; i < 16; ++i)
    {
        unsigned char r = static_cast<unsigned char>(i * 16);
        unsigned char g = static_cast<unsigned char>(128);
        unsigned char b = static_cast<unsigned char>(255 - i * 16);
        out.put(static_cast<char>(r));
        out.put(static_cast<char>(g));
        out.put(static_cast<char>(b));
    }
    out.close();
    return path;
}

// Test: qwen_merge_group strategy — default, produces [C*T, H, W] with permutation.
static void test_qwen_merge_group_strategy()
{
    // Create temp directory
    char temp_pattern[] = "/tmp/claude/trtf_test_img_XXXXXX";
    char* temp_dir = mkdtemp(temp_pattern);
    if (temp_dir == nullptr)
    {
        std::cerr << "SKIP: test_qwen_merge_group_strategy (cannot create temp dir)\n";
        return;
    }
    const std::string dir(temp_dir);

    // Write test image
    const std::string image_path = write_test_ppm(dir);

    // Configure for a small fixed size
    trtf::VLPreprocessConfig config;
    config.fixed_image_size = 8;  // small for testing
    config.temporal_patch_size = 2;
    config.in_channels = 3;
    config.preprocessor_type = "qwen_merge_group";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtf::load_and_preprocess_image(image_path, config);

    check(result.ok, "qwen_merge_group: image loaded successfully");
    check(result.channels == 6, "qwen_merge_group: channels = T*C = 2*3 = 6");
    check(result.height == 8, "qwen_merge_group: height = fixed_image_size = 8");
    check(result.width == 8, "qwen_merge_group: width = fixed_image_size = 8");

    const std::size_t expected_size = 6 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "qwen_merge_group: pixel_values size = channels * H * W");

    // Check normalization range: (pixel/255 - 0.5) / 0.5 is in [-1, 1]
    bool in_range = true;
    for (float v : result.pixel_values)
    {
        if (v < -1.1F || v > 1.1F)
        {
            in_range = false;
            break;
        }
    }
    check(in_range, "qwen_merge_group: all normalized values in [-1.1, 1.1]");

    // Cleanup
    std::filesystem::remove_all(dir);
}

// Test: simple_chw strategy — produces [C, H, W] without permutation.
static void test_simple_chw_strategy()
{
    // Create temp directory
    char temp_pattern[] = "/tmp/claude/trtf_test_img_chw_XXXXXX";
    char* temp_dir = mkdtemp(temp_pattern);
    if (temp_dir == nullptr)
    {
        std::cerr << "SKIP: test_simple_chw_strategy (cannot create temp dir)\n";
        return;
    }
    const std::string dir(temp_dir);

    // Write test image
    const std::string image_path = write_test_ppm(dir);

    trtf::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.in_channels = 3;
    config.preprocessor_type = "simple_chw";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtf::load_and_preprocess_image(image_path, config);

    check(result.ok, "simple_chw: image loaded successfully");
    check(result.channels == 3, "simple_chw: channels = C = 3 (no temporal)");
    check(result.height == 8, "simple_chw: height = fixed_image_size = 8");
    check(result.width == 8, "simple_chw: width = fixed_image_size = 8");

    const std::size_t expected_size = 3 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "simple_chw: pixel_values size = C * H * W");

    // Check normalization range: (pixel/255 - 0.5) / 0.5 is in [-1, 1]
    bool in_range = true;
    for (float v : result.pixel_values)
    {
        if (v < -1.1F || v > 1.1F)
        {
            in_range = false;
            break;
        }
    }
    check(in_range, "simple_chw: all normalized values in [-1.1, 1.1]");

    // Cleanup
    std::filesystem::remove_all(dir);
}

// Test: load non-existent image returns ok=false.
static void test_load_missing_image()
{
    trtf::VLPreprocessConfig config;
    config.fixed_image_size = 8;

    auto result = trtf::load_and_preprocess_image("/nonexistent/image.jpg", config);
    check(!result.ok, "missing image returns ok=false");
}

// Test: format_vl_prompt replaces {image_pads} and {prompt}.
static void test_format_vl_prompt()
{
    trtf::VLPreprocessConfig config;
    config.num_image_pad_tokens = 3;
    config.image_token_str = "<|pad|>";
    config.vl_prompt_template = "USER: {image_pads}\n{prompt}\nASST:";

    const std::string result = trtf::format_vl_prompt("Describe this", config);

    // Should contain 3 copies of <|pad|>
    check(result.find("<|pad|><|pad|><|pad|>") != std::string::npos,
          "3 image pad tokens present");

    // Should contain the user prompt
    check(result.find("Describe this") != std::string::npos,
          "user prompt present");

    // Should have the template structure
    check(result.find("USER: ") == 0, "starts with USER: ");
    check(result.find("ASST:") != std::string::npos, "ends with ASST:");
}

// Test: parse_vl_preprocess_config extracts fields correctly.
static void test_parse_vl_config()
{
    const std::string config_json = R"({
        "image_token_id": 151655,
        "fixed_image_size": 448,
        "num_image_pad_tokens": 256,
        "vision_output_dim": 2048,
        "vl_prompt_template": "test {image_pads} {prompt}",
        "image_token_str": "<|image_pad|>",
        "preprocessor_type": "qwen_merge_group"
    })";

    const std::string preproc_json = R"({
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711]
    })";

    auto cfg = trtf::parse_vl_preprocess_config(config_json, preproc_json);

    check(cfg.image_token_id == 151655, "image_token_id = 151655");
    check(cfg.fixed_image_size == 448, "fixed_image_size = 448");
    check(cfg.num_image_pad_tokens == 256, "num_image_pad_tokens = 256");
    check(cfg.vision_output_dim == 2048, "vision_output_dim = 2048");
    check(cfg.patch_size == 14, "patch_size = 14");
    check(cfg.merge_size == 2, "merge_size = 2");
    check(cfg.temporal_patch_size == 2, "temporal_patch_size = 2");
    check(cfg.image_token_str == "<|image_pad|>", "image_token_str parsed");
    check(cfg.preprocessor_type == "qwen_merge_group", "preprocessor_type parsed");

    // Check float array parsing
    check(std::abs(cfg.image_mean[0] - 0.48145466F) < 1e-5F, "image_mean[0]");
    check(std::abs(cfg.image_mean[1] - 0.4578275F) < 1e-5F, "image_mean[1]");
    check(std::abs(cfg.image_std[2] - 0.27577711F) < 1e-5F, "image_std[2]");
}

// Test: preprocessor_type defaults to "qwen_merge_group" when absent.
static void test_parse_vl_config_default_preprocessor_type()
{
    const std::string config_json = R"({
        "image_token_id": 100
    })";

    auto cfg = trtf::parse_vl_preprocess_config(config_json, "");
    check(cfg.preprocessor_type == "qwen_merge_group",
          "preprocessor_type defaults to qwen_merge_group");
}

// Test: preprocessor_type = "simple_chw" round-trips through parse.
static void test_parse_vl_config_simple_chw()
{
    const std::string config_json = R"({
        "preprocessor_type": "simple_chw"
    })";

    auto cfg = trtf::parse_vl_preprocess_config(config_json, "");
    check(cfg.preprocessor_type == "simple_chw",
          "preprocessor_type simple_chw parsed correctly");
}

// Write a non-square 6x4 PPM image (binary format) to a file.
static std::string write_test_ppm_nonsquare(const std::string& dir)
{
    const std::string path = dir + "/test_nonsquare.ppm";
    std::ofstream out(path, std::ios::binary);
    out << "P6\n6 4\n255\n";
    // 6x4 pixels, each RGB
    for (int i = 0; i < 24; ++i)
    {
        unsigned char r = static_cast<unsigned char>(i * 10);
        unsigned char g = static_cast<unsigned char>(128);
        unsigned char b = static_cast<unsigned char>(255 - i * 10);
        out.put(static_cast<char>(r));
        out.put(static_cast<char>(g));
        out.put(static_cast<char>(b));
    }
    out.close();
    return path;
}

// Test Gap 1: unknown preprocessor_type falls back to qwen_merge_group with ok=true.
static void test_unknown_preprocessor_type_fallback()
{
    char temp_pattern[] = "/tmp/claude/trtf_test_unknown_XXXXXX";
    char* temp_dir = mkdtemp(temp_pattern);
    if (temp_dir == nullptr)
    {
        std::cerr << "SKIP: test_unknown_preprocessor_type_fallback (cannot create temp dir)\n";
        return;
    }
    const std::string dir(temp_dir);
    const std::string image_path = write_test_ppm(dir);

    trtf::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.temporal_patch_size = 2;
    config.in_channels = 3;
    config.preprocessor_type = "bogus";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtf::load_and_preprocess_image(image_path, config);

    check(result.ok, "unknown type fallback: ok=true");
    // Should produce qwen_merge_group output (C*T channels)
    check(result.channels == 6, "unknown type fallback: channels = C*T = 6");
    check(result.height == 8, "unknown type fallback: height = 8");
    check(result.width == 8, "unknown type fallback: width = 8");

    std::filesystem::remove_all(dir);
}

// Test Gap 2: center_crop_chw strategy with non-square image.
static void test_center_crop_chw_strategy()
{
    char temp_pattern[] = "/tmp/claude/trtf_test_crop_XXXXXX";
    char* temp_dir = mkdtemp(temp_pattern);
    if (temp_dir == nullptr)
    {
        std::cerr << "SKIP: test_center_crop_chw_strategy (cannot create temp dir)\n";
        return;
    }
    const std::string dir(temp_dir);
    const std::string image_path = write_test_ppm_nonsquare(dir);

    trtf::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.in_channels = 3;
    config.preprocessor_type = "center_crop_chw";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtf::load_and_preprocess_image(image_path, config);

    check(result.ok, "center_crop_chw: ok=true");
    check(result.channels == 3, "center_crop_chw: channels = 3");
    check(result.height == 8, "center_crop_chw: height = 8");
    check(result.width == 8, "center_crop_chw: width = 8");

    const std::size_t expected_size = 3 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "center_crop_chw: pixel_values size = C * H * W");

    bool in_range = true;
    for (float v : result.pixel_values)
    {
        if (v < -1.1F || v > 1.1F)
        {
            in_range = false;
            break;
        }
    }
    check(in_range, "center_crop_chw: all normalized values in [-1.1, 1.1]");

    std::filesystem::remove_all(dir);
}

// Test Gap 3: aspect_preserve_chw strategy with non-square image.
static void test_aspect_preserve_chw_strategy()
{
    char temp_pattern[] = "/tmp/claude/trtf_test_aspect_XXXXXX";
    char* temp_dir = mkdtemp(temp_pattern);
    if (temp_dir == nullptr)
    {
        std::cerr << "SKIP: test_aspect_preserve_chw_strategy (cannot create temp dir)\n";
        return;
    }
    const std::string dir(temp_dir);
    const std::string image_path = write_test_ppm_nonsquare(dir);

    trtf::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.in_channels = 3;
    config.preprocessor_type = "aspect_preserve_chw";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtf::load_and_preprocess_image(image_path, config);

    check(result.ok, "aspect_preserve_chw: ok=true");
    check(result.channels == 3, "aspect_preserve_chw: channels = 3");
    check(result.height == 8, "aspect_preserve_chw: height = 8");
    check(result.width == 8, "aspect_preserve_chw: width = 8");

    const std::size_t expected_size = 3 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "aspect_preserve_chw: pixel_values size = C * H * W");

    // Padded region (bottom rows) should have normalized-zero values:
    // (0/255 - 0.5) / 0.5 = -1.0
    // The 6x4 image scaled to fit 8x8 -> new_w=8, new_h=5 (6/6*8=8, 4/6*8=5.33->5)
    // So rows 5-7 should be padded zeros -> normalized to -1.0
    // Check last row of first channel
    const float expected_pad = (0.0F / 255.0F - 0.5F) / 0.5F;  // -1.0
    bool pad_ok = true;
    for (int x = 0; x < 8; ++x)
    {
        // Channel 0, row 7, col x
        const std::size_t idx = static_cast<std::size_t>(0) * 64 + 7 * 8 + x;
        if (std::abs(result.pixel_values[idx] - expected_pad) > 0.01F)
        {
            pad_ok = false;
            break;
        }
    }
    check(pad_ok, "aspect_preserve_chw: padded rows have correct normalized zero value");

    std::filesystem::remove_all(dir);
}

// Test Gap 4: interpolation defaults to "bicubic".
static void test_parse_interpolation_default()
{
    const std::string config_json = R"({
        "preprocessor_type": "simple_chw"
    })";

    auto cfg = trtf::parse_vl_preprocess_config(config_json, "");
    check(cfg.interpolation == "bicubic",
          "interpolation defaults to bicubic");
}

// Test Gap 4: interpolation = "bilinear" round-trips.
static void test_parse_interpolation_bilinear()
{
    const std::string config_json = R"({
        "interpolation": "bilinear"
    })";

    auto cfg = trtf::parse_vl_preprocess_config(config_json, "");
    check(cfg.interpolation == "bilinear",
          "interpolation bilinear parsed from config.json");
}

// Test Gap 4: resample int from preprocessor_config.json maps correctly.
static void test_parse_resample_from_preprocessor()
{
    // config.json does NOT set interpolation -> fallback to resample int
    const std::string config_json = R"({
        "preprocessor_type": "simple_chw"
    })";

    const std::string preproc_json = R"({
        "resample": 2
    })";

    auto cfg = trtf::parse_vl_preprocess_config(config_json, preproc_json);
    check(cfg.interpolation == "bilinear",
          "resample=2 maps to bilinear");

    // Test resample=3 -> bicubic
    const std::string preproc_json3 = R"({
        "resample": 3
    })";
    auto cfg3 = trtf::parse_vl_preprocess_config(config_json, preproc_json3);
    check(cfg3.interpolation == "bicubic",
          "resample=3 maps to bicubic");

    // Test resample=0 -> nearest
    const std::string preproc_json0 = R"({
        "resample": 0
    })";
    auto cfg0 = trtf::parse_vl_preprocess_config(config_json, preproc_json0);
    check(cfg0.interpolation == "nearest",
          "resample=0 maps to nearest");

    // Test: explicit config.json interpolation overrides resample
    const std::string config_explicit = R"({
        "interpolation": "nearest"
    })";
    auto cfg_override = trtf::parse_vl_preprocess_config(config_explicit, preproc_json);
    check(cfg_override.interpolation == "nearest",
          "explicit interpolation overrides resample");
}

int main()
{
    test_qwen_merge_group_strategy();
    test_simple_chw_strategy();
    test_load_missing_image();
    test_format_vl_prompt();
    test_parse_vl_config();
    test_parse_vl_config_default_preprocessor_type();
    test_parse_vl_config_simple_chw();
    test_unknown_preprocessor_type_fallback();
    test_center_crop_chw_strategy();
    test_aspect_preserve_chw_strategy();
    test_parse_interpolation_default();
    test_parse_interpolation_bilinear();
    test_parse_resample_from_preprocessor();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All image preprocessor tests passed.\n";
    return 0;
}
