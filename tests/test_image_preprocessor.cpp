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

// Test: load and preprocess a small test image.
static void test_load_and_preprocess()
{
    // Create temp directory
    char temp_pattern[] = "/tmp/claude/trtf_test_img_XXXXXX";
    char* temp_dir = mkdtemp(temp_pattern);
    if (temp_dir == nullptr)
    {
        std::cerr << "SKIP: test_load_and_preprocess (cannot create temp dir)\n";
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
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtf::load_and_preprocess_image(image_path, config);

    check(result.ok, "image loaded successfully");
    check(result.channels == 6, "channels = T*C = 2*3 = 6");
    check(result.height == 8, "height = fixed_image_size = 8");
    check(result.width == 8, "width = fixed_image_size = 8");

    const std::size_t expected_size = 6 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "pixel_values size = channels * H * W");

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
    check(in_range, "all normalized values in [-1.1, 1.1]");

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
        "image_token_str": "<|image_pad|>"
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

    // Check float array parsing
    check(std::abs(cfg.image_mean[0] - 0.48145466F) < 1e-5F, "image_mean[0]");
    check(std::abs(cfg.image_mean[1] - 0.4578275F) < 1e-5F, "image_mean[1]");
    check(std::abs(cfg.image_std[2] - 0.27577711F) < 1e-5F, "image_std[2]");
}

int main()
{
    test_load_and_preprocess();
    test_load_missing_image();
    test_format_vl_prompt();
    test_parse_vl_config();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All image preprocessor tests passed.\n";
    return 0;
}
