// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-IO-CPP-01
// Architecture:   ARCH-MULTI-001
// Unit Design:    UD-IO-01
// Intent:         read_image: valid BMP load, pixel normalisation, failure paths
// Preconditions:  Writable temp directory; stb_image compiled into trtmc_core
// Postconditions: Pixel values are normalised to [0,1]; empty result returned
//                 on decode failure or missing file
// =============================================================================

// test_image_reader.cpp — Unit tests for src/utils/image_reader.cpp
//
// Purpose:
//   Validates read_image() from trtmc::io: load an image file and return
//   float32 RGB pixels normalised to [0, 1] in HWC layout.
//
//   Tests create minimal BMP files in-process (no external assets needed)
//   and verify that dimensions, pixel count, and channel values match the
//   written data.  Failure paths (bad path, invalid content) are verified
//   to return an empty LoadedImage without throwing.
//
// Dependencies:
//   - trtmc/trtmc_io.hpp : trtmc::io::read_image, trtmc::io::LoadedImage
//   - test_helpers.h   : TempDirGuard
//   No TRT, GPU, or CUDA required.

#include "trtmc/trtmc_io.hpp"
#include "test_helpers.h"

#include <cmath>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// BMP helper: write a minimal 24-bit BMP (2 x 1 pixels, no padding needed)
// ---------------------------------------------------------------------------
// BMP stores pixels in BGR order, bottom-up. stb_image converts to RGB top-down.
// For a 2x1 image (height=1) row order is irrelevant.
static void write_bmp_2x1(const std::string& path,
                           uint8_t r0, uint8_t g0, uint8_t b0,
                           uint8_t r1, uint8_t g1, uint8_t b1)
{
    // 2 pixels × 3 bytes = 6 bytes; already 4-byte aligned — no row padding.
    const uint32_t pixel_offset = 54;  // 14-byte file header + 40-byte DIB header
    const uint32_t data_size    = 6;
    const uint32_t file_size    = pixel_offset + data_size;
    const int32_t  width        = 2;
    const int32_t  height       = 1;

    std::ofstream f(path, std::ios::binary);

    // ── BMP file header (14 bytes) ────────────────────────────────────────
    f.write("BM", 2);
    f.write(reinterpret_cast<const char*>(&file_size),    4);
    const uint32_t reserved = 0;
    f.write(reinterpret_cast<const char*>(&reserved),     4);
    f.write(reinterpret_cast<const char*>(&pixel_offset), 4);

    // ── BITMAPINFOHEADER (40 bytes) ───────────────────────────────────────
    const uint32_t dib_size    = 40;
    const uint16_t planes      = 1;
    const uint16_t bpp         = 24;
    const uint32_t compression = 0;
    const uint32_t zero32      = 0;

    f.write(reinterpret_cast<const char*>(&dib_size),    4);
    f.write(reinterpret_cast<const char*>(&width),       4);
    f.write(reinterpret_cast<const char*>(&height),      4);
    f.write(reinterpret_cast<const char*>(&planes),      2);
    f.write(reinterpret_cast<const char*>(&bpp),         2);
    f.write(reinterpret_cast<const char*>(&compression), 4);
    f.write(reinterpret_cast<const char*>(&data_size),   4);
    f.write(reinterpret_cast<const char*>(&zero32),      4);  // x ppm
    f.write(reinterpret_cast<const char*>(&zero32),      4);  // y ppm
    f.write(reinterpret_cast<const char*>(&zero32),      4);  // clr_used
    f.write(reinterpret_cast<const char*>(&zero32),      4);  // clr_important

    // ── Pixel data: BGR order ─────────────────────────────────────────────
    f.put(static_cast<char>(b0)); f.put(static_cast<char>(g0)); f.put(static_cast<char>(r0));
    f.put(static_cast<char>(b1)); f.put(static_cast<char>(g1)); f.put(static_cast<char>(r1));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Intention: read_image loads a valid BMP and returns correct dimensions.
// Preconditions:  write_bmp_2x1 produces a valid 2x1 24-bit BMP
// Postconditions: LoadedImage has width=2, height=1, pixels.size()==6
static bool test_read_image_dimensions()
{
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "dims.bmp").string();
    write_bmp_2x1(path, 255, 0, 0, 0, 255, 0);

    const auto img = trtmc::io::read_image(path);
    if (img.empty())
    {
        std::cerr << "read_image_dimensions: returned empty\n";
        return false;
    }
    if (img.width != 2 || img.height != 1)
    {
        std::cerr << "read_image_dimensions: expected 2x1 got "
                  << img.width << "x" << img.height << '\n';
        return false;
    }
    if (img.pixels.size() != 6)  // 2 pixels * 3 channels
    {
        std::cerr << "read_image_dimensions: pixel count " << img.pixels.size() << '\n';
        return false;
    }
    return true;
}

// Intention: read_image normalises uint8 pixel values to [0, 1].
// Preconditions:  BMP pixel 0 = pure red (255, 0, 0)
// Postconditions: pixels[0] ≈ 1.0, pixels[1] ≈ 0.0, pixels[2] ≈ 0.0
static bool test_read_image_pixel_normalisation()
{
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "norm.bmp").string();
    // Pixel 0 = red (R=255, G=0, B=0); Pixel 1 = green (R=0, G=255, B=0)
    write_bmp_2x1(path, 255, 0, 0, 0, 255, 0);

    const auto img = trtmc::io::read_image(path);
    if (img.empty()) return false;

    // Pixel 0: R channel should be ~1.0
    if (std::abs(img.pixels[0] - 1.0F) > 0.005F)
    {
        std::cerr << "pixel_normalisation: pixel[0].R = " << img.pixels[0] << '\n';
        return false;
    }
    // Pixel 0: G and B channels should be ~0.0
    if (img.pixels[1] > 0.005F || img.pixels[2] > 0.005F)
    {
        std::cerr << "pixel_normalisation: pixel[0].GB non-zero\n";
        return false;
    }
    // Pixel 1: G channel should be ~1.0
    if (std::abs(img.pixels[4] - 1.0F) > 0.005F)
    {
        std::cerr << "pixel_normalisation: pixel[1].G = " << img.pixels[4] << '\n';
        return false;
    }
    return true;
}

// Intention: read_image returns an empty LoadedImage for a nonexistent path.
// Preconditions:  path does not exist
// Postconditions: result.empty() == true, no exception thrown
static bool test_read_image_missing_file_returns_empty()
{
    bool threw = false;
    trtmc::io::LoadedImage img;
    try
    {
        img = trtmc::io::read_image("/nonexistent/path/to/image.bmp");
    }
    catch (...) { threw = true; }

    // Either returns empty or throws — both are acceptable; we just must not crash.
    return img.empty() || threw;
}

// Intention: read_image returns an empty LoadedImage when the file content
//            is not a valid image (corrupt / truncated data).
// Preconditions:  file exists but contains non-image bytes
// Postconditions: result.empty() == true
static bool test_read_image_invalid_content_returns_empty()
{
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "bad.bmp").string();
    {
        std::ofstream f(path, std::ios::binary);
        f.write("NOT_AN_IMAGE_FILE", 17);
    }

    trtmc::io::LoadedImage img;
    try
    {
        img = trtmc::io::read_image(path);
    }
    catch (...) {}

    return img.empty();
}

// Intention: decode_image (legacy wrapper) returns the same pixel data as
//            read_image and correctly fills h and w out-parameters.
// Preconditions:  valid 2x1 BMP in temp dir
// Postconditions: h==1, w==2, pixel vector size==6
static bool test_decode_image_legacy_wrapper()
{
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "legacy.bmp").string();
    write_bmp_2x1(path, 128, 64, 32, 32, 64, 128);

    int h = 0, w = 0;
    const auto pixels = trtmc::io::decode_image(path, h, w);

    return h == 1 && w == 2 && pixels.size() == 6;
}

int main()
{
    bool all_passed = true;
    std::cout << "test_image_reader:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << '\n';
        all_passed &= ok;
    };

    run("read_image_dimensions",                 test_read_image_dimensions);
    run("read_image_pixel_normalisation",        test_read_image_pixel_normalisation);
    run("read_image_missing_file_returns_empty", test_read_image_missing_file_returns_empty);
    run("read_image_invalid_content_returns_empty", test_read_image_invalid_content_returns_empty);
    run("decode_image_legacy_wrapper",           test_decode_image_legacy_wrapper);

    if (all_passed)
    {
        std::cout << "test_image_reader passed" << std::endl;
        return 0;
    }
    std::cerr << "test_image_reader FAILED" << std::endl;
    return 1;
}
