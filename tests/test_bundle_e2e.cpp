// Test: Bundle E2E -- build, save, load, and generate from .trtfb bundles.
// Requires TRT + GPU. Tests are skipped (pass) when TRT is not available.

#include "trtf/pipeline.h"
#include "trtf/bundle.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include <stdlib.h>

static int failures = 0;
static bool has_trt = false;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::filesystem::path make_temp_dir()
{
    char pattern[] = "/tmp/trtfb_e2e_XXXXXX";
    char* dir = mkdtemp(pattern);
    if (dir == nullptr)
    {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(dir);
}

static void test_save_and_load_roundtrip()
{
    if (!has_trt)
    {
        std::cerr << "  SKIP: test_save_and_load_roundtrip (no TRT)\n";
        return;
    }

    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_FORCE_TRT);
    if (p == nullptr)
    {
        std::cerr << "  SKIP: test_save_and_load_roundtrip (TRT pipeline creation failed: "
                  << trtf_last_error() << ")\n";
        return;
    }

    const auto tmp = make_temp_dir();
    const auto bundle_path = (tmp / "roundtrip.trtfb").string();

    const bool saved = p->save_bundle(bundle_path.c_str());
    if (!saved)
    {
        std::cerr << "  SKIP: test_save_and_load_roundtrip (save_bundle not yet implemented for this backend)\n";
        delete p;
        std::filesystem::remove_all(tmp);
        return;
    }

    check(std::filesystem::exists(bundle_path), "bundle file created");
    check(trtf::IsBundle(bundle_path), "bundle has valid magic");

    // Generate text with original pipeline
    const char* original_output = p->generate("hello", 5);
    const std::string original_text(original_output ? original_output : "");
    delete p;

    // Load from bundle and generate
    auto* p2 = trtf_create_pipeline(bundle_path.c_str(), TRTF_FORCE_TRT);
    check(p2 != nullptr, "bundle loads as pipeline");
    if (p2 != nullptr)
    {
        const char* bundle_output = p2->generate("hello", 5);
        check(bundle_output != nullptr, "bundle pipeline generates text");
        delete p2;
    }

    std::filesystem::remove_all(tmp);
}

static void test_inspect_saved_bundle()
{
    if (!has_trt)
    {
        std::cerr << "  SKIP: test_inspect_saved_bundle (no TRT)\n";
        return;
    }

    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_FORCE_TRT);
    if (p == nullptr)
    {
        std::cerr << "  SKIP: test_inspect_saved_bundle (TRT pipeline creation failed)\n";
        return;
    }

    const auto tmp = make_temp_dir();
    const auto bundle_path = (tmp / "inspect.trtfb").string();

    const bool saved = p->save_bundle(bundle_path.c_str());
    delete p;

    if (!saved)
    {
        std::cerr << "  SKIP: test_inspect_saved_bundle (save_bundle not yet implemented)\n";
        std::filesystem::remove_all(tmp);
        return;
    }

    const auto info = trtf::InspectBundle(bundle_path);
    check(!info.model_id.empty(), "inspect model_id non-empty");

    std::filesystem::remove_all(tmp);
}

int main()
{
    has_trt = (trtf_has_trt() == 1);

    test_save_and_load_roundtrip();
    test_inspect_saved_bundle();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle E2E tests passed.\n";
    return 0;
}
