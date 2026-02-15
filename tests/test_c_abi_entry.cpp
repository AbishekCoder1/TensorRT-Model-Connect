// Test: C ABI entry point (trtf_create_pipeline, trtf_last_error, trtf_version, trtf_has_trt).
// Uses C++ but only calls through the extern "C" interface.

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_version_not_null()
{
    const char* ver = trtf_version();
    check(ver != nullptr, "trtf_version returns non-null");
    check(std::strlen(ver) > 0, "trtf_version is non-empty");
}

static void test_has_trt_returns_bool()
{
    const int val = trtf_has_trt();
    check(val == 0 || val == 1, "trtf_has_trt returns 0 or 1");
}

static void test_create_null_returns_null()
{
    auto* p = trtf_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after null input");
}

static void test_create_empty_returns_null()
{
    auto* p = trtf_create_pipeline("", 0);
    check(p == nullptr, "empty input returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after empty input");
}

static void test_create_bad_path_returns_null()
{
    auto* p = trtf_create_pipeline("/nonexistent/path/to/model", 0);
    check(p == nullptr, "bad path returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after bad path");
}

static void test_delete_null_safe()
{
    trtf::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

static void test_last_error_cleared_on_success()
{
    // Create failure first
    auto* p1 = trtf_create_pipeline("/nonexistent", 0);
    check(p1 == nullptr, "bad path fails");
    check(std::strlen(trtf_last_error()) > 0, "error set after failure");

    // Try QWEN3 — if it resolves, error should be cleared
    auto* p2 = trtf_create_pipeline("QWEN3", TRTF_CPU_ONLY);
    if (p2 != nullptr)
    {
        check(std::strlen(trtf_last_error()) == 0, "error cleared after success");
        delete p2;
    }
    else
    {
        // QWEN3 may not be available in test env — that's OK
        check(true, "QWEN3 not available in test env");
    }
}

static void test_pipeline_options_zero_init()
{
    // Verify that zero-initialized TrtfPipelineOptions is safe and backward-compatible
    TrtfPipelineOptions opts{};
    check(opts.flags == 0, "zero-init flags == 0 (TRTF_PREFER_TRT)");
    check(opts.max_new_tokens == 0, "zero-init max_new_tokens == 0");
    check(opts.max_cache_length == 0, "zero-init max_cache_length == 0");
    check(opts.hf_python == nullptr, "zero-init hf_python == nullptr");
    check(opts.engine_cache_dir == nullptr, "zero-init engine_cache_dir == nullptr");
    check(opts.no_engine_cache == 0, "zero-init no_engine_cache == 0");

    // Should work with null options (uses defaults)
    auto* p = trtf_create_pipeline_ex("/nonexistent", nullptr);
    check(p == nullptr, "null options with bad path returns null");
}

static void test_create_ex_with_options()
{
    TrtfPipelineOptions opts{};
    opts.flags = TRTF_FORCE_TRT;
    opts.max_new_tokens = 5;
    opts.max_cache_length = 128;
    opts.hf_python = "/nonexistent/python";
    opts.engine_cache_dir = "/tmp/nonexistent_cache";
    opts.no_engine_cache = 1;

    auto* p = trtf_create_pipeline_ex("/nonexistent/model", &opts);
    check(p == nullptr, "bad model with options returns null");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set with options");
}

static void test_bundle_path_not_bundle()
{
    // Test that passing a non-bundle file doesn't crash
    auto* p = trtf_create_pipeline("/dev/null", TRTF_PREFER_TRT);
    check(p == nullptr, "non-bundle file returns null");
}

int main()
{
    test_version_not_null();
    test_has_trt_returns_bool();
    test_create_null_returns_null();
    test_create_empty_returns_null();
    test_create_bad_path_returns_null();
    test_delete_null_safe();
    test_last_error_cleared_on_success();
    test_pipeline_options_zero_init();
    test_create_ex_with_options();
    test_bundle_path_not_bundle();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI entry tests passed.\n";
    return 0;
}
