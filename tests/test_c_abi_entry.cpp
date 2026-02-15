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

int main()
{
    test_version_not_null();
    test_has_trt_returns_bool();
    test_create_null_returns_null();
    test_create_empty_returns_null();
    test_create_bad_path_returns_null();
    test_delete_null_safe();
    test_last_error_cleared_on_success();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI entry tests passed.\n";
    return 0;
}
