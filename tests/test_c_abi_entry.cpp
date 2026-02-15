// Test: C ABI entry point (trtf_create_pipeline, trtf_last_error, trtf_version, trtf_has_trt).
// Uses C++ but only calls through the extern "C" interface.

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <memory>
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

static void test_create_builtin_succeeds()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "builtin model creates pipeline");
    if (p != nullptr)
    {
        delete p;
    }
}

static void test_generate_through_c_entry()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created for generate test");
    if (p != nullptr)
    {
        const char* result = p->generate("hello", 5);
        check(result != nullptr, "generate returns non-null");
        check(std::strlen(result) > 0, "generate returns non-empty string");
        delete p;
    }
}

static void test_model_id_through_c_entry()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created for model_id test");
    if (p != nullptr)
    {
        const char* mid = p->model_id();
        check(mid != nullptr, "model_id returns non-null");
        check(std::string(mid) == "trtf/tiny-cake-v1", "model_id matches input");
        delete p;
    }
}

static void test_backend_name_not_null()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created for backend_name test");
    if (p != nullptr)
    {
        const char* bn = p->backend_name();
        check(bn != nullptr, "backend_name returns non-null");
        check(std::strlen(bn) > 0, "backend_name is non-empty");
        delete p;
    }
}

static void test_delete_null_safe()
{
    // Deleting a null pointer should be safe (C++ standard guarantees this).
    trtf::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

static void test_flags_cpu_only()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "CPU_ONLY creates pipeline");
    if (p != nullptr)
    {
        const std::string bn = p->backend_name();
        check(bn == "cpu-reference", "CPU_ONLY uses cpu-reference backend");
        delete p;
    }
}

static void test_last_error_cleared_on_success()
{
    // First create failure
    auto* p1 = trtf_create_pipeline("/nonexistent", 0);
    check(p1 == nullptr, "bad path fails");
    check(std::strlen(trtf_last_error()) > 0, "error set after failure");

    // Then create success
    auto* p2 = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p2 != nullptr, "builtin succeeds");
    check(std::strlen(trtf_last_error()) == 0, "error cleared after success");
    delete p2;
}

static void test_fast_path_miss_falls_through()
{
    // No cached engine for the builtin model → fast path misses, slow path runs.
    // This verifies the fast path code doesn't crash when there's no cache hit.
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_PREFER_TRT);
    check(p != nullptr, "fast path miss falls through to slow path");
    if (p != nullptr)
    {
        const char* result = p->generate("hello", 3);
        check(result != nullptr, "generate works after fast path miss");
        check(std::strlen(result) > 0, "generate produces output after fast path miss");
        delete p;
    }
}

static void test_fast_path_no_config_skips()
{
    // trtf/tiny-cake-v1 has no config.json in the expected HF location →
    // fast path should skip gracefully and fall through to slow path without crash.
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_FORCE_TRT);
    // On CPU-only builds, force_trt may fail (no TRT available), which is fine.
    // On TRT builds, the builtin model should still work via slow path.
    // Either way, this must not crash.
    if (p != nullptr)
    {
        const char* bn = p->backend_name();
        check(bn != nullptr, "backend_name not null after fast path skip");
        delete p;
    }
    else
    {
        // Expected on non-TRT builds: force_trt fails for builtin model
        check(true, "fast path skip: null pipeline acceptable on non-TRT build");
    }
}

int main()
{
    test_version_not_null();
    test_has_trt_returns_bool();
    test_create_null_returns_null();
    test_create_empty_returns_null();
    test_create_bad_path_returns_null();
    test_create_builtin_succeeds();
    test_generate_through_c_entry();
    test_model_id_through_c_entry();
    test_backend_name_not_null();
    test_delete_null_safe();
    test_flags_cpu_only();
    test_last_error_cleared_on_success();
    test_fast_path_miss_falls_through();
    test_fast_path_no_config_skips();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI entry tests passed.\n";
    return 0;
}
