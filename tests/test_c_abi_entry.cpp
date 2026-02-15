// =============================================================================
// Test suite: C ABI entry points (trtf_create_pipeline, trtf_create_pipeline_ex,
//   trtf_last_error, trtf_version, trtf_has_trt).
//
// Purpose:
//   Validates the public C ABI surface exposed by trtf/pipeline.h. Although
//   this test file is compiled as C++, all pipeline interactions go through
//   the extern "C" functions to verify the ABI contract that language bindings
//   and the CLI depend on. Tests cover: version string availability, TRT
//   detection, error handling for invalid inputs, null safety, the extended
//   options struct (TrtfPipelineOptions), and error-state lifecycle.
//
// Dependencies:
//   - trtf/pipeline.h: C ABI functions and IPipeline, TrtfPipelineOptions.
//   - No TRT or GPU required for most tests (they exercise error paths with
//     invalid/nonexistent model paths). The QWEN3 test is environment-
//     dependent and skips gracefully if the model is unavailable.
//
// Approach:
//   Each test calls C ABI functions with specific inputs and checks return
//   values and side effects (error messages via trtf_last_error). Tests are
//   designed to succeed in any environment -- GPU, CPU-only, or CI sandbox --
//   by testing error paths and null handling rather than successful pipeline
//   creation (which would require model files and possibly TRT).
//
// Test categories:
//   - Version/capability queries: trtf_version, trtf_has_trt
//   - Error handling: null input, empty input, bad path, non-bundle file
//   - Null safety: deleting a null IPipeline pointer
//   - Error lifecycle: error set after failure, cleared after success
//   - Extended API: TrtfPipelineOptions zero-init safety, trtf_create_pipeline_ex
// =============================================================================

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

// -----------------------------------------------------------------------------
// Intention: Verify that trtf_version() returns a non-null, non-empty string
//   identifying the library version.
// Setup: None.
// Mechanism: Calls trtf_version(), checks the pointer is non-null and the
//   string has length > 0.
// -----------------------------------------------------------------------------
static void test_version_not_null()
{
    const char* ver = trtf_version();
    check(ver != nullptr, "trtf_version returns non-null");
    check(std::strlen(ver) > 0, "trtf_version is non-empty");
}

// -----------------------------------------------------------------------------
// Intention: Verify that trtf_has_trt() returns a valid boolean value (0 or 1),
//   indicating whether TensorRT support was compiled in and is available.
// Setup: None.
// Mechanism: Calls trtf_has_trt(), checks the return value is exactly 0 or 1.
// -----------------------------------------------------------------------------
static void test_has_trt_returns_bool()
{
    const int val = trtf_has_trt();
    check(val == 0 || val == 1, "trtf_has_trt returns 0 or 1");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing a null model_id to trtf_create_pipeline
//   returns nullptr and populates trtf_last_error() with a descriptive message.
// Setup: None.
// Mechanism: Calls trtf_create_pipeline(nullptr, 0), asserts the return is
//   nullptr, then checks trtf_last_error() is non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_create_null_returns_null()
{
    auto* p = trtf_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after null input");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing an empty string as model_id returns nullptr
//   and sets an error message.
// Setup: None.
// Mechanism: Calls trtf_create_pipeline("", 0), asserts nullptr return, checks
//   trtf_last_error() is non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_create_empty_returns_null()
{
    auto* p = trtf_create_pipeline("", 0);
    check(p == nullptr, "empty input returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after empty input");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing a nonexistent filesystem path as model_id
//   returns nullptr and sets an error message (model resolution fails).
// Setup: None.
// Mechanism: Calls trtf_create_pipeline("/nonexistent/path/to/model", 0),
//   asserts nullptr return, checks trtf_last_error() is non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_create_bad_path_returns_null()
{
    auto* p = trtf_create_pipeline("/nonexistent/path/to/model", 0);
    check(p == nullptr, "bad path returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after bad path");
}

// -----------------------------------------------------------------------------
// Intention: Verify that deleting a null IPipeline pointer is safe and does
//   not crash (C++ guarantees delete on nullptr is a no-op, but this
//   explicitly tests it through the ABI's usage pattern).
// Setup: A null IPipeline pointer.
// Mechanism: Calls delete on the null pointer, then asserts true (if we reach
//   the assertion, the delete did not crash).
// -----------------------------------------------------------------------------
static void test_delete_null_safe()
{
    trtf::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

// -----------------------------------------------------------------------------
// Intention: Verify the error-state lifecycle: trtf_last_error() should contain
//   a message after a failed pipeline creation, and should be cleared (empty
//   string) after a subsequent successful creation.
// Setup: First triggers a failure with a nonexistent path, then attempts to
//   create a QWEN3 pipeline with TRTF_CPU_ONLY to test success clearing.
// Mechanism:
//   1. Creates a pipeline with "/nonexistent" -> fails, error message is set.
//   2. Creates a pipeline with "QWEN3" + CPU_ONLY.
//   3. If QWEN3 succeeds, checks that trtf_last_error() is now empty.
//   4. If QWEN3 is unavailable (model not present), the test passes trivially
//      since error clearing can only be tested when a model resolves.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that zero-initializing TrtfPipelineOptions (via C++ value
//   initialization) produces safe default values for every field, ensuring
//   backward compatibility when new fields are added to the struct.
// Setup: A brace-initialized TrtfPipelineOptions{}.
// Mechanism: Checks each field: flags==0 (TRTF_PREFER_TRT), max_new_tokens==0,
//   max_cache_length==0, hf_python==nullptr, engine_cache_dir==nullptr,
//   no_engine_cache==0. Also verifies that trtf_create_pipeline_ex with null
//   options and a bad path returns nullptr (null options should use defaults).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that trtf_create_pipeline_ex correctly accepts a fully
//   populated TrtfPipelineOptions struct and propagates the error when the
//   model path is invalid.
// Setup: Creates a TrtfPipelineOptions with all fields set: flags=TRTF_FORCE_TRT,
//   max_new_tokens=5, max_cache_length=128, hf_python="/nonexistent/python",
//   engine_cache_dir="/tmp/nonexistent_cache", no_engine_cache=1.
// Mechanism: Calls trtf_create_pipeline_ex with a nonexistent model path and
//   the options struct. Asserts nullptr return and a non-empty error message.
//   This ensures the extended API processes all option fields without crashing.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that passing a non-bundle file path (e.g., /dev/null) to
//   trtf_create_pipeline does not crash and correctly returns nullptr.
// Setup: None (uses the always-available /dev/null as input).
// Mechanism: Calls trtf_create_pipeline("/dev/null", TRTF_PREFER_TRT), asserts
//   nullptr return. This ensures the bundle detection logic gracefully rejects
//   non-bundle files.
// -----------------------------------------------------------------------------
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
