// =============================================================================
// Test suite: Pipeline C ABI -- IPipeline virtual interface via trtf_create_pipeline
// =============================================================================
//
// Purpose:
//   Validates the public C ABI entry point trtf_create_pipeline() and the
//   IPipeline virtual interface it returns. Tests cover null/invalid input
//   handling, version queries, and ABI stability guarantees. Since the
//   runtime is now bundle-only (requires pre-built .trtfb files), tests
//   focus on error paths and compile-time interface properties rather than
//   successful generation.
//
// Dependencies:
//   - trtf/pipeline.h (IPipeline, trtf_create_pipeline, trtf_last_error,
//     trtf_version, trtf_has_trt)
//   - No TRT, GPU, or model files required.
//
// Approach:
//   Each test function exercises the C ABI entry points with invalid inputs
//   and checks return values. A simple check() helper tracks failure count.
//   Tests are designed to be order-independent but run sequentially in main().
// =============================================================================

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>

static int failures = 0;

// -----------------------------------------------------------------------------
// Helper: Assert a boolean condition and report failure with a descriptive name.
// Increments the global failure counter on false. Does not abort -- all tests
// run to completion so the full failure picture is visible.
// -----------------------------------------------------------------------------
static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// -----------------------------------------------------------------------------
// Test: null input returns nullptr with error message
//
// Intention:
//   Verify that passing nullptr as the bundle path to trtf_create_pipeline
//   returns nullptr and sets a descriptive error via trtf_last_error().
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_create_pipeline(nullptr, 0) and checks the return and error.
// -----------------------------------------------------------------------------
static void test_null_input_returns_null()
{
    auto* p = trtf_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after null input");
}

// -----------------------------------------------------------------------------
// Test: invalid path returns nullptr with error message
//
// Intention:
//   Verify that a nonexistent bundle path is rejected with a clear error.
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_create_pipeline with a path that does not exist, checks nullptr
//   return and non-empty error message.
// -----------------------------------------------------------------------------
static void test_invalid_path_returns_null()
{
    auto* p = trtf_create_pipeline("/nonexistent/path/to/bundle.trtfb", 0);
    check(p == nullptr, "invalid path returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after invalid path");
}

// -----------------------------------------------------------------------------
// Test: version string is available and non-empty
//
// Intention:
//   Verify that trtf_version() returns a usable version string.
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_version(), checks non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_version_available()
{
    const char* ver = trtf_version();
    check(ver != nullptr, "version is non-null");
    check(std::strlen(ver) > 0, "version is non-empty");
}

// -----------------------------------------------------------------------------
// Test: trtf_has_trt returns a valid boolean
//
// Intention:
//   Verify that the TRT detection function returns 0 or 1.
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_has_trt(), checks the value is 0 or 1.
// -----------------------------------------------------------------------------
static void test_has_trt_returns_bool()
{
    const int val = trtf_has_trt();
    check(val == 0 || val == 1, "trtf_has_trt returns 0 or 1");
}

// -----------------------------------------------------------------------------
// Test: sizeof(IPipeline) equals a single vtable pointer
//
// Intention:
//   Verify the ABI stability guarantee that IPipeline is a pure abstract
//   interface with no data members -- its size must equal exactly one pointer
//   (the vtable pointer). This ensures the C ABI boundary is safe: callers
//   compiled with different compilers/settings can use IPipeline* as long
//   as the vtable layout is stable.
//
// Setup:
//   None -- this is a compile-time/static property check.
//
// Mechanism:
//   Compares sizeof(trtf::IPipeline) against sizeof(void*). If they differ,
//   it means data members or multiple inheritance have been added, breaking
//   the ABI contract.
// -----------------------------------------------------------------------------
static void test_sizeof_ipipeline_is_vtable()
{
    check(sizeof(trtf::IPipeline) == sizeof(void*), "sizeof(IPipeline) equals vtable pointer size");
}

// -----------------------------------------------------------------------------
// Test: delete null IPipeline is safe
//
// Intention:
//   Verify that deleting a null IPipeline pointer does not crash.
//
// Setup: A null IPipeline pointer.
//
// Mechanism:
//   Calls delete on the null pointer, asserts true if we reach this point.
// -----------------------------------------------------------------------------
static void test_delete_null_safe()
{
    trtf::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

int main()
{
    test_null_input_returns_null();
    test_invalid_path_returns_null();
    test_version_available();
    test_has_trt_returns_bool();
    test_sizeof_ipipeline_is_vtable();
    test_delete_null_safe();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}
