// =============================================================================
// Test suite: Pipeline C ABI — IPipeline virtual interface via trtf_create_pipeline
// =============================================================================
//
// Purpose:
//   Validates the public C ABI entry point trtf_create_pipeline() and the
//   IPipeline virtual interface it returns. Tests cover basic generation
//   functionality, pointer lifetime semantics, and ABI stability guarantees.
//   Uses the QWEN3 built-in model, which falls back gracefully if the model
//   directory is not available (tests SKIP rather than FAIL).
//
// Dependencies:
//   - trtf/pipeline.h (IPipeline, trtf_create_pipeline, trtf_last_error,
//     TRTF_CPU_ONLY flag)
//   - QWEN3 built-in model (optional — tests skip if unavailable)
//
// Approach:
//   Each test function creates a pipeline via the C ABI, exercises the
//   generate() method, and checks return values. A simple check() helper
//   tracks failure count. Tests are designed to be order-independent but
//   run sequentially in main(). The TRTF_CPU_ONLY flag is used to avoid
//   requiring GPU/TRT for these API-level tests.
// =============================================================================

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>

static int failures = 0;

// -----------------------------------------------------------------------------
// Helper: Assert a boolean condition and report failure with a descriptive name.
// Increments the global failure counter on false. Does not abort — all tests
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
// Test: generate() returns non-null, non-empty text
//
// Intention:
//   Verify the most basic contract of IPipeline::generate() — that it returns
//   a valid C string with at least one character when given a valid prompt
//   and a positive max_new_tokens value.
//
// Setup:
//   Creates a QWEN3 pipeline with TRTF_CPU_ONLY flag. Skips (without failure)
//   if the model is not available.
//
// Mechanism:
//   Calls generate("hello", 3) and checks that the result pointer is non-null
//   and that strlen(result) > 0.
// -----------------------------------------------------------------------------
static void test_generate_returns_text()
{
    auto* p = trtf_create_pipeline("QWEN3", TRTF_CPU_ONLY);
    if (p == nullptr)
    {
        std::cerr << "SKIP: QWEN3 not available: " << trtf_last_error() << '\n';
        return;
    }
    const char* result = p->generate("hello", 3);
    check(result != nullptr, "generate returns non-null");
    if (result != nullptr)
    {
        check(std::strlen(result) > 0, "generate returns non-empty text");
    }
    delete p;
}

// -----------------------------------------------------------------------------
// Test: generate() with explicit max_tokens returns non-null
//
// Intention:
//   Verify that the max_new_tokens parameter is accepted and that generation
//   succeeds (returns non-null) when a specific token limit is provided.
//   This is a simpler variant of test_generate_returns_text focused on the
//   max_tokens parameter path.
//
// Setup:
//   Creates a QWEN3 pipeline with TRTF_CPU_ONLY. Skips if unavailable.
//
// Mechanism:
//   Calls generate("hello", 3) and checks that the returned pointer is
//   non-null.
// -----------------------------------------------------------------------------
static void test_generate_with_max_tokens()
{
    auto* p = trtf_create_pipeline("QWEN3", TRTF_CPU_ONLY);
    if (p == nullptr) return;
    const char* result = p->generate("hello", 3);
    check(result != nullptr, "generate with max_tokens returns non-null");
    delete p;
}

// -----------------------------------------------------------------------------
// Test: Result pointer lifetime — valid until next generate() call
//
// Intention:
//   Verify that the pointer returned by generate() remains valid at least
//   until the next call to generate(). This tests the implicit contract that
//   the pipeline owns the result buffer and may reuse/overwrite it on
//   subsequent calls, but the previous pointer is usable before that happens.
//
// Setup:
//   Creates a QWEN3 pipeline with TRTF_CPU_ONLY. Skips if unavailable.
//
// Mechanism:
//   Calls generate() twice with different prompts. After the first call,
//   copies the result string to verify it was non-empty. After the second
//   call, checks that the second result is also non-null. The copy of the
//   first result confirms it was readable before the second call.
// -----------------------------------------------------------------------------
static void test_generate_pointer_valid_until_next()
{
    auto* p = trtf_create_pipeline("QWEN3", TRTF_CPU_ONLY);
    if (p == nullptr) return;
    const char* first = p->generate("hello", 3);
    check(first != nullptr, "first generate result valid");
    const std::string first_copy(first ? first : "");

    const char* second = p->generate("world", 3);
    check(second != nullptr, "second generate result valid");
    check(!first_copy.empty(), "first result was non-empty");
    delete p;
}

// -----------------------------------------------------------------------------
// Test: sizeof(IPipeline) equals a single vtable pointer
//
// Intention:
//   Verify the ABI stability guarantee that IPipeline is a pure abstract
//   interface with no data members — its size must equal exactly one pointer
//   (the vtable pointer). This ensures the C ABI boundary is safe: callers
//   compiled with different compilers/settings can use IPipeline* as long
//   as the vtable layout is stable.
//
// Setup:
//   None — this is a compile-time/static property check.
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

int main()
{
    test_generate_returns_text();
    test_generate_with_max_tokens();
    test_generate_pointer_valid_until_next();
    test_sizeof_ipipeline_is_vtable();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}
