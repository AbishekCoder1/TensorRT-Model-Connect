// =============================================================================
// Test suite: TensorRT backend smoke test — end-to-end generation with TRT
// =============================================================================
//
// Purpose:
//   Validates that the TensorRT backend can be forced via the TRTF_FORCE_TRT
//   flag and that it produces correct output for a known built-in model
//   ("trtf/tiny-cake-v1"). This is a smoke test: it confirms the full
//   TRT pipeline works (model resolution -> engine build -> autoregressive
//   generation -> text output) without exhaustive correctness checks.
//
// Dependencies:
//   - trtf/pipeline.h (IPipeline, trtf_create_pipeline, TRTF_FORCE_TRT)
//   - Built-in model "trtf/tiny-cake-v1" (deterministic transition-based model)
//   - TensorRT runtime (test gracefully passes if TRT is unavailable)
//
// Approach:
//   Attempts to create a pipeline with TRTF_FORCE_TRT. If pipeline creation
//   returns nullptr, the test assumes TRT is not available in this build and
//   passes with a "TRT unavailable" message. If creation succeeds, the test
//   verifies:
//     1. The backend_name() is "trt" (not "hf-transformers" or other).
//     2. generate() returns a non-null, non-empty string.
//     3. The output contains the expected deterministic phrase from the
//        built-in model's transition table.
// =============================================================================

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>

int main()
{
    // -------------------------------------------------------------------------
    // Test: TRT backend end-to-end smoke test
    //
    // Intention:
    //   Verify that forcing the TRT backend on a built-in model produces the
    //   expected deterministic output, exercising the complete TRT code path:
    //   model resolution, TRT engine construction, autoregressive decoding,
    //   and text assembly.
    //
    // Setup:
    //   Uses the built-in "trtf/tiny-cake-v1" model, which is a small
    //   deterministic transition-based model bundled with the project.
    //   The TRTF_FORCE_TRT flag ensures the TRT backend is used (not the
    //   HF Python fallback).
    //
    // Mechanism:
    //   1. Calls trtf_create_pipeline with TRTF_FORCE_TRT. If nullptr is
    //      returned, assumes TRT is unavailable and passes gracefully.
    //   2. Checks backend_name() == "trt".
    //   3. Calls generate() with a known prompt and verifies the output
    //      contains the expected phrase "to use fresh butter and measure
    //      carefully" (deterministic from the model's transition table).
    // -------------------------------------------------------------------------
    auto* pipeline = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_FORCE_TRT);
    if (pipeline == nullptr)
    {
        // Expected on non-TRT builds: force_trt fails for builtin model
        std::cout << "test_trt_smoke passed with expected unavailable-TRT path" << std::endl;
        return 0;
    }

    const std::string bn = pipeline->backend_name();
    if (bn != "trt")
    {
        std::cerr << "expected backend=trt, got " << bn << std::endl;
        delete pipeline;
        return 1;
    }

    const char* result = pipeline->generate("the secret to baking a really good cake is", 20);
    if (result == nullptr || std::strlen(result) == 0)
    {
        std::cerr << "generate returned null or empty" << std::endl;
        delete pipeline;
        return 1;
    }

    const std::string expected_phrase = "to use fresh butter and measure carefully";
    if (std::string(result).find(expected_phrase) == std::string::npos)
    {
        std::cerr << "missing expected phrase in output: " << result << std::endl;
        delete pipeline;
        return 1;
    }

    delete pipeline;
    std::cout << "test_trt_smoke passed with backend=trt" << std::endl;
    return 0;
}
