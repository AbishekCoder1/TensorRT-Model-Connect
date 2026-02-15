// =============================================================================
// Test suite: Runtime factory — backend selection validation
// =============================================================================
//
// Purpose:
//   Validates that BuildRuntimeForTextGeneration() correctly rejects invalid
//   or unsupported backend selection configurations. This test exercises the
//   error-handling paths in the runtime assembly stage (stage 3 of the
//   pipeline flow), ensuring that contradictory or inapplicable flag
//   combinations are caught early with appropriate exceptions.
//
// Dependencies:
//   - trtf/model_resolver.h (ResolvedModelSpec, ResolvedModelKind)
//   - trtf/runtime_factory.h (BuildRuntimeForTextGeneration, BackendSelection)
//
// Approach:
//   Constructs synthetic ResolvedModelSpec objects (without real model data)
//   and calls BuildRuntimeForTextGeneration with intentionally invalid
//   BackendSelection configurations. Expects specific exception types/messages.
//   No GPU, model files, or TensorRT installation required.
// =============================================================================

#include "trtf/model_resolver.h"
#include "trtf/runtime_factory.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    // -------------------------------------------------------------------------
    // Test: Contradictory backend flags (force_trt=true, prefer_trt=false)
    //
    // Intention:
    //   Verify that setting force_trt=true while prefer_trt=false is rejected
    //   as a logical contradiction. The force_trt flag implies TRT is mandatory,
    //   but prefer_trt=false says "do not use TRT" — these cannot coexist.
    //
    // Setup:
    //   A dummy ResolvedModelSpec with kind=kDecoderDefinition (the kind that
    //   would normally route to TRT). A BackendSelection with the contradictory
    //   flag combination.
    //
    // Mechanism:
    //   Calls BuildRuntimeForTextGeneration and catches std::invalid_argument.
    //   If the exception is not thrown, the test fails.
    // -------------------------------------------------------------------------
    bool saw_invalid_argument = false;
    trtf::BackendSelection invalid_selection;
    invalid_selection.prefer_trt = false;
    invalid_selection.force_trt = true;

    trtf::ResolvedModelSpec dummy;
    dummy.model_id = "dummy";
    dummy.kind = trtf::ResolvedModelKind::kDecoderDefinition;

    try
    {
        (void) trtf::BuildRuntimeForTextGeneration(dummy, invalid_selection);
    }
    catch (const std::invalid_argument&)
    {
        saw_invalid_argument = true;
    }

    if (!saw_invalid_argument)
    {
        std::cerr << "expected invalid_argument when force_trt=true and prefer_trt=false" << std::endl;
        return 1;
    }

    // -------------------------------------------------------------------------
    // Test: force_trt with HuggingFace-local model (unsupported combination)
    //
    // Intention:
    //   Verify that requesting force_trt on a model resolved as
    //   kHuggingFaceLocal (i.e., a model that only has an HF Python backend,
    //   not a registered TRT model runtime) produces a clear runtime_error.
    //
    // Setup:
    //   A ResolvedModelSpec with kind=kHuggingFaceLocal pointing to a fake
    //   directory. A BackendSelection with both prefer_trt and force_trt set
    //   to true.
    //
    // Mechanism:
    //   Calls BuildRuntimeForTextGeneration and catches std::runtime_error.
    //   Inspects the error message for the substring "force_trt is not
    //   supported" to confirm the correct error path was taken.
    // -------------------------------------------------------------------------
    trtf::ResolvedModelSpec hf_spec;
    hf_spec.model_id = "/tmp/fake-hf";
    hf_spec.kind = trtf::ResolvedModelKind::kHuggingFaceLocal;
    hf_spec.huggingface_model_dir = "/tmp/fake-hf";

    bool saw_force_not_supported = false;
    trtf::BackendSelection hf_selection;
    hf_selection.prefer_trt = true;
    hf_selection.force_trt = true;
    try
    {
        (void) trtf::BuildRuntimeForTextGeneration(hf_spec, hf_selection);
    }
    catch (const std::runtime_error& e)
    {
        if (std::string(e.what()).find("force_trt is not supported") != std::string::npos)
        {
            saw_force_not_supported = true;
        }
    }

    if (!saw_force_not_supported)
    {
        std::cerr << "expected force_trt unsupported error for huggingface-local model" << std::endl;
        return 1;
    }

    std::cout << "test_runtime_factory passed" << std::endl;
    return 0;
}
