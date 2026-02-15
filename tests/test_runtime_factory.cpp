// Test: Runtime assembly validation.
// Verifies: Backend selection validation and force_trt error handling.

#include "trtf/model_resolver.h"
#include "trtf/runtime_factory.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    // Test invalid backend selection
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

    // Test HF spec with force_trt
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
