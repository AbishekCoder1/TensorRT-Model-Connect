#include "trtf/model_resolver.h"
#include "trtf/runtime_factory.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    const trtf::ResolvedModelSpec decoder = trtf::ResolveTextGenerationModel("trtf/tiny-cake-v1");

    trtf::BackendSelection selection;
    selection.prefer_trt = true;
    selection.force_trt = false;

    trtf::RuntimeAssembly runtime = trtf::BuildRuntimeForTextGeneration(decoder, selection);
    if (!runtime.backend)
    {
        std::cerr << "expected runtime backend to be initialized for decoder model" << std::endl;
        return 1;
    }
    if (!runtime.tokenizer)
    {
        std::cerr << "expected runtime tokenizer to be initialized for decoder model" << std::endl;
        return 1;
    }
    if (runtime.backend_name.empty())
    {
        std::cerr << "expected runtime backend name to be non-empty" << std::endl;
        return 1;
    }

    bool saw_invalid_argument = false;
    trtf::BackendSelection invalid_selection;
    invalid_selection.prefer_trt = false;
    invalid_selection.force_trt = true;
    try
    {
        (void) trtf::BuildRuntimeForTextGeneration(decoder, invalid_selection);
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

    std::cout << "test_runtime_factory passed with backend=" << runtime.backend_name << std::endl;
    return 0;
}
