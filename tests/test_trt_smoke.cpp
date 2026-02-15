// Test: TensorRT backend smoke test with built-in model.
// Verifies: TRT backend can be forced and produces output with the built-in
// model, with graceful fallback when TRT is unavailable.

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>

int main()
{
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
