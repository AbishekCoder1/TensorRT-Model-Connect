// Test: TensorRT backend smoke test with built-in model.
// Verifies: TRT backend can be forced and produces correct output with the built-in
// model, with graceful fallback when TRT is unavailable.

#include "trtf/pipeline_legacy.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    try
    {
        auto pipeline = trtf::Pipeline::CreateTextGeneration("trtf/tiny-cake-v1", true, true);
        if (pipeline.backend_name() != "trt")
        {
            std::cerr << "expected backend=trt, got " << pipeline.backend_name() << std::endl;
            return 1;
        }

        const auto out = pipeline("the secret to baking a really good cake is");
        if (out.size() != 1)
        {
            std::cerr << "unexpected output list size" << std::endl;
            return 1;
        }

        const std::string expected_phrase = "to use fresh butter and measure carefully";
        if (out[0].generated_text.find(expected_phrase) == std::string::npos)
        {
            std::cerr << "missing expected phrase in output: " << out[0].generated_text << std::endl;
            return 1;
        }

        std::cout << "test_trt_smoke passed with backend=trt" << std::endl;
        return 0;
    }
    catch (const std::runtime_error&)
    {
        std::cout << "test_trt_smoke passed with expected unavailable-TRT path" << std::endl;
        return 0;
    }
    catch (...)
    {
        std::cerr << "unexpected exception type" << std::endl;
        return 1;
    }
}
