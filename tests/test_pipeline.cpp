// Test: Pipeline API end-to-end with built-in tiny-cake-v1.
// Verifies: Text generation produces expected output, error handling for invalid
// backend selections, and unknown model IDs throw appropriate errors.

#include "trtf/pipeline_legacy.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    auto pipeline = trtf::Pipeline::CreateTextGeneration("trtf/tiny-cake-v1", true);
    const std::string prompt = "the secret to baking a really good cake is";
    const auto out = pipeline(prompt);

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

    if (pipeline.backend_name().empty())
    {
        std::cerr << "backend name must not be empty" << std::endl;
        return 1;
    }

    bool saw_invalid_argument = false;
    try
    {
        (void) trtf::Pipeline::CreateTextGeneration("trtf/tiny-cake-v1", false, true);
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

    bool saw_unknown_model = false;
    try
    {
        (void) trtf::Pipeline::CreateTextGeneration("Qwen/Qwen2.5-1.5B", true);
    }
    catch (const std::runtime_error& e)
    {
        if (std::string(e.what()).find("Unknown model_id") != std::string::npos)
        {
            saw_unknown_model = true;
        }
    }

    if (!saw_unknown_model)
    {
        std::cerr << "expected runtime_error for unknown model_id" << std::endl;
        return 1;
    }

    std::cout << "test_pipeline passed with backend=" << pipeline.backend_name() << std::endl;
    return 0;
}
