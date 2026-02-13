#include "trtf/pipeline.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    bool prefer_trt = true;
    bool force_trt = false;
    std::vector<std::string> positional_args;
    positional_args.reserve(static_cast<std::size_t>(argc > 1 ? argc - 1 : 0));

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--cpu-only")
        {
            prefer_trt = false;
            continue;
        }
        if (arg == "--force-trt")
        {
            force_trt = true;
            continue;
        }
        positional_args.push_back(arg);
    }

    if (force_trt && !prefer_trt)
    {
        std::cerr << "--force-trt cannot be combined with --cpu-only" << '\n';
        return EXIT_FAILURE;
    }

    const std::string model = positional_args.empty() ? "trtf/tiny-cake-v1" : positional_args[0];
    const std::string prompt
        = positional_args.size() < 2 ? "the secret to baking a really good cake is" : positional_args[1];

    try
    {
        auto pipeline = trtf::Pipeline::CreateTextGeneration(model, prefer_trt, force_trt);
        const auto out = pipeline(prompt);

        std::cout << "backend=" << pipeline.backend_name() << '\n';
        std::cout << "[{'generated_text': '" << out[0].generated_text << "'}]" << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "text_generation failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
