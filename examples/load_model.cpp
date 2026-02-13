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

    const std::string model_id = positional_args.empty() ? "QWEN3" : positional_args[0];
    const std::string prompt = positional_args.size() < 2 ? "Hello" : positional_args[1];

    try
    {
        auto model = trtf::loadModel(model_id, prefer_trt, force_trt);
        const std::string output = model.generate(prompt);
        std::cout << "backend=" << model.backend_name() << '\n';
        std::cout << output << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "load_model failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
