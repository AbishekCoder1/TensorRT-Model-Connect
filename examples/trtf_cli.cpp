// trtf CLI -- command-line interface using the library API.
//
// Usage:
//   trtf run     <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--hf-python PATH]
//   trtf inspect <bundle.trtfb>
//   trtf version

#include "trtf/pipeline.h"
#include "trtf/bundle.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

struct CliArgs {
    std::string command;
    std::string bundle_path;
    std::string prompt;
    std::string hf_python;
    std::string image_path;
    int max_new_tokens{0};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
};

void print_usage()
{
    std::cerr <<
        "Usage:\n"
        "  trtf run     <bundle.trtfb> --prompt \"text\" [--image PATH] [--max-new-tokens N] [--hf-python PATH]\n"
        "  trtf inspect <bundle.trtfb>\n"
        "  trtf version\n";
}

CliArgs parse_args(int argc, char** argv)
{
    CliArgs args;

    if (argc < 2)
    {
        args.show_help = true;
        return args;
    }

    args.command = argv[1];

    if (args.command == "version" || args.command == "--version" || args.command == "-v")
    {
        args.command = "version";
        return args;
    }

    if (args.command == "help" || args.command == "--help" || args.command == "-h")
    {
        args.show_help = true;
        return args;
    }

    if (args.command != "run" && args.command != "inspect")
    {
        args.parse_error = true;
        args.error_message = "Unknown command: " + args.command;
        return args;
    }

    // Parse remaining arguments
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--prompt" || arg == "-p")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.prompt = argv[++i];
            continue;
        }

        if (arg == "--max-new-tokens")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.max_new_tokens = std::atoi(argv[++i]);
            continue;
        }

        if (arg == "--hf-python")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.hf_python = argv[++i];
            continue;
        }

        if (arg == "--image")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.image_path = argv[++i];
            continue;
        }

        if (arg[0] == '-')
        {
            args.parse_error = true;
            args.error_message = "Unknown flag: " + arg;
            return args;
        }

        // Positional argument
        if (args.bundle_path.empty())
        {
            args.bundle_path = arg;
        }
        else
        {
            args.parse_error = true;
            args.error_message = "Unexpected positional argument: " + arg;
            return args;
        }
    }

    return args;
}

int cmd_version()
{
    std::cout << "trtf " << trtf_version() << '\n';
    std::cout << "TRT support: " << (trtf_has_trt() ? "yes" : "no") << '\n';
    return EXIT_SUCCESS;
}

int cmd_run(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: run requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.max_new_tokens = args.max_new_tokens;
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    opts.image_path = args.image_path.empty() ? nullptr : args.image_path.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    const std::string prompt = args.prompt.empty() ? "Hello" : args.prompt;
    const std::size_t max_tokens = args.max_new_tokens > 0
        ? static_cast<std::size_t>(args.max_new_tokens) : 0;

    const char* output = nullptr;
    if (!args.image_path.empty() && pipeline->supports_vision())
    {
        output = pipeline->generate(prompt.c_str(), args.image_path.c_str(), max_tokens);
    }
    else
    {
        output = pipeline->generate(prompt.c_str(), max_tokens);
    }
    if (output != nullptr)
    {
        std::cout << output << '\n';
    }
    else
    {
        std::cerr << "Error: generate returned null\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_inspect(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: inspect requires a bundle file path\n";
        return EXIT_FAILURE;
    }

    if (!trtf::IsBundle(args.bundle_path))
    {
        std::cerr << "Error: not a valid .trtfb bundle: " << args.bundle_path << '\n';
        return EXIT_FAILURE;
    }

    try
    {
        const auto info = trtf::InspectBundle(args.bundle_path);
        std::cout << "Model ID:           " << info.model_id << '\n';
        std::cout << "Model type:         " << info.model_type << '\n';
        std::cout << "Family:             " << info.family << '\n';
        std::cout << "TRT version:        " << info.trt_version << '\n';
        std::cout << "GPU:                " << info.gpu_name << '\n';
        std::cout << "Created:            " << info.created_at << '\n';
        std::cout << "Vocab size:         " << info.vocab_size << '\n';
        std::cout << "Hidden size:        " << info.hidden_size << '\n';
        std::cout << "Layers:             " << info.num_layers << '\n';
        std::cout << "Attention heads:    " << info.num_attention_heads << '\n';
        std::cout << "KV heads:           " << info.num_key_value_heads << '\n';
        std::cout << "Max cache length:   " << info.max_cache_length << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace

int main(int argc, char** argv)
{
    const CliArgs args = parse_args(argc, argv);

    if (args.show_help)
    {
        print_usage();
        return EXIT_SUCCESS;
    }

    if (args.parse_error)
    {
        std::cerr << "Error: " << args.error_message << '\n';
        print_usage();
        return EXIT_FAILURE;
    }

    if (args.command == "version")
    {
        return cmd_version();
    }
    if (args.command == "run")
    {
        return cmd_run(args);
    }
    if (args.command == "inspect")
    {
        return cmd_inspect(args);
    }

    print_usage();
    return EXIT_FAILURE;
}
