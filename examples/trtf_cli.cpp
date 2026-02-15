// trtf CLI -- command-line interface using the library API.
//
// Usage:
//   trtf build   <model-dir> -o <output.trtfb> [--max-cache-length N]
//   trtf run     <model-or-bundle> --prompt "text" [--max-new-tokens N] [--force-trt] [--cpu-only]
//   trtf inspect <bundle.trtfb>
//   trtf version

#include "trtf/pipeline.h"
#include "trtf/bundle.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CliArgs {
    std::string command;
    std::string model_or_bundle;
    std::string output_path;
    std::string prompt;
    int max_new_tokens{0};
    int max_cache_length{-1};
    int flags{TRTF_PREFER_TRT};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
};

void print_usage()
{
    std::cerr <<
        "Usage:\n"
        "  trtf build   <model-dir> -o <output.trtfb> [--max-cache-length N]\n"
        "  trtf run     <model-or-bundle> --prompt \"text\" [--max-new-tokens N] [--force-trt] [--cpu-only]\n"
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

    if (args.command != "build" && args.command != "run" && args.command != "inspect")
    {
        args.parse_error = true;
        args.error_message = "Unknown command: " + args.command;
        return args;
    }

    // Parse remaining arguments
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "-o" || arg == "--output")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.output_path = argv[++i];
            continue;
        }

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

        if (arg == "--max-cache-length")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.max_cache_length = std::atoi(argv[++i]);
            continue;
        }

        if (arg == "--force-trt")
        {
            args.flags = TRTF_FORCE_TRT;
            continue;
        }

        if (arg == "--cpu-only")
        {
            args.flags = TRTF_CPU_ONLY;
            continue;
        }

        if (arg[0] == '-')
        {
            args.parse_error = true;
            args.error_message = "Unknown flag: " + arg;
            return args;
        }

        // Positional argument
        if (args.model_or_bundle.empty())
        {
            args.model_or_bundle = arg;
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

int cmd_build(const CliArgs& args)
{
    if (args.model_or_bundle.empty())
    {
        std::cerr << "Error: build requires a model directory\n";
        return EXIT_FAILURE;
    }
    if (args.output_path.empty())
    {
        std::cerr << "Error: build requires -o <output.trtfb>\n";
        return EXIT_FAILURE;
    }

    try
    {
        trtf::BuildBundle(args.model_or_bundle, args.output_path, args.max_cache_length);
        std::cout << "Bundle saved to: " << args.output_path << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}

int cmd_run(const CliArgs& args)
{
    if (args.model_or_bundle.empty())
    {
        std::cerr << "Error: run requires a model path, alias, or bundle file\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.flags = args.flags;
    opts.max_new_tokens = args.max_new_tokens;
    opts.max_cache_length = args.max_cache_length;
    auto* pipeline = trtf_create_pipeline_ex(args.model_or_bundle.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    const std::string prompt = args.prompt.empty() ? "Hello" : args.prompt;
    const std::size_t max_tokens = args.max_new_tokens > 0
        ? static_cast<std::size_t>(args.max_new_tokens) : 0;

    const char* output = pipeline->generate(prompt.c_str(), max_tokens);
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
    if (args.model_or_bundle.empty())
    {
        std::cerr << "Error: inspect requires a bundle file path\n";
        return EXIT_FAILURE;
    }

    if (!trtf::IsBundle(args.model_or_bundle))
    {
        std::cerr << "Error: not a valid .trtfb bundle: " << args.model_or_bundle << '\n';
        return EXIT_FAILURE;
    }

    try
    {
        const auto info = trtf::InspectBundle(args.model_or_bundle);
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
    if (args.command == "build")
    {
        return cmd_build(args);
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
