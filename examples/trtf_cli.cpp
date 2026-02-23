// trtf CLI -- command-line interface using the library API.
//
// Usage:
//   trtf run             <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--hf-python PATH]
//   trtf transcribe      <bundle.trtfb> --audio FILE.wav [--max-new-tokens N] [--hf-python PATH]
//   trtf generate-video  <bundle.trtfb> --prompt "text" --output DIR [--num-steps N] [--guidance-scale S] [--hf-python PATH]
//   trtf inspect         <bundle.trtfb>
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
    std::string bundle_path;
    std::string prompt;
    std::string hf_python;
    std::string image_path;
    std::string output_dir;
    std::string output_path;  // for segment/generate-audio
    std::string branch_input; // comma-separated floats for solve (DeepONet)
    std::string trunk_input;  // comma-separated floats for solve (DeepONet)
    std::string field_input;  // comma-separated floats for solve (FNO field input)
    std::string document;     // document text for rerank
    std::string audio_in;     // input audio WAV for speak
    std::string audio_out;    // output audio WAV for speak
    float point_x{0.5F};     // SAM point prompt X (normalized 0-1)
    float point_y{0.5F};     // SAM point prompt Y (normalized 0-1)
    bool is_foreground{true}; // SAM point: foreground or background
    int max_new_tokens{0};
    int num_steps{-1};
    float guidance_scale{-1.0F};
    float conf_threshold{-1.0F};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
};

void print_usage()
{
    std::cerr <<
        "Usage:\n"
        "  trtf run             <bundle.trtfb> --prompt \"text\" [--image PATH] [--max-new-tokens N] [--hf-python PATH]\n"
        "  trtf encode          <bundle.trtfb> --prompt \"text\" [--hf-python PATH]\n"
        "  trtf segment         <bundle.trtfb> --image PATH --output PATH [--hf-python PATH]\n"
        "  trtf segment-sam     <bundle.trtfb> --image PATH --output DIR [--point-x 0.5] [--point-y 0.5] [--background]\n"
        "  trtf generate-audio  <bundle.trtfb> --prompt \"text\" --output PATH [--max-new-tokens N] [--hf-python PATH]\n"
        "  trtf generate-video  <bundle.trtfb> --prompt \"text\" --output DIR [--num-steps N] [--guidance-scale S] [--hf-python PATH]\n"
        "  trtf detect          <bundle.trtfb> --image PATH --output PATH [--threshold 0.5] [--hf-python PATH]\n"
        "  trtf embed           <bundle.trtfb> --prompt \"text\" [--hf-python PATH]\n"
        "  trtf rerank          <bundle.trtfb> --prompt \"query\" --document \"text\" [--hf-python PATH]\n"
        "  trtf transcribe      <bundle.trtfb> --audio FILE.wav [--max-new-tokens N] [--hf-python PATH]\n"
        "  trtf speak           <bundle.trtfb> --audio-in INPUT.wav --audio-out OUTPUT.wav [--max-new-tokens N]\n"
        "  trtf solve           <bundle.trtfb> --branch-input \"0.1,0.2,...\" --trunk-input \"0.5,0.5\"  (DeepONet)\n"
        "  trtf solve           <bundle.trtfb> --field-input \"0.1,0.2,...\"                          (FNO)\n"
        "  trtf inspect         <bundle.trtfb>\n"
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

    if (args.command != "run" && args.command != "inspect" &&
        args.command != "generate-video" &&
        args.command != "segment" && args.command != "generate-audio" &&
        args.command != "encode" && args.command != "solve" &&
        args.command != "detect" && args.command != "embed" &&
        args.command != "rerank" && args.command != "speak" &&
        args.command != "transcribe" && args.command != "segment-sam")
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

        if (arg == "--output" || arg == "-o")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.output_dir = argv[++i];
            continue;
        }

        if (arg == "--num-steps")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.num_steps = std::atoi(argv[++i]);
            continue;
        }

        if (arg == "--guidance-scale")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.guidance_scale = static_cast<float>(std::atof(argv[++i]));
            continue;
        }

        if (arg == "--threshold")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.conf_threshold = static_cast<float>(std::atof(argv[++i]));
            continue;
        }

        if (arg == "--branch-input")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.branch_input = argv[++i];
            continue;
        }

        if (arg == "--trunk-input")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.trunk_input = argv[++i];
            continue;
        }

        if (arg == "--field-input")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.field_input = argv[++i];
            continue;
        }

        if (arg == "--document")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.document = argv[++i];
            continue;
        }

        if (arg == "--audio-in")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.audio_in = argv[++i];
            continue;
        }

        if (arg == "--audio-out")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.audio_out = argv[++i];
            continue;
        }

        if (arg == "--audio")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.audio_in = argv[++i];
            continue;
        }

        if (arg == "--point-x")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.point_x = static_cast<float>(std::atof(argv[++i]));
            continue;
        }

        if (arg == "--point-y")
        {
            if (i + 1 >= argc)
            {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.point_y = static_cast<float>(std::atof(argv[++i]));
            continue;
        }

        if (arg == "--background")
        {
            args.is_foreground = false;
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

int cmd_generate_video(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: generate-video requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.prompt.empty())
    {
        std::cerr << "Error: generate-video requires --prompt\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_video())
    {
        std::cerr << "Error: this bundle does not support video generation\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const std::string out_dir = args.output_dir.empty()
        ? "/tmp/trtf_frames" : args.output_dir;

    const int32_t num_frames = pipeline->generate_video(
        args.prompt.c_str(), out_dir.c_str(),
        args.num_steps, args.guidance_scale);

    if (num_frames < 0)
    {
        std::cerr << "Error: video generation failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << "Generated " << num_frames << " frames in " << out_dir << '\n';
    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_segment(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: segment requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.image_path.empty())
    {
        std::cerr << "Error: segment requires --image\n";
        return EXIT_FAILURE;
    }

    const std::string out_path = args.output_dir.empty()
        ? "/tmp/segmentation_output.png" : args.output_dir;

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_segmentation())
    {
        std::cerr << "Error: this bundle does not support segmentation\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const int32_t rc = pipeline->segment(args.image_path.c_str(), out_path.c_str());
    if (rc != 0)
    {
        std::cerr << "Error: segmentation failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << "Segmentation saved to " << out_path << '\n';
    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_segment_sam(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: segment-sam requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.image_path.empty())
    {
        std::cerr << "Error: segment-sam requires --image\n";
        return EXIT_FAILURE;
    }

    const std::string out_dir = args.output_dir.empty()
        ? "/tmp/sam_masks" : args.output_dir;

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_prompted_segmentation())
    {
        std::cerr << "Error: this bundle does not support prompted segmentation (SAM)\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const int32_t num_masks = pipeline->segment_sam(
        args.image_path.c_str(), out_dir.c_str(),
        args.point_x, args.point_y, args.is_foreground);

    if (num_masks < 0)
    {
        std::cerr << "Error: SAM segmentation failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << "Generated " << num_masks << " masks in " << out_dir << '\n';
    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_detect(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: detect requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.image_path.empty())
    {
        std::cerr << "Error: detect requires --image\n";
        return EXIT_FAILURE;
    }

    const std::string out_path = args.output_dir.empty()
        ? "/tmp/detections.json" : args.output_dir;

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_detection())
    {
        std::cerr << "Error: this bundle does not support object detection\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const int32_t num_dets = pipeline->detect(
        args.image_path.c_str(), out_path.c_str(), args.conf_threshold);
    if (num_dets < 0)
    {
        std::cerr << "Error: detection failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << "Detected " << num_dets << " objects -> " << out_path << '\n';
    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_generate_audio(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: generate-audio requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.prompt.empty())
    {
        std::cerr << "Error: generate-audio requires --prompt\n";
        return EXIT_FAILURE;
    }

    const std::string out_path = args.output_dir.empty()
        ? "/tmp/generated_audio.wav" : args.output_dir;

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_audio())
    {
        std::cerr << "Error: this bundle does not support audio generation\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const int32_t num_samples = pipeline->generate_audio(
        args.prompt.c_str(), out_path.c_str(), args.max_new_tokens);

    if (num_samples < 0)
    {
        std::cerr << "Error: audio generation failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << "Generated " << num_samples << " audio samples -> " << out_path << '\n';
    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_encode(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: encode requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.prompt.empty())
    {
        std::cerr << "Error: encode requires --prompt\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_encoding())
    {
        std::cerr << "Error: this bundle does not support encoding\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    int32_t seq_len = 0;
    int32_t hidden_size = 0;
    const float* hidden_states = pipeline->encode(
        args.prompt.c_str(), &seq_len, &hidden_size);

    if (hidden_states == nullptr)
    {
        std::cerr << "Error: encoding failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    // Print summary: shape and first few values of [CLS] token embedding
    std::cout << "Hidden states shape: [" << seq_len << ", " << hidden_size << "]\n";
    std::cout << "[CLS] embedding (first 8 dims):";
    const int show = std::min(hidden_size, static_cast<int32_t>(8));
    for (int i = 0; i < show; ++i)
    {
        std::cout << " " << hidden_states[i];
    }
    std::cout << " ...\n";

    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_embed(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: embed requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.prompt.empty())
    {
        std::cerr << "Error: embed requires --prompt\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_embedding())
    {
        std::cerr << "Error: this bundle does not support embedding\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    int32_t emb_dim = 0;
    const float* embedding = pipeline->embed(args.prompt.c_str(), &emb_dim);

    if (embedding == nullptr)
    {
        std::cerr << "Error: embedding failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << "Embedding dim: " << emb_dim << '\n';
    std::cout << "Values (first 8 dims):";
    const int show = std::min(emb_dim, static_cast<int32_t>(8));
    for (int i = 0; i < show; ++i)
    {
        std::cout << " " << embedding[i];
    }
    if (emb_dim > show) std::cout << " ...";
    std::cout << '\n';

    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_rerank(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: rerank requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.prompt.empty())
    {
        std::cerr << "Error: rerank requires --prompt (query)\n";
        return EXIT_FAILURE;
    }

    if (args.document.empty())
    {
        std::cerr << "Error: rerank requires --document\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_reranking())
    {
        std::cerr << "Error: this bundle does not support reranking\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const float score = pipeline->rerank(args.prompt.c_str(), args.document.c_str());
    std::cout << "Relevance score: " << score << '\n';

    delete pipeline;
    return EXIT_SUCCESS;
}

std::vector<float> parse_csv_floats(const std::string& csv)
{
    std::vector<float> result;
    std::string token;
    for (std::size_t i = 0; i <= csv.size(); ++i)
    {
        if (i == csv.size() || csv[i] == ',')
        {
            if (!token.empty())
            {
                result.push_back(static_cast<float>(std::atof(token.c_str())));
                token.clear();
            }
        }
        else if (csv[i] != ' ')
        {
            token += csv[i];
        }
    }
    return result;
}

int cmd_solve(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: solve requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    const bool has_branch_trunk = !args.branch_input.empty() && !args.trunk_input.empty();
    const bool has_field = !args.field_input.empty();

    if (!has_branch_trunk && !has_field)
    {
        std::cerr << "Error: solve requires either --branch-input + --trunk-input (DeepONet) "
                     "or --field-input (FNO)\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_solve())
    {
        std::cerr << "Error: this bundle does not support solve (neural operator)\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    if (has_field)
    {
        // FNO field-based solve
        auto field_vals = parse_csv_floats(args.field_input);

        int32_t out_channels = 0, out_h = 0, out_w = 0;
        const float* output = pipeline->solve_field(
            field_vals.data(), static_cast<int32_t>(field_vals.size()),
            &out_channels, &out_h, &out_w);

        if (output == nullptr)
        {
            std::cerr << "Error: solve_field failed\n";
            delete pipeline;
            return EXIT_FAILURE;
        }

        const int32_t total = out_channels * out_h * out_w;
        std::cout << "Output shape: [" << out_channels << ", " << out_h << ", " << out_w << "]\n";
        const int32_t show = std::min(total, static_cast<int32_t>(16));
        std::cout << "First " << show << " values:";
        for (int32_t i = 0; i < show; ++i)
        {
            std::cout << " " << output[i];
        }
        if (total > show)
            std::cout << " ...";
        std::cout << '\n';
    }
    else
    {
        // DeepONet branch/trunk solve
        auto branch_vals = parse_csv_floats(args.branch_input);
        auto trunk_vals = parse_csv_floats(args.trunk_input);

        int32_t out_dim = 0;
        const float* output = pipeline->solve(
            branch_vals.data(), static_cast<int32_t>(branch_vals.size()),
            trunk_vals.data(), static_cast<int32_t>(trunk_vals.size()),
            &out_dim);

        if (output == nullptr)
        {
            std::cerr << "Error: solve failed\n";
            delete pipeline;
            return EXIT_FAILURE;
        }

        std::cout << "Output [" << out_dim << "]:";
        for (int32_t i = 0; i < out_dim; ++i)
        {
            std::cout << " " << output[i];
        }
        std::cout << '\n';
    }

    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_speak(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: speak requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.audio_in.empty())
    {
        std::cerr << "Error: speak requires --audio-in\n";
        return EXIT_FAILURE;
    }

    const std::string out_path = args.audio_out.empty()
        ? "/tmp/speech_output.wav" : args.audio_out;

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_speech())
    {
        std::cerr << "Error: this bundle does not support speech-to-speech\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const int32_t max_frames = args.max_new_tokens > 0
        ? args.max_new_tokens : -1;

    const int32_t num_samples = pipeline->speak(
        args.audio_in.c_str(), out_path.c_str(), max_frames);

    if (num_samples < 0)
    {
        std::cerr << "Error: speech-to-speech failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << "Generated " << num_samples << " audio samples -> "
              << out_path << '\n';
    delete pipeline;
    return EXIT_SUCCESS;
}

int cmd_transcribe(const CliArgs& args)
{
    if (args.bundle_path.empty())
    {
        std::cerr << "Error: transcribe requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    if (args.audio_in.empty())
    {
        std::cerr << "Error: transcribe requires --audio\n";
        return EXIT_FAILURE;
    }

    TrtfPipelineOptions opts{};
    opts.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    auto* pipeline = trtf_create_pipeline_ex(args.bundle_path.c_str(), &opts);
    if (pipeline == nullptr)
    {
        std::cerr << "Error: " << trtf_last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!pipeline->supports_transcription())
    {
        std::cerr << "Error: this bundle does not support transcription\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    const int32_t max_tokens = args.max_new_tokens > 0
        ? args.max_new_tokens : 224;

    const char* text = pipeline->transcribe(args.audio_in.c_str(), max_tokens);
    if (text == nullptr)
    {
        std::cerr << "Error: transcription failed\n";
        delete pipeline;
        return EXIT_FAILURE;
    }

    std::cout << text << '\n';
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
        if (!info.runtime_strategy.empty())
        {
            std::cout << "Runtime strategy:   " << info.runtime_strategy << '\n';
        }
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
    if (args.command == "encode")
    {
        return cmd_encode(args);
    }
    if (args.command == "segment")
    {
        return cmd_segment(args);
    }
    if (args.command == "segment-sam")
    {
        return cmd_segment_sam(args);
    }
    if (args.command == "detect")
    {
        return cmd_detect(args);
    }
    if (args.command == "generate-audio")
    {
        return cmd_generate_audio(args);
    }
    if (args.command == "generate-video")
    {
        return cmd_generate_video(args);
    }
    if (args.command == "solve")
    {
        return cmd_solve(args);
    }
    if (args.command == "embed")
    {
        return cmd_embed(args);
    }
    if (args.command == "rerank")
    {
        return cmd_rerank(args);
    }
    if (args.command == "speak")
    {
        return cmd_speak(args);
    }
    if (args.command == "transcribe")
    {
        return cmd_transcribe(args);
    }
    if (args.command == "inspect")
    {
        return cmd_inspect(args);
    }

    print_usage();
    return EXIT_FAILURE;
}
