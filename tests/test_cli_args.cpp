// Test: CLI argument parsing for trtf_cli.
// Verifies: subcommand parsing, flag handling, error on unknown flags.
// Approach: We replicate the parse_args struct+function from trtf_cli.cpp here
// to test in isolation (trtf_cli has its own main()).

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

namespace {

struct CliArgs {
    std::string command;
    std::string model_or_bundle;
    std::string output_path;
    std::string prompt;
    std::string hf_python;
    std::string engine_cache_dir;
    int max_new_tokens{0};
    int max_cache_length{-1};
    int flags{TRTF_PREFER_TRT};
    bool no_engine_cache{false};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
};

CliArgs parse_args(int argc, const char** argv)
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

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "-o" || arg == "--output")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.output_path = argv[++i];
            continue;
        }
        if (arg == "--prompt" || arg == "-p")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.prompt = argv[++i];
            continue;
        }
        if (arg == "--max-new-tokens")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.max_new_tokens = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--max-cache-length")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.max_cache_length = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--hf-python")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.hf_python = argv[++i];
            continue;
        }
        if (arg == "--engine-cache-dir")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.engine_cache_dir = argv[++i];
            continue;
        }
        if (arg == "--no-engine-cache") { args.no_engine_cache = true; continue; }
        if (arg == "--force-trt") { args.flags = TRTF_FORCE_TRT; continue; }
        if (arg == "--cpu-only") { args.flags = TRTF_CPU_ONLY; continue; }
        if (arg[0] == '-') { args.parse_error = true; args.error_message = "Unknown flag: " + arg; return args; }

        if (args.model_or_bundle.empty()) { args.model_or_bundle = arg; }
        else { args.parse_error = true; args.error_message = "Unexpected positional argument: " + arg; return args; }
    }

    return args;
}

CliArgs parse(std::vector<const char*> argv_vec)
{
    return parse_args(static_cast<int>(argv_vec.size()), argv_vec.data());
}

} // namespace

static void test_build_subcommand()
{
    auto args = parse({"trtf", "build", "model_dir", "-o", "out.trtfb"});
    check(args.command == "build", "build command");
    check(args.model_or_bundle == "model_dir", "build model_dir");
    check(args.output_path == "out.trtfb", "build output_path");
    check(!args.parse_error, "build no parse error");
}

static void test_build_with_max_cache()
{
    auto args = parse({"trtf", "build", "model_dir", "-o", "out.trtfb", "--max-cache-length", "256"});
    check(args.command == "build", "build+cache command");
    check(args.max_cache_length == 256, "build max_cache_length");
}

static void test_run_with_prompt()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hello world"});
    check(args.command == "run", "run command");
    check(args.model_or_bundle == "model", "run model");
    check(args.prompt == "hello world", "run prompt");
}

static void test_run_max_tokens()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--max-new-tokens", "50"});
    check(args.max_new_tokens == 50, "run max_new_tokens");
}

static void test_run_force_trt()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--force-trt"});
    check(args.flags == TRTF_FORCE_TRT, "run force_trt flag");
}

static void test_run_cpu_only()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--cpu-only"});
    check(args.flags == TRTF_CPU_ONLY, "run cpu_only flag");
}

static void test_inspect_subcommand()
{
    auto args = parse({"trtf", "inspect", "file.trtfb"});
    check(args.command == "inspect", "inspect command");
    check(args.model_or_bundle == "file.trtfb", "inspect file path");
}

static void test_no_args_shows_usage()
{
    auto args = parse({"trtf"});
    check(args.show_help, "no args shows help");
}

static void test_help_flag()
{
    auto args = parse({"trtf", "--help"});
    check(args.show_help, "--help shows help");
}

static void test_version_subcommand()
{
    auto args = parse({"trtf", "version"});
    check(args.command == "version", "version command");
}

static void test_unknown_flag_errors()
{
    auto args = parse({"trtf", "run", "model", "--bogus"});
    check(args.parse_error, "unknown flag causes error");
    check(args.error_message.find("--bogus") != std::string::npos, "error message mentions flag");
}

static void test_unknown_command_errors()
{
    auto args = parse({"trtf", "foobar"});
    check(args.parse_error, "unknown command causes error");
    check(args.error_message.find("foobar") != std::string::npos, "error message mentions command");
}

static void test_hf_python_flag()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--hf-python", "/usr/bin/python3"});
    check(!args.parse_error, "hf-python no parse error");
    check(args.hf_python == "/usr/bin/python3", "hf-python value");
}

static void test_engine_cache_dir_flag()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--engine-cache-dir", "/tmp/cache"});
    check(!args.parse_error, "engine-cache-dir no parse error");
    check(args.engine_cache_dir == "/tmp/cache", "engine-cache-dir value");
}

static void test_no_engine_cache_flag()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--no-engine-cache"});
    check(!args.parse_error, "no-engine-cache no parse error");
    check(args.no_engine_cache, "no-engine-cache is true");
}

static void test_build_with_hf_python()
{
    auto args = parse({"trtf", "build", "model_dir", "-o", "out.trtfb", "--hf-python", "/opt/python"});
    check(!args.parse_error, "build+hf-python no parse error");
    check(args.hf_python == "/opt/python", "build hf-python value");
}

static void test_all_run_flags_combined()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hello",
        "--max-new-tokens", "10", "--force-trt",
        "--hf-python", "/usr/bin/python3",
        "--engine-cache-dir", "/tmp/cache",
        "--no-engine-cache"});
    check(!args.parse_error, "combined flags no parse error");
    check(args.model_or_bundle == "model", "combined model");
    check(args.prompt == "hello", "combined prompt");
    check(args.max_new_tokens == 10, "combined max_new_tokens");
    check(args.flags == TRTF_FORCE_TRT, "combined force_trt");
    check(args.hf_python == "/usr/bin/python3", "combined hf-python");
    check(args.engine_cache_dir == "/tmp/cache", "combined engine-cache-dir");
    check(args.no_engine_cache, "combined no-engine-cache");
}

int main()
{
    test_build_subcommand();
    test_build_with_max_cache();
    test_run_with_prompt();
    test_run_max_tokens();
    test_run_force_trt();
    test_run_cpu_only();
    test_inspect_subcommand();
    test_no_args_shows_usage();
    test_help_flag();
    test_version_subcommand();
    test_unknown_flag_errors();
    test_unknown_command_errors();
    test_hf_python_flag();
    test_engine_cache_dir_flag();
    test_no_engine_cache_flag();
    test_build_with_hf_python();
    test_all_run_flags_combined();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All cli_args tests passed.\n";
    return 0;
}
