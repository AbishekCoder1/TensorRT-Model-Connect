// =============================================================================
// Test suite: CLI argument parsing for the `trtf` command-line interface.
//
// Purpose:
//   Validates the CLI argument parser that powers the `trtf` executable. The
//   parser handles subcommands (run, build, inspect, version, help), positional
//   arguments (model/bundle path), and option flags (--prompt, --force-trt,
//   --max-new-tokens, --hf-python, --engine-cache-dir, --no-engine-cache, etc.).
//
// Dependencies:
//   - trtf/pipeline.h: only for TRTF_PREFER_TRT / TRTF_FORCE_TRT / TRTF_CPU_ONLY
//     flag constants. No GPU, TRT, or filesystem access required.
//
// Approach:
//   The production CLI parser lives inside trtf_cli.cpp, which has its own
//   main(). To test in isolation without linking two main() symbols, this file
//   replicates the CliArgs struct and parse_args() function verbatim from the
//   production code. A convenience wrapper parse(vector<const char*>) converts
//   a brace-init list to argc/argv for concise test invocations.
//
//   Each test function simulates a specific command-line invocation, parses it,
//   and asserts that the resulting CliArgs fields match expected values.
//
// Test categories:
//   - Subcommand parsing: build, run, inspect, version, help
//   - Flag handling: --prompt, --force-trt, --cpu-only, --max-new-tokens,
//     --max-cache-length, --hf-python, --engine-cache-dir, --no-engine-cache
//   - Error handling: unknown flags, unknown commands, missing args
//   - Combination: all flags used together in a single invocation
// =============================================================================

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

// -----------------------------------------------------------------------------
// Intention: Verify that "trtf build model_dir -o out.trtfb" correctly parses
//   the "build" subcommand, extracts the positional model directory, and
//   captures the -o output path.
// Setup: Simulated argv: {"trtf", "build", "model_dir", "-o", "out.trtfb"}.
// Mechanism: Calls parse(), checks command=="build", model_or_bundle=="model_dir",
//   output_path=="out.trtfb", and no parse error occurred.
// -----------------------------------------------------------------------------
static void test_build_subcommand()
{
    auto args = parse({"trtf", "build", "model_dir", "-o", "out.trtfb"});
    check(args.command == "build", "build command");
    check(args.model_or_bundle == "model_dir", "build model_dir");
    check(args.output_path == "out.trtfb", "build output_path");
    check(!args.parse_error, "build no parse error");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --max-cache-length is correctly parsed with the build
//   subcommand, producing the expected integer value.
// Setup: Simulated argv with "build" + "--max-cache-length 256".
// Mechanism: Calls parse(), checks command=="build" and max_cache_length==256.
// -----------------------------------------------------------------------------
static void test_build_with_max_cache()
{
    auto args = parse({"trtf", "build", "model_dir", "-o", "out.trtfb", "--max-cache-length", "256"});
    check(args.command == "build", "build+cache command");
    check(args.max_cache_length == 256, "build max_cache_length");
}

// -----------------------------------------------------------------------------
// Intention: Verify that "trtf run model --prompt 'hello world'" correctly
//   parses the run subcommand and captures the prompt string with spaces.
// Setup: Simulated argv with "run", a model name, and a multi-word prompt.
// Mechanism: Calls parse(), checks command=="run", model_or_bundle=="model",
//   and prompt=="hello world".
// -----------------------------------------------------------------------------
static void test_run_with_prompt()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hello world"});
    check(args.command == "run", "run command");
    check(args.model_or_bundle == "model", "run model");
    check(args.prompt == "hello world", "run prompt");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --max-new-tokens is parsed as an integer for the run
//   subcommand.
// Setup: Simulated argv with "run" + "--max-new-tokens 50".
// Mechanism: Calls parse(), checks max_new_tokens==50.
// -----------------------------------------------------------------------------
static void test_run_max_tokens()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--max-new-tokens", "50"});
    check(args.max_new_tokens == 50, "run max_new_tokens");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --force-trt sets the flags field to TRTF_FORCE_TRT,
//   overriding the default TRTF_PREFER_TRT.
// Setup: Simulated argv with "run" + "--force-trt".
// Mechanism: Calls parse(), checks flags==TRTF_FORCE_TRT.
// -----------------------------------------------------------------------------
static void test_run_force_trt()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--force-trt"});
    check(args.flags == TRTF_FORCE_TRT, "run force_trt flag");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --cpu-only sets the flags field to TRTF_CPU_ONLY,
//   overriding the default TRTF_PREFER_TRT.
// Setup: Simulated argv with "run" + "--cpu-only".
// Mechanism: Calls parse(), checks flags==TRTF_CPU_ONLY.
// -----------------------------------------------------------------------------
static void test_run_cpu_only()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--cpu-only"});
    check(args.flags == TRTF_CPU_ONLY, "run cpu_only flag");
}

// -----------------------------------------------------------------------------
// Intention: Verify that the "inspect" subcommand correctly parses and captures
//   the positional bundle file path.
// Setup: Simulated argv: {"trtf", "inspect", "file.trtfb"}.
// Mechanism: Calls parse(), checks command=="inspect" and
//   model_or_bundle=="file.trtfb".
// -----------------------------------------------------------------------------
static void test_inspect_subcommand()
{
    auto args = parse({"trtf", "inspect", "file.trtfb"});
    check(args.command == "inspect", "inspect command");
    check(args.model_or_bundle == "file.trtfb", "inspect file path");
}

// -----------------------------------------------------------------------------
// Intention: Verify that invoking "trtf" with no arguments sets show_help=true
//   (the expected behavior for a bare invocation with no subcommand).
// Setup: Simulated argv: {"trtf"} (argc==1, no subcommand).
// Mechanism: Calls parse(), checks show_help==true.
// -----------------------------------------------------------------------------
static void test_no_args_shows_usage()
{
    auto args = parse({"trtf"});
    check(args.show_help, "no args shows help");
}

// -----------------------------------------------------------------------------
// Intention: Verify that "trtf --help" sets show_help=true.
// Setup: Simulated argv: {"trtf", "--help"}.
// Mechanism: Calls parse(), checks show_help==true.
// -----------------------------------------------------------------------------
static void test_help_flag()
{
    auto args = parse({"trtf", "--help"});
    check(args.show_help, "--help shows help");
}

// -----------------------------------------------------------------------------
// Intention: Verify that "trtf version" parses the version subcommand.
// Setup: Simulated argv: {"trtf", "version"}.
// Mechanism: Calls parse(), checks command=="version".
// -----------------------------------------------------------------------------
static void test_version_subcommand()
{
    auto args = parse({"trtf", "version"});
    check(args.command == "version", "version command");
}

// -----------------------------------------------------------------------------
// Intention: Verify that an unknown flag (e.g., --bogus) is rejected with a
//   parse error whose message mentions the offending flag.
// Setup: Simulated argv with "run" + "--bogus".
// Mechanism: Calls parse(), checks parse_error==true and error_message contains
//   "--bogus".
// -----------------------------------------------------------------------------
static void test_unknown_flag_errors()
{
    auto args = parse({"trtf", "run", "model", "--bogus"});
    check(args.parse_error, "unknown flag causes error");
    check(args.error_message.find("--bogus") != std::string::npos, "error message mentions flag");
}

// -----------------------------------------------------------------------------
// Intention: Verify that an unknown subcommand (e.g., "foobar") is rejected
//   with a parse error whose message mentions the unknown command name.
// Setup: Simulated argv: {"trtf", "foobar"}.
// Mechanism: Calls parse(), checks parse_error==true and error_message contains
//   "foobar".
// -----------------------------------------------------------------------------
static void test_unknown_command_errors()
{
    auto args = parse({"trtf", "foobar"});
    check(args.parse_error, "unknown command causes error");
    check(args.error_message.find("foobar") != std::string::npos, "error message mentions command");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --hf-python correctly captures the path to a Python
//   interpreter, used for the HuggingFace tokenizer bridge.
// Setup: Simulated argv with "run" + "--hf-python /usr/bin/python3".
// Mechanism: Calls parse(), checks no parse error and hf_python=="/usr/bin/python3".
// -----------------------------------------------------------------------------
static void test_hf_python_flag()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--hf-python", "/usr/bin/python3"});
    check(!args.parse_error, "hf-python no parse error");
    check(args.hf_python == "/usr/bin/python3", "hf-python value");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --engine-cache-dir correctly captures the cache
//   directory path for on-disk TRT engine plan caching.
// Setup: Simulated argv with "run" + "--engine-cache-dir /tmp/cache".
// Mechanism: Calls parse(), checks no parse error and
//   engine_cache_dir=="/tmp/cache".
// -----------------------------------------------------------------------------
static void test_engine_cache_dir_flag()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--engine-cache-dir", "/tmp/cache"});
    check(!args.parse_error, "engine-cache-dir no parse error");
    check(args.engine_cache_dir == "/tmp/cache", "engine-cache-dir value");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --no-engine-cache sets the boolean flag to true,
//   indicating engine plan caching should be disabled.
// Setup: Simulated argv with "run" + "--no-engine-cache".
// Mechanism: Calls parse(), checks no parse error and no_engine_cache==true.
// -----------------------------------------------------------------------------
static void test_no_engine_cache_flag()
{
    auto args = parse({"trtf", "run", "model", "--prompt", "hi", "--no-engine-cache"});
    check(!args.parse_error, "no-engine-cache no parse error");
    check(args.no_engine_cache, "no-engine-cache is true");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --hf-python works correctly with the "build"
//   subcommand (not just "run"), since build also needs tokenizer access.
// Setup: Simulated argv with "build" + "--hf-python /opt/python".
// Mechanism: Calls parse(), checks no parse error and hf_python=="/opt/python".
// -----------------------------------------------------------------------------
static void test_build_with_hf_python()
{
    auto args = parse({"trtf", "build", "model_dir", "-o", "out.trtfb", "--hf-python", "/opt/python"});
    check(!args.parse_error, "build+hf-python no parse error");
    check(args.hf_python == "/opt/python", "build hf-python value");
}

// -----------------------------------------------------------------------------
// Intention: Verify that all run flags can be combined in a single invocation
//   without conflicts or parse errors. This is the integration test for flag
//   coexistence: prompt, max-new-tokens, force-trt, hf-python, engine-cache-dir,
//   and no-engine-cache all specified together.
// Setup: Simulated argv with "run" and every supported flag.
// Mechanism: Calls parse(), checks every field matches the expected value
//   and no parse error occurred.
// -----------------------------------------------------------------------------
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
