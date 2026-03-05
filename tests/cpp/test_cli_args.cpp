// =============================================================================
// Test suite: CLI argument parsing for the `trtf` command-line interface.
//
// Purpose:
//   Validates the CLI argument parser that powers the `trtf` executable. The
//   parser handles subcommands (run, detect, inspect, version, help),
//   positional arguments (bundle path), and option flags (--prompt,
//   --max-new-tokens, --hf-python, detection aliases).
//
// Dependencies:
//   - trtf/pipeline.h: only for basic type references. No GPU, TRT, or
//     filesystem access required.
//
// Approach:
//   The production CLI parser lives inside trtf_cli.cpp, which has its own
//   main(). To test in isolation without linking two main() symbols, this file
//   replicates the CliArgs struct and parse_args() function matching the
//   simplified production code. A convenience wrapper parse(vector<const char*>)
//   converts a brace-init list to argc/argv for concise test invocations.
//
//   Each test function simulates a specific command-line invocation, parses it,
//   and asserts that the resulting CliArgs fields match expected values.
//
// Test categories:
//   - Subcommand parsing: run, detect, inspect, version, help
//   - Flag handling: --prompt, --max-new-tokens, --hf-python, detection aliases
//   - Error handling: unknown flags, unknown commands
//   - No-args: bare invocation shows help
// =============================================================================

#include "trtf/pipeline.h"

#include <cstdlib>
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
    std::string prompt;
    std::string hf_python;
    std::string image_path;
    std::string output_path;
    int max_new_tokens{0};
    float conf_threshold{-1.0F};
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

    if (args.command != "run" && args.command != "inspect" && args.command != "detect")
    {
        args.parse_error = true;
        args.error_message = "Unknown command: " + args.command;
        return args;
    }

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];

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
        if (arg == "--hf-python")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.hf_python = argv[++i];
            continue;
        }
        if (arg == "--image")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.image_path = argv[++i];
            continue;
        }
        if (arg == "--output" || arg == "--output-json" || arg == "-o")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.output_path = argv[++i];
            continue;
        }
        if (arg == "--threshold" || arg == "--score-threshold")
        {
            if (i + 1 >= argc) { args.parse_error = true; args.error_message = arg + " requires a value"; return args; }
            args.conf_threshold = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
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
// Intention: Verify that "trtf run bundle.trtfb --prompt 'hello world'"
//   correctly parses the run subcommand and captures the prompt string.
// Setup: Simulated argv with "run", a bundle path, and a multi-word prompt.
// Mechanism: Calls parse(), checks command=="run", model_or_bundle=="bundle.trtfb",
//   and prompt=="hello world".
// -----------------------------------------------------------------------------
static void test_run_with_prompt()
{
    auto args = parse({"trtf", "run", "bundle.trtfb", "--prompt", "hello world"});
    check(args.command == "run", "run command");
    check(args.model_or_bundle == "bundle.trtfb", "run bundle path");
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
    auto args = parse({"trtf", "run", "bundle.trtfb", "--prompt", "hi", "--max-new-tokens", "50"});
    check(args.max_new_tokens == 50, "run max_new_tokens");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --hf-python correctly captures the path to a Python
//   interpreter, used for the HuggingFace tokenizer bridge.
// Setup: Simulated argv with "run" + "--hf-python /usr/bin/python3".
// Mechanism: Calls parse(), checks no parse error and hf_python=="/usr/bin/python3".
// -----------------------------------------------------------------------------
static void test_hf_python_flag()
{
    auto args = parse({"trtf", "run", "bundle.trtfb", "--prompt", "hi", "--hf-python", "/usr/bin/python3"});
    check(!args.parse_error, "hf-python no parse error");
    check(args.hf_python == "/usr/bin/python3", "hf-python value");
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
    auto args = parse({"trtf", "run", "bundle.trtfb", "--bogus"});
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
// Intention: Verify that all supported run flags can be combined in a single
//   invocation without conflicts or parse errors.
// Setup: Simulated argv with "run" and every supported flag.
// Mechanism: Calls parse(), checks every field matches the expected value
//   and no parse error occurred.
// -----------------------------------------------------------------------------
static void test_all_run_flags_combined()
{
    auto args = parse({"trtf", "run", "bundle.trtfb", "--prompt", "hello",
        "--max-new-tokens", "10",
        "--hf-python", "/usr/bin/python3"});
    check(!args.parse_error, "combined flags no parse error");
    check(args.model_or_bundle == "bundle.trtfb", "combined bundle path");
    check(args.prompt == "hello", "combined prompt");
    check(args.max_new_tokens == 10, "combined max_new_tokens");
    check(args.hf_python == "/usr/bin/python3", "combined hf-python");
}

// -----------------------------------------------------------------------------
// Intention: Verify detect alias flags parse exactly like canonical names.
// Setup: Simulated argv with detect + --output-json + --score-threshold.
// Mechanism: Calls parse(), checks parsed command and values.
// -----------------------------------------------------------------------------
static void test_detect_alias_flags()
{
    auto args = parse({"trtf", "detect", "bundle.trtfb",
        "--image", "img.jpg",
        "--output-json", "det.json",
        "--score-threshold", "0.25"});
    check(!args.parse_error, "detect aliases no parse error");
    check(args.command == "detect", "detect command");
    check(args.image_path == "img.jpg", "detect image path");
    check(args.output_path == "det.json", "detect output path");
    check(args.conf_threshold == 0.25F, "detect threshold");
}

// -----------------------------------------------------------------------------
// Intention: Verify strict unknown-flag behavior is preserved after adding
//   alias support for known detection flags.
// Setup: Simulated argv with detect + aliases + unknown flag.
// Mechanism: Calls parse(), checks parse_error and unknown flag message.
// -----------------------------------------------------------------------------
static void test_detect_unknown_flag_still_errors()
{
    auto args = parse({"trtf", "detect", "bundle.trtfb",
        "--image", "img.jpg",
        "--output-json", "det.json",
        "--score-threshold", "0.25",
        "--not-a-real-flag"});
    check(args.parse_error, "detect unknown flag causes error");
    check(args.error_message.find("--not-a-real-flag") != std::string::npos,
          "detect unknown flag message mentions flag");
}

int main()
{
    test_run_with_prompt();
    test_run_max_tokens();
    test_hf_python_flag();
    test_inspect_subcommand();
    test_no_args_shows_usage();
    test_help_flag();
    test_version_subcommand();
    test_unknown_flag_errors();
    test_unknown_command_errors();
    test_all_run_flags_combined();
    test_detect_alias_flags();
    test_detect_unknown_flag_still_errors();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All cli_args tests passed.\n";
    return 0;
}
