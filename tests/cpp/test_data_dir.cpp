// =============================================================================
// test_data_dir.cpp — Unit tests for src/utils/data_dir.cpp
// =============================================================================
//
// Purpose:
//   Validates the centralized source-directory resolution logic used by the
//   entire codebase to locate runtime assets (scripts, model directories,
//   vocabulary files). The data_dir module resolves paths through a two-tier
//   strategy:
//     1. TRTF_DATA_DIR environment variable (if set and non-empty).
//     2. TRTF_SOURCE_DIR compile-time define (baked in by CMake).
//   The tests verify both tiers, the derived path helpers (scripts_dir,
//   models_dir, script_path, model_path), and edge cases (empty env var).
//
// Dependencies:
//   - utils/data_dir.h (source_dir, scripts_dir, models_dir, script_path,
//     model_path)
//   - <filesystem> for verifying that resolved paths actually exist on disk.
//   - <cstdlib> for setenv/unsetenv to manipulate TRTF_DATA_DIR.
//
// Approach:
//   Each test manipulates (or clears) the TRTF_DATA_DIR environment variable,
//   then calls the appropriate data_dir function and checks the returned path
//   against expectations. Tests that check filesystem existence (e.g.,
//   hf_tokenizer.py) rely on the real source tree being available at the
//   compile-time TRTF_SOURCE_DIR.
//
// Environment:
//   CPU-only, no TRT/CUDA dependencies. Requires the source tree to be
//   present at the compiled-in TRTF_SOURCE_DIR for filesystem-based tests.
// =============================================================================

#include "utils/data_dir.h"
#include "test_helpers.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

static int failures = 0;

// Helper: records a failure with a descriptive test name if condition is false.
static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Intention: Verify that without TRTF_DATA_DIR set, source_dir() returns the
//            compiled-in TRTF_SOURCE_DIR (a non-empty, non-"." path).
// Setup:     Unsets TRTF_DATA_DIR to force the compile-time fallback.
// Mechanism: Calls source_dir(), checks the result is non-empty and not the
//            fallback "." (which would indicate TRTF_SOURCE_DIR was not set).
static void test_default_resolves_to_source_dir()
{
    // Without TRTF_DATA_DIR set, should return the compiled-in TRTF_SOURCE_DIR.
    trtf_test::EnvVarGuard guard("TRTF_DATA_DIR");  // unsets the var
    const std::string dir = trtf::source_dir();
    check(!dir.empty(), "source_dir not empty");
    check(dir != ".", "source_dir is not fallback dot");
}

// Intention: Verify that TRTF_DATA_DIR environment variable overrides the
//            compiled-in source directory.
// Setup:     Sets TRTF_DATA_DIR to a known fake path, then unsets it after.
// Mechanism: Calls source_dir(), checks the result exactly matches the env
//            var value "/tmp/claude/fake_trtf_root".
static void test_env_override()
{
    trtf_test::EnvVarGuard guard("TRTF_DATA_DIR", "/tmp/claude/fake_trtf_root");
    const std::string dir = trtf::source_dir();
    check(dir == "/tmp/claude/fake_trtf_root", "source_dir matches env override");
}

// Intention: Verify that scripts_dir() returns a path ending in "/scripts".
// Setup:     Unsets TRTF_DATA_DIR to use the compiled-in source directory.
// Mechanism: Calls scripts_dir(), checks the returned string ends with the
//            suffix "/scripts".
static void test_scripts_dir_suffix()
{
    trtf_test::EnvVarGuard guard("TRTF_DATA_DIR");
    const std::string dir = trtf::scripts_dir();
    const std::string suffix = "/scripts";
    check(dir.size() >= suffix.size() && dir.compare(dir.size() - suffix.size(), suffix.size(), suffix) == 0,
        "scripts_dir ends with /scripts");
}

// Intention: Verify that models_dir() returns a path ending in "/models".
// Setup:     Unsets TRTF_DATA_DIR to use the compiled-in source directory.
// Mechanism: Calls models_dir(), checks the returned string ends with the
//            suffix "/models".
static void test_models_dir_suffix()
{
    trtf_test::EnvVarGuard guard("TRTF_DATA_DIR");
    const std::string dir = trtf::models_dir();
    const std::string suffix = "/models";
    check(dir.size() >= suffix.size() && dir.compare(dir.size() - suffix.size(), suffix.size(), suffix) == 0,
        "models_dir ends with /models");
}

// Intention: Verify that script_path() resolves to an actual file on disk,
//            confirming the source directory is correctly pointing at the
//            real project tree.
// Setup:     Unsets TRTF_DATA_DIR to use the compiled-in source directory.
// Mechanism: Calls script_path("hf_tokenizer.py"), then checks the resolved
//            path exists on the filesystem using std::filesystem::exists().
static void test_script_file_exists()
{
    trtf_test::EnvVarGuard guard("TRTF_DATA_DIR");
    const std::string path = trtf::script_path("hf_tokenizer.py");
    check(std::filesystem::exists(path), "hf_tokenizer.py exists at resolved path");
}

// Intention: Verify that model_path() returns a non-empty path for a known
//            relative model directory.
// Setup:     Unsets TRTF_DATA_DIR to use the compiled-in source directory.
// Mechanism: Calls model_path("hf"), checks the result is non-empty.
static void test_model_path_resolution()
{
    trtf_test::EnvVarGuard guard("TRTF_DATA_DIR");
    const std::string path = trtf::model_path("hf");
    check(!path.empty(), "model_path returns non-empty path");
}

// Intention: Verify that setting TRTF_DATA_DIR to an empty string is treated
//            the same as not setting it (the empty override is ignored and
//            the compiled-in path is used instead).
// Setup:     Sets TRTF_DATA_DIR to "", then unsets after the check.
// Mechanism: Calls source_dir(), checks the result is non-empty (meaning the
//            empty env var was ignored in favor of the compiled-in path).
static void test_env_empty_string_ignored()
{
    trtf_test::EnvVarGuard guard("TRTF_DATA_DIR", "");
    const std::string dir = trtf::source_dir();
    check(dir != "", "empty TRTF_DATA_DIR is ignored, returns compiled-in path");
}

int main()
{
    test_default_resolves_to_source_dir();
    test_env_override();
    test_scripts_dir_suffix();
    test_models_dir_suffix();
    test_script_file_exists();
    test_model_path_resolution();
    test_env_empty_string_ignored();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All data_dir tests passed.\n";
    return 0;
}
