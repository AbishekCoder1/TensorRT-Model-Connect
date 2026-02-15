#include "utils/data_dir.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_default_resolves_to_source_dir()
{
    // Without TRTF_DATA_DIR set, should return the compiled-in TRTF_SOURCE_DIR.
    unsetenv("TRTF_DATA_DIR");
    const std::string dir = trtf::source_dir();
    check(!dir.empty(), "source_dir not empty");
    check(dir != ".", "source_dir is not fallback dot");
}

static void test_env_override()
{
    setenv("TRTF_DATA_DIR", "/tmp/claude/fake_trtf_root", 1);
    const std::string dir = trtf::source_dir();
    check(dir == "/tmp/claude/fake_trtf_root", "source_dir matches env override");
    unsetenv("TRTF_DATA_DIR");
}

static void test_scripts_dir_suffix()
{
    unsetenv("TRTF_DATA_DIR");
    const std::string dir = trtf::scripts_dir();
    const std::string suffix = "/scripts";
    check(dir.size() >= suffix.size() && dir.compare(dir.size() - suffix.size(), suffix.size(), suffix) == 0,
        "scripts_dir ends with /scripts");
}

static void test_models_dir_suffix()
{
    unsetenv("TRTF_DATA_DIR");
    const std::string dir = trtf::models_dir();
    const std::string suffix = "/models";
    check(dir.size() >= suffix.size() && dir.compare(dir.size() - suffix.size(), suffix.size(), suffix) == 0,
        "models_dir ends with /models");
}

static void test_script_file_exists()
{
    unsetenv("TRTF_DATA_DIR");
    const std::string path = trtf::script_path("hf_tokenizer.py");
    check(std::filesystem::exists(path), "hf_tokenizer.py exists at resolved path");
}

static void test_model_path_resolution()
{
    unsetenv("TRTF_DATA_DIR");
    const std::string path = trtf::model_path("hf");
    check(!path.empty(), "model_path returns non-empty path");
}

static void test_env_empty_string_ignored()
{
    setenv("TRTF_DATA_DIR", "", 1);
    const std::string dir = trtf::source_dir();
    check(dir != "", "empty TRTF_DATA_DIR is ignored, returns compiled-in path");
    unsetenv("TRTF_DATA_DIR");
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
