// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-UTIL-CPP-03
// Architecture:   ARCH-BDL-001
// Unit Design:    UD-UTIL-01
// Intent:         Source/scripts dir resolution, runtime setting
// Preconditions:  set_source_dir("") for isolation
// Postconditions: source_dir/scripts_dir return valid paths
// =============================================================================
//
// After the config-registry migration, TRTF_DATA_DIR is gone. Resolution is:
//   1. Runtime value set via set_source_dir(...) (populated by
//      pipeline_factory from the platform.* config namespace).
//   2. TRTF_SOURCE_DIR compile-time define (baked in by CMake).
// Tests reset the setting to "" between cases to restore the compile-time
// fallback.
// =============================================================================

#include "utils/data_dir.h"

#include <filesystem>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_default_resolves_to_source_dir() {
    trtf::set_source_dir("");
    const std::string dir = trtf::source_dir();
    check(!dir.empty(), "source_dir not empty");
    check(dir != ".", "source_dir is not fallback dot");
}

static void test_runtime_setting() {
    trtf::set_source_dir("/tmp/claude/fake_trtf_root");
    const std::string dir = trtf::source_dir();
    check(dir == "/tmp/claude/fake_trtf_root", "source_dir matches runtime setting");
    trtf::set_source_dir("");
}

static void test_scripts_dir_suffix() {
    trtf::set_source_dir("");
    const std::string dir = trtf::scripts_dir();
    const std::string suffix = "/scripts";
    check(dir.size() >= suffix.size() &&
              dir.compare(dir.size() - suffix.size(), suffix.size(), suffix) == 0,
          "scripts_dir ends with /scripts");
}

static void test_models_dir_suffix() {
    trtf::set_source_dir("");
    const std::string dir = trtf::models_dir();
    const std::string suffix = "/models";
    check(dir.size() >= suffix.size() &&
              dir.compare(dir.size() - suffix.size(), suffix.size(), suffix) == 0,
          "models_dir ends with /models");
}

static void test_script_file_exists() {
    trtf::set_source_dir("");
    const std::string path = trtf::script_path("hf_tokenizer.py");
    check(std::filesystem::exists(path), "hf_tokenizer.py exists at resolved path");
}

static void test_model_path_resolution() {
    trtf::set_source_dir("");
    const std::string path = trtf::model_path("hf");
    check(!path.empty(), "model_path returns non-empty path");
}

static void test_empty_setting_uses_compile_default() {
    trtf::set_source_dir("");
    const std::string dir = trtf::source_dir();
    check(dir != "", "empty setting falls back to compile-time path");
}

int main() {
    test_default_resolves_to_source_dir();
    test_runtime_setting();
    test_scripts_dir_suffix();
    test_models_dir_suffix();
    test_script_file_exists();
    test_model_path_resolution();
    test_empty_setting_uses_compile_default();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All data_dir tests passed.\n";
    return 0;
}
