// =============================================================================
// C ABI runtime regression tests for bundle -> TRT runtime -> deserialize path.
//
// These tests build a syntactically valid .trtfb file with an invalid
// engine_plan payload, then call trtf_create_pipeline repeatedly. This catches
// crashes in runtime/logger lifetime and validates that failures are reported
// through trtf_last_error().
// =============================================================================

#include "trtf/pipeline.h"
#include "test_helpers.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void write_u64_le(std::ofstream& out, uint64_t value)
{
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i)
    {
        bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFFU);
    }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

void write_invalid_engine_bundle(const std::filesystem::path& path)
{
    // Internal .trtfb magic: "TRTFB\0\1\0"
    static constexpr unsigned char kBundleMagic[8] = {'T', 'R', 'T', 'F', 'B', '\0', '\x01', '\0'};

    const std::string header = R"({
  "model_id": "runtime-regression-test",
  "model_type": "unit-test",
  "family": "unit",
  "hidden_size": 64,
  "num_layers": 1,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "max_cache_length": 32,
  "sections": {
    "engine_plan": {"offset": 0, "size": 16}
  }
})";

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(kBundleMagic), sizeof(kBundleMagic));
    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    // Intentionally invalid TensorRT plan payload.
    static constexpr char kInvalidPlan[16] = {
        'N', 'O', 'T', '_', 'A', '_', 'P', 'L', 'A', 'N', '_', 'B', 'L', 'O', 'B', '!'};
    out.write(kInvalidPlan, sizeof(kInvalidPlan));
}

bool message_contains_any_expected_failure(const std::string& msg)
{
    return msg.find("deserialize engine") != std::string::npos
        || msg.find("execution context") != std::string::npos
        || msg.find("Failed to load bundle") != std::string::npos;
}

void expect_invalid_bundle_creation_fails(const std::string& bundle_path, const char* test_name)
{
    auto* pipeline = trtf_create_pipeline(bundle_path.c_str(), 0);
    check(pipeline == nullptr, test_name);

    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "trtf_last_error set for invalid plan bundle");
    if (err != nullptr)
    {
        check(message_contains_any_expected_failure(err), "error message indicates TRT runtime failure");
    }
}

void test_invalid_plan_bundle_reports_error()
{
#if TRTF_HAS_TRT
    trtf_test::TempDirGuard dir;
    const std::filesystem::path bundle_path = std::filesystem::path(dir.path()) / "invalid_engine_plan.trtfb";
    write_invalid_engine_bundle(bundle_path);
    expect_invalid_bundle_creation_fails(bundle_path.string(), "invalid plan bundle returns nullptr");
#else
    std::cerr << "SKIP: TRTF_HAS_TRT=0\n";
#endif
}

void test_invalid_plan_bundle_repeatable()
{
#if TRTF_HAS_TRT
    trtf_test::TempDirGuard dir;
    const std::filesystem::path bundle_path = std::filesystem::path(dir.path()) / "invalid_engine_plan_loop.trtfb";
    write_invalid_engine_bundle(bundle_path);

    for (int i = 0; i < 25; ++i)
    {
        expect_invalid_bundle_creation_fails(bundle_path.string(), "invalid plan bundle repeated returns nullptr");
    }
#else
    std::cerr << "SKIP: TRTF_HAS_TRT=0\n";
#endif
}

} // namespace

int main()
{
    test_invalid_plan_bundle_reports_error();
    test_invalid_plan_bundle_repeatable();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI runtime regression tests passed.\n";
    return 0;
}
