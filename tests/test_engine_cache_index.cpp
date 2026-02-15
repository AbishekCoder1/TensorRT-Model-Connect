// Test: BuildModelDirIndexKey, SaveModelDirIndex, LookupModelDirIndex from engine_cache.cpp.
// CPU-only tests using temp dirs with synthetic config.json + safetensors.

#include "utils/trt/engine_cache.h"
#include "test_helpers.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
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

// Helper: set an env var, returning previous value (empty string = was unset).
static std::string set_env(const char* name, const char* value)
{
    const char* prev = std::getenv(name);
    std::string old_val = prev ? prev : "";
    if (value != nullptr)
    {
        setenv(name, value, 1);
    }
    else
    {
        unsetenv(name);
    }
    return old_val;
}

static void restore_env(const char* name, const std::string& old_val)
{
    if (old_val.empty())
    {
        unsetenv(name);
    }
    else
    {
        setenv(name, old_val.c_str(), 1);
    }
}

// Create a minimal model dir with config.json and model.safetensors
static void write_minimal_model_dir(const std::filesystem::path& dir, const std::string& config_json)
{
    trtf_test::write_file(dir / "config.json", config_json);
    // Write a tiny safetensors file (just one tensor)
    trtf_test::write_safetensors_f32(dir / "model.safetensors",
        {{"dummy", {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}}});
}

static void test_index_key_deterministic()
{
    auto dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_det_XXXXXX");
    write_minimal_model_dir(dir, R"({"model_type":"test","hidden_size":64})");

    const std::string key1 = trtf::BuildModelDirIndexKey(dir.string(), 128);
    const std::string key2 = trtf::BuildModelDirIndexKey(dir.string(), 128);
    check(key1 == key2, "same model_dir + cache_length → same key");
    check(!key1.empty(), "key is non-empty");

    std::filesystem::remove_all(dir);
}

static void test_index_key_varies_with_cache_length()
{
    auto dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_cl_XXXXXX");
    write_minimal_model_dir(dir, R"({"model_type":"test","hidden_size":64})");

    const std::string key_128 = trtf::BuildModelDirIndexKey(dir.string(), 128);
    const std::string key_256 = trtf::BuildModelDirIndexKey(dir.string(), 256);
    check(key_128 != key_256, "different cache_length → different key");

    std::filesystem::remove_all(dir);
}

static void test_index_key_varies_with_config()
{
    auto dir1 = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_cfg1_XXXXXX");
    auto dir2 = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_cfg2_XXXXXX");
    write_minimal_model_dir(dir1, R"({"model_type":"test","hidden_size":64})");
    write_minimal_model_dir(dir2, R"({"model_type":"test","hidden_size":128})");

    const std::string key1 = trtf::BuildModelDirIndexKey(dir1.string(), 128);
    const std::string key2 = trtf::BuildModelDirIndexKey(dir2.string(), 128);
    check(key1 != key2, "different config.json content → different key");

    std::filesystem::remove_all(dir1);
    std::filesystem::remove_all(dir2);
}

static void test_save_and_lookup_roundtrip()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_rt_XXXXXX");
    auto model_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_rt_m_XXXXXX");
    write_minimal_model_dir(model_dir, R"({"model_type":"test","hidden_size":64})");

    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const std::string index_key = trtf::BuildModelDirIndexKey(model_dir.string(), 128);
    const std::string cache_key = "abcdef0123456789";

    // Create a dummy .plan file so LookupModelDirIndex finds it
    std::filesystem::create_directories(cache_dir);
    {
        std::ofstream out(cache_dir / (cache_key + ".plan"), std::ios::binary);
        out << "fake-plan-data";
    }

    trtf::SaveModelDirIndex(index_key, cache_key);
    const auto result = trtf::LookupModelDirIndex(index_key);
    check(result.has_value(), "lookup after save returns value");
    check(result.value_or("") == cache_key, "lookup returns same cache_key");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
    std::filesystem::remove_all(model_dir);
}

static void test_lookup_missing_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_miss_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const auto result = trtf::LookupModelDirIndex("nonexistent_index_key_12345");
    check(!result.has_value(), "no saved index → nullopt");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_lookup_stale_plan_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_stale_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const std::string index_key = "test_stale_index";
    const std::string cache_key = "stale_cache_key_999";

    // Create .plan file, save index, verify lookup works
    std::filesystem::create_directories(cache_dir);
    const auto plan_path = cache_dir / (cache_key + ".plan");
    {
        std::ofstream out(plan_path, std::ios::binary);
        out << "fake-plan-data";
    }
    trtf::SaveModelDirIndex(index_key, cache_key);
    check(trtf::LookupModelDirIndex(index_key).has_value(), "lookup works with plan present");

    // Delete the .plan file → lookup should return nullopt
    std::filesystem::remove(plan_path);
    const auto result = trtf::LookupModelDirIndex(index_key);
    check(!result.has_value(), "index exists but .plan deleted → nullopt");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_save_creates_directory()
{
    auto base_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_mkdir_XXXXXX");
    auto nested_cache = base_dir / "nested" / "cache" / "dir";
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", nested_cache.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    check(!std::filesystem::exists(nested_cache), "nested dir does not exist before save");
    trtf::SaveModelDirIndex("test_mkdir_key", "some_cache_key");
    check(std::filesystem::exists(nested_cache), "save auto-creates cache directory");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(base_dir);
}

static void test_cache_disabled_skips_save()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_dis_s_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", "1");

    trtf::SaveModelDirIndex("disabled_save_key", "disabled_cache_key");

    // No .idx file should have been created
    bool found_idx = false;
    for (const auto& entry : std::filesystem::directory_iterator(cache_dir))
    {
        if (entry.path().extension() == ".idx")
        {
            found_idx = true;
        }
    }
    check(!found_idx, "TRTF_DISABLE_ENGINE_CACHE=1 → save is no-op");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_cache_disabled_skips_lookup()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_dis_l_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());

    // First save with cache enabled
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);
    const std::string cache_key = "disabled_lookup_ck";
    std::filesystem::create_directories(cache_dir);
    {
        std::ofstream out(cache_dir / (cache_key + ".plan"), std::ios::binary);
        out << "fake-plan";
    }
    trtf::SaveModelDirIndex("disabled_lookup_key", cache_key);
    check(trtf::LookupModelDirIndex("disabled_lookup_key").has_value(), "lookup works when enabled");

    // Now disable cache → lookup should return nullopt
    set_env("TRTF_DISABLE_ENGINE_CACHE", "1");
    const auto result = trtf::LookupModelDirIndex("disabled_lookup_key");
    check(!result.has_value(), "TRTF_DISABLE_ENGINE_CACHE=1 → lookup returns nullopt");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_overwrite_existing_index()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_ow_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const std::string index_key = "overwrite_index";
    const std::string cache_key1 = "first_cache_key_111";
    const std::string cache_key2 = "second_cache_key_222";

    std::filesystem::create_directories(cache_dir);
    // Create plan files for both cache keys
    {
        std::ofstream out(cache_dir / (cache_key1 + ".plan"), std::ios::binary);
        out << "plan1";
    }
    {
        std::ofstream out(cache_dir / (cache_key2 + ".plan"), std::ios::binary);
        out << "plan2";
    }

    trtf::SaveModelDirIndex(index_key, cache_key1);
    check(trtf::LookupModelDirIndex(index_key).value_or("") == cache_key1, "first save → first key");

    trtf::SaveModelDirIndex(index_key, cache_key2);
    check(trtf::LookupModelDirIndex(index_key).value_or("") == cache_key2, "second save overwrites → second key");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

int main()
{
    test_index_key_deterministic();
    test_index_key_varies_with_cache_length();
    test_index_key_varies_with_config();
    test_save_and_lookup_roundtrip();
    test_lookup_missing_returns_nullopt();
    test_lookup_stale_plan_returns_nullopt();
    test_save_creates_directory();
    test_cache_disabled_skips_save();
    test_cache_disabled_skips_lookup();
    test_overwrite_existing_index();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All engine cache index tests passed.\n";
    return 0;
}
