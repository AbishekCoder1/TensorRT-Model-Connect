// Test: SaveTrtEnginePlanToCache and LoadTrtEnginePlanFromCache from engine_cache.cpp.
// CPU-only tests using temp dirs with synthetic plan data.

#include "utils/trt/engine_cache.h"
#include "test_helpers.h"

#include <cstdlib>
#include <cstring>
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

static void test_save_and_load_roundtrip()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_rt_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const std::string cache_key = "test_io_roundtrip_key";
    const std::string plan_data = "This is synthetic TRT engine plan data for testing.";

    trtf::SaveTrtEnginePlanToCache(cache_key, plan_data.data(), plan_data.size());
    const auto loaded = trtf::LoadTrtEnginePlanFromCache(cache_key);

    check(loaded.has_value(), "load after save returns value");
    if (loaded.has_value())
    {
        check(loaded->size() == plan_data.size(), "loaded size matches saved size");
        check(std::memcmp(loaded->data(), plan_data.data(), plan_data.size()) == 0,
            "loaded data matches saved data byte-for-byte");
    }

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_load_nonexistent_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_ne_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const auto result = trtf::LoadTrtEnginePlanFromCache("nonexistent_key_xyz");
    check(!result.has_value(), "missing file → nullopt");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_load_empty_file_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_empty_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const std::string cache_key = "empty_plan_key";
    // Create an empty .plan file directly
    std::filesystem::create_directories(cache_dir);
    {
        std::ofstream out(cache_dir / (cache_key + ".plan"), std::ios::binary | std::ios::trunc);
        // Write nothing — 0-byte file
    }

    const auto result = trtf::LoadTrtEnginePlanFromCache(cache_key);
    check(!result.has_value(), "0-byte file → nullopt");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_load_large_file()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_large_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);

    const std::string cache_key = "large_plan_key";
    constexpr std::size_t kSize = 10 * 1024 * 1024; // 10 MB
    std::vector<char> large_data(kSize);
    // Fill with a pattern
    for (std::size_t i = 0; i < kSize; ++i)
    {
        large_data[i] = static_cast<char>(i % 251); // prime modulus for diversity
    }

    trtf::SaveTrtEnginePlanToCache(cache_key, large_data.data(), large_data.size());
    const auto loaded = trtf::LoadTrtEnginePlanFromCache(cache_key);

    check(loaded.has_value(), "10MB plan loads successfully");
    if (loaded.has_value())
    {
        check(loaded->size() == kSize, "loaded size matches 10MB");
        check(std::memcmp(loaded->data(), large_data.data(), kSize) == 0,
            "10MB loaded data matches byte-for-byte");
    }

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_cache_disabled_save_noop()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_dis_s_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", "1");

    const std::string cache_key = "disabled_save_plan_key";
    const std::string data = "should not be saved";
    trtf::SaveTrtEnginePlanToCache(cache_key, data.data(), data.size());

    // No .plan file should exist
    bool found_plan = false;
    if (std::filesystem::exists(cache_dir))
    {
        for (const auto& entry : std::filesystem::directory_iterator(cache_dir))
        {
            if (entry.path().extension() == ".plan")
            {
                found_plan = true;
            }
        }
    }
    check(!found_plan, "TRTF_DISABLE_ENGINE_CACHE=1 → save does nothing");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

static void test_cache_disabled_load_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_dis_l_XXXXXX");
    const std::string old_cache = set_env("TRTF_TRT_ENGINE_CACHE_DIR", cache_dir.c_str());

    // Save with cache enabled
    const std::string old_disable = set_env("TRTF_DISABLE_ENGINE_CACHE", nullptr);
    const std::string cache_key = "disabled_load_plan_key";
    const std::string data = "plan data present";
    trtf::SaveTrtEnginePlanToCache(cache_key, data.data(), data.size());
    check(trtf::LoadTrtEnginePlanFromCache(cache_key).has_value(), "load works when enabled");

    // Disable → load returns nullopt
    set_env("TRTF_DISABLE_ENGINE_CACHE", "1");
    const auto result = trtf::LoadTrtEnginePlanFromCache(cache_key);
    check(!result.has_value(), "TRTF_DISABLE_ENGINE_CACHE=1 → load returns nullopt");

    restore_env("TRTF_TRT_ENGINE_CACHE_DIR", old_cache);
    restore_env("TRTF_DISABLE_ENGINE_CACHE", old_disable);
    std::filesystem::remove_all(cache_dir);
}

int main()
{
    test_save_and_load_roundtrip();
    test_load_nonexistent_returns_nullopt();
    test_load_empty_file_returns_nullopt();
    test_load_large_file();
    test_cache_disabled_save_noop();
    test_cache_disabled_load_nullopt();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All engine cache I/O tests passed.\n";
    return 0;
}
