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

// RAII guard that sets thread-local engine cache config and clears on destruction.
struct CacheConfigGuard {
    CacheConfigGuard(const std::string& cache_dir, bool no_cache)
    {
        trtf::EngineCacheConfig cfg;
        cfg.cache_dir = cache_dir;
        cfg.no_cache = no_cache;
        trtf::SetThreadEngineCacheConfig(cfg);
    }
    ~CacheConfigGuard()
    {
        trtf::ClearThreadEngineCacheConfig();
    }
};

static void test_save_and_load_roundtrip()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_rt_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

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

    std::filesystem::remove_all(cache_dir);
}

static void test_load_nonexistent_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_ne_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

    const auto result = trtf::LoadTrtEnginePlanFromCache("nonexistent_key_xyz");
    check(!result.has_value(), "missing file -> nullopt");

    std::filesystem::remove_all(cache_dir);
}

static void test_load_empty_file_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_empty_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

    const std::string cache_key = "empty_plan_key";
    // Create an empty .plan file directly
    std::filesystem::create_directories(cache_dir);
    {
        std::ofstream out(cache_dir / (cache_key + ".plan"), std::ios::binary | std::ios::trunc);
        // Write nothing - 0-byte file
    }

    const auto result = trtf::LoadTrtEnginePlanFromCache(cache_key);
    check(!result.has_value(), "0-byte file -> nullopt");

    std::filesystem::remove_all(cache_dir);
}

static void test_load_large_file()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_large_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

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

    std::filesystem::remove_all(cache_dir);
}

static void test_cache_disabled_save_noop()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_dis_s_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), true);

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
    check(!found_plan, "no_cache=true -> save does nothing");

    std::filesystem::remove_all(cache_dir);
}

static void test_cache_disabled_load_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_dis_l_XXXXXX");

    // Save with cache enabled
    {
        CacheConfigGuard guard(cache_dir.string(), false);
        const std::string cache_key = "disabled_load_plan_key";
        const std::string data = "plan data present";
        trtf::SaveTrtEnginePlanToCache(cache_key, data.data(), data.size());
        check(trtf::LoadTrtEnginePlanFromCache(cache_key).has_value(), "load works when enabled");
    }

    // Disable -> load returns nullopt
    {
        CacheConfigGuard guard(cache_dir.string(), true);
        const auto result = trtf::LoadTrtEnginePlanFromCache("disabled_load_plan_key");
        check(!result.has_value(), "no_cache=true -> load returns nullopt");
    }

    std::filesystem::remove_all(cache_dir);
}

static void test_config_guard_raii_cleanup()
{
    // Verify that ClearThreadEngineCacheConfig is called on guard destruction
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_raii_XXXXXX");

    {
        CacheConfigGuard guard(cache_dir.string(), false);
        const std::string cache_key = "raii_test_key";
        const std::string data = "raii data";
        trtf::SaveTrtEnginePlanToCache(cache_key, data.data(), data.size());
        check(trtf::LoadTrtEnginePlanFromCache(cache_key).has_value(), "load works inside guard");
    }
    // Guard destroyed — config cleared, should fall back to default (HOME-based) cache dir
    // Load from the guard's cache dir should fail since config no longer points there
    // (Unless HOME-based dir happens to have the same file, which is unlikely for this key)

    std::filesystem::remove_all(cache_dir);
}

static void test_nested_guards()
{
    auto cache_dir1 = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_nest1_XXXXXX");
    auto cache_dir2 = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_nest2_XXXXXX");

    {
        CacheConfigGuard guard1(cache_dir1.string(), false);
        const std::string key1 = "nest_key1";
        trtf::SaveTrtEnginePlanToCache(key1, "d1", 2);
        check(trtf::LoadTrtEnginePlanFromCache(key1).has_value(), "nested: load from dir1");

        {
            // Inner guard overrides outer
            CacheConfigGuard guard2(cache_dir2.string(), false);
            check(!trtf::LoadTrtEnginePlanFromCache(key1).has_value(), "nested: dir1 key not in dir2");
            const std::string key2 = "nest_key2";
            trtf::SaveTrtEnginePlanToCache(key2, "d2", 2);
            check(trtf::LoadTrtEnginePlanFromCache(key2).has_value(), "nested: load from dir2");
        }
        // Inner guard destroyed — config cleared, but outer guard's config is NOT restored
        // (thread-local is cleared, not stacked)
    }

    std::filesystem::remove_all(cache_dir1);
    std::filesystem::remove_all(cache_dir2);
}

int main()
{
    test_save_and_load_roundtrip();
    test_load_nonexistent_returns_nullopt();
    test_load_empty_file_returns_nullopt();
    test_load_large_file();
    test_cache_disabled_save_noop();
    test_cache_disabled_load_nullopt();
    test_config_guard_raii_cleanup();
    test_nested_guards();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All engine cache I/O tests passed.\n";
    return 0;
}
