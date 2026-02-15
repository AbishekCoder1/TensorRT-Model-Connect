// =============================================================================
// Test suite: TRT engine plan cache I/O -- SaveTrtEnginePlanToCache and
//   LoadTrtEnginePlanFromCache.
//
// Purpose:
//   Validates the on-disk engine plan caching system that stores serialized
//   TensorRT engine plans to avoid expensive rebuilds across process restarts.
//   Tests cover save/load roundtrips, missing file handling, empty file
//   rejection, large payload integrity, cache-disabled behavior (both save
//   and load), and RAII config guard semantics.
//
// Dependencies:
//   - utils/trt/engine_cache.h: SaveTrtEnginePlanToCache,
//     LoadTrtEnginePlanFromCache, EngineCacheConfig, SetThreadEngineCacheConfig,
//     ClearThreadEngineCacheConfig.
//   - test_helpers.h: make_temp_dir_or_throw utility.
//   - Filesystem access (temp directories).
//   - No TRT or GPU required -- all plan data is synthetic byte strings.
//
// Approach:
//   Each test creates a fresh temp directory and uses the CacheConfigGuard
//   RAII helper to set the thread-local engine cache configuration (cache_dir
//   and no_cache flag). The guard automatically clears the config on
//   destruction. Tests use synthetic "plan data" strings to exercise the
//   save/load API without needing actual TRT engine artifacts.
//
// Key helper:
//   CacheConfigGuard -- RAII struct that calls SetThreadEngineCacheConfig on
//   construction and ClearThreadEngineCacheConfig on destruction, ensuring
//   test isolation.
//
// Test categories:
//   - Roundtrip: save then load, verify byte-for-byte equality
//   - Missing data: load nonexistent key -> nullopt
//   - Empty file: manually created 0-byte .plan file -> nullopt
//   - Large payload: 10 MB roundtrip with pattern verification
//   - Cache disabled: save is no-op when no_cache=true; load returns nullopt
//   - RAII semantics: guard cleanup, nested guard override behavior
// =============================================================================

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

// -----------------------------------------------------------------------------
// Intention: Verify that saving a plan and immediately loading it back returns
//   identical data (byte-for-byte roundtrip).
// Setup: Creates a temp directory, configures the cache to use it via
//   CacheConfigGuard(dir, no_cache=false). Uses "test_io_roundtrip_key" as
//   the cache key and a short synthetic string as plan data.
// Mechanism: Calls SaveTrtEnginePlanToCache to write the plan, then
//   LoadTrtEnginePlanFromCache to read it back. Asserts the returned optional
//   has a value, the size matches, and memcmp confirms byte equality.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that loading a cache key that was never saved returns
//   std::nullopt (not a crash or empty vector).
// Setup: Creates a temp directory with caching enabled. Does not save anything.
// Mechanism: Calls LoadTrtEnginePlanFromCache with a nonexistent key, asserts
//   the result does not have a value.
// -----------------------------------------------------------------------------
static void test_load_nonexistent_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_io_ne_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

    const auto result = trtf::LoadTrtEnginePlanFromCache("nonexistent_key_xyz");
    check(!result.has_value(), "missing file -> nullopt");

    std::filesystem::remove_all(cache_dir);
}

// -----------------------------------------------------------------------------
// Intention: Verify that a manually created 0-byte .plan file is treated as
//   invalid and returns nullopt (not an empty vector), since a valid TRT
//   engine plan always has content.
// Setup: Creates a temp directory with caching enabled. Manually creates an
//   empty file named "<key>.plan" in the cache directory.
// Mechanism: Calls LoadTrtEnginePlanFromCache with the key matching the empty
//   file. Asserts the result does not have a value.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that the cache correctly handles large payloads (10 MB),
//   ensuring no truncation or corruption occurs during the write/read cycle.
// Setup: Creates a 10 MB buffer filled with a repeating pattern (i % 251,
//   using a prime modulus for byte diversity). Configures cache and saves.
// Mechanism: Saves the 10 MB buffer via SaveTrtEnginePlanToCache, loads it
//   back, asserts the size matches and memcmp confirms full byte equality.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that SaveTrtEnginePlanToCache is a no-op when the cache is
//   disabled (no_cache=true), meaning no .plan file is created on disk.
// Setup: Creates a temp directory, configures cache with no_cache=true.
//   Attempts to save data.
// Mechanism: After calling SaveTrtEnginePlanToCache, iterates the cache
//   directory and checks that no .plan files were created.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that LoadTrtEnginePlanFromCache returns nullopt when the
//   cache is disabled (no_cache=true), even if a valid plan file exists on
//   disk from a prior save.
// Setup: First saves a plan with caching enabled (guard with no_cache=false).
//   Verifies the load succeeds. Then creates a second guard with no_cache=true
//   pointing to the same directory.
// Mechanism: Loads the same key with caching disabled. Asserts the result is
//   nullopt despite the file being physically present on disk.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify that CacheConfigGuard properly clears the thread-local
//   engine cache configuration upon destruction, so that subsequent operations
//   fall back to the default cache behavior (typically HOME-based directory).
// Setup: Creates a temp directory and a CacheConfigGuard in an inner scope.
//   Saves and loads data inside the guard's scope.
// Mechanism: After the guard is destroyed (scope exit), the thread-local config
//   is cleared. The test asserts the save/load works inside the guard's scope.
//   After destruction, the config no longer points to the temp directory. (We
//   do not assert post-destruction behavior since the fallback depends on HOME.)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention: Verify the behavior of nested CacheConfigGuard instances, where
//   an inner guard overrides the outer guard's cache directory, and destruction
//   of the inner guard clears the config (thread-local is cleared, not stacked).
// Setup: Creates two temp directories (cache_dir1 and cache_dir2). Outer guard
//   points to dir1, inner guard points to dir2. Saves a key in each directory.
// Mechanism:
//   1. In outer guard scope: saves "nest_key1" to dir1, verifies load succeeds.
//   2. In inner guard scope: verifies dir1's key is NOT found in dir2 (isolation).
//      Saves "nest_key2" to dir2, verifies load succeeds.
//   3. Inner guard destroyed: thread-local config is cleared (not restored to
//      outer guard's value, since the config is not stacked).
// -----------------------------------------------------------------------------
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
