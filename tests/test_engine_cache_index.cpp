// =============================================================================
// Test suite: Engine cache index operations
// =============================================================================
//
// Purpose:
//   Validates the model-directory index layer of the TRT engine plan cache.
//   The index maps a composite key (derived from model directory content +
//   cache length) to a cache key that identifies a serialized .plan file on
//   disk. This test suite exercises BuildModelDirIndexKey, SaveModelDirIndex,
//   and LookupModelDirIndex from engine_cache.cpp.
//
// Dependencies:
//   - utils/trt/engine_cache.h  (BuildModelDirIndexKey, SaveModelDirIndex,
//                                 LookupModelDirIndex, EngineCacheConfig,
//                                 SetThreadEngineCacheConfig, ClearThreadEngineCacheConfig)
//   - test_helpers.h            (make_temp_dir_or_throw, write_file,
//                                 write_safetensors_f32)
//
// Approach:
//   CPU-only tests using temporary directories with synthetic config.json and
//   safetensors fixtures. Each test creates its own isolated temp dir(s),
//   exercises the index API, and cleans up after itself. The RAII
//   CacheConfigGuard sets/clears thread-local engine cache configuration so
//   that save/lookup operations target the test's temp cache directory.
//
// Environment:
//   Requires writable /tmp (uses mkdtemp). May fail in read-only sandboxes.
//   No GPU or TensorRT runtime required.
// =============================================================================

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

// Create a minimal model dir with config.json and model.safetensors
static void write_minimal_model_dir(const std::filesystem::path& dir, const std::string& config_json)
{
    trtf_test::write_file(dir / "config.json", config_json);
    // Write a tiny safetensors file (just one tensor)
    trtf_test::write_safetensors_f32(dir / "model.safetensors",
        {{"dummy", {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}}});
}

// -----------------------------------------------------------------------------
// Intention:  Verify that BuildModelDirIndexKey is deterministic — calling it
//             twice with the same model directory and cache length produces the
//             same non-empty key string.
// Setup:      A temp directory containing a minimal config.json and safetensors.
// Mechanism:  Calls BuildModelDirIndexKey twice with identical arguments and
//             asserts that both returned keys are equal and non-empty.
// -----------------------------------------------------------------------------
static void test_index_key_deterministic()
{
    auto dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_det_XXXXXX");
    write_minimal_model_dir(dir, R"({"model_type":"test","hidden_size":64})");

    const std::string key1 = trtf::BuildModelDirIndexKey(dir.string(), 128);
    const std::string key2 = trtf::BuildModelDirIndexKey(dir.string(), 128);
    check(key1 == key2, "same model_dir + cache_length -> same key");
    check(!key1.empty(), "key is non-empty");

    std::filesystem::remove_all(dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that changing the cache_length parameter produces a
//             different index key, even when the model directory is the same.
// Setup:      A single temp model directory with fixed config.json content.
// Mechanism:  Calls BuildModelDirIndexKey with cache_length=128 and 256 on the
//             same directory, then asserts the two keys differ.
// -----------------------------------------------------------------------------
static void test_index_key_varies_with_cache_length()
{
    auto dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_cl_XXXXXX");
    write_minimal_model_dir(dir, R"({"model_type":"test","hidden_size":64})");

    const std::string key_128 = trtf::BuildModelDirIndexKey(dir.string(), 128);
    const std::string key_256 = trtf::BuildModelDirIndexKey(dir.string(), 256);
    check(key_128 != key_256, "different cache_length -> different key");

    std::filesystem::remove_all(dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that different config.json content (different model
//             configurations) produces different index keys, even when the
//             cache length is the same.
// Setup:      Two separate temp model directories with different hidden_size
//             values in their config.json files.
// Mechanism:  Calls BuildModelDirIndexKey on both directories with the same
//             cache_length and asserts the resulting keys differ.
// -----------------------------------------------------------------------------
static void test_index_key_varies_with_config()
{
    auto dir1 = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_cfg1_XXXXXX");
    auto dir2 = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_cfg2_XXXXXX");
    write_minimal_model_dir(dir1, R"({"model_type":"test","hidden_size":64})");
    write_minimal_model_dir(dir2, R"({"model_type":"test","hidden_size":128})");

    const std::string key1 = trtf::BuildModelDirIndexKey(dir1.string(), 128);
    const std::string key2 = trtf::BuildModelDirIndexKey(dir2.string(), 128);
    check(key1 != key2, "different config.json content -> different key");

    std::filesystem::remove_all(dir1);
    std::filesystem::remove_all(dir2);
}

// -----------------------------------------------------------------------------
// Intention:  Verify the full save-then-lookup round-trip: an index entry saved
//             via SaveModelDirIndex can be retrieved via LookupModelDirIndex
//             with the correct cache key.
// Setup:      A temp cache directory with a dummy .plan file and a temp model
//             directory. CacheConfigGuard points the thread-local config at the
//             cache directory with caching enabled.
// Mechanism:  Builds an index key, saves a mapping from that key to a cache key,
//             then looks up the index key and asserts the returned value matches
//             the originally saved cache key.
// -----------------------------------------------------------------------------
static void test_save_and_lookup_roundtrip()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_rt_XXXXXX");
    auto model_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_rt_m_XXXXXX");
    write_minimal_model_dir(model_dir, R"({"model_type":"test","hidden_size":64})");

    CacheConfigGuard guard(cache_dir.string(), false);

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

    std::filesystem::remove_all(cache_dir);
    std::filesystem::remove_all(model_dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that looking up an index key that was never saved returns
//             std::nullopt rather than crashing or returning stale data.
// Setup:      An empty temp cache directory with caching enabled.
// Mechanism:  Calls LookupModelDirIndex with a fabricated key that has no
//             corresponding .idx file on disk, and asserts the result has no
//             value.
// -----------------------------------------------------------------------------
static void test_lookup_missing_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_miss_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

    const auto result = trtf::LookupModelDirIndex("nonexistent_index_key_12345");
    check(!result.has_value(), "no saved index -> nullopt");

    std::filesystem::remove_all(cache_dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that if an index entry exists but the corresponding .plan
//             file has been deleted from disk, lookup returns std::nullopt
//             (stale index detection).
// Setup:      A temp cache directory with a dummy .plan file and a saved index
//             entry pointing to it. After confirming lookup works, the .plan
//             file is deleted.
// Mechanism:  Saves an index entry, confirms lookup succeeds, deletes the .plan
//             file, then re-runs lookup and asserts it returns nullopt.
// -----------------------------------------------------------------------------
static void test_lookup_stale_plan_returns_nullopt()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_stale_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

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

    // Delete the .plan file -> lookup should return nullopt
    std::filesystem::remove(plan_path);
    const auto result = trtf::LookupModelDirIndex(index_key);
    check(!result.has_value(), "index exists but .plan deleted -> nullopt");

    std::filesystem::remove_all(cache_dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that SaveModelDirIndex auto-creates the cache directory
//             (including nested intermediate directories) if it does not exist.
// Setup:      A base temp directory with a deeply nested cache path that does
//             not yet exist on disk.
// Mechanism:  Asserts the nested directory does not exist, calls
//             SaveModelDirIndex, then asserts the directory was created.
// -----------------------------------------------------------------------------
static void test_save_creates_directory()
{
    auto base_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_mkdir_XXXXXX");
    auto nested_cache = base_dir / "nested" / "cache" / "dir";
    CacheConfigGuard guard(nested_cache.string(), false);

    check(!std::filesystem::exists(nested_cache), "nested dir does not exist before save");
    trtf::SaveModelDirIndex("test_mkdir_key", "some_cache_key");
    check(std::filesystem::exists(nested_cache), "save auto-creates cache directory");

    std::filesystem::remove_all(base_dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that SaveModelDirIndex is a no-op when caching is disabled
//             (no_cache=true) — no .idx file should be written.
// Setup:      A temp cache directory with caching disabled via CacheConfigGuard.
// Mechanism:  Calls SaveModelDirIndex, then iterates the cache directory
//             looking for any .idx files. Asserts none were created.
// -----------------------------------------------------------------------------
static void test_cache_disabled_skips_save()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_dis_s_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), true);

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
    check(!found_idx, "no_cache=true -> save is no-op");

    std::filesystem::remove_all(cache_dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that LookupModelDirIndex returns nullopt when caching is
//             disabled, even if a valid index entry was previously saved while
//             caching was enabled.
// Setup:      A temp cache directory. First, an index entry is saved with
//             caching enabled and a corresponding .plan file. Then caching is
//             disabled via a new CacheConfigGuard.
// Mechanism:  Confirms lookup succeeds with caching enabled, then switches to
//             no_cache=true and asserts lookup returns nullopt.
// -----------------------------------------------------------------------------
static void test_cache_disabled_skips_lookup()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_dis_l_XXXXXX");

    // First save with cache enabled
    {
        CacheConfigGuard guard(cache_dir.string(), false);
        const std::string cache_key = "disabled_lookup_ck";
        std::filesystem::create_directories(cache_dir);
        {
            std::ofstream out(cache_dir / (cache_key + ".plan"), std::ios::binary);
            out << "fake-plan";
        }
        trtf::SaveModelDirIndex("disabled_lookup_key", cache_key);
        check(trtf::LookupModelDirIndex("disabled_lookup_key").has_value(), "lookup works when enabled");
    }

    // Now disable cache -> lookup should return nullopt
    {
        CacheConfigGuard guard(cache_dir.string(), true);
        const auto result = trtf::LookupModelDirIndex("disabled_lookup_key");
        check(!result.has_value(), "no_cache=true -> lookup returns nullopt");
    }

    std::filesystem::remove_all(cache_dir);
}

// -----------------------------------------------------------------------------
// Intention:  Verify that saving a new cache key for an existing index key
//             overwrites the previous mapping — the latest save wins.
// Setup:      A temp cache directory with two distinct .plan files and one
//             index key.
// Mechanism:  Saves the index key pointing to cache_key1, asserts lookup
//             returns cache_key1. Then saves the same index key pointing to
//             cache_key2, and asserts lookup now returns cache_key2.
// -----------------------------------------------------------------------------
static void test_overwrite_existing_index()
{
    auto cache_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_test_idx_ow_XXXXXX");
    CacheConfigGuard guard(cache_dir.string(), false);

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
    check(trtf::LookupModelDirIndex(index_key).value_or("") == cache_key1, "first save -> first key");

    trtf::SaveModelDirIndex(index_key, cache_key2);
    check(trtf::LookupModelDirIndex(index_key).value_or("") == cache_key2, "second save overwrites -> second key");

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
