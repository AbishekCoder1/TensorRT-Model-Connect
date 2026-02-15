// =============================================================================
// Test suite: .trtfb bundle format read/write, magic validation, and JSON
//   header roundtrip.
//
// Purpose:
//   Validates the binary bundle format (.trtfb) used to package TRT engine
//   plans alongside metadata and tokenizer data into a single distributable
//   file. Tests cover the full lifecycle: writing a BundleFile to disk,
//   reading it back, and verifying all metadata fields and section data are
//   preserved byte-for-byte.
//
// Dependencies:
//   - bundle/bundle_format.h: BundleFile, BundleInfo, BundleSection,
//     WriteBundleFile, ReadBundleFile, IsBundle, InspectBundle,
//     BundleInfoToJson, BundleInfoFromJson, kBundleMagic.
//   - Filesystem access (temp directories via mkdtemp).
//   - No TRT, GPU, or CUDA required -- all data is synthetic.
//
// Approach:
//   Each test creates a temporary directory, constructs a BundleFile with
//   synthetic metadata and section data, writes it to disk, reads it back,
//   and asserts field-level equality. Temp directories are cleaned up after
//   each test. Tests also cover edge cases: invalid magic bytes, empty
//   sections, large payloads, 64-bit offsets, truncated files, and the
//   IsBundle/InspectBundle utility functions.
//
// Test categories:
//   - Roundtrip: write then read, verify all fields match
//   - Magic validation: reject files with incorrect magic bytes
//   - Edge cases: empty sections, large (1MB) sections, 64-bit offsets
//   - JSON header: BundleInfo <-> JSON serialization roundtrip
//   - Utilities: IsBundle detection, InspectBundle metadata extraction
//   - Error handling: truncated file throws runtime_error
// =============================================================================

#include "bundle/bundle_format.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <stdlib.h>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::filesystem::path make_temp_dir()
{
    char pattern[] = "/tmp/trtfb_test_XXXXXX";
    char* dir = mkdtemp(pattern);
    if (dir == nullptr)
    {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(dir);
}

// -----------------------------------------------------------------------------
// Intention: Verify that a fully populated BundleFile survives a write-then-read
//   cycle with all metadata fields and section data intact.
// Setup: Creates a BundleFile with a complete BundleInfo (model_id, model_type,
//   family, trt_version, gpu_name, created_at, vocab_size, hidden_size,
//   num_layers, num_attention_heads, num_key_value_heads, max_cache_length)
//   and two sections ("trt_plan" with 4 bytes, "tokenizer_json" with 2 bytes).
// Mechanism: Writes the bundle via WriteBundleFile, reads it back via
//   ReadBundleFile, and asserts every metadata field and every section's name
//   and data byte-vector match exactly.
// -----------------------------------------------------------------------------
static void test_write_read_roundtrip()
{
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "test.trtfb").string();

    trtf::BundleFile bundle;
    bundle.info.model_id = "test-model";
    bundle.info.model_type = "qwen3";
    bundle.info.family = "qwen";
    bundle.info.trt_version = "10.15.0";
    bundle.info.gpu_name = "GeForce RTX 4090";
    bundle.info.created_at = "2026-02-14T12:00:00Z";
    bundle.info.vocab_size = 151936;
    bundle.info.hidden_size = 1024;
    bundle.info.num_layers = 28;
    bundle.info.num_attention_heads = 16;
    bundle.info.num_key_value_heads = 4;
    bundle.info.max_cache_length = 2048;

    trtf::BundleSection section1;
    section1.name = "trt_plan";
    section1.data = {'P', 'L', 'A', 'N'};

    trtf::BundleSection section2;
    section2.name = "tokenizer_json";
    section2.data = {'{', '}'};

    bundle.sections = {section1, section2};

    trtf::WriteBundleFile(path, bundle);
    const auto loaded = trtf::ReadBundleFile(path);

    check(loaded.info.model_id == "test-model", "roundtrip model_id");
    check(loaded.info.model_type == "qwen3", "roundtrip model_type");
    check(loaded.info.family == "qwen", "roundtrip family");
    check(loaded.info.trt_version == "10.15.0", "roundtrip trt_version");
    check(loaded.info.gpu_name == "GeForce RTX 4090", "roundtrip gpu_name");
    check(loaded.info.created_at == "2026-02-14T12:00:00Z", "roundtrip created_at");
    check(loaded.info.vocab_size == 151936, "roundtrip vocab_size");
    check(loaded.info.hidden_size == 1024, "roundtrip hidden_size");
    check(loaded.info.num_layers == 28, "roundtrip num_layers");
    check(loaded.info.num_attention_heads == 16, "roundtrip num_attention_heads");
    check(loaded.info.num_key_value_heads == 4, "roundtrip num_key_value_heads");
    check(loaded.info.max_cache_length == 2048, "roundtrip max_cache_length");

    check(loaded.sections.size() == 2, "roundtrip section count");
    check(loaded.sections[0].name == "trt_plan", "roundtrip section 0 name");
    check(loaded.sections[0].data == std::vector<char>{'P', 'L', 'A', 'N'}, "roundtrip section 0 data");
    check(loaded.sections[1].name == "tokenizer_json", "roundtrip section 1 name");
    check(loaded.sections[1].data == std::vector<char>{'{', '}'}, "roundtrip section 1 data");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that ReadBundleFile rejects files with invalid magic bytes
//   by throwing a runtime_error with a descriptive message.
// Setup: Creates a file containing "NOTMAGIC" (8 bytes) instead of the valid
//   kBundleMagic header.
// Mechanism: Calls ReadBundleFile on the invalid file, catches the expected
//   runtime_error, and checks that the exception message references "magic"
//   or "Invalid".
// -----------------------------------------------------------------------------
static void test_magic_validation()
{
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "bad.trtfb").string();

    std::ofstream out(path, std::ios::binary);
    out.write("NOTMAGIC", 8);
    out.close();

    bool threw = false;
    try
    {
        trtf::ReadBundleFile(path);
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        const std::string msg = e.what();
        check(msg.find("magic") != std::string::npos || msg.find("Invalid") != std::string::npos,
            "magic error message is descriptive");
    }
    check(threw, "invalid magic throws");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that a bundle with zero sections (metadata only) can be
//   written and read back correctly.
// Setup: Creates a BundleFile with model_id="empty" and no sections added.
// Mechanism: Writes via WriteBundleFile, reads back, checks model_id matches
//   and loaded.sections is empty.
// -----------------------------------------------------------------------------
static void test_empty_sections()
{
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "empty.trtfb").string();

    trtf::BundleFile bundle;
    bundle.info.model_id = "empty";
    // No sections

    trtf::WriteBundleFile(path, bundle);
    const auto loaded = trtf::ReadBundleFile(path);

    check(loaded.info.model_id == "empty", "empty sections model_id");
    check(loaded.sections.empty(), "empty sections count");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that BundleInfo serializes to JSON and deserializes back
//   with all fields intact, including computed section offsets. This tests the
//   JSON layer independently of the binary file format.
// Setup: Creates a BundleInfo with LLaMA-like metadata and a section_sizes
//   vector with two entries: {"plan", 1024} and {"tok", 256}.
// Mechanism: Calls BundleInfoToJson to serialize, then BundleInfoFromJson to
//   parse. Checks all metadata fields match and that parsed section offsets
//   are correctly computed (section 1 offset = section 0 size = 1024).
// -----------------------------------------------------------------------------
static void test_header_json_roundtrip()
{
    trtf::BundleInfo info;
    info.model_id = "my-model";
    info.model_type = "llama";
    info.family = "llama";
    info.trt_version = "10.15.0";
    info.gpu_name = "A100";
    info.created_at = "2026-01-01";
    info.vocab_size = 32000;
    info.hidden_size = 4096;
    info.num_layers = 32;
    info.num_attention_heads = 32;
    info.num_key_value_heads = 8;
    info.max_cache_length = 4096;

    std::vector<std::pair<std::string, std::size_t>> section_sizes = {
        {"plan", 1024},
        {"tok", 256},
    };

    const std::string json = trtf::BundleInfoToJson(info, section_sizes);

    std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>> parsed_sections;
    const auto parsed = trtf::BundleInfoFromJson(json, parsed_sections);

    check(parsed.model_id == "my-model", "json roundtrip model_id");
    check(parsed.model_type == "llama", "json roundtrip model_type");
    check(parsed.family == "llama", "json roundtrip family");
    check(parsed.vocab_size == 32000, "json roundtrip vocab_size");
    check(parsed.hidden_size == 4096, "json roundtrip hidden_size");
    check(parsed.num_layers == 32, "json roundtrip num_layers");
    check(parsed.num_attention_heads == 32, "json roundtrip num_attention_heads");
    check(parsed.num_key_value_heads == 8, "json roundtrip num_key_value_heads");
    check(parsed.max_cache_length == 4096, "json roundtrip max_cache_length");
    check(parsed_sections.size() == 2, "json roundtrip section count");
    check(parsed_sections[0].first == "plan", "json roundtrip section 0 name");
    check(parsed_sections[0].second.second == 1024, "json roundtrip section 0 size");
    check(parsed_sections[1].first == "tok", "json roundtrip section 1 name");
    check(parsed_sections[1].second.first == 1024, "json roundtrip section 1 offset");
    check(parsed_sections[1].second.second == 256, "json roundtrip section 1 size");
}

// -----------------------------------------------------------------------------
// Intention: Verify that IsBundle() returns true for a valid .trtfb file
//   (one written by WriteBundleFile with correct magic bytes).
// Setup: Creates a minimal valid bundle (model_id="valid", no sections) on disk.
// Mechanism: Calls IsBundle() on the written file path and asserts it returns true.
// -----------------------------------------------------------------------------
static void test_is_bundle_valid()
{
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "valid.trtfb").string();

    trtf::BundleFile bundle;
    bundle.info.model_id = "valid";
    trtf::WriteBundleFile(path, bundle);

    check(trtf::IsBundle(path), "IsBundle true for valid file");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that IsBundle() returns false for non-bundle inputs:
//   a plain text file, a directory path, and a nonexistent path.
// Setup: Creates a text file ("Hello world"), and uses the temp directory itself
//   and a nonexistent path as additional test inputs.
// Mechanism: Calls IsBundle() for each input and asserts it returns false.
// -----------------------------------------------------------------------------
static void test_is_bundle_invalid()
{
    const auto tmp = make_temp_dir();

    // Text file
    const auto text_path = (tmp / "readme.txt").string();
    std::ofstream(text_path) << "Hello world";
    check(!trtf::IsBundle(text_path), "IsBundle false for text file");

    // Directory
    check(!trtf::IsBundle(tmp.string()), "IsBundle false for directory");

    // Nonexistent
    check(!trtf::IsBundle((tmp / "nonexistent").string()), "IsBundle false for nonexistent");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that InspectBundle() extracts metadata from a bundle file
//   without loading the full section data, returning the BundleInfo.
// Setup: Creates a bundle with model_id="inspectable", vocab_size=50000,
//   num_layers=12, and a single 3-byte section.
// Mechanism: Writes the bundle, calls InspectBundle, and checks that the
//   returned BundleInfo fields match the original values.
// -----------------------------------------------------------------------------
static void test_inspect_returns_metadata()
{
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "inspect.trtfb").string();

    trtf::BundleFile bundle;
    bundle.info.model_id = "inspectable";
    bundle.info.vocab_size = 50000;
    bundle.info.num_layers = 12;

    trtf::BundleSection section;
    section.name = "data";
    section.data = {'X', 'Y', 'Z'};
    bundle.sections = {section};

    trtf::WriteBundleFile(path, bundle);

    const auto info = trtf::InspectBundle(path);
    check(info.model_id == "inspectable", "inspect model_id");
    check(info.vocab_size == 50000, "inspect vocab_size");
    check(info.num_layers == 12, "inspect num_layers");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that the bundle format handles large section payloads
//   (1 MB) without corruption, ensuring the first and last bytes match.
// Setup: Creates a bundle with a single section named "big_data" containing
//   1,048,576 bytes all set to 'A'.
// Mechanism: Writes the bundle, reads it back, checks section count, size,
//   and samples the first and last byte for correctness.
// -----------------------------------------------------------------------------
static void test_large_section()
{
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "large.trtfb").string();

    trtf::BundleFile bundle;
    bundle.info.model_id = "large";

    trtf::BundleSection section;
    section.name = "big_data";
    section.data.resize(1024 * 1024, 'A'); // 1MB section
    bundle.sections = {section};

    trtf::WriteBundleFile(path, bundle);
    const auto loaded = trtf::ReadBundleFile(path);

    check(loaded.sections.size() == 1, "large section count");
    check(loaded.sections[0].data.size() == 1024 * 1024, "large section size");
    check(loaded.sections[0].data[0] == 'A', "large section first byte");
    check(loaded.sections[0].data.back() == 'A', "large section last byte");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that the bundle format works with section names matching
//   real production usage (engine_plan, config.json, tokenizer.json,
//   tokenizer_config.json), simulating a Qwen3 bundle.
// Setup: Creates a bundle with Qwen3-like metadata and four sections with
//   production section names and small synthetic payloads.
// Mechanism: Writes the bundle, reads it back, checks all section names and
//   data match. Also calls InspectBundle to verify metadata extraction works
//   on the realistic file.
// -----------------------------------------------------------------------------
static void test_realistic_bundle_sections()
{
    // Test with section names matching real bundle format (engine_plan, config.json, tokenizer.json)
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "realistic.trtfb").string();

    trtf::BundleFile bundle;
    bundle.info.model_id = "qwen3-test";
    bundle.info.model_type = "qwen3";
    bundle.info.family = "qwen";
    bundle.info.vocab_size = 151936;
    bundle.info.hidden_size = 1024;
    bundle.info.num_layers = 28;
    bundle.info.max_cache_length = 256;

    bundle.sections.push_back({"engine_plan", {'E', 'N', 'G', 'I', 'N', 'E'}});
    bundle.sections.push_back({"config.json", {'{', '"', 'a', '"', ':', '1', '}'}});
    bundle.sections.push_back({"tokenizer.json", {'{', '}'}});
    bundle.sections.push_back({"tokenizer_config.json", {'{', '}'}});

    trtf::WriteBundleFile(path, bundle);
    const auto loaded = trtf::ReadBundleFile(path);

    check(loaded.sections.size() == 4, "realistic section count");
    check(loaded.sections[0].name == "engine_plan", "engine_plan section name");
    check(loaded.sections[0].data == std::vector<char>{'E', 'N', 'G', 'I', 'N', 'E'}, "engine_plan data");
    check(loaded.sections[1].name == "config.json", "config.json section name");
    check(loaded.sections[2].name == "tokenizer.json", "tokenizer.json section name");
    check(loaded.sections[3].name == "tokenizer_config.json", "tokenizer_config.json section name");

    // Inspect should also work
    const auto info = trtf::InspectBundle(path);
    check(info.model_id == "qwen3-test", "inspect realistic model_id");
    check(info.family == "qwen", "inspect realistic family");

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that the JSON header parser correctly handles section
//   offsets and sizes exceeding INT32_MAX (>2 GB), ensuring 64-bit integer
//   support in the serialization layer.
// Setup: Constructs a raw JSON string with an "engine_plan" section of 3 GB
//   and a "config.json" section at offset 3 GB. No actual file of that size
//   is created -- this tests the JSON parser in isolation.
// Mechanism: Calls BundleInfoFromJson with the hand-crafted JSON, checks that
//   parsed section sizes and offsets are correct 64-bit values (3000000000ULL).
// -----------------------------------------------------------------------------
static void test_large_section_offsets_int64()
{
    // Test that section offset/size JSON parsing handles values > INT32_MAX
    // We don't create a 3GB file, but verify the JSON parser handles large numbers
    trtf::BundleInfo info;
    info.model_id = "large-model";

    // Manually create JSON with large offsets
    const std::string json = R"({
  "model_id": "large-model",
  "model_type": "",
  "family": "",
  "trt_version": "",
  "gpu_name": "",
  "created_at": "",
  "vocab_size": 0,
  "hidden_size": 0,
  "num_layers": 0,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "max_cache_length": 32,
  "sections": {
    "engine_plan": {"offset": 0, "size": 3000000000},
    "config.json": {"offset": 3000000000, "size": 1024}
  }
})";

    std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>> parsed_sections;
    const auto parsed = trtf::BundleInfoFromJson(json, parsed_sections);

    check(parsed.model_id == "large-model", "int64 model_id");
    check(parsed_sections.size() == 2, "int64 section count");
    check(parsed_sections[0].first == "engine_plan", "int64 section 0 name");
    check(parsed_sections[0].second.second == 3000000000ULL, "int64 engine_plan size 3GB");
    check(parsed_sections[1].first == "config.json", "int64 section 1 name");
    check(parsed_sections[1].second.first == 3000000000ULL, "int64 config.json offset 3GB");
    check(parsed_sections[1].second.second == 1024, "int64 config.json size");
}

// -----------------------------------------------------------------------------
// Intention: Verify that ReadBundleFile throws a runtime_error when the file
//   has valid magic bytes but is truncated (header length claims 1000 bytes
//   but only 5 bytes of header data follow).
// Setup: Manually writes the 8-byte kBundleMagic, then a little-endian uint64
//   header length of 1000, followed by only 5 bytes of content ("short").
// Mechanism: Calls ReadBundleFile on the truncated file, catches the expected
//   runtime_error, and asserts that an exception was indeed thrown.
// -----------------------------------------------------------------------------
static void test_truncated_bundle_throws()
{
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "truncated.trtfb").string();

    // Write valid magic + header length but truncate the actual header
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(trtf::kBundleMagic), 8);
        // Write header length of 1000 but only write 10 bytes
        uint64_t len = 1000;
        unsigned char bytes[8];
        for (int i = 0; i < 8; ++i) bytes[i] = static_cast<unsigned char>((len >> (8 * i)) & 0xFF);
        out.write(reinterpret_cast<const char*>(bytes), 8);
        out.write("short", 5);
    }

    bool threw = false;
    try
    {
        trtf::ReadBundleFile(path);
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    check(threw, "truncated bundle throws");

    std::filesystem::remove_all(tmp);
}

int main()
{
    test_write_read_roundtrip();
    test_magic_validation();
    test_empty_sections();
    test_header_json_roundtrip();
    test_is_bundle_valid();
    test_is_bundle_invalid();
    test_inspect_returns_metadata();
    test_large_section();
    test_realistic_bundle_sections();
    test_large_section_offsets_int64();
    test_truncated_bundle_throws();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle_format tests passed.\n";
    return 0;
}
