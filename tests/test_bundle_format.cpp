// Test: Bundle format (.trtfb) read/write, magic validation, JSON roundtrip.
// No TRT/GPU needed -- tests binary format with synthetic data.

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

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle_format tests passed.\n";
    return 0;
}
