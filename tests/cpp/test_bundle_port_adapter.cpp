#include "bundle/bundle_format.h"
#include "runtime/adapters/bundle/bundle_port_adapter.h"

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

static std::vector<char> to_bytes(const std::string& text)
{
    return std::vector<char>(text.begin(), text.end());
}

static void test_fetch_section_present()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"engine_plan", {'P', 'L', 'A', 'N'}});
    trtf::runtime::BundlePortAdapter adapter(bundle);

    check(adapter.has_section("engine_plan"), "present section: has_section true");

    const auto result = adapter.fetch_section_bytes("engine_plan");
    check(result.ok(), "present section: fetch ok");
    check(result.status == trtf::runtime::BundlePortStatus::kOk, "present section: status kOk");
    check(result.value.size() == 4, "present section: payload size");
    check(result.value[0] == 'P', "present section: payload data");
}

static void test_fetch_section_missing()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"config.json", to_bytes("{\"hidden_size\":64}")});
    trtf::runtime::BundlePortAdapter adapter(bundle);

    check(!adapter.has_section("engine_plan"), "missing section: has_section false");

    const auto result = adapter.fetch_section_bytes("engine_plan");
    check(!result.ok(), "missing section: fetch not ok");
    check(result.status == trtf::runtime::BundlePortStatus::kMissingSection,
        "missing section: status kMissingSection");
    check(result.message.find("engine_plan") != std::string::npos,
        "missing section: message contains section name");
}

static void test_fetch_section_invalid_empty_payload()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"engine_plan", {}});
    trtf::runtime::BundlePortAdapter adapter(bundle);

    check(adapter.has_section("engine_plan"), "invalid section: still present");

    const auto result = adapter.fetch_section_bytes("engine_plan");
    check(!result.ok(), "invalid section: fetch not ok");
    check(result.status == trtf::runtime::BundlePortStatus::kInvalidSection,
        "invalid section: status kInvalidSection");
}

static void test_parse_fast_path_config_present()
{
    trtf::BundleFile bundle;
    const std::string config_text = R"({
        "runtime_strategy": "encoder_only",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 2048
    })";
    bundle.sections.push_back({"config.json", to_bytes(config_text)});
    trtf::runtime::BundlePortAdapter adapter(bundle);

    const auto result = adapter.parse_fast_path_config(128);
    check(result.ok(), "parse config present: ok");
    check(result.status == trtf::runtime::BundlePortStatus::kOk, "parse config present: status kOk");
    check(result.value.hidden_size == 768, "parse config present: hidden_size parsed");
    check(result.value.runtime_strategy == "encoder_only", "parse config present: runtime_strategy parsed");
    check(result.value.max_cache_length == 128, "parse config present: override applied");
}

static void test_parse_fast_path_config_missing()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"engine_plan", {'P'}});
    trtf::runtime::BundlePortAdapter adapter(bundle);

    const auto result = adapter.parse_fast_path_config(-1);
    check(!result.ok(), "parse config missing: not ok");
    check(result.status == trtf::runtime::BundlePortStatus::kMissingSection,
        "parse config missing: status kMissingSection");
}

static void test_parse_fast_path_config_invalid()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"config.json", to_bytes("this-is-not-json")});
    trtf::runtime::BundlePortAdapter adapter(bundle);

    const auto result = adapter.parse_fast_path_config(-1);
    check(!result.ok(), "parse config invalid: not ok");
    check(result.status == trtf::runtime::BundlePortStatus::kInvalidSection,
        "parse config invalid: status kInvalidSection");
}

int main()
{
    test_fetch_section_present();
    test_fetch_section_missing();
    test_fetch_section_invalid_empty_payload();
    test_parse_fast_path_config_present();
    test_parse_fast_path_config_missing();
    test_parse_fast_path_config_invalid();

    if (failures != 0)
    {
        std::cerr << "[test_bundle_port_adapter] " << failures << " failure(s)\n";
        return 1;
    }

    std::cout << "[test_bundle_port_adapter] all checks passed\n";
    return 0;
}
