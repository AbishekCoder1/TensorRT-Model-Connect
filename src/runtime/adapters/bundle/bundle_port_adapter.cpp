#include "runtime/adapters/bundle/bundle_port_adapter.h"

#if TRTF_HAS_TRT
#include "cabi/bundle/bundle_helpers.h"
#endif

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>
#include <string_view>

namespace trtf::runtime {
namespace {

bool is_json_object_text(const std::string& text)
{
    auto first = std::find_if(text.begin(), text.end(), [](char ch) {
        return !std::isspace(static_cast<unsigned char>(ch));
    });
    if (first == text.end())
        return false;

    auto last = std::find_if(text.rbegin(), text.rend(), [](char ch) {
        return !std::isspace(static_cast<unsigned char>(ch));
    });
    return last != text.rend() && *first == '{' && *last == '}';
}

} // namespace

BundlePortAdapter::BundlePortAdapter(const BundleFile& bundle)
    : mBundle(bundle)
{
}

const std::vector<char>* BundlePortAdapter::find_section(std::string_view section_name) const
{
    for (const auto& section : mBundle.sections)
    {
        if (section.name == section_name)
            return &section.data;
    }
    return nullptr;
}

bool BundlePortAdapter::has_section(std::string_view section_name) const
{
    return find_section(section_name) != nullptr;
}

BundlePortResult<std::vector<char>> BundlePortAdapter::fetch_section_bytes(std::string_view section_name) const
{
    const auto* section_data = find_section(section_name);
    if (section_data == nullptr)
    {
        return BundlePortResult<std::vector<char>>::missing(
            "Bundle section not found: " + std::string(section_name));
    }
    if (section_data->empty())
    {
        return BundlePortResult<std::vector<char>>::invalid(
            "Bundle section is empty: " + std::string(section_name));
    }
    return BundlePortResult<std::vector<char>>::success(*section_data);
}

BundlePortResult<FastPathModelConfig> BundlePortAdapter::parse_fast_path_config(int32_t max_cache_length_override) const
{
    const std::vector<char>* config_section = nullptr;

#if TRTF_HAS_TRT
    try
    {
        const trtf::BundleSections sections = trtf::find_bundle_sections(mBundle);
        config_section = sections.config_json_data;
    }
    catch (const std::exception& ex)
    {
        return BundlePortResult<FastPathModelConfig>::invalid(
            std::string("Failed to inspect bundle sections: ") + ex.what());
    }
#else
    config_section = find_section("config.json");
#endif

    if (config_section == nullptr)
    {
        return BundlePortResult<FastPathModelConfig>::missing("Bundle section not found: config.json");
    }
    if (config_section->empty())
    {
        return BundlePortResult<FastPathModelConfig>::invalid("Bundle section is empty: config.json");
    }

    const std::string config_text(config_section->data(), config_section->size());
    if (!is_json_object_text(config_text))
    {
        return BundlePortResult<FastPathModelConfig>::invalid("Bundle section is invalid JSON object: config.json");
    }

    try
    {
        return BundlePortResult<FastPathModelConfig>::success(
            trtf::parse_fast_path_config(config_text, max_cache_length_override));
    }
    catch (const std::exception& ex)
    {
        return BundlePortResult<FastPathModelConfig>::invalid(
            std::string("Failed to parse fast-path config: ") + ex.what());
    }
}

} // namespace trtf::runtime
