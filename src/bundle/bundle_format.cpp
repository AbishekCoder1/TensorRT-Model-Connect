#include "bundle/bundle_format.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trtf {

namespace {

uint64_t read_u64_le(std::ifstream& in)
{
    unsigned char bytes[8];
    in.read(reinterpret_cast<char*>(bytes), 8);
    if (!in)
    {
        throw std::runtime_error("Failed to read uint64 from bundle file");
    }
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i)
    {
        value = (value << 8) | bytes[i];
    }
    return value;
}

BundleInfo BundleInfoFromJson(const std::string& json,
    std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>>& sections_out)
{
    BundleInfo info;
    info.model_id = extract_json_string(json, "model_id", "");
    info.model_type = extract_json_string(json, "model_type", "");
    info.family = extract_json_string(json, "family", "");
    info.trt_version = extract_json_string(json, "trt_version", "");
    info.gpu_name = extract_json_string(json, "gpu_name", "");
    info.created_at = extract_json_string(json, "created_at", "");
    info.vocab_size = extract_json_int(json, "vocab_size", 0);
    info.hidden_size = extract_json_int(json, "hidden_size", 0);
    info.num_layers = extract_json_int(json, "num_layers", 0);
    info.num_attention_heads = extract_json_int(json, "num_attention_heads", 1);
    info.num_key_value_heads = extract_json_int(json, "num_key_value_heads", 1);
    info.max_cache_length = extract_json_int(json, "max_cache_length", 32);
    info.runtime_strategy = extract_json_string(json, "runtime_strategy", "decoder_kv_cache");

    // Parse sections block: look for "sections": { ... }
    sections_out.clear();
    const std::string sections_key = "\"sections\"";
    auto sections_pos = json.find(sections_key);
    if (sections_pos != std::string::npos)
    {
        auto brace_start = json.find('{', sections_pos + sections_key.size());
        if (brace_start != std::string::npos)
        {
            // Find matching closing brace
            int depth = 1;
            auto pos = brace_start + 1;
            while (pos < json.size() && depth > 0)
            {
                if (json[pos] == '{') ++depth;
                else if (json[pos] == '}') --depth;
                ++pos;
            }
            const std::string sections_json = json.substr(brace_start, pos - brace_start);

            // Parse each section entry: "name": {"offset": N, "size": M}
            std::size_t search_pos = 0;
            while (search_pos < sections_json.size())
            {
                auto quote_start = sections_json.find('"', search_pos);
                if (quote_start == std::string::npos) break;
                auto quote_end = sections_json.find('"', quote_start + 1);
                if (quote_end == std::string::npos) break;

                const std::string section_name = sections_json.substr(quote_start + 1, quote_end - quote_start - 1);

                auto inner_brace = sections_json.find('{', quote_end + 1);
                if (inner_brace == std::string::npos) break;
                auto inner_brace_end = sections_json.find('}', inner_brace + 1);
                if (inner_brace_end == std::string::npos) break;

                const std::string inner = sections_json.substr(inner_brace, inner_brace_end - inner_brace + 1);

                // Parse offset/size as int64 (stoll) since engine plans can exceed 2 GB.
                const auto parse_int64 = [&](const std::string& key) -> int64_t {
                    const std::string needle = "\"" + key + "\"";
                    auto kpos = inner.find(needle);
                    if (kpos == std::string::npos) return 0;
                    auto colon = inner.find(':', kpos + needle.size());
                    if (colon == std::string::npos) return 0;
                    auto start = inner.find_first_of("-0123456789", colon + 1);
                    if (start == std::string::npos) return 0;
                    try { return std::stoll(inner.substr(start)); } catch (...) { return 0; }
                };
                const int64_t offset_val = parse_int64("offset");
                const int64_t size_val = parse_int64("size");
                sections_out.push_back({section_name, {static_cast<std::size_t>(offset_val), static_cast<std::size_t>(size_val)}});

                search_pos = inner_brace_end + 1;
            }
        }
    }

    return info;
}

} // namespace

BundleFile ReadBundleFile(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open bundle file: " + path);
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0)
    {
        throw std::runtime_error("Invalid bundle magic in: " + path);
    }

    const uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024)
    {
        throw std::runtime_error("Bundle header too large: " + path);
    }

    std::string header_json(static_cast<std::size_t>(header_length), '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_length));
    if (!in)
    {
        throw std::runtime_error("Failed to read bundle header: " + path);
    }

    std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>> section_table;
    BundleFile bundle;
    bundle.info = BundleInfoFromJson(header_json, section_table);

    const std::size_t data_start = kBundleHeaderOffset + static_cast<std::size_t>(header_length);

    for (const auto& [name, offset_size] : section_table)
    {
        const auto& [offset, size] = offset_size;
        BundleSection section;
        section.name = name;
        section.data.resize(size);

        in.seekg(static_cast<std::streamoff>(data_start + offset));
        in.read(section.data.data(), static_cast<std::streamsize>(size));
        if (!in)
        {
            throw std::runtime_error("Failed to read bundle section '" + name + "' from: " + path);
        }

        bundle.sections.push_back(std::move(section));
    }

    return bundle;
}

BundleInfo ReadBundleHeader(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open bundle file: " + path);
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0)
    {
        throw std::runtime_error("Invalid bundle magic in: " + path);
    }

    const uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024)
    {
        throw std::runtime_error("Bundle header too large: " + path);
    }

    std::string header_json(static_cast<std::size_t>(header_length), '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_length));
    if (!in)
    {
        throw std::runtime_error("Failed to read bundle header: " + path);
    }

    std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>> sections_ignored;
    return BundleInfoFromJson(header_json, sections_ignored);
}

bool HasBundleMagic(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in)
    {
        return false;
    }

    return std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) == 0;
}

// Public API implementations from bundle.h

bool IsBundle(const std::string& path)
{
    return HasBundleMagic(path);
}

BundleInfo InspectBundle(const std::string& bundle_path)
{
    return ReadBundleHeader(bundle_path);
}

} // namespace trtf
