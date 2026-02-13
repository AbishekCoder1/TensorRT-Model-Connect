#include "trtf/hf_family_registry.h"
#include "qwen3_decoder_model_loader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#ifndef TRTF_SOURCE_DIR
#define TRTF_SOURCE_DIR "."
#endif

namespace trtf {
namespace {

constexpr char kTrtfDecoderSubdir[] = "trtf_decoder";

bool is_hf_transformers_model_dir(const std::string& model_id)
{
    if (model_id.empty())
    {
        return false;
    }
    const std::filesystem::path dir(model_id);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir)
        && std::filesystem::exists(dir / "config.json")
        && (std::filesystem::exists(dir / "model.safetensors")
            || std::filesystem::exists(dir / "model.safetensors.index.json"));
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string extract_json_string(const std::string& text, const std::string& key, const std::string& fallback)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return fallback;
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return fallback;
    }

    const std::size_t first_quote = text.find('"', colon + 1);
    if (first_quote == std::string::npos)
    {
        return fallback;
    }
    const std::size_t second_quote = text.find('"', first_quote + 1);
    if (second_quote == std::string::npos || second_quote <= first_quote + 1)
    {
        return fallback;
    }
    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::vector<std::string> extract_json_string_array(const std::string& text, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return {};
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return {};
    }

    const std::size_t open_bracket = text.find('[', colon + 1);
    if (open_bracket == std::string::npos)
    {
        return {};
    }

    std::vector<std::string> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size())
    {
        while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) != 0 || text[pos] == ','))
        {
            ++pos;
        }
        if (pos >= text.size() || text[pos] == ']')
        {
            break;
        }
        if (text[pos] != '"')
        {
            break;
        }

        const std::size_t first_quote = pos;
        const std::size_t second_quote = text.find('"', first_quote + 1);
        if (second_quote == std::string::npos || second_quote <= first_quote + 1)
        {
            break;
        }

        out.push_back(text.substr(first_quote + 1, second_quote - first_quote - 1));
        pos = second_quote + 1;
    }

    return out;
}

bool starts_with(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool iequals_ascii(std::string_view a, std::string_view b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const unsigned char ac = static_cast<unsigned char>(a[i]);
        const unsigned char bc = static_cast<unsigned char>(b[i]);
        if (std::tolower(ac) != std::tolower(bc))
        {
            return false;
        }
    }
    return true;
}

std::string to_lower_ascii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool is_qwen_model_type(const std::string& model_type)
{
    if (model_type.empty())
    {
        return false;
    }
    const std::string lowered = to_lower_ascii(model_type);
    return starts_with(lowered, "qwen") || starts_with(lowered, "qwq");
}

std::filesystem::path qwen_decoder_dir(const HfModelMetadata& metadata)
{
    return std::filesystem::path(metadata.model_dir) / kTrtfDecoderSubdir;
}

std::filesystem::path builtin_qwen3_hf_demo_dir()
{
    return std::filesystem::path(TRTF_SOURCE_DIR) / "models" / "hf" / "qwen3";
}

std::filesystem::path builtin_qwen3_hf_real_dir()
{
    return std::filesystem::path(TRTF_SOURCE_DIR) / "models" / "hf" / "Qwen__Qwen3-0.6B";
}

bool is_hf_checkpoint_dir(const std::filesystem::path& dir)
{
    return std::filesystem::exists(dir / "config.json")
        && (std::filesystem::exists(dir / "model.safetensors")
            || std::filesystem::exists(dir / "model.safetensors.index.json"));
}

std::string resolve_model_dir_from_alias(const std::string& model_id)
{
    if (iequals_ascii(model_id, "QWEN3") || iequals_ascii(model_id, "qwen3")
        || iequals_ascii(model_id, "Qwen/QWEN3"))
    {
        const std::filesystem::path real = builtin_qwen3_hf_real_dir();
        if (is_hf_checkpoint_dir(real) && std::filesystem::exists(real / "tokenizer.json"))
        {
            return real.string();
        }
        return builtin_qwen3_hf_demo_dir().string();
    }
    return model_id;
}

bool has_decoder_definition_files(const std::filesystem::path& decoder_dir)
{
    return std::filesystem::exists(decoder_dir / "config.json")
        && std::filesystem::exists(decoder_dir / "vocab.txt")
        && std::filesystem::exists(decoder_dir / "transitions.txt");
}

bool has_qwen_root_checkpoint(const HfModelMetadata& metadata)
{
    const std::filesystem::path model_dir(metadata.model_dir);
    return std::filesystem::exists(model_dir / "model.safetensors")
        || std::filesystem::exists(model_dir / "model.safetensors.index.json");
}

DecoderModel load_decoder_definition_model(const HfModelMetadata& metadata)
{
    const std::filesystem::path decoder_dir = qwen_decoder_dir(metadata);
    const std::filesystem::path model_dir(metadata.model_dir);

    // Prefer HF-root loading when tokenizer assets exist so TRT path uses real upstream tokenization.
    if (has_qwen_root_checkpoint(metadata) && std::filesystem::exists(model_dir / "tokenizer.json"))
    {
        return LoadQwen3DecoderModel(metadata.model_dir);
    }

    if (has_decoder_definition_files(decoder_dir))
    {
        return LoadDecoderModel(decoder_dir.string());
    }

    if (has_qwen_root_checkpoint(metadata))
    {
        // Upstream path: load directly from HF root config + model.safetensors (or sharded index).
        return LoadQwen3DecoderModel(metadata.model_dir);
    }

    throw std::runtime_error("Missing required decoder-definition files for family model at "
        + decoder_dir.string()
        + " and no model.safetensors (or model.safetensors.index.json) found in HF root dir " + metadata.model_dir + ".");
}

HfModelMetadata load_hf_metadata(const std::string& model_dir)
{
    const std::filesystem::path dir(model_dir);
    const std::string config_text = read_file(dir / "config.json");

    HfModelMetadata metadata;
    metadata.model_dir = model_dir;
    metadata.model_type = extract_json_string(config_text, "model_type", "");
    metadata.architectures = extract_json_string_array(config_text, "architectures");
    return metadata;
}

std::mutex& families_mutex()
{
    static std::mutex m;
    return m;
}

std::vector<HfModelFamilyRegistration>& families()
{
    static std::vector<HfModelFamilyRegistration> registrations;
    return registrations;
}

std::once_flag& builtins_once()
{
    static std::once_flag flag;
    return flag;
}

} // namespace

void RegisterHfModelFamily(HfModelFamilyRegistration registration)
{
    if (registration.family_name.empty())
    {
        throw std::invalid_argument("RegisterHfModelFamily requires non-empty family_name.");
    }
    if (!registration.matcher)
    {
        throw std::invalid_argument("RegisterHfModelFamily requires a valid matcher.");
    }
    if (!registration.model_definition_loader)
    {
        throw std::invalid_argument("RegisterHfModelFamily requires a valid model_definition_loader.");
    }

    std::lock_guard<std::mutex> lock(families_mutex());
    families().push_back(std::move(registration));
}

void RegisterBuiltinHfModelFamilies()
{
    RegisterHfModelFamily({
        "qwen-decoder-definition",
        100,
        [](const HfModelMetadata& metadata) {
            if (!is_qwen_model_type(metadata.model_type))
            {
                return false;
            }
            return has_decoder_definition_files(qwen_decoder_dir(metadata)) || has_qwen_root_checkpoint(metadata);
        },
        [](const HfModelMetadata& metadata) { return load_decoder_definition_model(metadata); },
    });
}

std::optional<ResolvedModelSpec> ResolveHfModelViaFamilyRegistry(const std::string& model_id)
{
    const std::string effective_model_dir = resolve_model_dir_from_alias(model_id);
    if (!is_hf_transformers_model_dir(effective_model_dir))
    {
        return std::nullopt;
    }

    std::call_once(builtins_once(), []() { RegisterBuiltinHfModelFamilies(); });

    std::vector<HfModelFamilyRegistration> families_snapshot;
    {
        std::lock_guard<std::mutex> lock(families_mutex());
        families_snapshot = families();
    }
    if (families_snapshot.empty())
    {
        return std::nullopt;
    }

    std::stable_sort(families_snapshot.begin(), families_snapshot.end(),
        [](const HfModelFamilyRegistration& a, const HfModelFamilyRegistration& b) {
            return a.priority > b.priority;
        });

    const HfModelMetadata metadata = load_hf_metadata(effective_model_dir);
    for (const auto& family : families_snapshot)
    {
        if (!family.matcher(metadata))
        {
            continue;
        }

        DecoderModel model = family.model_definition_loader(metadata);
        model.model_id = model_id;

        ResolvedModelSpec spec;
        spec.model_id = model_id;
        spec.kind = ResolvedModelKind::kDecoderDefinition;
        spec.decoder_model = std::move(model);
        return spec;
    }

    return std::nullopt;
}

} // namespace trtf
