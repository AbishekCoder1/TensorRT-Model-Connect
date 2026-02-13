#include "trtf/hf_family_registry.h"
#include "qwen3_decoder_model_loader.h"
#include "models/qwen/registration.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"

#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef TRTF_SOURCE_DIR
#define TRTF_SOURCE_DIR "."
#endif

namespace trtf {
namespace {

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
    qwen::RegisterQwenFamily();
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
