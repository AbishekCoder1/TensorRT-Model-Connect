#include "trtf/hf_family_registry.h"
#include "trtf/model_resolver.h"

#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mutex>

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

std::mutex& custom_resolvers_mutex()
{
    static std::mutex m;
    return m;
}

std::vector<TextGenerationModelResolver>& custom_resolvers()
{
    static std::vector<TextGenerationModelResolver> resolvers;
    return resolvers;
}

} // namespace

void RegisterTextGenerationModelResolver(TextGenerationModelResolver resolver)
{
    if (!resolver)
    {
        throw std::invalid_argument("RegisterTextGenerationModelResolver requires a valid resolver.");
    }

    std::lock_guard<std::mutex> lock(custom_resolvers_mutex());
    custom_resolvers().push_back(std::move(resolver));
}

ResolvedModelSpec ResolveTextGenerationModel(const std::string& model_id)
{
    std::vector<TextGenerationModelResolver> resolvers_snapshot;
    {
        std::lock_guard<std::mutex> lock(custom_resolvers_mutex());
        resolvers_snapshot = custom_resolvers();
    }

    for (const auto& resolver : resolvers_snapshot)
    {
        std::optional<ResolvedModelSpec> resolved = resolver(model_id);
        if (resolved.has_value())
        {
            return std::move(*resolved);
        }
    }

    if (auto spec = ResolveHfModelViaFamilyRegistry(model_id))
    {
        return std::move(*spec);
    }

    if (is_hf_transformers_model_dir(model_id))
    {
        ResolvedModelSpec spec;
        spec.model_id = model_id;
        spec.kind = ResolvedModelKind::kHuggingFaceLocal;
        spec.huggingface_model_dir = model_id;
        return spec;
    }

    ResolvedModelSpec spec;
    spec.model_id = model_id;
    spec.kind = ResolvedModelKind::kDecoderDefinition;
    spec.decoder_model = LoadDecoderModel(model_id);
    return spec;
}

} // namespace trtf
