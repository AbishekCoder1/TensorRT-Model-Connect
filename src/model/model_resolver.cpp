#include "trtf/hf_family_registry.h"
#include "trtf/model_resolver.h"

#include <filesystem>
#include <stdexcept>
#include <utility>

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

} // namespace

ResolvedModelSpec ResolveTextGenerationModel(const std::string& model_id)
{
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
