#include "models/mistral/registration.h"
#include "models/mistral/checkpoint_mapper.h"
#include "trtf/hf_family_registry.h"
#include "model/checkpoint_mapper.h"
#include "runtime/trt/model_runtime_fwd.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"

#include <filesystem>
#include <memory>
#include <string>

namespace trtf {
namespace mistral {
namespace {

bool is_mistral_model_type(const std::string& model_type)
{
    if (model_type.empty())
    {
        return false;
    }
    const std::string lowered = to_lower_ascii(model_type);
    return starts_with(lowered, "mistral");
}

bool has_mistral_root_checkpoint(const HfModelMetadata& metadata)
{
    const std::filesystem::path model_dir(metadata.model_dir);
    return std::filesystem::exists(model_dir / "model.safetensors")
        || std::filesystem::exists(model_dir / "model.safetensors.index.json");
}

DecoderModel load_mistral_model(const HfModelMetadata& metadata)
{
    DecoderModel model = LoadDecoderModel(metadata.model_dir);

    const int32_t env_max_cache = parse_positive_env_int("TRTF_MAX_CACHE_LENGTH", -1);
    if (env_max_cache <= 0 && model.max_cache_length > 4096)
    {
        model.max_cache_length = 4096;
    }
    return model;
}

} // namespace

void RegisterMistralFamily()
{
    RegisterCheckpointMapper("mistral", 100, std::make_unique<MistralCheckpointMapper>());

#if TRTF_HAS_TRT
    RegisterModelRuntime("mistral", CreateStandardDecoderRuntime());
    if (!FindModelRuntime("standard-decoder"))
    {
        RegisterModelRuntime("standard-decoder", CreateStandardDecoderRuntime());
    }
#endif

    RegisterHfModelFamily({
        "mistral-safetensors",
        100,
        [](const HfModelMetadata& metadata) {
            if (!is_mistral_model_type(metadata.model_type))
            {
                return false;
            }
            return has_mistral_root_checkpoint(metadata);
        },
        [](const HfModelMetadata& metadata) { return load_mistral_model(metadata); },
    });
}

} // namespace mistral
} // namespace trtf
