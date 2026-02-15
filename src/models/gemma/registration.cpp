#include "models/gemma/registration.h"
#include "models/gemma/checkpoint_mapper.h"
#include "trtf/hf_family_registry.h"
#include "model/checkpoint_mapper.h"
#include "runtime/trt/model_runtime_fwd.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"

#include <filesystem>
#include <memory>
#include <string>

namespace trtf {
namespace gemma {
namespace {

bool is_gemma_model_type(const std::string& model_type)
{
    if (model_type.empty())
    {
        return false;
    }
    const std::string lowered = to_lower_ascii(model_type);
    return starts_with(lowered, "gemma");
}

bool has_gemma_root_checkpoint(const HfModelMetadata& metadata)
{
    const std::filesystem::path model_dir(metadata.model_dir);
    return std::filesystem::exists(model_dir / "model.safetensors")
        || std::filesystem::exists(model_dir / "model.safetensors.index.json");
}

DecoderModel load_gemma_model(const HfModelMetadata& metadata)
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

void RegisterGemmaFamily()
{
    RegisterCheckpointMapper("gemma", 100, std::make_unique<GemmaCheckpointMapper>());

#if TRTF_HAS_TRT
    RegisterModelRuntime("gemma", CreateStandardDecoderRuntime());
    if (!FindModelRuntime("standard-decoder"))
    {
        RegisterModelRuntime("standard-decoder", CreateStandardDecoderRuntime());
    }
#endif

    RegisterHfModelFamily({
        "gemma-safetensors",
        100,
        [](const HfModelMetadata& metadata) {
            if (!is_gemma_model_type(metadata.model_type))
            {
                return false;
            }
            return has_gemma_root_checkpoint(metadata);
        },
        [](const HfModelMetadata& metadata) { return load_gemma_model(metadata); },
    });
}

} // namespace gemma
} // namespace trtf
