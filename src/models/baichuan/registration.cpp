#include "models/baichuan/registration.h"
#include "models/baichuan/checkpoint_mapper.h"
#include "trtf/hf_family_registry.h"
#include "model/checkpoint_mapper.h"
#include "runtime/trt/trt_graph_builder.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/standard_decoder_graph_builder.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"

#include <filesystem>
#include <memory>
#include <string>

namespace trtf {
namespace baichuan {
namespace {

bool is_baichuan_model_type(const std::string& model_type)
{
    if (model_type.empty())
    {
        return false;
    }
    const std::string lowered = to_lower_ascii(model_type);
    return starts_with(lowered, "baichuan");
}

bool has_baichuan_root_checkpoint(const HfModelMetadata& metadata)
{
    const std::filesystem::path model_dir(metadata.model_dir);
    return std::filesystem::exists(model_dir / "model.safetensors")
        || std::filesystem::exists(model_dir / "model.safetensors.index.json");
}

DecoderModel load_baichuan_model(const HfModelMetadata& metadata)
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

void RegisterBaichuanFamily()
{
    RegisterCheckpointMapper("baichuan", 100, std::make_unique<BaichuanCheckpointMapper>());

#if TRTF_HAS_TRT
    RegisterTrtGraphBuilder("baichuan", std::make_unique<StandardDecoderGraphBuilder>());
    if (!FindTrtGraphBuilder("standard-decoder"))
    {
        RegisterTrtGraphBuilder("standard-decoder", std::make_unique<StandardDecoderGraphBuilder>());
    }
#endif

    RegisterHfModelFamily({
        "baichuan-safetensors",
        100,
        [](const HfModelMetadata& metadata) {
            if (!is_baichuan_model_type(metadata.model_type))
            {
                return false;
            }
            return has_baichuan_root_checkpoint(metadata);
        },
        [](const HfModelMetadata& metadata) { return load_baichuan_model(metadata); },
    });
}

} // namespace baichuan
} // namespace trtf
