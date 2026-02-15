#include "models/qwen/registration.h"
#include "models/qwen/checkpoint_mapper.h"
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
namespace qwen {
namespace {

constexpr char kTrtfDecoderSubdir[] = "trtf_decoder";

bool is_qwen_model_type(const std::string& model_type)
{
    if (model_type.empty())
    {
        return false;
    }
    const std::string lowered = to_lower_ascii(model_type);
    return starts_with(lowered, "qwen") || starts_with(lowered, "qwq");
}

bool is_qwen_family(std::string_view family)
{
    const std::string lowered = to_lower_ascii(std::string(family));
    return starts_with(lowered, "qwen") || starts_with(lowered, "qwq");
}

std::filesystem::path qwen_decoder_dir(const HfModelMetadata& metadata)
{
    return std::filesystem::path(metadata.model_dir) / kTrtfDecoderSubdir;
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

DecoderModel load_qwen_decoder_model(const std::string& model_dir)
{
    DecoderModel model = LoadDecoderModel(model_dir);
    if (is_qwen_family(model.architecture.family))
    {
        // Default cap keeps TRT step-engine memory practical for large upstream checkpoints
        // when no explicit config or env override is set.
        const int32_t env_max_cache = parse_positive_env_int("TRTF_MAX_CACHE_LENGTH", -1);
        if (env_max_cache <= 0 && model.max_cache_length > 4096)
        {
            model.max_cache_length = 4096;
        }
        return model;
    }

    // Normalized family-owned trtf_decoder fixtures may omit architecture metadata
    // while still being resolved through the Qwen family registry path.
    const std::filesystem::path dir(model_dir);
    if (dir.filename() == "trtf_decoder")
    {
        model.architecture.family = "qwen3";
        return model;
    }

    throw std::runtime_error("Qwen loader received non-Qwen model family for dir: " + model_dir);
}

DecoderModel load_decoder_definition_model(const HfModelMetadata& metadata)
{
    const std::filesystem::path decoder_dir = qwen_decoder_dir(metadata);
    const std::filesystem::path model_dir(metadata.model_dir);

    if (has_qwen_root_checkpoint(metadata) && std::filesystem::exists(model_dir / "tokenizer.json"))
    {
        return load_qwen_decoder_model(metadata.model_dir);
    }

    if (has_decoder_definition_files(decoder_dir))
    {
        return LoadDecoderModel(decoder_dir.string());
    }

    if (has_qwen_root_checkpoint(metadata))
    {
        return load_qwen_decoder_model(metadata.model_dir);
    }

    throw std::runtime_error("Missing required decoder-definition files for family model at "
        + decoder_dir.string()
        + " and no model.safetensors (or model.safetensors.index.json) found in HF root dir " + metadata.model_dir + ".");
}

} // namespace

void RegisterQwenFamily()
{
    RegisterCheckpointMapper("qwen", 100, std::make_unique<QwenCheckpointMapper>());

#if TRTF_HAS_TRT
    RegisterTrtGraphBuilder("qwen", std::make_unique<StandardDecoderGraphBuilder>());
    RegisterTrtGraphBuilder("qwen2", std::make_unique<StandardDecoderGraphBuilder>());
    RegisterTrtGraphBuilder("qwen3", std::make_unique<StandardDecoderGraphBuilder>());
    RegisterTrtGraphBuilder("qwq", std::make_unique<StandardDecoderGraphBuilder>());
    if (!FindTrtGraphBuilder("standard-decoder"))
    {
        RegisterTrtGraphBuilder("standard-decoder", std::make_unique<StandardDecoderGraphBuilder>());
    }
#endif

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

} // namespace qwen
} // namespace trtf
