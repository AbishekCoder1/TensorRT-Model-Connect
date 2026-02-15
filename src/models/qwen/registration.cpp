#include "models/qwen/registration.h"
#include "models/qwen/checkpoint_mapper.h"
#include "trtf/hf_family_registry.h"
#include "model/checkpoint_mapper.h"
#include "runtime/trt/model_runtime_fwd.h"
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
        if (model.max_cache_length > 4096)
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
    RegisterModelRuntime("qwen", CreateStandardDecoderRuntime());
    RegisterModelRuntime("qwen2", CreateStandardDecoderRuntime());
    RegisterModelRuntime("qwen3", CreateStandardDecoderRuntime());
    RegisterModelRuntime("qwq", CreateStandardDecoderRuntime());
    if (!FindModelRuntime("standard-decoder"))
    {
        RegisterModelRuntime("standard-decoder", CreateStandardDecoderRuntime());
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
