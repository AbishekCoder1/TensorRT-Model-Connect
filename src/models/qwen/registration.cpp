#include "models/qwen/registration.h"
#include "trtf/hf_family_registry.h"
#include "model/qwen3_decoder_model_loader.h"
#include "runtime/trt/trt_graph_builder.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_graph_ops.h"
#include "runtime/trt/trt_backend_shared.h"
#include "model/trt_model_definition.h"
#include "utils/text_parsers.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#endif

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

DecoderModel load_decoder_definition_model(const HfModelMetadata& metadata)
{
    const std::filesystem::path decoder_dir = qwen_decoder_dir(metadata);
    const std::filesystem::path model_dir(metadata.model_dir);

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
        return LoadQwen3DecoderModel(metadata.model_dir);
    }

    throw std::runtime_error("Missing required decoder-definition files for family model at "
        + decoder_dir.string()
        + " and no model.safetensors (or model.safetensors.index.json) found in HF root dir " + metadata.model_dir + ".");
}

} // namespace

void RegisterQwenFamily()
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

} // namespace qwen
} // namespace trtf
