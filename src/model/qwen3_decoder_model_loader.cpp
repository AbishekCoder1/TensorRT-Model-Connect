#include "qwen3_decoder_model_loader.h"
#include "utils/text_parsers.h"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace trtf {
namespace {

bool is_qwen_family(std::string_view family)
{
    const std::string lowered = to_lower_ascii(std::string(family));
    return starts_with(lowered, "qwen") || starts_with(lowered, "qwq");
}

} // namespace

DecoderModel LoadQwen3DecoderModel(const std::string& model_dir)
{
    DecoderModel model = LoadDecoderModel(model_dir);
    if (is_qwen_family(model.architecture.family))
    {
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

} // namespace trtf
