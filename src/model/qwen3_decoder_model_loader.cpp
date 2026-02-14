#include "qwen3_decoder_model_loader.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"

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

} // namespace trtf
