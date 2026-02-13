#include "trt_backend_qwen_impl.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace trtf {
namespace {

std::string to_lower_ascii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool starts_with(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

std::unique_ptr<IGenerationBackend> CreateTrtBackend(const ITokenizer& tokenizer, const DecoderModel& model)
{
    const std::string family = to_lower_ascii(model.architecture.family);

    // Keep dispatch explicit so new families can register backend implementations
    // without modifying TRT runtime internals.
    if (starts_with(family, "qwen") || starts_with(family, "qwq") || model.checkpoint.has_qwen_layers)
    {
        return CreateTrtQwenBackend(tokenizer, model);
    }

    // Temporary fallback: non-family-tagged decoder definitions still use the
    // existing TRT implementation path until dedicated builders are split out.
    return CreateTrtQwenBackend(tokenizer, model);
}

} // namespace trtf
