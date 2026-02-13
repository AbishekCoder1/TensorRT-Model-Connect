#include "trt_backend_qwen_impl.h"
#include "utils/text_parsers.h"

#include <string>

namespace trtf {

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
