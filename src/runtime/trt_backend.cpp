#include "runtime/trt/model_runtime.h"
#include "runtime/trt/trt_backend_shared.h"
#include "utils/text_parsers.h"

#include <string>

namespace trtf {

std::unique_ptr<IGenerationBackend> CreateTrtBackend(const ITokenizer& tokenizer, const DecoderModel& model)
{
#if TRTF_HAS_TRT
    const std::string family = to_lower_ascii(model.architecture.family);

    // 1. Family-specific runtime
    if (auto* runtime = FindModelRuntime(family))
    {
        return CreateTrtBackendWithRuntime(tokenizer, model, *runtime);
    }

    // 2. Standard decoder fallback (works for most LLM families)
    if (auto* runtime = FindModelRuntime("standard-decoder"))
    {
        return CreateTrtBackendWithRuntime(tokenizer, model, *runtime);
    }

    return nullptr;
#else
    (void) tokenizer;
    (void) model;
    return nullptr;
#endif
}

} // namespace trtf
