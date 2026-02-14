#include "runtime/trt/trt_graph_builder.h"
#include "runtime/trt/trt_backend_shared.h"
#include "utils/text_parsers.h"

#include <string>

namespace trtf {

std::unique_ptr<IGenerationBackend> CreateTrtBackend(const ITokenizer& tokenizer, const DecoderModel& model)
{
#if TRTF_HAS_TRT
    const std::string family = to_lower_ascii(model.architecture.family);

    // 1. Family-specific builder
    if (auto* builder = FindTrtGraphBuilder(family))
    {
        return CreateTrtBackendWithBuilder(tokenizer, model, *builder);
    }

    // 2. Standard decoder fallback (works for most LLM families)
    if (auto* builder = FindTrtGraphBuilder("standard-decoder"))
    {
        return CreateTrtBackendWithBuilder(tokenizer, model, *builder);
    }

    return nullptr;
#else
    (void) tokenizer;
    (void) model;
    return nullptr;
#endif
}

} // namespace trtf
