#include "models/llama/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace llama {

bool LlamaCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string family = to_lower_ascii(architecture.family);
    return starts_with(family, "llama");
}

} // namespace llama
} // namespace trtf
