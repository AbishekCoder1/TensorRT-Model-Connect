#include "models/qwen/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace qwen {

bool QwenCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string family = to_lower_ascii(architecture.family);
    return starts_with(family, "qwen") || starts_with(family, "qwq");
}

} // namespace qwen
} // namespace trtf
