#include "models/internlm/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace internlm {

bool InternLMCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string fam = to_lower_ascii(architecture.family);
    return starts_with(fam, "internlm");
}

} // namespace internlm
} // namespace trtf
