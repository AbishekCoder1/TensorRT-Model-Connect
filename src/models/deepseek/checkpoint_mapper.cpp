#include "models/deepseek/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace deepseek {

bool DeepSeekCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string fam = to_lower_ascii(architecture.family);
    return starts_with(fam, "deepseek");
}

} // namespace deepseek
} // namespace trtf
