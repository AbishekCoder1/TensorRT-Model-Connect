#include "models/mistral/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace mistral {

bool MistralCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string fam = to_lower_ascii(architecture.family);
    return starts_with(fam, "mistral");
}

} // namespace mistral
} // namespace trtf
