#include "models/yi/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace yi {

bool YiCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string fam = to_lower_ascii(architecture.family);
    return starts_with(fam, "yi");
}

} // namespace yi
} // namespace trtf
