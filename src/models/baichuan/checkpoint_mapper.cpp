#include "models/baichuan/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace baichuan {

bool BaichuanCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string fam = to_lower_ascii(architecture.family);
    return starts_with(fam, "baichuan");
}

} // namespace baichuan
} // namespace trtf
