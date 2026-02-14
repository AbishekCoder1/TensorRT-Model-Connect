#pragma once

#include "model/standard_checkpoint_mapper.h"

namespace trtf {
namespace baichuan {

class BaichuanCheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
};

} // namespace baichuan
} // namespace trtf
