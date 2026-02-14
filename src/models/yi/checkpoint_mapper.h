#pragma once

#include "model/standard_checkpoint_mapper.h"

namespace trtf {
namespace yi {

class YiCheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
};

} // namespace yi
} // namespace trtf
