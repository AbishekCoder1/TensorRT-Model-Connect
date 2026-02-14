#pragma once

#include "model/standard_checkpoint_mapper.h"

namespace trtf {
namespace deepseek {

class DeepSeekCheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
};

} // namespace deepseek
} // namespace trtf
