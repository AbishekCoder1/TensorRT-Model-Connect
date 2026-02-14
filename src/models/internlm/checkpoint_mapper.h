#pragma once

#include "model/standard_checkpoint_mapper.h"

namespace trtf {
namespace internlm {

class InternLMCheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
};

} // namespace internlm
} // namespace trtf
