#pragma once

#include "model/standard_checkpoint_mapper.h"

namespace trtf {
namespace qwen {

class QwenCheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
};

} // namespace qwen
} // namespace trtf
