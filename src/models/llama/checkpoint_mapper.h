#pragma once

#include "model/standard_checkpoint_mapper.h"

namespace trtf {
namespace llama {

class LlamaCheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
};

} // namespace llama
} // namespace trtf
