#pragma once

#include "model/standard_checkpoint_mapper.h"

#include <cstddef>
#include <filesystem>

namespace trtf {
namespace gemma {

class GemmaCheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
    DecoderCheckpoint map_checkpoint(
        const TensorSource& reader, std::size_t vocab_size,
        const std::filesystem::path& path,
        const DecoderArchitectureConfig& architecture) const override;
};

} // namespace gemma
} // namespace trtf
