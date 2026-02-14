#pragma once

#include "trtf/model.h"
#include "safetensors_loader.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace trtf {

class ICheckpointMapper {
public:
    virtual ~ICheckpointMapper() = default;
    virtual bool can_map(const DecoderArchitectureConfig& architecture) const = 0;
    virtual DecoderCheckpoint map_checkpoint(
        const TensorSource& reader, std::size_t vocab_size,
        const std::filesystem::path& path,
        const DecoderArchitectureConfig& architecture) const = 0;
};

void RegisterCheckpointMapper(const std::string& family, int priority,
                              std::unique_ptr<ICheckpointMapper> mapper);
ICheckpointMapper* FindCheckpointMapper(const DecoderArchitectureConfig& architecture);

} // namespace trtf
