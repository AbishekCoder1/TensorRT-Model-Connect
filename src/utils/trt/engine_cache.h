#pragma once

#include "model/trt_model_definition.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trtf {

struct TrtEngineCacheKeyParams {
    bool requires_position_input{false};
    int32_t num_layers{0};
};

std::string BuildTrtEngineCacheKey(const TrtDecoderDefinition& definition, const TrtEngineCacheKeyParams& params);
std::optional<std::vector<char>> LoadTrtEnginePlanFromCache(const std::string& cache_key);
void SaveTrtEnginePlanToCache(const std::string& cache_key, const void* plan_data, std::size_t plan_size);

} // namespace trtf
