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

// Model-dir index: maps a model directory + cache_length to a cache key so we can
// find the cached engine without loading weights.
// index_key = hash(canonical_model_dir + config.json content + safetensors file sizes + max_cache_length)
std::string BuildModelDirIndexKey(const std::string& model_dir, int32_t max_cache_length);
void SaveModelDirIndex(const std::string& index_key, const std::string& cache_key);
std::optional<std::string> LookupModelDirIndex(const std::string& index_key);

} // namespace trtf
