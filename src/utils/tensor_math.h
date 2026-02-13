#pragma once

#include "trtf/model.h"

#include <cstdint>
#include <vector>

namespace trtf {

std::vector<float> transpose_2d(const std::vector<float>& src, int32_t rows, int32_t cols, const char* name);
std::vector<float> repeat_head_norm(const std::vector<float>& norm, int32_t num_heads);
std::vector<float> expand_kv_projection(const std::vector<float>& in_hidden_kv, int32_t hidden, int32_t kv_hidden,
    int32_t target_hidden, const DecoderArchitectureConfig& architecture, const char* tensor_name);

} // namespace trtf
