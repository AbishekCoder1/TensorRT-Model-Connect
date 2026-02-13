#include "utils/tensor_math.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtf {

std::vector<float> transpose_2d(const std::vector<float>& src, int32_t rows, int32_t cols, const char* name)
{
    const std::size_t expected = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (src.size() != expected)
    {
        throw std::runtime_error(std::string("Invalid tensor size for transpose: ") + name);
    }

    std::vector<float> dst(expected, 0.0F);
    for (int32_t r = 0; r < rows; ++r)
    {
        for (int32_t c = 0; c < cols; ++c)
        {
            const std::size_t src_idx
                = static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c);
            const std::size_t dst_idx
                = static_cast<std::size_t>(c) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r);
            dst[dst_idx] = src[src_idx];
        }
    }
    return dst;
}

std::vector<float> repeat_head_norm(const std::vector<float>& norm, int32_t num_heads)
{
    if (norm.empty() || num_heads <= 0)
    {
        return {};
    }

    std::vector<float> out(static_cast<std::size_t>(num_heads) * norm.size(), 1.0F);
    for (int32_t h = 0; h < num_heads; ++h)
    {
        const std::size_t offset = static_cast<std::size_t>(h) * norm.size();
        std::copy(norm.begin(), norm.end(), out.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    return out;
}

std::vector<float> expand_kv_projection(const std::vector<float>& in_hidden_kv, int32_t hidden, int32_t kv_hidden,
    int32_t target_hidden, const DecoderArchitectureConfig& architecture, const char* tensor_name)
{
    const std::size_t expected = static_cast<std::size_t>(hidden) * static_cast<std::size_t>(kv_hidden);
    if (in_hidden_kv.size() != expected)
    {
        throw std::runtime_error("Unexpected tensor size while expanding " + std::string(tensor_name));
    }
    if (target_hidden <= 0)
    {
        throw std::runtime_error("Invalid target hidden size while expanding " + std::string(tensor_name));
    }

    if (kv_hidden == target_hidden)
    {
        return in_hidden_kv;
    }

    const int32_t num_heads = std::max(architecture.num_attention_heads, 1);
    const int32_t num_kv_heads = std::max(architecture.num_key_value_heads, 1);
    const bool has_head_mapping = target_hidden % num_heads == 0 && num_heads % num_kv_heads == 0
        && kv_hidden == (target_hidden / num_heads) * num_kv_heads;
    const int32_t head_dim = has_head_mapping ? (target_hidden / num_heads) : 0;
    const int32_t group_size = has_head_mapping ? (num_heads / num_kv_heads) : 1;

    std::vector<float> out(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(target_hidden), 0.0F);
    for (int32_t row = 0; row < hidden; ++row)
    {
        const float* src_row = in_hidden_kv.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(kv_hidden);
        float* dst_row = out.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(target_hidden);
        for (int32_t col = 0; col < target_hidden; ++col)
        {
            int32_t src_col = col % kv_hidden;
            if (has_head_mapping)
            {
                const int32_t q_head = col / head_dim;
                const int32_t offset = col % head_dim;
                const int32_t kv_head = std::min(num_kv_heads - 1, q_head / group_size);
                src_col = kv_head * head_dim + offset;
            }
            dst_row[static_cast<std::size_t>(col)] = src_row[static_cast<std::size_t>(src_col)];
        }
    }
    return out;
}

} // namespace trtf
