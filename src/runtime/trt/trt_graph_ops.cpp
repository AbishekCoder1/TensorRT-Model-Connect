#include "runtime/trt/trt_graph_ops.h"

#include <cmath>

namespace trtf {

#if TRTF_HAS_TRT

nvinfer1::Dims make_dims_1d(int32_t d0)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = d0;
    return dims;
}

nvinfer1::Dims make_dims_2d(int32_t d0, int32_t d1)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 2;
    dims.d[0] = d0;
    dims.d[1] = d1;
    return dims;
}

nvinfer1::Dims make_dims_3d(int32_t d0, int32_t d1, int32_t d2)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = d0;
    dims.d[1] = d1;
    dims.d[2] = d2;
    return dims;
}

std::string layer_tensor_name(const char* stem, int32_t layer)
{
    return std::string(stem) + "_" + std::to_string(layer);
}

nvinfer1::ITensor* add_constant_tensor(
    nvinfer1::INetworkDefinition& network, nvinfer1::Dims dims, const std::vector<float>& values)
{
    const nvinfer1::Weights weights{
        nvinfer1::DataType::kFLOAT, values.data(), static_cast<int64_t>(values.size())};
    auto* layer = network.addConstant(dims, weights);
    if (layer == nullptr)
    {
        return nullptr;
    }
    return layer->getOutput(0);
}

nvinfer1::ITensor* add_matmul_rhs_constant(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& lhs,
    int32_t lhs_width, int32_t rhs_width, const std::vector<float>& rhs_weights)
{
    nvinfer1::ITensor* rhs = add_constant_tensor(network, make_dims_2d(lhs_width, rhs_width), rhs_weights);
    if (rhs == nullptr)
    {
        return nullptr;
    }

    auto* matmul
        = network.addMatrixMultiply(lhs, nvinfer1::MatrixOperation::kNONE, *rhs, nvinfer1::MatrixOperation::kNONE);
    if (matmul == nullptr)
    {
        return nullptr;
    }
    return matmul->getOutput(0);
}

nvinfer1::ITensor* add_bias_sum(
    nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input, int32_t width, const std::vector<float>& bias)
{
    nvinfer1::ITensor* bias_tensor = add_constant_tensor(network, make_dims_2d(1, width), bias);
    if (bias_tensor == nullptr)
    {
        return nullptr;
    }

    auto* sum = network.addElementWise(input, *bias_tensor, nvinfer1::ElementWiseOperation::kSUM);
    if (sum == nullptr)
    {
        return nullptr;
    }
    return sum->getOutput(0);
}

nvinfer1::ITensor* add_rms_norm(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input, int32_t hidden_size,
    const std::vector<float>& gamma, nvinfer1::ITensor& eps_tensor)
{
    auto* sq = network.addElementWise(input, input, nvinfer1::ElementWiseOperation::kPROD);
    if (sq == nullptr)
    {
        return nullptr;
    }

    auto* mean = network.addReduce(*sq->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 1, true);
    if (mean == nullptr)
    {
        return nullptr;
    }

    auto* denom_in = network.addElementWise(*mean->getOutput(0), eps_tensor, nvinfer1::ElementWiseOperation::kSUM);
    if (denom_in == nullptr)
    {
        return nullptr;
    }

    auto* sqrt_layer = network.addUnary(*denom_in->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    if (sqrt_layer == nullptr)
    {
        return nullptr;
    }

    auto* recip = network.addUnary(*sqrt_layer->getOutput(0), nvinfer1::UnaryOperation::kRECIP);
    if (recip == nullptr)
    {
        return nullptr;
    }

    auto* normalized = network.addElementWise(input, *recip->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (normalized == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* gamma_tensor = add_constant_tensor(network, make_dims_2d(1, hidden_size), gamma);
    if (gamma_tensor == nullptr)
    {
        return nullptr;
    }

    auto* scaled
        = network.addElementWise(*normalized->getOutput(0), *gamma_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled == nullptr)
    {
        return nullptr;
    }
    return scaled->getOutput(0);
}

nvinfer1::ITensor* add_rms_norm_per_head(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input,
    int32_t num_heads, int32_t head_dim, const std::vector<float>& gamma, nvinfer1::ITensor& eps_tensor)
{
    if (num_heads <= 0 || head_dim <= 0 || gamma.size() != static_cast<std::size_t>(num_heads) * static_cast<std::size_t>(head_dim))
    {
        return nullptr;
    }

    auto* reshape_in = network.addShuffle(input);
    if (reshape_in == nullptr)
    {
        return nullptr;
    }
    reshape_in->setReshapeDimensions(make_dims_2d(num_heads, head_dim));

    nvinfer1::ITensor* reshaped = reshape_in->getOutput(0);
    auto* sq = network.addElementWise(*reshaped, *reshaped, nvinfer1::ElementWiseOperation::kPROD);
    if (sq == nullptr)
    {
        return nullptr;
    }

    auto* mean = network.addReduce(*sq->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 1, true);
    if (mean == nullptr)
    {
        return nullptr;
    }

    auto* denom_in = network.addElementWise(*mean->getOutput(0), eps_tensor, nvinfer1::ElementWiseOperation::kSUM);
    if (denom_in == nullptr)
    {
        return nullptr;
    }

    auto* sqrt_layer = network.addUnary(*denom_in->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    if (sqrt_layer == nullptr)
    {
        return nullptr;
    }

    auto* recip = network.addUnary(*sqrt_layer->getOutput(0), nvinfer1::UnaryOperation::kRECIP);
    if (recip == nullptr)
    {
        return nullptr;
    }

    auto* normalized = network.addElementWise(*reshaped, *recip->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (normalized == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* gamma_tensor = add_constant_tensor(network, make_dims_2d(num_heads, head_dim), gamma);
    if (gamma_tensor == nullptr)
    {
        return nullptr;
    }

    auto* scaled
        = network.addElementWise(*normalized->getOutput(0), *gamma_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled == nullptr)
    {
        return nullptr;
    }

    auto* reshape_out = network.addShuffle(*scaled->getOutput(0));
    if (reshape_out == nullptr)
    {
        return nullptr;
    }
    reshape_out->setReshapeDimensions(make_dims_2d(1, num_heads * head_dim));
    return reshape_out->getOutput(0);
}

std::vector<float> make_rope_table(int32_t max_cache_length, int32_t hidden_size, int32_t num_attention_heads,
    float rope_theta, bool cosine)
{
    std::vector<float> table(
        static_cast<std::size_t>(max_cache_length) * static_cast<std::size_t>(hidden_size), cosine ? 1.0F : 0.0F);
    if (max_cache_length <= 0 || hidden_size <= 0 || num_attention_heads <= 0 || hidden_size % num_attention_heads != 0)
    {
        return table;
    }

    const int32_t head_dim = hidden_size / num_attention_heads;
    const int32_t half_head_dim = head_dim / 2;
    if (half_head_dim <= 0 || rope_theta <= 0.0F)
    {
        return table;
    }

    for (int32_t pos = 0; pos < max_cache_length; ++pos)
    {
        for (int32_t head = 0; head < num_attention_heads; ++head)
        {
            // HF Qwen3 rotary embedding builds emb = cat(freqs, freqs) across the head dimension.
            const int32_t rope_dims = half_head_dim * 2;
            for (int32_t dim = 0; dim < rope_dims; ++dim)
            {
                const int32_t freq_idx = dim % half_head_dim;
                const float exponent = (2.0F * static_cast<float>(freq_idx)) / static_cast<float>(head_dim);
                const float inv_freq = std::pow(rope_theta, -exponent);
                const float angle = static_cast<float>(pos) * inv_freq;
                const float value = cosine ? std::cos(angle) : std::sin(angle);
                const std::size_t offset = static_cast<std::size_t>(pos) * static_cast<std::size_t>(hidden_size)
                    + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim)
                    + static_cast<std::size_t>(dim);
                table[offset] = value;
            }
        }
    }

    return table;
}

std::vector<float> make_rotate_half_matrix(int32_t hidden_size, int32_t num_attention_heads)
{
    std::vector<float> matrix(
        static_cast<std::size_t>(hidden_size) * static_cast<std::size_t>(hidden_size), 0.0F);
    if (hidden_size <= 0 || num_attention_heads <= 0 || hidden_size % num_attention_heads != 0)
    {
        for (int32_t i = 0; i < hidden_size; ++i)
        {
            matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden_size) + static_cast<std::size_t>(i)]
                = 1.0F;
        }
        return matrix;
    }

    const int32_t head_dim = hidden_size / num_attention_heads;
    const int32_t half_head_dim = head_dim / 2;

    for (int32_t head = 0; head < num_attention_heads; ++head)
    {
        const int32_t base = head * head_dim;
        for (int32_t i = 0; i < half_head_dim; ++i)
        {
            const int32_t out_left = base + i;
            const int32_t out_right = base + half_head_dim + i;

            // For row-vector matmul (x * M), set coefficients so output is rotate_half(x).
            matrix[static_cast<std::size_t>(out_left) * static_cast<std::size_t>(hidden_size)
                + static_cast<std::size_t>(out_right)] = 1.0F;
            matrix[static_cast<std::size_t>(out_right) * static_cast<std::size_t>(hidden_size)
                + static_cast<std::size_t>(out_left)] = -1.0F;
        }

        if (head_dim % 2 != 0)
        {
            const int32_t tail = base + (2 * half_head_dim);
            matrix[static_cast<std::size_t>(tail) * static_cast<std::size_t>(hidden_size)
                + static_cast<std::size_t>(tail)] = 1.0F;
        }
    }
    return matrix;
}

nvinfer1::ITensor* add_apply_rope(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input,
    nvinfer1::ITensor& position_id, nvinfer1::ITensor& cos_table, nvinfer1::ITensor& sin_table,
    nvinfer1::ITensor& rotate_half_matrix)
{
    auto* cos_gather = network.addGather(cos_table, position_id, 0);
    auto* sin_gather = network.addGather(sin_table, position_id, 0);
    if (cos_gather == nullptr || sin_gather == nullptr)
    {
        return nullptr;
    }

    auto* rotated = network.addMatrixMultiply(
        input, nvinfer1::MatrixOperation::kNONE, rotate_half_matrix, nvinfer1::MatrixOperation::kNONE);
    if (rotated == nullptr)
    {
        return nullptr;
    }

    auto* x_cos = network.addElementWise(
        input, *cos_gather->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (x_cos == nullptr)
    {
        return nullptr;
    }

    auto* rot_sin = network.addElementWise(
        *rotated->getOutput(0), *sin_gather->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (rot_sin == nullptr)
    {
        return nullptr;
    }

    auto* sum = network.addElementWise(
        *x_cos->getOutput(0), *rot_sin->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    if (sum == nullptr)
    {
        return nullptr;
    }
    return sum->getOutput(0);
}

#endif // TRTF_HAS_TRT

} // namespace trtf
