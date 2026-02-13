#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#endif

namespace trtf {

#if TRTF_HAS_TRT

nvinfer1::Dims make_dims_1d(int32_t d0);
nvinfer1::Dims make_dims_2d(int32_t d0, int32_t d1);
nvinfer1::Dims make_dims_3d(int32_t d0, int32_t d1, int32_t d2);

std::string layer_tensor_name(const char* stem, int32_t layer);

nvinfer1::ITensor* add_constant_tensor(
    nvinfer1::INetworkDefinition& network, nvinfer1::Dims dims, const std::vector<float>& values);

nvinfer1::ITensor* add_matmul_rhs_constant(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& lhs,
    int32_t lhs_width, int32_t rhs_width, const std::vector<float>& rhs_weights);

nvinfer1::ITensor* add_bias_sum(
    nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input, int32_t width, const std::vector<float>& bias);

nvinfer1::ITensor* add_rms_norm(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input, int32_t hidden_size,
    const std::vector<float>& gamma, nvinfer1::ITensor& eps_tensor);

nvinfer1::ITensor* add_rms_norm_per_head(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input,
    int32_t num_heads, int32_t head_dim, const std::vector<float>& gamma, nvinfer1::ITensor& eps_tensor);

std::vector<float> make_rope_table(int32_t max_cache_length, int32_t hidden_size, int32_t num_attention_heads,
    float rope_theta, bool cosine);

std::vector<float> make_rotate_half_matrix(int32_t hidden_size, int32_t num_attention_heads);

nvinfer1::ITensor* add_apply_rope(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input,
    nvinfer1::ITensor& position_id, nvinfer1::ITensor& cos_table, nvinfer1::ITensor& sin_table,
    nvinfer1::ITensor& rotate_half_matrix);

#endif // TRTF_HAS_TRT

} // namespace trtf
