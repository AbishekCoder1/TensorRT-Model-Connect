#include "runtime/trt/embedding_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

EmbeddingBackend::EmbeddingBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    EmbeddingConfig config)
    : mEngine(std::move(engine))
    , mContext(std::move(context))
    , mConfig(std::move(config))
{
}

EmbeddingBackend::~EmbeddingBackend() = default;

bool EmbeddingBackend::is_available() const
{
    return mEngine && mContext;
}

EmbeddingResult EmbeddingBackend::embed(const std::vector<int32_t>& input_ids)
{
    EmbeddingResult result;
    result.embedding_dim = mConfig.embedding_dim;

    const auto seq_len = static_cast<std::size_t>(mConfig.max_seq_length);
    const auto hidden = static_cast<std::size_t>(mConfig.hidden_size);

    // Pad or truncate input_ids to max_seq_length
    std::vector<int32_t> padded_ids(seq_len, 0);
    const auto actual_len = std::min(input_ids.size(), seq_len);
    std::memcpy(padded_ids.data(), input_ids.data(),
                actual_len * sizeof(int32_t));

    // Build attention mask: 1 for real tokens, 0 for padding
    std::vector<int32_t> attn_mask(seq_len, 0);
    for (std::size_t i = 0; i < actual_len; ++i)
        attn_mask[i] = 1;

    // Allocate GPU buffers
    const auto ids_bytes = seq_len * sizeof(int32_t);
    CudaBuffer input_ids_buf(ids_bytes);
    if (!input_ids_buf.ok())
        throw std::runtime_error("Failed to allocate GPU input buffer for embedding");

    CudaBuffer attn_mask_buf(ids_bytes);
    if (!attn_mask_buf.ok())
        throw std::runtime_error("Failed to allocate GPU attention mask buffer for embedding");

    const auto output_size = seq_len * hidden;
    CudaBuffer output_buf(output_size * sizeof(float));
    if (!output_buf.ok())
        throw std::runtime_error("Failed to allocate GPU output buffer for embedding");

    // H2D
    cudaMemcpyAsync(input_ids_buf.data(), padded_ids.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(attn_mask_buf.data(), attn_mask.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());

    // Run TRT
    mContext->setTensorAddress("input_ids", input_ids_buf.data());
    mContext->setTensorAddress("attention_mask", attn_mask_buf.data());
    mContext->setTensorAddress("hidden_states", output_buf.data());
    mContext->enqueueV3(mStream.get());

    // D2H
    std::vector<float> all_hidden(output_size);
    cudaMemcpyAsync(all_hidden.data(), output_buf.data(),
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    // Mean pooling over non-padding positions
    result.embedding.resize(hidden, 0.0F);
    const auto pool_len = actual_len > 0 ? actual_len : 1;

    for (std::size_t pos = 0; pos < pool_len; ++pos)
    {
        for (std::size_t d = 0; d < hidden; ++d)
        {
            result.embedding[d] += all_hidden[pos * hidden + d];
        }
    }

    const float inv_len = 1.0F / static_cast<float>(pool_len);
    for (std::size_t d = 0; d < hidden; ++d)
    {
        result.embedding[d] *= inv_len;
    }

    // L2 normalization
    float norm_sq = 0.0F;
    for (std::size_t d = 0; d < hidden; ++d)
    {
        norm_sq += result.embedding[d] * result.embedding[d];
    }
    const float inv_norm = (norm_sq > 1e-12F) ? (1.0F / std::sqrt(norm_sq)) : 1.0F;
    for (std::size_t d = 0; d < hidden; ++d)
    {
        result.embedding[d] *= inv_norm;
    }

    return result;
}

std::unique_ptr<EmbeddingBackend> CreateEmbeddingBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg)
{
    EmbeddingConfig emb_cfg;
    emb_cfg.max_seq_length = cfg.max_cache_length;
    emb_cfg.hidden_size = cfg.hidden_size;
    emb_cfg.embedding_dim = (cfg.embedding_dim > 0) ? cfg.embedding_dim : cfg.hidden_size;

    return std::make_unique<EmbeddingBackend>(
        std::move(engine), std::move(context), std::move(emb_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
