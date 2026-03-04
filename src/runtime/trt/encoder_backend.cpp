#include "runtime/trt/encoder_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

EncoderBackend::EncoderBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    EncoderConfig config)
    : mEngine(std::move(engine))
    , mContext(std::move(context))
    , mConfig(std::move(config))
{
}

EncoderBackend::~EncoderBackend() = default;

bool EncoderBackend::is_available() const
{
    return mEngine && mContext;
}

EncoderResult EncoderBackend::encode(const std::vector<int32_t>& input_ids)
{
    // Default token_type_ids: all zeros (single segment)
    std::vector<int32_t> token_type_ids(
        static_cast<std::size_t>(mConfig.max_seq_length), 0);
    return encode(input_ids, token_type_ids);
}

EncoderResult EncoderBackend::encode(
    const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& token_type_ids)
{
    EncoderResult result;
    result.seq_length = mConfig.max_seq_length;
    result.hidden_size = mConfig.hidden_size;

    const auto seq_len = static_cast<std::size_t>(mConfig.max_seq_length);
    const auto hidden = static_cast<std::size_t>(mConfig.hidden_size);

    // Pad or truncate input_ids to max_seq_length
    std::vector<int32_t> padded_ids(seq_len, 0);
    const auto copy_len = std::min(input_ids.size(), seq_len);
    std::memcpy(padded_ids.data(), input_ids.data(),
                copy_len * sizeof(int32_t));

    // Pad or truncate token_type_ids
    std::vector<int32_t> padded_tt(seq_len, 0);
    const auto tt_copy_len = std::min(token_type_ids.size(), seq_len);
    std::memcpy(padded_tt.data(), token_type_ids.data(),
                tt_copy_len * sizeof(int32_t));

    // Build attention mask: 1 for real tokens, 0 for padding
    std::vector<int32_t> attn_mask(seq_len, 0);
    for (std::size_t i = 0; i < copy_len; ++i)
        attn_mask[i] = 1;

    // Allocate GPU buffers
    const auto ids_bytes = seq_len * sizeof(int32_t);
    CudaBuffer input_ids_buf(ids_bytes);
    CudaBuffer token_type_buf(ids_bytes);
    CudaBuffer attn_mask_buf(ids_bytes);

    if (!input_ids_buf.ok() || !token_type_buf.ok() || !attn_mask_buf.ok())
        throw std::runtime_error("Failed to allocate GPU input buffers for encoder");

    const auto output_size = seq_len * hidden;
    CudaBuffer output_buf(output_size * sizeof(float));
    if (!output_buf.ok())
        throw std::runtime_error("Failed to allocate GPU output buffer for encoder");

    // H2D
    cudaMemcpyAsync(input_ids_buf.data(), padded_ids.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(token_type_buf.data(), padded_tt.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(attn_mask_buf.data(), attn_mask.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());

    // Run TRT
    mContext->setTensorAddress("input_ids", input_ids_buf.data());
    mContext->setTensorAddress("token_type_ids", token_type_buf.data());
    mContext->setTensorAddress("attention_mask", attn_mask_buf.data());
    mContext->setTensorAddress("hidden_states", output_buf.data());
    mContext->enqueueV3(mStream.get());

    // D2H
    result.hidden_states.resize(output_size);
    cudaMemcpyAsync(result.hidden_states.data(), output_buf.data(),
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    return result;
}

std::unique_ptr<EncoderBackend> CreateEncoderBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg)
{
    EncoderConfig enc_cfg;
    enc_cfg.max_seq_length = cfg.max_cache_length;  // reuse max_cache_length for seq_len
    enc_cfg.hidden_size = cfg.hidden_size;
    enc_cfg.type_vocab_size = cfg.type_vocab_size;

    return std::make_unique<EncoderBackend>(
        std::move(engine), std::move(context), std::move(enc_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
