#include "runtime/trt/reranking_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

RerankingBackend::RerankingBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    RerankingConfig config)
    : mEngine(std::move(engine))
    , mContext(std::move(context))
    , mConfig(std::move(config))
{
}

RerankingBackend::~RerankingBackend() = default;

bool RerankingBackend::is_available() const
{
    return mEngine && mContext;
}

RerankingResult RerankingBackend::rerank(const std::vector<int32_t>& input_ids)
{
    RerankingResult result;

    const auto seq_len = static_cast<std::size_t>(mConfig.max_seq_length);

    // Pad or truncate input_ids to max_seq_length
    std::vector<int32_t> padded_ids(seq_len, 0);
    const auto copy_len = std::min(input_ids.size(), seq_len);
    std::memcpy(padded_ids.data(), input_ids.data(),
                copy_len * sizeof(int32_t));

    // Allocate GPU buffers
    const auto ids_bytes = seq_len * sizeof(int32_t);
    CudaBuffer input_ids_buf(ids_bytes);
    if (!input_ids_buf.ok())
        throw std::runtime_error("Failed to allocate GPU input buffer for reranking");

    // Score output: [seq_len, 1] (we take the last non-padding position)
    const auto score_output_size = seq_len;
    CudaBuffer output_buf(score_output_size * sizeof(float));
    if (!output_buf.ok())
        throw std::runtime_error("Failed to allocate GPU output buffer for reranking");

    // H2D
    cudaMemcpyAsync(input_ids_buf.data(), padded_ids.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());

    // Run TRT
    mContext->setTensorAddress("input_ids", input_ids_buf.data());
    mContext->setTensorAddress("score", output_buf.data());
    mContext->enqueueV3(mStream.get());

    // D2H
    std::vector<float> scores(score_output_size);
    cudaMemcpyAsync(scores.data(), output_buf.data(),
        score_output_size * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    // Take the score at the last non-padding position
    const auto last_pos = copy_len > 0 ? copy_len - 1 : 0;
    result.score = scores[last_pos];

    return result;
}

std::unique_ptr<RerankingBackend> CreateRerankingBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg)
{
    RerankingConfig rerank_cfg;
    rerank_cfg.max_seq_length = cfg.max_cache_length;
    rerank_cfg.hidden_size = cfg.hidden_size;

    return std::make_unique<RerankingBackend>(
        std::move(engine), std::move(context), std::move(rerank_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
