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

    // input_embed and use_input_embed buffers (zeros for text-only path)
    // CudaBuffer has no default constructor — allocate conditionally
    const auto embed_buf_size = seq_len * hidden * sizeof(float);
    const auto use_embed_buf_size = seq_len * sizeof(float);
    CudaBuffer input_embed_buf(mConfig.has_input_embed ? embed_buf_size : 1);
    CudaBuffer use_input_embed_buf(mConfig.has_input_embed ? use_embed_buf_size : 1);
    if (mConfig.has_input_embed)
    {
        if (!input_embed_buf.ok() || !use_input_embed_buf.ok())
            throw std::runtime_error("Failed to allocate GPU input_embed buffers");
        cudaMemsetAsync(input_embed_buf.data(), 0, embed_buf_size, mStream.get());
        cudaMemsetAsync(use_input_embed_buf.data(), 0, use_embed_buf_size, mStream.get());
    }

    // H2D
    cudaMemcpyAsync(input_ids_buf.data(), padded_ids.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(attn_mask_buf.data(), attn_mask.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());

    // Run TRT
    mContext->setTensorAddress("input_ids", input_ids_buf.data());
    mContext->setTensorAddress("attention_mask", attn_mask_buf.data());
    if (mConfig.has_input_embed)
    {
        mContext->setTensorAddress("input_embed", input_embed_buf.data());
        mContext->setTensorAddress("use_input_embed", use_input_embed_buf.data());
    }
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

void EmbeddingBackend::set_vision_engine(
    std::unique_ptr<VisionStepEngine> vision_engine,
    VLPreprocessConfig vl_config)
{
    mVisionEngine = std::move(vision_engine);
    mVLConfig = std::move(vl_config);
}

bool EmbeddingBackend::has_vision() const
{
    return mVisionEngine != nullptr;
}

EmbeddingResult EmbeddingBackend::embed_with_image(
    const std::vector<int32_t>& input_ids,
    const std::string& image_path)
{
    if (!mVisionEngine)
    {
        // No vision engine; fall back to text-only
        return embed(input_ids);
    }

    // 1. Preprocess image
    auto preprocessed = load_and_preprocess_image(image_path, mVLConfig);
    if (!preprocessed.ok)
    {
        throw std::runtime_error("Failed to preprocess image: " + image_path);
    }

    // 2. Run vision encoder
    std::vector<float> image_features;
    std::string vision_error;
    if (!run_vision_encoder(
            *mVisionEngine,
            preprocessed.pixel_values.data(),
            preprocessed.pixel_values.size() * sizeof(float),
            image_features, vision_error))
    {
        throw std::runtime_error("Vision encoder failed: " + vision_error);
    }

    const auto num_patches = static_cast<std::size_t>(mVisionEngine->num_output_features);
    const auto feat_dim = static_cast<std::size_t>(mVisionEngine->feature_dim);

    // 3. Check if engine has input_embed support
    if (!mConfig.has_input_embed)
    {
        // No input_embed bypass — fall back to text-only embedding.
        // The vision features cannot be injected into the text backbone.
        std::cerr << "[trtf] Warning: embedding engine lacks input_embed support; "
                  << "falling back to text-only" << std::endl;
        return embed(input_ids);
    }

    const auto seq_len = static_cast<std::size_t>(mConfig.max_seq_length);
    const auto hidden = static_cast<std::size_t>(mConfig.hidden_size);

    // 4. Pad or truncate input_ids
    std::vector<int32_t> padded_ids(seq_len, 0);
    const auto actual_len = std::min(input_ids.size(), seq_len);
    std::memcpy(padded_ids.data(), input_ids.data(),
                actual_len * sizeof(int32_t));

    // 5. Build attention mask
    std::vector<int32_t> attn_mask(seq_len, 0);
    for (std::size_t i = 0; i < actual_len; ++i)
        attn_mask[i] = 1;

    // 6. Build input_embed and use_input_embed arrays
    std::vector<float> input_embed(seq_len * hidden, 0.0F);
    std::vector<float> use_input_embed(seq_len, 0.0F);

    std::size_t feat_idx = 0;
    for (std::size_t i = 0; i < actual_len && feat_idx < num_patches; ++i)
    {
        if (padded_ids[i] == mConfig.img_context_token_id)
        {
            // Replace this position with vision features
            use_input_embed[i] = 1.0F;
            if (feat_idx < num_patches)
            {
                std::memcpy(&input_embed[i * hidden],
                            &image_features[feat_idx * feat_dim],
                            std::min(hidden, feat_dim) * sizeof(float));
                ++feat_idx;
            }
        }
    }

    // 7. Allocate GPU buffers
    const auto ids_bytes = seq_len * sizeof(int32_t);
    CudaBuffer input_ids_buf(ids_bytes);
    CudaBuffer attn_mask_buf(ids_bytes);
    CudaBuffer input_embed_buf(seq_len * hidden * sizeof(float));
    CudaBuffer use_input_embed_buf(seq_len * sizeof(float));
    CudaBuffer output_buf(seq_len * hidden * sizeof(float));

    if (!input_ids_buf.ok() || !attn_mask_buf.ok() || !input_embed_buf.ok() ||
        !use_input_embed_buf.ok() || !output_buf.ok())
    {
        throw std::runtime_error("Failed to allocate GPU buffers for VL embedding");
    }

    // H2D
    cudaMemcpyAsync(input_ids_buf.data(), padded_ids.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(attn_mask_buf.data(), attn_mask.data(), ids_bytes,
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(input_embed_buf.data(), input_embed.data(),
        seq_len * hidden * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(use_input_embed_buf.data(), use_input_embed.data(),
        seq_len * sizeof(float), cudaMemcpyHostToDevice, mStream.get());

    // Run TRT
    mContext->setTensorAddress("input_ids", input_ids_buf.data());
    mContext->setTensorAddress("attention_mask", attn_mask_buf.data());
    mContext->setTensorAddress("input_embed", input_embed_buf.data());
    mContext->setTensorAddress("use_input_embed", use_input_embed_buf.data());
    mContext->setTensorAddress("hidden_states", output_buf.data());
    mContext->enqueueV3(mStream.get());

    // D2H
    const auto output_size = seq_len * hidden;
    std::vector<float> all_hidden(output_size);
    cudaMemcpyAsync(all_hidden.data(), output_buf.data(),
        output_size * sizeof(float), cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    // Mean pooling + L2 normalization
    EmbeddingResult result;
    result.embedding_dim = mConfig.embedding_dim;
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
        result.embedding[d] *= inv_len;

    float norm_sq = 0.0F;
    for (std::size_t d = 0; d < hidden; ++d)
        norm_sq += result.embedding[d] * result.embedding[d];

    const float inv_norm = (norm_sq > 1e-12F) ? (1.0F / std::sqrt(norm_sq)) : 1.0F;
    for (std::size_t d = 0; d < hidden; ++d)
        result.embedding[d] *= inv_norm;

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

    // Detect input_embed support from engine tensor names
    if (engine)
    {
        const int num_tensors = engine->getNbIOTensors();
        for (int i = 0; i < num_tensors; ++i)
        {
            const char* name = engine->getIOTensorName(i);
            if (name != nullptr && std::string(name) == "input_embed")
            {
                emb_cfg.has_input_embed = true;
                break;
            }
        }
    }

    return std::make_unique<EmbeddingBackend>(
        std::move(engine), std::move(context), std::move(emb_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
