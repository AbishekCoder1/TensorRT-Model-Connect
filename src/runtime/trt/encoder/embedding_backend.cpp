#include "runtime/trt/encoder/embedding_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

namespace {

struct TextInputs {
    std::vector<int32_t> padded_ids;
    std::vector<int32_t> attention_mask;
    std::size_t actual_len{0};
};

TextInputs build_text_inputs(const std::vector<int32_t>& input_ids, std::size_t seq_len)
{
    TextInputs text_inputs;
    text_inputs.padded_ids.assign(seq_len, 0);
    text_inputs.attention_mask.assign(seq_len, 0);

    text_inputs.actual_len = std::min(input_ids.size(), seq_len);
    std::memcpy(
        text_inputs.padded_ids.data(),
        input_ids.data(),
        text_inputs.actual_len * sizeof(int32_t));

    for (std::size_t i = 0; i < text_inputs.actual_len; ++i)
    {
        text_inputs.attention_mask[i] = 1;
    }
    return text_inputs;
}

void ensure_buffer_ok(const CudaBuffer& buffer, const char* message)
{
    if (!buffer.ok())
    {
        throw std::runtime_error(message);
    }
}

void maybe_zero_text_only_embed_buffers(
    bool has_input_embed,
    CudaBuffer& input_embed_buf,
    CudaBuffer& use_input_embed_buf,
    std::size_t embed_buf_size,
    std::size_t use_embed_buf_size,
    CudaStream& stream)
{
    if (!has_input_embed)
    {
        return;
    }

    if (!input_embed_buf.ok() || !use_input_embed_buf.ok())
    {
        throw std::runtime_error("Failed to allocate GPU input_embed buffers");
    }
    cudaMemsetAsync(input_embed_buf.data(), 0, embed_buf_size, stream.get());
    cudaMemsetAsync(use_input_embed_buf.data(), 0, use_embed_buf_size, stream.get());
}

void copy_text_inputs_to_device(
    const TextInputs& text_inputs,
    CudaBuffer& input_ids_buf,
    CudaBuffer& attn_mask_buf,
    CudaStream& stream)
{
    const auto ids_bytes = text_inputs.padded_ids.size() * sizeof(int32_t);
    cudaMemcpyAsync(
        input_ids_buf.data(), text_inputs.padded_ids.data(), ids_bytes,
        cudaMemcpyHostToDevice, stream.get());
    cudaMemcpyAsync(
        attn_mask_buf.data(), text_inputs.attention_mask.data(), ids_bytes,
        cudaMemcpyHostToDevice, stream.get());
}

void bind_optional_embed_tensors(
    nvinfer1::IExecutionContext& context,
    const void* input_embed,
    const void* use_input_embed)
{
    if (input_embed == nullptr || use_input_embed == nullptr)
    {
        return;
    }

    context.setTensorAddress("input_embed", const_cast<void*>(input_embed));
    context.setTensorAddress("use_input_embed", const_cast<void*>(use_input_embed));
}

std::vector<float> run_embedding_network(
    nvinfer1::IExecutionContext& context,
    CudaStream& stream,
    CudaBuffer& input_ids_buf,
    CudaBuffer& attn_mask_buf,
    CudaBuffer& output_buf,
    std::size_t output_size,
    const void* input_embed,
    const void* use_input_embed)
{
    context.setTensorAddress("input_ids", input_ids_buf.data());
    context.setTensorAddress("attention_mask", attn_mask_buf.data());
    bind_optional_embed_tensors(context, input_embed, use_input_embed);
    context.setTensorAddress("hidden_states", output_buf.data());
    context.enqueueV3(stream.get());

    std::vector<float> all_hidden(output_size);
    cudaMemcpyAsync(
        all_hidden.data(), output_buf.data(), output_size * sizeof(float),
        cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());
    return all_hidden;
}

EmbeddingResult pool_and_normalize(
    const std::vector<float>& all_hidden,
    std::size_t hidden,
    std::size_t actual_len,
    int32_t embedding_dim)
{
    EmbeddingResult result;
    result.embedding_dim = embedding_dim;
    result.embedding.assign(hidden, 0.0F);

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

    float norm_sq = 0.0F;
    for (float value : result.embedding)
    {
        norm_sq += value * value;
    }
    const float inv_norm = norm_sq > 1e-12F ? (1.0F / std::sqrt(norm_sq)) : 1.0F;
    for (float& value : result.embedding)
    {
        value *= inv_norm;
    }

    return result;
}

void inject_vision_features(
    const EmbeddingConfig& config,
    const std::vector<int32_t>& padded_ids,
    std::size_t actual_len,
    std::size_t hidden,
    const std::vector<float>& image_features,
    std::size_t feat_dim,
    std::size_t num_patches,
    std::vector<float>& input_embed,
    std::vector<float>& use_input_embed)
{
    std::size_t feat_idx = 0;
    const std::size_t copy_width = std::min(hidden, feat_dim);

    for (std::size_t i = 0; i < actual_len && feat_idx < num_patches; ++i)
    {
        if (padded_ids[i] != config.img_context_token_id)
        {
            continue;
        }

        use_input_embed[i] = 1.0F;
        std::memcpy(
            &input_embed[i * hidden],
            &image_features[feat_idx * feat_dim],
            copy_width * sizeof(float));
        ++feat_idx;
    }
}

std::vector<float> run_vision_or_throw(
    const VisionStepEngine& vision_engine,
    const std::string& image_path,
    const VLPreprocessConfig& vl_config)
{
    auto preprocessed = load_and_preprocess_image(image_path, vl_config);
    if (!preprocessed.ok)
    {
        throw std::runtime_error("Failed to preprocess image: " + image_path);
    }

    std::vector<float> image_features;
    std::string vision_error;
    if (!run_vision_encoder(
            vision_engine,
            preprocessed.pixel_values.data(),
            preprocessed.pixel_values.size() * sizeof(float),
            image_features,
            vision_error))
    {
        throw std::runtime_error("Vision encoder failed: " + vision_error);
    }
    return image_features;
}

} // namespace

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
    const auto seq_len = static_cast<std::size_t>(mConfig.max_seq_length);
    const auto hidden = static_cast<std::size_t>(mConfig.hidden_size);
    const auto output_size = seq_len * hidden;
    const auto ids_bytes = seq_len * sizeof(int32_t);

    const TextInputs text_inputs = build_text_inputs(input_ids, seq_len);

    CudaBuffer input_ids_buf(ids_bytes);
    CudaBuffer attn_mask_buf(ids_bytes);
    CudaBuffer output_buf(output_size * sizeof(float));
    ensure_buffer_ok(input_ids_buf, "Failed to allocate GPU input buffer for embedding");
    ensure_buffer_ok(attn_mask_buf, "Failed to allocate GPU attention mask buffer for embedding");
    ensure_buffer_ok(output_buf, "Failed to allocate GPU output buffer for embedding");

    const auto embed_buf_size = output_size * sizeof(float);
    const auto use_embed_buf_size = seq_len * sizeof(float);
    CudaBuffer input_embed_buf(mConfig.has_input_embed ? embed_buf_size : 1);
    CudaBuffer use_input_embed_buf(mConfig.has_input_embed ? use_embed_buf_size : 1);
    maybe_zero_text_only_embed_buffers(
        mConfig.has_input_embed,
        input_embed_buf,
        use_input_embed_buf,
        embed_buf_size,
        use_embed_buf_size,
        mStream);

    copy_text_inputs_to_device(text_inputs, input_ids_buf, attn_mask_buf, mStream);

    const void* input_embed_ptr = mConfig.has_input_embed ? input_embed_buf.data() : nullptr;
    const void* use_input_embed_ptr =
        mConfig.has_input_embed ? use_input_embed_buf.data() : nullptr;
    const std::vector<float> all_hidden = run_embedding_network(
        *mContext,
        mStream,
        input_ids_buf,
        attn_mask_buf,
        output_buf,
        output_size,
        input_embed_ptr,
        use_input_embed_ptr);

    return pool_and_normalize(
        all_hidden,
        hidden,
        text_inputs.actual_len,
        mConfig.embedding_dim);
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
        return embed(input_ids);
    }

    if (!mConfig.has_input_embed)
    {
        std::cerr << "[trtf] Warning: embedding engine lacks input_embed support; "
                  << "falling back to text-only" << std::endl;
        return embed(input_ids);
    }

    const auto seq_len = static_cast<std::size_t>(mConfig.max_seq_length);
    const auto hidden = static_cast<std::size_t>(mConfig.hidden_size);
    const auto output_size = seq_len * hidden;
    const auto ids_bytes = seq_len * sizeof(int32_t);

    const std::vector<float> image_features =
        run_vision_or_throw(*mVisionEngine, image_path, mVLConfig);
    const auto feat_dim = static_cast<std::size_t>(mVisionEngine->feature_dim);
    const auto num_patches = static_cast<std::size_t>(mVisionEngine->num_output_features);

    const TextInputs text_inputs = build_text_inputs(input_ids, seq_len);
    std::vector<float> input_embed(output_size, 0.0F);
    std::vector<float> use_input_embed(seq_len, 0.0F);
    inject_vision_features(
        mConfig,
        text_inputs.padded_ids,
        text_inputs.actual_len,
        hidden,
        image_features,
        feat_dim,
        num_patches,
        input_embed,
        use_input_embed);

    CudaBuffer input_ids_buf(ids_bytes);
    CudaBuffer attn_mask_buf(ids_bytes);
    CudaBuffer input_embed_buf(output_size * sizeof(float));
    CudaBuffer use_input_embed_buf(seq_len * sizeof(float));
    CudaBuffer output_buf(output_size * sizeof(float));
    if (!input_ids_buf.ok() || !attn_mask_buf.ok() || !input_embed_buf.ok() ||
        !use_input_embed_buf.ok() || !output_buf.ok())
    {
        throw std::runtime_error("Failed to allocate GPU buffers for VL embedding");
    }

    copy_text_inputs_to_device(text_inputs, input_ids_buf, attn_mask_buf, mStream);
    cudaMemcpyAsync(
        input_embed_buf.data(), input_embed.data(), output_size * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(
        use_input_embed_buf.data(), use_input_embed.data(), seq_len * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());

    const std::vector<float> all_hidden = run_embedding_network(
        *mContext,
        mStream,
        input_ids_buf,
        attn_mask_buf,
        output_buf,
        output_size,
        input_embed_buf.data(),
        use_input_embed_buf.data());

    return pool_and_normalize(
        all_hidden,
        hidden,
        text_inputs.actual_len,
        mConfig.embedding_dim);
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
