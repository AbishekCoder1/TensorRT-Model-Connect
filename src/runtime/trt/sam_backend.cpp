#include "runtime/trt/sam_backend.h"

#if TRTF_HAS_TRT

#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

SamBackend::SamBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_ctx,
    TrtUniquePtr<nvinfer1::ICudaEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> decoder_ctx,
    SamConfig config)
    : mEncoderEngine(std::move(encoder_engine))
    , mEncoderCtx(std::move(encoder_ctx))
    , mDecoderEngine(std::move(decoder_engine))
    , mDecoderCtx(std::move(decoder_ctx))
    , mConfig(std::move(config))
    , mImageEmbeddings(static_cast<std::size_t>(mConfig.decoder_hidden_size)
          * mConfig.image_embedding_size * mConfig.image_embedding_size * sizeof(float))
{
}

SamBackend::~SamBackend() = default;

bool SamBackend::is_available() const
{
    return mEncoderEngine && mEncoderCtx && mDecoderEngine && mDecoderCtx;
}

bool SamBackend::encode_image(const std::string& image_path)
{
    const int32_t H = mConfig.image_size;
    const int32_t W = mConfig.image_size;

    // Load image via stb_image
    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &img_w, &img_h, &img_c, 3);
    if (raw == nullptr)
    {
        std::cerr << "[trtf] SAM: Failed to load image: " << image_path << std::endl;
        return false;
    }

    // Resize and normalize to CHW float32
    std::vector<float> pixel_values(static_cast<std::size_t>(3) * H * W);
    for (int32_t y = 0; y < H; ++y)
    {
        for (int32_t x = 0; x < W; ++x)
        {
            const float src_y = static_cast<float>(y) * static_cast<float>(img_h) / static_cast<float>(H);
            const float src_x = static_cast<float>(x) * static_cast<float>(img_w) / static_cast<float>(W);
            const int32_t y0 = std::min(static_cast<int32_t>(src_y), img_h - 1);
            const int32_t x0 = std::min(static_cast<int32_t>(src_x), img_w - 1);
            const auto src_idx = static_cast<std::size_t>((y0 * img_w + x0) * 3);
            for (int32_t c = 0; c < 3; ++c)
            {
                float val = static_cast<float>(raw[src_idx + c]) / 255.0F;
                val = (val - mConfig.image_mean[c]) / mConfig.image_std[c];
                pixel_values[static_cast<std::size_t>(c) * H * W +
                             static_cast<std::size_t>(y) * W + x] = val;
            }
        }
    }
    stbi_image_free(raw);

    // Allocate GPU input buffer
    const auto input_bytes = pixel_values.size() * sizeof(float);
    CudaBuffer input_buf(input_bytes);
    if (!input_buf.ok())
    {
        std::cerr << "[trtf] SAM: Failed to allocate GPU input buffer" << std::endl;
        return false;
    }

    // H2D
    cudaMemcpyAsync(input_buf.data(), pixel_values.data(), input_bytes,
        cudaMemcpyHostToDevice, mStream.get());

    // Run encoder
    mEncoderCtx->setTensorAddress("pixel_values", input_buf.data());
    mEncoderCtx->setTensorAddress("image_embeddings", mImageEmbeddings.data());
    if (!mEncoderCtx->enqueueV3(mStream.get()))
    {
        std::cerr << "[trtf] SAM: Encoder execution failed" << std::endl;
        return false;
    }
    cudaStreamSynchronize(mStream.get());

    mHasImage = true;
    std::cerr << "[trtf] SAM: Image encoded (" << H << "x" << W << ")" << std::endl;
    return true;
}

std::vector<float> SamBackend::encode_point(float x, float y, bool is_foreground)
{
    const int32_t dim = mConfig.decoder_hidden_size;

    // Build sparse prompt: single point embedding [1, dim]
    std::vector<float> sparse(static_cast<std::size_t>(dim), 0.0F);

    // Use point embeddings from config (fg=index 1, bg=index 0)
    const auto& point_embed = is_foreground
        ? mConfig.point_embed_fg : mConfig.point_embed_bg;

    if (static_cast<int32_t>(point_embed.size()) >= dim)
    {
        for (int32_t i = 0; i < dim; ++i)
        {
            sparse[i] = point_embed[i];
        }
    }

    // Add positional encoding for the point location
    // Simple sinusoidal encoding based on normalized coordinates
    const float scale = 2.0F * 3.14159265F;
    for (int32_t i = 0; i < dim / 2 && i < dim; ++i)
    {
        float freq = std::pow(2.0F, -static_cast<float>(i) / static_cast<float>(dim / 2));
        sparse[i] += std::sin(x * scale * freq);
        if (i + dim / 2 < dim)
            sparse[i + dim / 2] += std::sin(y * scale * freq);
    }

    return sparse;
}

std::vector<float> SamBackend::encode_box(float x1, float y1, float x2, float y2)
{
    // A box is encoded as two points: top-left (type=2) and bottom-right (type=3)
    // For simplicity, we average them into a single point prompt
    const float cx = (x1 + x2) / 2.0F;
    const float cy = (y1 + y2) / 2.0F;
    return encode_point(cx, cy, true);
}

SamResult SamBackend::run_decoder(const std::vector<float>& sparse_prompt)
{
    SamResult result;
    result.num_masks = mConfig.num_mask_outputs;
    result.mask_height = 256;
    result.mask_width = 256;

    const int32_t dim = mConfig.decoder_hidden_size;
    const int32_t num_masks = mConfig.num_mask_outputs;
    const int32_t embed_size = mConfig.image_embedding_size;

    // Sparse prompt -> GPU
    CudaBuffer sparse_buf(sparse_prompt.size() * sizeof(float));
    if (!sparse_buf.ok())
        throw std::runtime_error("SAM: Failed to allocate sparse prompt GPU buffer");

    cudaMemcpyAsync(sparse_buf.data(), sparse_prompt.data(),
        sparse_prompt.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());

    // Dense prompt: zeros [1, dim, embed_size, embed_size]
    const auto dense_size = static_cast<std::size_t>(dim) * embed_size * embed_size;
    CudaBuffer dense_buf(dense_size * sizeof(float));
    if (!dense_buf.ok())
        throw std::runtime_error("SAM: Failed to allocate dense prompt GPU buffer");
    cudaMemsetAsync(dense_buf.data(), 0, dense_size * sizeof(float), mStream.get());

    // Output buffers
    const auto masks_size = static_cast<std::size_t>(num_masks) * 256 * 256;
    CudaBuffer masks_buf(masks_size * sizeof(float));
    CudaBuffer iou_buf(static_cast<std::size_t>(num_masks) * sizeof(float));
    if (!masks_buf.ok() || !iou_buf.ok())
        throw std::runtime_error("SAM: Failed to allocate output GPU buffers");

    // Set tensor addresses
    mDecoderCtx->setTensorAddress("image_embeddings", mImageEmbeddings.data());
    mDecoderCtx->setTensorAddress("sparse_prompt_embeddings", sparse_buf.data());
    mDecoderCtx->setTensorAddress("masks", masks_buf.data());
    mDecoderCtx->setTensorAddress("iou_scores", iou_buf.data());

    if (!mDecoderCtx->enqueueV3(mStream.get()))
    {
        throw std::runtime_error("SAM: Mask decoder execution failed");
    }

    // D2H
    result.masks.resize(masks_size);
    result.iou_scores.resize(static_cast<std::size_t>(num_masks));
    cudaMemcpyAsync(result.masks.data(), masks_buf.data(),
        masks_size * sizeof(float), cudaMemcpyDeviceToHost, mStream.get());
    cudaMemcpyAsync(result.iou_scores.data(), iou_buf.data(),
        static_cast<std::size_t>(num_masks) * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    return result;
}

SamResult SamBackend::segment_point(float point_x, float point_y, bool is_foreground)
{
    if (!mHasImage)
        throw std::runtime_error("SAM: No image encoded. Call encode_image() first.");

    auto sparse = encode_point(point_x, point_y, is_foreground);
    return run_decoder(sparse);
}

SamResult SamBackend::segment_box(float x1, float y1, float x2, float y2)
{
    if (!mHasImage)
        throw std::runtime_error("SAM: No image encoded. Call encode_image() first.");

    auto sparse = encode_box(x1, y1, x2, y2);
    return run_decoder(sparse);
}

std::unique_ptr<SamBackend> CreateSamBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_ctx,
    TrtUniquePtr<nvinfer1::ICudaEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> decoder_ctx,
    const FastPathModelConfig& cfg)
{
    SamConfig sam_cfg;
    sam_cfg.image_size = cfg.input_image_h > 0 ? cfg.input_image_h : 1024;
    sam_cfg.image_embedding_size = cfg.sam_image_embedding_size > 0
        ? cfg.sam_image_embedding_size : 64;
    sam_cfg.decoder_hidden_size = cfg.sam_decoder_hidden_size > 0
        ? cfg.sam_decoder_hidden_size : 256;
    sam_cfg.num_mask_outputs = cfg.sam_num_mask_outputs > 0
        ? cfg.sam_num_mask_outputs : 4;
    sam_cfg.num_multimask_outputs = cfg.sam_num_multimask_outputs > 0
        ? cfg.sam_num_multimask_outputs : 3;

    if (cfg.seg_image_mean.size() >= 3)
    {
        for (int i = 0; i < 3; ++i) sam_cfg.image_mean[i] = cfg.seg_image_mean[i];
    }
    if (cfg.seg_image_std.size() >= 3)
    {
        for (int i = 0; i < 3; ++i) sam_cfg.image_std[i] = cfg.seg_image_std[i];
    }

    sam_cfg.point_embed_fg = cfg.sam_point_embed_fg;
    sam_cfg.point_embed_bg = cfg.sam_point_embed_bg;

    return std::make_unique<SamBackend>(
        std::move(encoder_engine), std::move(encoder_ctx),
        std::move(decoder_engine), std::move(decoder_ctx),
        std::move(sam_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
