#include "runtime/trt/perception/sam_backend.h"
#include "runtime/trt/perception/sam_image_preprocess_seam.h"
#include "runtime/trt/perception/sam_output_selection.h"
#include "runtime/trt/perception/sam_postprocess_seam.h"
#include "runtime/trt/perception/sam_prompt_seam.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

namespace {

int32_t positive_or_default(int32_t value, int32_t default_value)
{
    if (value > 0)
    {
        return value;
    }
    return default_value;
}

void copy_first_three_values_if_present(
    const std::vector<float>& src, std::vector<float>& dst)
{
    if (src.size() < 3)
    {
        return;
    }

    for (std::size_t i = 0; i < 3; ++i)
    {
        dst[i] = src[i];
    }
}

} // namespace

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

bool SamBackend::encode_image(const runtime::adapters::io::DecodedImage& image)
{
    const auto plan = build_sam_image_encode_plan(image, mConfig);
    if (!plan.ok())
    {
        std::cerr << "[trtf] SAM: Failed to preprocess decoded image" << std::endl;
        return false;
    }

    const int32_t H = mConfig.image_size;
    const int32_t W = mConfig.image_size;
    mRescaledW = plan.rescaled_width;
    mRescaledH = plan.rescaled_height;
    mOriginalW = plan.original_width;
    mOriginalH = plan.original_height;

    // Allocate GPU input buffer
    const auto input_bytes = plan.pixel_values.size() * sizeof(float);
    CudaBuffer input_buf(input_bytes);
    if (!input_buf.ok())
    {
        std::cerr << "[trtf] SAM: Failed to allocate GPU input buffer" << std::endl;
        return false;
    }

    // H2D
    cudaMemcpyAsync(input_buf.data(), plan.pixel_values.data(), input_bytes,
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
    return encode_sam_point_embedding(
        x,
        y,
        is_foreground,
        mConfig.image_size,
        mConfig.decoder_hidden_size,
        mConfig.shared_image_pe,
        mConfig.point_embed_fg,
        mConfig.point_embed_bg);
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

    const int32_t num_masks = mConfig.num_mask_outputs;

    // Sparse prompt -> GPU
    CudaBuffer sparse_buf(sparse_prompt.size() * sizeof(float));
    if (!sparse_buf.ok())
        throw std::runtime_error("SAM: Failed to allocate sparse prompt GPU buffer");

    cudaMemcpyAsync(sparse_buf.data(), sparse_prompt.data(),
        sparse_prompt.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());

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

    const auto sparse = build_sam_point_sparse_prompt(
        point_x,
        point_y,
        is_foreground,
        mRescaledW,
        mRescaledH,
        mConfig.image_size,
        mConfig.decoder_hidden_size,
        mConfig.shared_image_pe,
        mConfig.point_embed_fg,
        mConfig.point_embed_bg,
        mConfig.not_a_point_embed);
    return postprocess_sam_result(
        select_sam_multimask_outputs(run_decoder(sparse), mConfig.num_multimask_outputs),
        mConfig.image_size,
        mRescaledW,
        mRescaledH,
        mOriginalW,
        mOriginalH);
}

SamResult SamBackend::segment_box(float x1, float y1, float x2, float y2)
{
    if (!mHasImage)
        throw std::runtime_error("SAM: No image encoded. Call encode_image() first.");

    auto sparse = encode_box(x1, y1, x2, y2);
    return postprocess_sam_result(
        select_sam_multimask_outputs(run_decoder(sparse), mConfig.num_multimask_outputs),
        mConfig.image_size,
        mRescaledW,
        mRescaledH,
        mOriginalW,
        mOriginalH);
}

std::unique_ptr<SamBackend> CreateSamBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_ctx,
    TrtUniquePtr<nvinfer1::ICudaEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> decoder_ctx,
    const FastPathModelConfig& cfg)
{
    SamConfig sam_cfg;
    sam_cfg.image_size = positive_or_default(cfg.input_image_h, 1024);
    sam_cfg.image_embedding_size = positive_or_default(cfg.sam_image_embedding_size, 64);
    sam_cfg.decoder_hidden_size = positive_or_default(cfg.sam_decoder_hidden_size, 256);
    sam_cfg.num_mask_outputs = positive_or_default(cfg.sam_num_mask_outputs, 4);
    sam_cfg.num_multimask_outputs = positive_or_default(cfg.sam_num_multimask_outputs, 3);

    copy_first_three_values_if_present(cfg.seg_image_mean, sam_cfg.image_mean);
    copy_first_three_values_if_present(cfg.seg_image_std, sam_cfg.image_std);

    sam_cfg.point_embed_fg = cfg.sam_point_embed_fg;
    sam_cfg.point_embed_bg = cfg.sam_point_embed_bg;
    sam_cfg.not_a_point_embed = cfg.sam_not_a_point_embed;
    sam_cfg.shared_image_pe = cfg.sam_shared_image_pe;

    return std::make_unique<SamBackend>(
        std::move(encoder_engine), std::move(encoder_ctx),
        std::move(decoder_engine), std::move(decoder_ctx),
        std::move(sam_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
