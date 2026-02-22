#include "runtime/trt/segmentation_backend.h"

#if TRTF_HAS_TRT

#include "stb_image.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

SegmentationBackend::SegmentationBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    SegmentationConfig config)
    : mEngine(std::move(engine))
    , mContext(std::move(context))
    , mConfig(std::move(config))
{
}

SegmentationBackend::~SegmentationBackend() = default;

bool SegmentationBackend::is_available() const
{
    return mEngine && mContext;
}

SegmentationResult SegmentationBackend::segment_image(const std::string& image_path)
{
    SegmentationResult result;
    result.height = mConfig.output_h;
    result.width = mConfig.output_w;
    result.num_classes = mConfig.num_classes;

    const int32_t H = mConfig.input_image_h;
    const int32_t W = mConfig.input_image_w;
    const int32_t out_H = mConfig.output_h;
    const int32_t out_W = mConfig.output_w;

    // Load image via stb_image
    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &img_w, &img_h, &img_c, 3);
    if (raw == nullptr)
    {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // Resize to H x W and normalize to CHW float32
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

    // Allocate GPU buffers
    const auto input_bytes = pixel_values.size() * sizeof(float);
    CudaBuffer input_buf(input_bytes);
    if (!input_buf.ok())
        throw std::runtime_error("Failed to allocate GPU input buffer");

    const auto logits_size = static_cast<std::size_t>(mConfig.num_classes) * out_H * out_W;
    CudaBuffer output_buf(logits_size * sizeof(float));
    if (!output_buf.ok())
        throw std::runtime_error("Failed to allocate GPU output buffer");

    // H2D
    cudaMemcpyAsync(input_buf.data(), pixel_values.data(), input_bytes,
        cudaMemcpyHostToDevice, mStream.get());

    // Run TRT
    mContext->setTensorAddress("pixel_values", input_buf.data());
    mContext->setTensorAddress("logits", output_buf.data());
    mContext->enqueueV3(mStream.get());

    // D2H
    std::vector<float> logits(logits_size);
    cudaMemcpyAsync(logits.data(), output_buf.data(), logits_size * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    // Argmax over classes
    result.class_map.resize(static_cast<std::size_t>(out_H) * out_W);
    for (int32_t y = 0; y < out_H; ++y)
    {
        for (int32_t x = 0; x < out_W; ++x)
        {
            int32_t best_class = 0;
            float best_val = -1e30F;
            for (int32_t c = 0; c < mConfig.num_classes; ++c)
            {
                const float val = logits[
                    static_cast<std::size_t>(c) * out_H * out_W +
                    static_cast<std::size_t>(y) * out_W + x];
                if (val > best_val)
                {
                    best_val = val;
                    best_class = c;
                }
            }
            result.class_map[static_cast<std::size_t>(y) * out_W + x] = best_class;
        }
    }

    return result;
}

std::unique_ptr<SegmentationBackend> CreateSegmentationBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg)
{
    SegmentationConfig seg_cfg;
    seg_cfg.num_classes = cfg.num_classes;
    seg_cfg.input_image_h = cfg.input_image_h;
    seg_cfg.input_image_w = cfg.input_image_w;
    seg_cfg.output_h = cfg.output_h;
    seg_cfg.output_w = cfg.output_w;
    if (cfg.seg_image_mean.size() >= 3)
    {
        for (int i = 0; i < 3; ++i) seg_cfg.image_mean[i] = cfg.seg_image_mean[i];
    }
    if (cfg.seg_image_std.size() >= 3)
    {
        for (int i = 0; i < 3; ++i) seg_cfg.image_std[i] = cfg.seg_image_std[i];
    }

    return std::make_unique<SegmentationBackend>(
        std::move(engine), std::move(context), std::move(seg_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
