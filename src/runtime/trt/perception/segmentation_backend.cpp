#include "runtime/trt/perception/segmentation_backend.h"
#include "runtime/trt/perception/segmentation_preprocess_seam.h"
#include "runtime/trt/perception/segmentation_postprocess_seam.h"

#if TRTF_HAS_TRT

#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

namespace {

std::vector<float> run_segmentation_inference(
    nvinfer1::IExecutionContext& context,
    CudaStream& stream,
    const SegmentationConfig& config,
    const std::vector<float>& pixel_values)
{
    const int32_t out_H = config.output_h;
    const int32_t out_W = config.output_w;

    const auto input_bytes = pixel_values.size() * sizeof(float);
    CudaBuffer input_buf(input_bytes);
    if (!input_buf.ok())
    {
        throw std::runtime_error("Failed to allocate GPU input buffer");
    }

    const auto logits_size = static_cast<std::size_t>(config.num_classes) * out_H * out_W;
    CudaBuffer output_buf(logits_size * sizeof(float));
    if (!output_buf.ok())
    {
        throw std::runtime_error("Failed to allocate GPU output buffer");
    }

    cudaMemcpyAsync(
        input_buf.data(),
        pixel_values.data(),
        input_bytes,
        cudaMemcpyHostToDevice,
        stream.get());

    context.setTensorAddress("pixel_values", input_buf.data());
    context.setTensorAddress("logits", output_buf.data());
    context.enqueueV3(stream.get());

    std::vector<float> logits(logits_size);
    cudaMemcpyAsync(
        logits.data(),
        output_buf.data(),
        logits_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get());
    cudaStreamSynchronize(stream.get());
    return logits;
}

} // namespace

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

SegmentationResult SegmentationBackend::segment_image(const runtime::adapters::io::DecodedImage& image)
{
    SegmentationResult result;
    result.height = mConfig.output_h;
    result.width = mConfig.output_w;
    result.num_classes = mConfig.num_classes;

    const std::vector<float> pixel_values = preprocess_segmentation_image(image, mConfig);
    const std::vector<float> logits = run_segmentation_inference(
        *mContext, mStream, mConfig, pixel_values);
    const SegmentationLogitsShape logits_shape{
        mConfig.num_classes,
        mConfig.output_h,
        mConfig.output_w};
    const SegmentationPostprocessStatus postprocess_status =
        compute_segmentation_class_map_from_logits(logits, logits_shape, result.class_map);
    if (postprocess_status != SegmentationPostprocessStatus::kOk)
    {
        throw std::runtime_error("Segmentation postprocess failed due to invalid logits shape");
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
