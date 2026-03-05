#include "cabi/factories/factories_vision.h"

#include "runtime/trt/perception/detection_backend.h"
#include "runtime/trt/perception/neural_operator_backend.h"
#include "runtime/trt/perception/sam_backend.h"
#include "runtime/trt/perception/segmentation_backend.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

trtf::IPipeline* create_segmentation_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path)
{
    (void) bundle_path;

    auto seg_backend = trtf::CreateSegmentationBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!seg_backend || !seg_backend->is_available())
    {
        throw std::runtime_error("Failed to create segmentation backend from bundle engine");
    }

    auto* pipeline = detail::create_vision_pipeline_impl(
        model_id, "trt_segmentation");
    detail::set_segmentation_backend(pipeline, std::move(seg_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_segmentation, strategy=segmentation)" << std::endl;
    return pipeline;
}

trtf::IPipeline* create_detection_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path)
{
    (void) bundle_path;

    auto det_backend = trtf::CreateDetectionBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!det_backend || !det_backend->is_available())
    {
        throw std::runtime_error("Failed to create detection backend from bundle engine");
    }

    auto* pipeline = detail::create_vision_pipeline_impl(
        model_id, "trt_detection");
    detail::set_detection_backend(pipeline, std::move(det_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_detection, strategy=object_detection)" << std::endl;
    return pipeline;
}

trtf::IPipeline* create_sam_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& bundle_path)
{
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> decoder_engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> decoder_ctx;
    if (sections.vision_plan_data != nullptr && !sections.vision_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing SAM mask decoder TRT engine ("
                  << sections.vision_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

        decoder_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.vision_plan_data->data(), sections.vision_plan_data->size()));
        if (!decoder_engine)
        {
            throw std::runtime_error("Failed to deserialize SAM mask decoder engine: " + bundle_path);
        }
        decoder_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            decoder_engine->createExecutionContext());
        if (!decoder_ctx)
        {
            throw std::runtime_error("Failed to create SAM mask decoder execution context");
        }
    }
    else
    {
        throw std::runtime_error(
            "SAM bundle missing vision_engine_plan (mask decoder): " + bundle_path);
    }

    auto sam_backend = trtf::CreateSamBackend(
        std::move(trt_engine), std::move(exec_ctx),
        std::move(decoder_engine), std::move(decoder_ctx), fp_cfg);
    if (!sam_backend || !sam_backend->is_available())
    {
        throw std::runtime_error("Failed to create SAM backend from bundle engines");
    }

    auto* pipeline = detail::create_vision_pipeline_impl(
        model_id, "trt_sam");
    detail::set_sam_backend(pipeline, std::move(sam_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_sam, strategy=prompted_segmentation)" << std::endl;
    return pipeline;
}

trtf::IPipeline* create_neural_operator_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path)
{
    (void) bundle_path;

    auto no_backend = trtf::CreateNeuralOperatorBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!no_backend || !no_backend->is_available())
    {
        throw std::runtime_error("Failed to create neural operator backend from bundle engine");
    }

    auto* pipeline = detail::create_vision_pipeline_impl(
        model_id, "trt_neural_operator");
    detail::set_neural_operator_backend(pipeline, std::move(no_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_neural_operator, strategy=neural_operator)" << std::endl;
    return pipeline;
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
