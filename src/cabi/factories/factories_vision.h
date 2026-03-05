#pragma once

#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "trtf/pipeline.h"

#include <memory>
#include <string>

namespace trtf {

class DetectionBackend;
class NeuralOperatorBackend;
class SamBackend;
class SegmentationBackend;

namespace cabi {

#if TRTF_HAS_TRT

namespace detail {

// Internal bridge to construct/configure PipelineImpl without moving its class.
trtf::IPipeline* create_vision_pipeline_impl(
    const std::string& model_id,
    const std::string& backend_name);

void set_segmentation_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::SegmentationBackend> backend);

void set_detection_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::DetectionBackend> backend);

void set_sam_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::SamBackend> backend);

void set_neural_operator_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::NeuralOperatorBackend> backend);

} // namespace detail

trtf::IPipeline* create_segmentation_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path);

trtf::IPipeline* create_detection_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path);

trtf::IPipeline* create_sam_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& bundle_path);

trtf::IPipeline* create_neural_operator_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path);

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
