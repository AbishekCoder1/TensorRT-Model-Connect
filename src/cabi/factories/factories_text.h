#pragma once

#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "trtf/backend.h"
#include "trtf/pipeline.h"

#include <memory>
#include <string>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

namespace detail {

// Internal bridge to construct/configure PipelineImpl without moving its class.
trtf::IPipeline* create_text_pipeline_impl(
    const std::string& model_id,
    trtf::TokenizerResult tok,
    std::unique_ptr<trtf::IGenerationBackend> backend,
    const std::string& backend_name);

} // namespace detail

trtf::IPipeline* create_mamba_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_rwkv_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_hybrid_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_decoder_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
