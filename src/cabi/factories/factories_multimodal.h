#pragma once

#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "trtf/pipeline.h"

#include <memory>
#include <string>

namespace trtf {

class WhisperBackend;

namespace cabi {

#if TRTF_HAS_TRT

namespace detail {

void set_whisper_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::WhisperBackend> backend,
    trtf::MelFilterbank mel_filterbank,
    const trtf::FastPathModelConfig& fp_cfg);

void set_hf_python(
    trtf::IPipeline* pipeline,
    std::string hf_python);

} // namespace detail

trtf::IPipeline* create_whisper_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_vl_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
