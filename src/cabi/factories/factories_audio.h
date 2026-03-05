#pragma once

#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "trtf/pipeline.h"

#include <memory>
#include <string>

namespace trtf {

class BarkBackend;
class IGenerationBackend;
class ITokenizer;
class MagpieTTSBackend;
class OmniBackend;
class SpeechToSpeechBackend;

namespace cabi {

#if TRTF_HAS_TRT

namespace detail {

// Internal bridge to construct/configure PipelineImpl without moving its class.
trtf::IPipeline* create_pipeline_impl(
    const std::string& model_id,
    std::unique_ptr<trtf::ITokenizer> tokenizer,
    std::unique_ptr<trtf::IGenerationBackend> backend,
    const std::string& backend_name);

void set_bundle_temp_dir(trtf::IPipeline* pipeline, std::string dir);

void set_bark_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::BarkBackend> backend);

void set_magpie_tts_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::MagpieTTSBackend> backend);

void set_omni_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::OmniBackend> backend);

void set_speech_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::SpeechToSpeechBackend> backend);

} // namespace detail

trtf::IPipeline* create_magpie_tts_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_bark_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_omni_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_speech_pipeline(
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
