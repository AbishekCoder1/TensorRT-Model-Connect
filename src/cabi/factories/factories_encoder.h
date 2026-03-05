#pragma once

#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"

#include <memory>
#include <string>

namespace trtf {

class EmbeddingBackend;
class EncoderBackend;
class RerankingBackend;

namespace cabi {

#if TRTF_HAS_TRT

namespace detail {

// Internal bridge to construct/configure PipelineImpl without moving its class.
trtf::IPipeline* create_encoder_pipeline_impl(
    const std::string& model_id,
    std::unique_ptr<trtf::ITokenizer> tokenizer,
    const std::string& backend_name);

void set_bundle_temp_dir(
    trtf::IPipeline* pipeline,
    std::string temp_dir);

void set_encoder_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::EncoderBackend> backend);

void set_embedding_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::EmbeddingBackend> backend);

void set_reranking_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::RerankingBackend> backend);

} // namespace detail

trtf::IPipeline* create_encoder_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_embedding_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

trtf::IPipeline* create_reranking_pipeline(
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
