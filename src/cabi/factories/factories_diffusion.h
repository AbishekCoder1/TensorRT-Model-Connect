#pragma once

#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "trtf/pipeline.h"

#include <memory>
#include <string>

namespace trtf {

class IGenerationBackend;
class ITokenizer;

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

} // namespace detail

trtf::IPipeline* create_diffusion_pipeline(
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path);

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
