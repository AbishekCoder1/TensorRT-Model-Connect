#pragma once

#include "cabi/bundle/bundle_helpers.h"

#include <string>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

struct BackendRegistryDispatchContext {
    trtf::TrtUniquePtr<nvinfer1::IRuntime>* runtime_ptr{nullptr};
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine>* trt_engine{nullptr};
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext>* exec_ctx{nullptr};
    const trtf::FastPathModelConfig* fp_cfg{nullptr};
    const trtf::BundleSections* sections{nullptr};
    const std::string* model_id{nullptr};
    const std::string* hf_python{nullptr};
    const std::string* bundle_path{nullptr};
};

void register_builtin_backend_factories_once();

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
