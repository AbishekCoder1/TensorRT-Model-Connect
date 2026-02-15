#pragma once

// Full header — includes TRT types needed by the autoregressive loop and
// factory helpers.  Family registrations should include model_runtime_fwd.h
// instead to avoid pulling in TRT/CUDA headers.

#include "runtime/trt/model_runtime_fwd.h"
#include "runtime/trt/step_state.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "model/trt_model_definition.h"

#include <functional>
#include <memory>
#include <string>

namespace trtf {

#if TRTF_HAS_TRT

// Function type for creating a decoder step engine from weights.
using EngineFactory = std::function<std::unique_ptr<DecoderStepEngine>(
    const TrtDecoderDefinition&, TrtLogger&)>;

// Custom engine builder + standard KV-cache state/step.
std::unique_ptr<IModelRuntime> CreateKvCacheRuntime(EngineFactory engine_factory);

#endif // TRTF_HAS_TRT

} // namespace trtf
