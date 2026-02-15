#pragma once

#include "trtf/backend.h"
#include "trtf/model.h"
#include "trtf/tokenizer.h"
#include "runtime/trt/step_state.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "model/trt_model_definition.h"

#include <memory>
#include <string>

namespace trtf {

#if TRTF_HAS_TRT

class IModelRuntime;

// Creates a TRT backend using an IModelRuntime (family-owned forward pass).
std::unique_ptr<IGenerationBackend> CreateTrtBackendWithRuntime(
    const ITokenizer& tokenizer, const DecoderModel& model, IModelRuntime& runtime);

// Creates a TRT backend from a pre-built engine (fast path — no weight loading).
std::unique_ptr<IGenerationBackend> CreateTrtBackendFromEngine(
    std::unique_ptr<DecoderStepEngine> engine);

#endif // TRTF_HAS_TRT

} // namespace trtf
