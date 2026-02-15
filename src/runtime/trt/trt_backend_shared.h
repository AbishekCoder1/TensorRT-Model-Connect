#pragma once

#include "trtf/backend.h"
#include "trtf/model.h"
#include "trtf/tokenizer.h"
#include "runtime/trt/step_state.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_graph_builder.h"
#include "model/trt_model_definition.h"

#include <functional>
#include <memory>
#include <string>

namespace trtf {

#if TRTF_HAS_TRT

// Function type for creating a decoder step engine from weights.
using DecoderStepEngineFactory = std::function<std::unique_ptr<DecoderStepEngine>(
    const TrtDecoderDefinition&, TrtLogger&)>;

// Creates a TRT backend using the provided engine factory.
std::unique_ptr<IGenerationBackend> CreateTrtBackendWithFactory(
    const ITokenizer& tokenizer, const DecoderModel& model, DecoderStepEngineFactory factory);

// Creates a TRT backend using an ITrtGraphBuilder.
std::unique_ptr<IGenerationBackend> CreateTrtBackendWithBuilder(
    const ITokenizer& tokenizer, const DecoderModel& model, ITrtGraphBuilder& builder);

// Creates a TRT backend from a pre-built engine (fast path — no weight loading).
std::unique_ptr<IGenerationBackend> CreateTrtBackendFromEngine(
    std::unique_ptr<DecoderStepEngine> engine);

#endif // TRTF_HAS_TRT

} // namespace trtf
