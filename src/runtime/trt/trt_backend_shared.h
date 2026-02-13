#pragma once

#include "trtf/backend.h"
#include "trtf/model.h"
#include "trtf/tokenizer.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "model/trt_model_definition.h"

#include <memory>
#include <string>

namespace trtf {

#if TRTF_HAS_TRT

// Function type for creating a decoder step engine from weights.
using DecoderStepEngineFactory = std::unique_ptr<DecoderStepEngine>(*)(const TrtDecoderDefinition&, TrtLogger&);

// Creates a TRT backend using the provided engine factory.
std::unique_ptr<IGenerationBackend> CreateTrtBackendWithFactory(
    const ITokenizer& tokenizer, const DecoderModel& model, DecoderStepEngineFactory factory);

#endif // TRTF_HAS_TRT

} // namespace trtf
