#pragma once

#include "trtf/backend.h"
#include "runtime/trt/trt_engine_lifecycle.h"

#include <memory>

namespace trtf {

#if TRTF_HAS_TRT

// Creates a TRT backend from a pre-built engine (bundle load path).
std::unique_ptr<IGenerationBackend> CreateTrtBackendFromEngine(
    std::unique_ptr<DecoderStepEngine> engine);

#endif // TRTF_HAS_TRT

} // namespace trtf
