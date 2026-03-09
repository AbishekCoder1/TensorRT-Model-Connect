#pragma once

#include "runtime/trt/core/generation_backend.h"
#include "runtime/trt/recurrent/rwkv_decode_runtime.h"

#include <memory>

namespace trtf {

#if TRTF_HAS_TRT

// Creates an RWKV TRT backend from a pre-built engine (bundle load path).
std::unique_ptr<IGenerationBackend> CreateRwkvBackendFromEngine(
    std::unique_ptr<RwkvStepEngine> engine);

#endif // TRTF_HAS_TRT

} // namespace trtf
