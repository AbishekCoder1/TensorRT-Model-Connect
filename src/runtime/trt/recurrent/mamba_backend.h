#pragma once

#include "runtime/trt/core/generation_backend.h"
#include "runtime/trt/recurrent/mamba_decode_runtime.h"

#include <memory>

namespace trtf {

#if TRTF_HAS_TRT

// Creates a Mamba/SSM TRT backend from a pre-built engine (bundle load path).
std::unique_ptr<IGenerationBackend> CreateMambaBackendFromEngine(
    std::unique_ptr<MambaStepEngine> engine);

#endif // TRTF_HAS_TRT

} // namespace trtf
