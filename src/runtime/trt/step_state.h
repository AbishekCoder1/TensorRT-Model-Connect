#pragma once

namespace trtf {

// Opaque base for per-step state during autoregressive generation.
// KV-cache models use KvCacheStepState; Mamba/SSM models can provide
// a recurrent state implementation; hybrid models can combine both.
class IStepState {
public:
    virtual ~IStepState() = default;
};

} // namespace trtf
