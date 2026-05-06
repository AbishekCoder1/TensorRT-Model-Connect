#pragma once

namespace trtmc {

// Opaque base for per-step state during autoregressive generation.
// KV-cache models use DeviceKvCache (device-resident, not an IStepState
// subclass); Mamba/SSM models use MambaStepState; hybrid models can
// combine both.
class IStepState {
public:
    virtual ~IStepState() = default;
};

} // namespace trtmc
