#pragma once

// RecurrentState: generic SSM/RWKV state manager.
// Replaces MambaStepState and RwkvStepState with a single config-driven class.
//
// HF equivalent: MambaCache (manages conv + ssm state per layer).
//
// Usage:
//   RecurrentState state(num_layers, {{"conv", {d_inner*3}}, {"ssm", {state*d_inner}}}, stream);
//   state.bind_to(module);
//   module.forward(...);
//   state.advance();

#include "trtf/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#if TRTF_HAS_TRT

namespace trtf {

class TrtModule;

class RecurrentState {
public:
    // Specification for one named state tensor per layer.
    struct TensorSpec {
        std::string name;              // Input tensor name prefix, e.g. "conv_state"
        std::vector<int64_t> shape;
        std::string output_prefix;     // Output tensor name prefix, e.g. "present_conv"
                                       // If empty, defaults to "present_" + name.
    };

    // Allocate state buffers for all layers.
    // For Mamba: specs = {{"conv_state", {d_inner*conv_kernel}, "present_conv"},
    //                     {"ssm_state", {state_size*d_inner}, "present_ssm"}}
    // For RWKV:  specs = {{"attn_state", {hidden}, "present_attn"}, ...}
    RecurrentState(int32_t num_layers, std::vector<TensorSpec> specs, cudaStream_t stream);

    // Bind all state tensors to a TrtModule.
    // For each spec and layer i:
    //   input name:  "{spec.name}_{i}"
    //   output name: "{spec.output_prefix}_{i}"  (or "present_{spec.name}_{i}" if output_prefix is empty)
    void bind_to(TrtModule& module);

    // After each step: copy present→state for all tensors (D2D async).
    void advance();

    // Reset all state to zeros.
    void reset();

    int32_t num_layers() const { return num_layers_; }
    const std::vector<TensorSpec>& specs() const { return specs_; }
    bool ok() const;

private:
    std::vector<TensorSpec> specs_;
    // state_[spec_index][layer_index] → DeviceTensor
    std::vector<std::vector<DeviceTensor>> state_;
    std::vector<std::vector<DeviceTensor>> present_;
    int32_t num_layers_{0};
    cudaStream_t stream_{nullptr};
};

} // namespace trtf

#endif // TRTF_HAS_TRT
