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
#include "trtf/runtime/inference_state.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtf {

class TrtModule;

class RecurrentState : public IInferenceState {
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

    // --- IInferenceState overrides ---
    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return -1; }  // unbounded
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "recurrent"; }
    bool ok() const override;

    // --- RecurrentState-specific methods ---
    const std::vector<TensorSpec>& specs() const { return specs_; }

private:
    std::vector<TensorSpec> specs_;
    // state_[spec_index][layer_index] -> DeviceTensor
    std::vector<std::vector<DeviceTensor>> state_;
    std::vector<std::vector<DeviceTensor>> present_;
    int32_t num_layers_{0};
    int32_t position_{0};
    cudaStream_t stream_{nullptr};
};

} // namespace trtf
