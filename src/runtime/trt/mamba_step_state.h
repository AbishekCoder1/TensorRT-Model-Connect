#pragma once

#include "runtime/trt/step_state.h"

#include <cstdint>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Mamba/SSM recurrent state implementation.
// Manages conv_state and ssm_state per layer (constant size, no growth).
class MambaStepState final : public IStepState {
public:
    MambaStepState(int32_t num_layers, int32_t d_inner, int32_t state_size, int32_t conv_kernel);

    const std::vector<std::vector<float>>& conv_state_by_layer() const;
    const std::vector<std::vector<float>>& ssm_state_by_layer() const;

    void update_after_step(
        const std::vector<std::vector<float>>& present_conv_by_layer,
        const std::vector<std::vector<float>>& present_ssm_by_layer);

    int32_t d_inner() const { return mDInner; }
    int32_t state_size() const { return mStateSize; }
    int32_t conv_kernel() const { return mConvKernel; }
    int32_t num_layers() const { return mNumLayers; }

private:
    int32_t mNumLayers;
    int32_t mDInner;
    int32_t mStateSize;
    int32_t mConvKernel;
    std::vector<std::vector<float>> mConvState;  // [layer][d_inner * conv_kernel]
    std::vector<std::vector<float>> mSsmState;   // [layer][d_inner * state_size]
};

#endif // TRTF_HAS_TRT

} // namespace trtf
