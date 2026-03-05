#pragma once

#include "runtime/trt/core/step_state.h"

#include <cstdint>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// RWKV recurrent state implementation.
// Manages 5 state vectors per layer (constant size, no growth):
//   attn_state:  previous token hidden for time-mixing time-shift
//   ff_state:    previous token hidden for channel-mixing time-shift
//   num_state:   WKV numerator accumulator
//   den_state:   WKV denominator accumulator
//   max_state:   WKV max value tracker (numerical stability)
class RwkvStepState final : public IStepState {
public:
    RwkvStepState(int32_t num_layers, int32_t hidden_size);

    const std::vector<std::vector<float>>& attn_state_by_layer() const;
    const std::vector<std::vector<float>>& ff_state_by_layer() const;
    const std::vector<std::vector<float>>& num_state_by_layer() const;
    const std::vector<std::vector<float>>& den_state_by_layer() const;
    const std::vector<std::vector<float>>& max_state_by_layer() const;

    void update_after_step(
        const std::vector<std::vector<float>>& present_attn_by_layer,
        const std::vector<std::vector<float>>& present_ff_by_layer,
        const std::vector<std::vector<float>>& present_num_by_layer,
        const std::vector<std::vector<float>>& present_den_by_layer,
        const std::vector<std::vector<float>>& present_max_by_layer);

    int32_t hidden_size() const { return mHiddenSize; }
    int32_t num_layers() const { return mNumLayers; }

private:
    int32_t mNumLayers;
    int32_t mHiddenSize;
    std::vector<std::vector<float>> mAttnState;  // [layer][hidden_size]
    std::vector<std::vector<float>> mFfState;    // [layer][hidden_size]
    std::vector<std::vector<float>> mNumState;   // [layer][hidden_size]
    std::vector<std::vector<float>> mDenState;   // [layer][hidden_size]
    std::vector<std::vector<float>> mMaxState;   // [layer][hidden_size]
};

#endif // TRTF_HAS_TRT

} // namespace trtf
