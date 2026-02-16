#include "runtime/trt/mamba_step_state.h"

namespace trtf {

#if TRTF_HAS_TRT

MambaStepState::MambaStepState(int32_t num_layers, int32_t d_inner, int32_t state_size, int32_t conv_kernel)
    : mNumLayers(num_layers)
    , mDInner(d_inner)
    , mStateSize(state_size)
    , mConvKernel(conv_kernel)
{
    const auto conv_elems = static_cast<std::size_t>(d_inner) * static_cast<std::size_t>(conv_kernel);
    const auto ssm_elems = static_cast<std::size_t>(d_inner) * static_cast<std::size_t>(state_size);
    mConvState.assign(static_cast<std::size_t>(num_layers), std::vector<float>(conv_elems, 0.0F));
    mSsmState.assign(static_cast<std::size_t>(num_layers), std::vector<float>(ssm_elems, 0.0F));
}

const std::vector<std::vector<float>>& MambaStepState::conv_state_by_layer() const
{
    return mConvState;
}

const std::vector<std::vector<float>>& MambaStepState::ssm_state_by_layer() const
{
    return mSsmState;
}

void MambaStepState::update_after_step(
    const std::vector<std::vector<float>>& present_conv_by_layer,
    const std::vector<std::vector<float>>& present_ssm_by_layer)
{
    // Direct replacement: the engine outputs the full updated state.
    for (int32_t layer = 0; layer < mNumLayers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        mConvState[idx] = present_conv_by_layer[idx];
        mSsmState[idx] = present_ssm_by_layer[idx];
    }
}

#endif // TRTF_HAS_TRT

} // namespace trtf
