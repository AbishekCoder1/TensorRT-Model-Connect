#include "runtime/trt/rwkv_step_state.h"

#include <limits>

namespace trtf {

#if TRTF_HAS_TRT

RwkvStepState::RwkvStepState(int32_t num_layers, int32_t hidden_size)
    : mNumLayers(num_layers)
    , mHiddenSize(hidden_size)
{
    const auto elems = static_cast<std::size_t>(hidden_size);
    const auto layers = static_cast<std::size_t>(num_layers);

    // Initialize all states to zero except max_state which starts at -inf
    // (so the first token's key dominates).
    mAttnState.assign(layers, std::vector<float>(elems, 0.0F));
    mFfState.assign(layers, std::vector<float>(elems, 0.0F));
    mNumState.assign(layers, std::vector<float>(elems, 0.0F));
    mDenState.assign(layers, std::vector<float>(elems, 0.0F));
    mMaxState.assign(layers, std::vector<float>(elems, -1.0e38F));
}

const std::vector<std::vector<float>>& RwkvStepState::attn_state_by_layer() const
{
    return mAttnState;
}

const std::vector<std::vector<float>>& RwkvStepState::ff_state_by_layer() const
{
    return mFfState;
}

const std::vector<std::vector<float>>& RwkvStepState::num_state_by_layer() const
{
    return mNumState;
}

const std::vector<std::vector<float>>& RwkvStepState::den_state_by_layer() const
{
    return mDenState;
}

const std::vector<std::vector<float>>& RwkvStepState::max_state_by_layer() const
{
    return mMaxState;
}

void RwkvStepState::update_after_step(
    const std::vector<std::vector<float>>& present_attn_by_layer,
    const std::vector<std::vector<float>>& present_ff_by_layer,
    const std::vector<std::vector<float>>& present_num_by_layer,
    const std::vector<std::vector<float>>& present_den_by_layer,
    const std::vector<std::vector<float>>& present_max_by_layer)
{
    // Direct replacement: the engine outputs the full updated state.
    for (int32_t layer = 0; layer < mNumLayers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        mAttnState[idx] = present_attn_by_layer[idx];
        mFfState[idx] = present_ff_by_layer[idx];
        mNumState[idx] = present_num_by_layer[idx];
        mDenState[idx] = present_den_by_layer[idx];
        mMaxState[idx] = present_max_by_layer[idx];
    }
}

#endif // TRTF_HAS_TRT

} // namespace trtf
