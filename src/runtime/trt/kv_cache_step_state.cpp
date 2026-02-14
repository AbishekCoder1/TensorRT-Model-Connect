#include "runtime/trt/kv_cache_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>

namespace trtf {

#if TRTF_HAS_TRT

KvCacheStepState::KvCacheStepState(const DecoderStepEngine& engine)
    : mCacheStateSize(engine.cache_state_size)
    , mMaxCacheLength(engine.max_cache_length)
    , mNumLayers(engine.num_layers)
    , mIncludeCurrentSlot(engine.requires_position_input)
    , mPositionLimit(mIncludeCurrentSlot ? mMaxCacheLength : std::max(mMaxCacheLength - 1, 0))
{
    const std::size_t cache_elems
        = static_cast<std::size_t>(mMaxCacheLength) * static_cast<std::size_t>(mCacheStateSize);
    mCacheK.assign(static_cast<std::size_t>(mNumLayers), std::vector<float>(cache_elems, 0.0F));
    mCacheV.assign(static_cast<std::size_t>(mNumLayers), std::vector<float>(cache_elems, 0.0F));
}

bool KvCacheStepState::prepare_step(int32_t& out_position_id,
                                     std::vector<float>& out_attention_mask)
{
    out_position_id = std::min(mCacheLength, mPositionLimit);
    out_attention_mask = build_attention_mask(mCacheLength, mMaxCacheLength, mIncludeCurrentSlot);
    return true;
}

const std::vector<std::vector<float>>& KvCacheStepState::cache_k_by_layer() const
{
    return mCacheK;
}

const std::vector<std::vector<float>>& KvCacheStepState::cache_v_by_layer() const
{
    return mCacheV;
}

void KvCacheStepState::update_after_step(
    const std::vector<std::vector<float>>& present_k_by_layer,
    const std::vector<std::vector<float>>& present_v_by_layer)
{
    for (int32_t layer = 0; layer < mNumLayers; ++layer)
    {
        const std::size_t idx = static_cast<std::size_t>(layer);
        append_cache_state(mCacheK[idx], present_k_by_layer[idx], mCacheStateSize, mMaxCacheLength, mCacheLength);
        append_cache_state(mCacheV[idx], present_v_by_layer[idx], mCacheStateSize, mMaxCacheLength, mCacheLength);
    }
    mCacheLength = std::min(mCacheLength + 1, mMaxCacheLength);
}

#endif // TRTF_HAS_TRT

} // namespace trtf
