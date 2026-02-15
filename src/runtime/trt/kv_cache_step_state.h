#pragma once

#include "runtime/trt/step_state.h"
#include "runtime/trt/trt_engine_lifecycle.h"

#include <cstdint>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// KV-cache state implementation for standard attention-based decoders.
// Extracts the cache management logic previously inlined in TrtBackendShared::generate().
class KvCacheStepState final : public IStepState {
public:
    explicit KvCacheStepState(const DecoderStepEngine& engine);

    bool prepare_step(int32_t& out_position_id,
                      std::vector<float>& out_attention_mask);

    const std::vector<std::vector<float>>& cache_k_by_layer() const;
    const std::vector<std::vector<float>>& cache_v_by_layer() const;

    void update_after_step(
        const std::vector<std::vector<float>>& present_k_by_layer,
        const std::vector<std::vector<float>>& present_v_by_layer);

private:
    int32_t mCacheStateSize;
    int32_t mMaxCacheLength;
    int32_t mNumLayers;
    bool mIncludeCurrentSlot;
    int32_t mPositionLimit;
    int32_t mCacheLength{0};
    std::vector<std::vector<float>> mCacheK;
    std::vector<std::vector<float>> mCacheV;
};

#endif // TRTF_HAS_TRT

} // namespace trtf
