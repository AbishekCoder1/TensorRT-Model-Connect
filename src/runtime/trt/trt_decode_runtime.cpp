#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>

namespace trtf {

#if TRTF_HAS_TRT

int32_t select_argmax_token(const std::vector<float>& logits)
{
    if (logits.empty())
    {
        return 0;
    }
    const auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<int32_t>(std::distance(logits.begin(), it));
}

std::vector<int32_t> select_topk_tokens(const std::vector<float>& logits, int32_t k)
{
    if (logits.empty() || k <= 0)
    {
        return {};
    }

    const int32_t capped = std::min(k, static_cast<int32_t>(logits.size()));
    std::vector<int32_t> indices(logits.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = static_cast<int32_t>(i);
    }

    std::partial_sort(indices.begin(), indices.begin() + capped, indices.end(),
        [&](int32_t a, int32_t b) { return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)]; });
    indices.resize(static_cast<std::size_t>(capped));
    return indices;
}

std::vector<float> build_attention_mask(int32_t cache_length, int32_t max_cache_length, bool include_current_slot)
{
    const int32_t width = max_cache_length + (include_current_slot ? 1 : 0);
    if (width <= 0)
    {
        return {};
    }

    std::vector<float> mask(static_cast<std::size_t>(width), kMaskedScore);
    const int32_t valid = std::max(0, std::min(cache_length, max_cache_length));
    for (int32_t i = 0; i < valid; ++i)
    {
        mask[static_cast<std::size_t>(i)] = 0.0F;
    }

    if (include_current_slot)
    {
        mask.back() = 0.0F;
    }
    else if (valid <= 0)
    {
        mask[0] = 0.0F;
    }

    return mask;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
