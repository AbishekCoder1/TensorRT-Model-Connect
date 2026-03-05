#pragma once

#include "runtime/trt/core/trt_engine_lifecycle.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

int32_t select_argmax_token(const std::vector<float>& logits);

/// Sample a token using temperature scaling and top-k filtering.
/// Divides logits by temperature, zeros out all but top_k, applies softmax,
/// then samples from the resulting probability distribution.
/// rng_state is updated in place (simple xorshift64).
int32_t sample_token_topk(const std::vector<float>& logits,
                          float temperature, int32_t top_k,
                          uint64_t& rng_state);

std::vector<int32_t> select_topk_tokens(const std::vector<float>& logits, int32_t k);

std::vector<float> build_attention_mask(int32_t cache_length, int32_t max_cache_length, bool include_current_slot);

#endif // TRTF_HAS_TRT

} // namespace trtf
