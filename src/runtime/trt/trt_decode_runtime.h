#pragma once

#include "runtime/trt/trt_engine_lifecycle.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

int32_t select_argmax_token(const std::vector<float>& logits);

std::vector<int32_t> select_topk_tokens(const std::vector<float>& logits, int32_t k);

std::vector<float> build_attention_mask(int32_t cache_length, int32_t max_cache_length, bool include_current_slot);

#endif // TRTF_HAS_TRT

} // namespace trtf
