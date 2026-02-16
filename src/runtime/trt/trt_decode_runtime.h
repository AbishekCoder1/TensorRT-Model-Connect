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

void append_cache_state(
    std::vector<float>& cache, const std::vector<float>& state, int32_t hidden_size, int32_t max_cache_length,
    int32_t write_index);

bool run_decoder_step(const DecoderStepEngine& engine, int32_t token_id, int32_t position_id,
    const std::vector<std::vector<float>>& cache_k_by_layer, const std::vector<std::vector<float>>& cache_v_by_layer,
    const std::vector<float>& attention_mask, std::vector<float>& logits,
    std::vector<std::vector<float>>& present_k_by_layer, std::vector<std::vector<float>>& present_v_by_layer,
    std::string& error,
    const float* input_embed = nullptr, int32_t embed_dim = 0, float use_input_embed = 0.0F);

#endif // TRTF_HAS_TRT

} // namespace trtf
