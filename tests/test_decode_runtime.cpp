// Unit tests for src/runtime/trt/trt_decode_runtime.cpp CPU-side functions.
// Tests argmax, topk, attention mask, and cache append.
// Guarded by TRTF_HAS_TRT — skips gracefully when TRT not available.

#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#if TRTF_HAS_TRT

namespace {

bool test_argmax_basic()
{
    const std::vector<float> logits = {0.1F, 0.9F, 0.3F};
    const int32_t result = trtf::select_argmax_token(logits);
    if (result != 1)
    {
        std::cerr << "argmax_basic: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_argmax_single()
{
    const std::vector<float> logits = {5.0F};
    const int32_t result = trtf::select_argmax_token(logits);
    if (result != 0)
    {
        std::cerr << "argmax_single: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_argmax_empty()
{
    const std::vector<float> logits;
    const int32_t result = trtf::select_argmax_token(logits);
    if (result != 0)
    {
        std::cerr << "argmax_empty: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_argmax_all_negative()
{
    const std::vector<float> logits = {-3.0F, -1.0F, -5.0F, -2.0F};
    const int32_t result = trtf::select_argmax_token(logits);
    if (result != 1)
    {
        std::cerr << "argmax_all_negative: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_argmax_tie()
{
    // std::max_element returns first occurrence
    const std::vector<float> logits = {1.0F, 5.0F, 5.0F, 2.0F};
    const int32_t result = trtf::select_argmax_token(logits);
    if (result != 1)
    {
        std::cerr << "argmax_tie: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_topk_basic()
{
    const std::vector<float> logits = {0.1F, 0.9F, 0.3F, 0.7F};
    const auto result = trtf::select_topk_tokens(logits, 2);
    if (result.size() != 2)
    {
        std::cerr << "topk_basic: size=" << result.size() << std::endl;
        return false;
    }
    // Top-2 by value: indices 1 (0.9) and 3 (0.7)
    if (result[0] != 1 || result[1] != 3)
    {
        std::cerr << "topk_basic: got [" << result[0] << ", " << result[1] << "]" << std::endl;
        return false;
    }
    return true;
}

bool test_topk_k_greater_than_size()
{
    const std::vector<float> logits = {0.1F, 0.9F};
    const auto result = trtf::select_topk_tokens(logits, 5);
    if (result.size() != 2)
    {
        std::cerr << "topk_k_greater: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_topk_k_zero()
{
    const std::vector<float> logits = {0.1F, 0.9F};
    const auto result = trtf::select_topk_tokens(logits, 0);
    if (!result.empty())
    {
        std::cerr << "topk_k_zero: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_topk_empty()
{
    const std::vector<float> logits;
    const auto result = trtf::select_topk_tokens(logits, 3);
    if (!result.empty())
    {
        std::cerr << "topk_empty: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_mask_cache0_no_current()
{
    // cache_length=0, max=4, include_current=false
    // First position visible (else clause), rest masked
    const auto mask = trtf::build_attention_mask(0, 4, false);
    if (mask.size() != 4)
    {
        std::cerr << "mask_cache0: size=" << mask.size() << std::endl;
        return false;
    }
    if (mask[0] != 0.0F)
    {
        std::cerr << "mask_cache0: [0]=" << mask[0] << std::endl;
        return false;
    }
    for (int i = 1; i < 4; ++i)
    {
        if (mask[static_cast<std::size_t>(i)] >= 0.0F)
        {
            std::cerr << "mask_cache0: [" << i << "]=" << mask[static_cast<std::size_t>(i)] << std::endl;
            return false;
        }
    }
    return true;
}

bool test_mask_cache3_no_current()
{
    // cache_length=3, max=4, include_current=false -> 3 visible, 1 masked
    const auto mask = trtf::build_attention_mask(3, 4, false);
    if (mask.size() != 4)
    {
        std::cerr << "mask_cache3: size=" << mask.size() << std::endl;
        return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        if (mask[static_cast<std::size_t>(i)] != 0.0F)
        {
            std::cerr << "mask_cache3: [" << i << "]=" << mask[static_cast<std::size_t>(i)] << std::endl;
            return false;
        }
    }
    if (mask[3] >= 0.0F)
    {
        std::cerr << "mask_cache3: [3]=" << mask[3] << std::endl;
        return false;
    }
    return true;
}

bool test_mask_with_current_slot()
{
    // cache_length=0, max=4, include_current=true -> width=5, last slot visible
    const auto mask = trtf::build_attention_mask(0, 4, true);
    if (mask.size() != 5)
    {
        std::cerr << "mask_current: size=" << mask.size() << std::endl;
        return false;
    }
    // Last slot (current) should be visible
    if (mask[4] != 0.0F)
    {
        std::cerr << "mask_current: [4]=" << mask[4] << std::endl;
        return false;
    }
    return true;
}

bool test_mask_full_cache()
{
    // cache_length >= max -> all positions visible
    const auto mask = trtf::build_attention_mask(5, 4, false);
    if (mask.size() != 4)
    {
        std::cerr << "mask_full: size=" << mask.size() << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < mask.size(); ++i)
    {
        if (mask[i] != 0.0F)
        {
            std::cerr << "mask_full: [" << i << "]=" << mask[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool test_cache_append_normal()
{
    // write_index=0, hidden=2, max_cache=3
    const int32_t hidden = 2;
    const int32_t max_cache = 3;
    std::vector<float> cache(static_cast<std::size_t>(max_cache) * static_cast<std::size_t>(hidden), 0.0F);
    const std::vector<float> state = {1.0F, 2.0F};
    trtf::append_cache_state(cache, state, hidden, max_cache, 0);
    // Row 0 should be [1.0, 2.0]
    if (cache[0] != 1.0F || cache[1] != 2.0F)
    {
        std::cerr << "cache_normal: [0]=" << cache[0] << " [1]=" << cache[1] << std::endl;
        return false;
    }
    // Row 1, 2 should still be zero
    if (cache[2] != 0.0F || cache[4] != 0.0F)
    {
        std::cerr << "cache_normal: rows 1,2 not zero" << std::endl;
        return false;
    }
    return true;
}

bool test_cache_append_overflow()
{
    // write_index >= max_cache -> shifts rows left, writes at tail
    const int32_t hidden = 2;
    const int32_t max_cache = 3;
    std::vector<float> cache = {1,2, 3,4, 5,6};
    const std::vector<float> state = {7.0F, 8.0F};
    trtf::append_cache_state(cache, state, hidden, max_cache, 3);
    // After shift: row0=3,4 row1=5,6 row2=7,8
    const std::vector<float> expected = {3,4, 5,6, 7,8};
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        if (cache[i] != expected[i])
        {
            std::cerr << "cache_overflow: [" << i << "]=" << cache[i]
                      << " expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool test_cache_append_size_mismatch()
{
    // state.size != hidden_size -> no-op
    const int32_t hidden = 2;
    const int32_t max_cache = 3;
    std::vector<float> cache(static_cast<std::size_t>(max_cache) * static_cast<std::size_t>(hidden), 0.0F);
    const std::vector<float> state = {1.0F, 2.0F, 3.0F}; // wrong size
    trtf::append_cache_state(cache, state, hidden, max_cache, 0);
    // Cache should be unchanged (all zeros)
    for (std::size_t i = 0; i < cache.size(); ++i)
    {
        if (cache[i] != 0.0F)
        {
            std::cerr << "cache_mismatch: [" << i << "]=" << cache[i] << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    bool all_passed = true;
    std::cout << "test_decode_runtime:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("argmax_basic", test_argmax_basic);
    run("argmax_single", test_argmax_single);
    run("argmax_empty", test_argmax_empty);
    run("argmax_all_negative", test_argmax_all_negative);
    run("argmax_tie", test_argmax_tie);
    run("topk_basic", test_topk_basic);
    run("topk_k_greater", test_topk_k_greater_than_size);
    run("topk_k_zero", test_topk_k_zero);
    run("topk_empty", test_topk_empty);
    run("mask_cache0_no_current", test_mask_cache0_no_current);
    run("mask_cache3_no_current", test_mask_cache3_no_current);
    run("mask_with_current_slot", test_mask_with_current_slot);
    run("mask_full_cache", test_mask_full_cache);
    run("cache_append_normal", test_cache_append_normal);
    run("cache_append_overflow", test_cache_append_overflow);
    run("cache_append_size_mismatch", test_cache_append_size_mismatch);

    if (all_passed)
    {
        std::cout << "test_decode_runtime passed" << std::endl;
        return 0;
    }
    std::cerr << "test_decode_runtime FAILED" << std::endl;
    return 1;
#else
    std::cout << "test_decode_runtime: SKIPPED (TRTF_HAS_TRT=0)" << std::endl;
    return 0;
#endif
}
