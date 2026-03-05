// =============================================================================
// Test suite: TRT decode runtime CPU-side helper functions
// =============================================================================
//
// Purpose:
//   Validates the CPU-side utility functions from trt_decode_runtime.cpp that
//   support the autoregressive decoding loop: token selection (argmax, top-k)
//   and causal attention mask construction. These functions are exercised
//   without a GPU — they operate on CPU vectors and return CPU results.
//
// Dependencies:
//   - runtime/trt/core/trt_decode_runtime.h (select_argmax_token, select_topk_tokens,
//                                        build_attention_mask)
//
// Approach:
//   All tests construct small input vectors, call the target function, and
//   verify the output against expected values. The test groups are:
//
//   1. Argmax tests — verify select_argmax_token returns the index of the
//      maximum logit value, including edge cases (single element, empty input,
//      all negatives, ties).
//
//   2. Top-k tests — verify select_topk_tokens returns the k indices with
//      highest logit values in descending order, including edge cases (k > size,
//      k=0, empty input).
//
//   3. Attention mask tests — verify build_attention_mask produces correct
//      causal masks for various cache occupancy levels (empty, partial, full,
//      with/without current-token slot).
//
// Environment:
//   Guarded by TRTF_HAS_TRT. Skips gracefully (exit 0) when TensorRT headers
//   are not available. No GPU execution — tests only exercise CPU logic.
// =============================================================================

#include "runtime/trt/core/trt_decode_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#if TRTF_HAS_TRT

namespace {

// -----------------------------------------------------------------------------
// Intention:  Verify basic argmax — selecting the index of the highest logit
//             from a 3-element vector.
// Setup:      logits = {0.1, 0.9, 0.3}. Maximum is at index 1.
// Mechanism:  Calls select_argmax_token and asserts the result is 1.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify argmax with a single-element vector — the only valid
//             index is 0.
// Setup:      logits = {5.0}. Only one element, so index must be 0.
// Mechanism:  Calls select_argmax_token and asserts the result is 0.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify argmax with an empty vector — should return 0 as a safe
//             default rather than crashing.
// Setup:      An empty logits vector.
// Mechanism:  Calls select_argmax_token and asserts the result is 0.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify argmax when all logit values are negative — the function
//             should still return the index of the maximum (least negative).
// Setup:      logits = {-3.0, -1.0, -5.0, -2.0}. Maximum is -1.0 at index 1.
// Mechanism:  Calls select_argmax_token and asserts the result is 1.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify argmax tie-breaking behavior — when multiple elements
//             share the maximum value, the function should return the first
//             occurrence (consistent with std::max_element).
// Setup:      logits = {1.0, 5.0, 5.0, 2.0}. Tie at indices 1 and 2.
// Mechanism:  Calls select_argmax_token and asserts the result is 1 (the first
//             occurrence of the maximum).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify basic top-k selection — the 2 indices with the highest
//             logit values should be returned in descending order of value.
// Setup:      logits = {0.1, 0.9, 0.3, 0.7}. Top-2: index 1 (0.9), index 3
//             (0.7).
// Mechanism:  Calls select_topk_tokens with k=2, asserts the result has 2
//             elements, and verifies the indices are [1, 3].
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify top-k behavior when k exceeds the vector size — the
//             function should return all elements (clamped to vector size)
//             rather than crashing.
// Setup:      logits = {0.1, 0.9} with k=5.
// Mechanism:  Calls select_topk_tokens and asserts the result has exactly 2
//             elements (the full vector).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify top-k behavior with k=0 — should return an empty vector.
// Setup:      logits = {0.1, 0.9} with k=0.
// Mechanism:  Calls select_topk_tokens and asserts the result is empty.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify top-k behavior with an empty logits vector — should return
//             an empty result without crashing.
// Setup:      An empty logits vector with k=3.
// Mechanism:  Calls select_topk_tokens and asserts the result is empty.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify the causal attention mask when the cache is empty
//             (cache_length=0) and include_current is false. Only the first
//             position should be visible (0.0); the rest should be masked
//             (large negative).
// Setup:      cache_length=0, max_cache=4, include_current=false.
// Mechanism:  Calls build_attention_mask and asserts: size is 4, mask[0]=0.0
//             (visible), mask[1..3] < 0.0 (masked).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify the causal attention mask when 3 of 4 cache slots are
//             occupied and include_current is false. The first 3 positions
//             should be visible; the last should be masked.
// Setup:      cache_length=3, max_cache=4, include_current=false.
// Mechanism:  Calls build_attention_mask and asserts: size is 4, mask[0..2]=0.0,
//             mask[3] < 0.0.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify that include_current=true appends an extra slot to the
//             mask (for the current token being decoded), and that this slot is
//             visible (0.0).
// Setup:      cache_length=0, max_cache=4, include_current=true.
// Mechanism:  Calls build_attention_mask and asserts: size is 5 (4 cache + 1
//             current), and mask[4] (the current-token slot) equals 0.0.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify the attention mask when cache_length >= max_cache_length —
//             all positions should be visible (the cache is fully occupied).
// Setup:      cache_length=5, max_cache=4, include_current=false. The cache
//             length exceeds max, so all 4 positions should be unmasked.
// Mechanism:  Calls build_attention_mask and asserts: size is 4, every element
//             equals 0.0.
// -----------------------------------------------------------------------------
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
