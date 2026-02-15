// =============================================================================
// test_tensor_math.cpp — Unit tests for src/utils/tensor_math.cpp
// =============================================================================
//
// Purpose:
//   Validates the CPU-side tensor math utilities used during model weight
//   loading and preprocessing. These functions operate on flat float vectors
//   representing 2D tensors and are used to transform HuggingFace checkpoint
//   weights into the layout required by TensorRT engine construction:
//     - transpose_2d: Row-major matrix transposition.
//     - repeat_head_norm: Replicates a normalization vector across attention
//       heads (used for per-head RMSNorm weights).
//     - expand_kv_projection: Expands KV-projection weight matrices from
//       GQA (grouped query attention) format to full multi-head format by
//       repeating each KV head's weights for the number of query heads it
//       serves.
//
// Dependencies:
//   - utils/tensor_math.h (transpose_2d, repeat_head_norm, expand_kv_projection)
//   - trtf/model.h (DecoderArchitectureConfig — used by expand_kv_projection)
//
// Approach:
//   Each test constructs small numeric inputs with known expected outputs,
//   calls the function under test, and uses check_close() for element-wise
//   floating-point comparison with an absolute tolerance. Error-path tests
//   verify that size mismatches throw std::runtime_error.
//
// Environment:
//   CPU-only, no TRT/CUDA dependencies. No filesystem access required.
// =============================================================================

#include "utils/tensor_math.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

// Helper: element-wise comparison of two float vectors with absolute tolerance.
// Reports the first mismatch (index, actual, expected) on failure.
bool check_close(const std::vector<float>& actual, const std::vector<float>& expected,
    float atol, const char* label)
{
    if (actual.size() != expected.size())
    {
        std::cerr << label << ": size mismatch actual=" << actual.size()
                  << " expected=" << expected.size() << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (std::abs(actual[i] - expected[i]) > atol)
        {
            std::cerr << label << ": mismatch at index " << i
                      << " actual=" << actual[i] << " expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// transpose_2d tests
// ---------------------------------------------------------------------------

// Intention: Verify basic 2x3 matrix transposition produces the correct 3x2
//            result in row-major order.
// Setup:     Flat vector representing [[1,2,3],[4,5,6]] (2 rows, 3 cols).
// Mechanism: Calls transpose_2d(src, 2, 3), checks the output matches the
//            expected transposition [[1,4],[2,5],[3,6]] stored as {1,4,2,5,3,6}.
bool test_transpose_2x3()
{
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    const std::vector<float> src = {1, 2, 3, 4, 5, 6};
    const std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    const auto result = trtf::transpose_2d(src, 2, 3, "test");
    return check_close(result, expected, 1e-6F, "transpose_2x3");
}

// Intention: Verify that transposing a 3x3 identity matrix returns the same
//            identity matrix (symmetric matrix invariant).
// Setup:     Flat vector representing the 3x3 identity matrix.
// Mechanism: Calls transpose_2d(eye, 3, 3), checks the output equals the input.
bool test_transpose_identity()
{
    // 3x3 identity stays identity
    const std::vector<float> eye = {1,0,0, 0,1,0, 0,0,1};
    const auto result = trtf::transpose_2d(eye, 3, 3, "eye");
    return check_close(result, eye, 1e-6F, "transpose_identity");
}

// Intention: Verify that transpose_2d throws std::runtime_error when the
//            input vector size does not match rows * cols.
// Setup:     Input vector of 3 elements with declared dimensions 2x3 (expects 6).
// Mechanism: Calls transpose_2d inside a try/catch, expects std::runtime_error
//            to be thrown. Returns false if no exception is raised.
bool test_transpose_size_mismatch()
{
    const std::vector<float> src = {1, 2, 3};
    try
    {
        trtf::transpose_2d(src, 2, 3, "bad");
        std::cerr << "transpose_size_mismatch: expected exception" << std::endl;
        return false;
    }
    catch (const std::runtime_error&)
    {
        return true;
    }
}

// ---------------------------------------------------------------------------
// repeat_head_norm tests
// ---------------------------------------------------------------------------

// Intention: Verify that repeat_head_norm correctly replicates a normalization
//            vector N times (simulating per-head norm weight expansion).
// Setup:     Norm vector {1.0, 2.0, 3.0, 4.0} with num_heads=3.
// Mechanism: Calls repeat_head_norm(norm, 3), checks the output is 3
//            concatenated copies of the input: {1,2,3,4, 1,2,3,4, 1,2,3,4}.
bool test_repeat_head_norm_basic()
{
    const std::vector<float> norm = {1.0F, 2.0F, 3.0F, 4.0F};
    const auto result = trtf::repeat_head_norm(norm, 3);
    const std::vector<float> expected = {1,2,3,4, 1,2,3,4, 1,2,3,4};
    return check_close(result, expected, 1e-6F, "repeat_head_norm_basic");
}

// Intention: Verify that repeat_head_norm handles an empty input vector
//            gracefully by returning an empty vector.
// Setup:     Empty norm vector, num_heads=3.
// Mechanism: Calls repeat_head_norm, checks the result is empty.
bool test_repeat_head_norm_empty()
{
    const std::vector<float> empty;
    const auto result = trtf::repeat_head_norm(empty, 3);
    if (!result.empty())
    {
        std::cerr << "repeat_head_norm_empty: expected empty result" << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that repeat_head_norm with num_heads=0 returns an empty
//            vector (zero repetitions).
// Setup:     Norm vector {1.0, 2.0}, num_heads=0.
// Mechanism: Calls repeat_head_norm, checks the result is empty.
bool test_repeat_head_norm_zero_heads()
{
    const std::vector<float> norm = {1.0F, 2.0F};
    const auto result = trtf::repeat_head_norm(norm, 0);
    if (!result.empty())
    {
        std::cerr << "repeat_head_norm_zero_heads: expected empty result" << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// expand_kv_projection tests
// ---------------------------------------------------------------------------

// Intention: Verify that expand_kv_projection returns the input unchanged
//            when kv_hidden equals target_hidden (MHA case, no GQA expansion
//            needed because every query head has its own KV head).
// Setup:     Architecture with num_attention_heads=4, num_key_value_heads=4.
//            Input is a [4 x 8] weight matrix (hidden=4, kv_hidden=8).
//            target_hidden=8 matches kv_hidden, so no expansion is needed.
// Mechanism: Calls expand_kv_projection, checks output equals input.
bool test_expand_kv_no_expansion()
{
    // kv_hidden == target_hidden -> return input unchanged
    trtf::DecoderArchitectureConfig arch;
    arch.num_attention_heads = 4;
    arch.num_key_value_heads = 4;
    const int32_t hidden = 4;
    const int32_t kv_hidden = 8;
    const int32_t target_hidden = 8;
    std::vector<float> src(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(kv_hidden), 1.5F);
    const auto result = trtf::expand_kv_projection(src, hidden, kv_hidden, target_hidden, arch, "test");
    return check_close(result, src, 1e-6F, "expand_kv_no_expansion");
}

// Intention: Verify GQA (Grouped Query Attention) KV-projection expansion.
//            When there are fewer KV heads than query heads, each KV head's
//            weight columns must be repeated to match the full query head count.
// Setup:     Architecture with num_attention_heads=4, num_key_value_heads=2,
//            head_dim=2. Input is a [2 x 4] weight matrix (2 KV heads of dim 2).
//            Target is [2 x 8] (4 query heads of dim 2).
// Mechanism: Calls expand_kv_projection. Each of the 2 KV heads should be
//            repeated twice (4 Q heads / 2 KV heads = repeat factor 2).
//            Verifies the output matches the hand-computed expected expansion:
//            Row 0: head0(1,2) head0(1,2) head1(3,4) head1(3,4)
//            Row 1: head0(5,6) head0(5,6) head1(7,8) head1(7,8)
bool test_expand_kv_gqa()
{
    // 2 KV heads -> 4 Q heads, head_dim=2, hidden=2
    trtf::DecoderArchitectureConfig arch;
    arch.num_attention_heads = 4;
    arch.num_key_value_heads = 2;
    const int32_t hidden = 2;
    const int32_t kv_hidden = 4;  // 2 kv_heads * 2 head_dim
    const int32_t target_hidden = 8;  // 4 q_heads * 2 head_dim
    // src: [hidden=2, kv_hidden=4] identity-like
    const std::vector<float> src = {
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F
    };
    // Expected: each KV head repeated 2x -> [hidden=2, target_hidden=8]
    // Row 0: head0(1,2) head0(1,2) head1(3,4) head1(3,4)
    // Row 1: head0(5,6) head0(5,6) head1(7,8) head1(7,8)
    const std::vector<float> expected = {
        1.0F, 2.0F, 1.0F, 2.0F, 3.0F, 4.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 5.0F, 6.0F, 7.0F, 8.0F, 7.0F, 8.0F
    };
    const auto result = trtf::expand_kv_projection(src, hidden, kv_hidden, target_hidden, arch, "test");
    return check_close(result, expected, 1e-6F, "expand_kv_gqa");
}

// Intention: Verify that expand_kv_projection throws std::runtime_error when
//            the input vector size does not match hidden * kv_hidden.
// Setup:     Architecture with num_attention_heads=4, num_key_value_heads=2.
//            Input vector has only 2 elements, but hidden=4, kv_hidden=4
//            expects 16.
// Mechanism: Calls expand_kv_projection inside a try/catch, expects
//            std::runtime_error. Returns false if no exception is raised.
bool test_expand_kv_size_mismatch()
{
    trtf::DecoderArchitectureConfig arch;
    arch.num_attention_heads = 4;
    arch.num_key_value_heads = 2;
    const std::vector<float> src = {1.0F, 2.0F};  // wrong size
    try
    {
        trtf::expand_kv_projection(src, 4, 4, 8, arch, "bad");
        std::cerr << "expand_kv_size_mismatch: expected exception" << std::endl;
        return false;
    }
    catch (const std::runtime_error&)
    {
        return true;
    }
}

} // namespace

int main()
{
    bool all_passed = true;
    std::cout << "test_tensor_math:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("transpose_2x3", test_transpose_2x3);
    run("transpose_identity", test_transpose_identity);
    run("transpose_size_mismatch", test_transpose_size_mismatch);
    run("repeat_head_norm_basic", test_repeat_head_norm_basic);
    run("repeat_head_norm_empty", test_repeat_head_norm_empty);
    run("repeat_head_norm_zero_heads", test_repeat_head_norm_zero_heads);
    run("expand_kv_no_expansion", test_expand_kv_no_expansion);
    run("expand_kv_gqa", test_expand_kv_gqa);
    run("expand_kv_size_mismatch", test_expand_kv_size_mismatch);

    if (all_passed)
    {
        std::cout << "test_tensor_math passed" << std::endl;
        return 0;
    }
    std::cerr << "test_tensor_math FAILED" << std::endl;
    return 1;
}
