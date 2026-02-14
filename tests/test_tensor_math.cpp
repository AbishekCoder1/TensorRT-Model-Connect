// Unit tests for src/utils/tensor_math.cpp (transpose, repeat_head_norm, expand_kv).
// CPU-only, no TRT/CUDA deps.

#include "utils/tensor_math.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

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

bool test_transpose_2x3()
{
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    const std::vector<float> src = {1, 2, 3, 4, 5, 6};
    const std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    const auto result = trtf::transpose_2d(src, 2, 3, "test");
    return check_close(result, expected, 1e-6F, "transpose_2x3");
}

bool test_transpose_identity()
{
    // 3x3 identity stays identity
    const std::vector<float> eye = {1,0,0, 0,1,0, 0,0,1};
    const auto result = trtf::transpose_2d(eye, 3, 3, "eye");
    return check_close(result, eye, 1e-6F, "transpose_identity");
}

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

bool test_repeat_head_norm_basic()
{
    const std::vector<float> norm = {1.0F, 2.0F, 3.0F, 4.0F};
    const auto result = trtf::repeat_head_norm(norm, 3);
    const std::vector<float> expected = {1,2,3,4, 1,2,3,4, 1,2,3,4};
    return check_close(result, expected, 1e-6F, "repeat_head_norm_basic");
}

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
