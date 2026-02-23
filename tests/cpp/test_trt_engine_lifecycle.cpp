// =============================================================================
// test_trt_engine_lifecycle.cpp — Unit tests for layer_tensor_name()
// =============================================================================
//
// Purpose:
//   Validates the layer_tensor_name() helper from trt_engine_lifecycle.h/cpp,
//   which generates per-layer tensor names (e.g. "cache_k_0", "present_v_23").
//   When TRTF_HAS_TRT is disabled, the test skips gracefully (exit 0).
//
// Dependencies:
//   - runtime/trt/trt_engine_lifecycle.h (layer_tensor_name)
//
// Environment:
//   CPU-only. No GPU, CUDA, or TRT runtime required.
//   The function itself only does string concatenation, but it is guarded
//   behind TRTF_HAS_TRT in the header.
// =============================================================================

#include "runtime/trt/trt_engine_lifecycle.h"

#include <iostream>
#include <string>

#if TRTF_HAS_TRT

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_layer_tensor_name_zero()
{
    const std::string result = trtf::layer_tensor_name("cache_k", 0);
    check(result == "cache_k_0", "layer_tensor_name cache_k_0");
}

static void test_layer_tensor_name_mid()
{
    const std::string result = trtf::layer_tensor_name("present_v", 23);
    check(result == "present_v_23", "layer_tensor_name present_v_23");
}

static void test_layer_tensor_name_large()
{
    const std::string result = trtf::layer_tensor_name("cache_k", 127);
    check(result == "cache_k_127", "layer_tensor_name cache_k_127");
}

static void test_layer_tensor_name_single_digit()
{
    const std::string result = trtf::layer_tensor_name("state", 7);
    check(result == "state_7", "layer_tensor_name state_7");
}

static void test_layer_tensor_name_empty_stem()
{
    const std::string result = trtf::layer_tensor_name("", 5);
    check(result == "_5", "layer_tensor_name empty_stem _5");
}

static void test_constants()
{
    check(trtf::kDefaultMaxCacheLength == 32, "kDefaultMaxCacheLength == 32");
    check(trtf::kMaskedScore == -1.0e4F, "kMaskedScore == -1.0e4");
}

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    test_layer_tensor_name_zero();
    test_layer_tensor_name_mid();
    test_layer_tensor_name_large();
    test_layer_tensor_name_single_digit();
    test_layer_tensor_name_empty_stem();
    test_constants();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All trt_engine_lifecycle tests passed.\n";
#else
    std::cerr << "test_trt_engine_lifecycle: TRTF_HAS_TRT=0, skipping TRT-dependent tests.\n";
#endif
    return 0;
}
