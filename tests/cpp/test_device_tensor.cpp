// =============================================================================
// Test suite: DeviceTensor GPU-resident tensor
// =============================================================================
//
// Validates DeviceTensor: allocation, H2D/D2H transfers, D2D copy, zeros
// factory, move semantics, shape/dtype/nbytes accessors.
//
// Requires CUDA GPU at runtime. Skips gracefully without TRT.
// =============================================================================

#include "trtf/runtime/device_tensor.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#if TRTF_HAS_TRT
#include <cuda_runtime_api.h>
#endif

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

#if TRTF_HAS_TRT

static void test_basic_allocation()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::DeviceTensor t({2, 3}, trtf::DType::kFloat32, stream);
    check(t.ok(), "allocation succeeds");
    check(t.numel() == 6, "numel = 6");
    check(t.nbytes() == 24, "nbytes = 24 (6 * 4)");
    check(t.shape().size() == 2, "shape has 2 dims");
    check(t.shape()[0] == 2, "shape[0] = 2");
    check(t.shape()[1] == 3, "shape[1] = 3");
    check(t.dtype() == trtf::DType::kFloat32, "dtype = float32");
    check(t.data() != nullptr, "data is not null");

    cudaStreamDestroy(stream);
}

static void test_h2d_d2h_roundtrip()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::DeviceTensor t({4}, trtf::DType::kFloat32, stream);

    // Upload
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    check(t.copy_from_host(src), "H2D copy succeeds");
    cudaStreamSynchronize(stream);

    // Download
    float dst[4] = {0};
    check(t.copy_to_host(dst), "D2H copy succeeds");

    check(dst[0] == 1.0f, "roundtrip[0] = 1.0");
    check(dst[1] == 2.0f, "roundtrip[1] = 2.0");
    check(dst[2] == 3.0f, "roundtrip[2] = 3.0");
    check(dst[3] == 4.0f, "roundtrip[3] = 4.0");

    cudaStreamDestroy(stream);
}

static void test_d2d_copy()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::DeviceTensor a({3}, trtf::DType::kFloat32, stream);
    trtf::DeviceTensor b({3}, trtf::DType::kFloat32, stream);

    float src[3] = {10.0f, 20.0f, 30.0f};
    a.copy_from_host(src);
    cudaStreamSynchronize(stream);

    check(b.copy_from(a), "D2D copy succeeds");
    cudaStreamSynchronize(stream);

    float dst[3] = {0};
    b.copy_to_host(dst);
    check(dst[0] == 10.0f, "D2D[0] = 10.0");
    check(dst[1] == 20.0f, "D2D[1] = 20.0");
    check(dst[2] == 30.0f, "D2D[2] = 30.0");

    cudaStreamDestroy(stream);
}

static void test_zeros_factory()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto t = trtf::DeviceTensor::zeros({2, 2}, trtf::DType::kFloat32, stream);
    check(t.ok(), "zeros allocation ok");
    cudaStreamSynchronize(stream);

    float dst[4] = {1, 1, 1, 1};
    t.copy_to_host(dst);
    check(dst[0] == 0.0f, "zeros[0] = 0");
    check(dst[1] == 0.0f, "zeros[1] = 0");
    check(dst[2] == 0.0f, "zeros[2] = 0");
    check(dst[3] == 0.0f, "zeros[3] = 0");

    cudaStreamDestroy(stream);
}

static void test_move_semantics()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::DeviceTensor a({5}, trtf::DType::kFloat32, stream);
    check(a.ok(), "a is ok before move");

    trtf::DeviceTensor b(std::move(a));
    check(b.ok(), "b is ok after move-construct");
    check(!a.ok(), "a is empty after move-construct");
    check(b.numel() == 5, "b has 5 elements");

    trtf::DeviceTensor c({1}, trtf::DType::kFloat32, stream);
    c = std::move(b);
    check(c.ok(), "c is ok after move-assign");
    check(!b.ok(), "b is empty after move-assign");
    check(c.numel() == 5, "c has 5 elements");

    cudaStreamDestroy(stream);
}

static void test_int32_dtype()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::DeviceTensor t({3}, trtf::DType::kInt32, stream);
    check(t.ok(), "int32 allocation ok");
    check(t.nbytes() == 12, "int32 nbytes = 12");

    int32_t src[3] = {100, 200, 300};
    t.copy_from_host(src);
    cudaStreamSynchronize(stream);

    int32_t dst[3] = {0};
    t.copy_to_host(dst);
    check(dst[0] == 100, "int32[0] = 100");
    check(dst[1] == 200, "int32[1] = 200");
    check(dst[2] == 300, "int32[2] = 300");

    cudaStreamDestroy(stream);
}

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    test_basic_allocation();
    test_h2d_d2h_roundtrip();
    test_d2d_copy();
    test_zeros_factory();
    test_move_semantics();
    test_int32_dtype();
#else
    std::cerr << "TRT not available, skipping DeviceTensor tests\n";
#endif

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
    }
    return failures;
}
