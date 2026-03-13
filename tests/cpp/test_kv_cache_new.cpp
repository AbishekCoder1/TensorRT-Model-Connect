// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-KVC-CPP-02
// Architecture:   ARCH-KVC-001
// Unit Design:    UD-KVC-01
// Intent:         KvCache construction, position tracking, attention mask building, advance/reset
// Preconditions:  CUDA GPU available
// Postconditions: Position advances correctly, mask is causal, reset zeros state
// =============================================================================

// =============================================================================
// Test suite: KvCache — autoregressive KV state manager
// =============================================================================
//
// Validates KvCache: construction, position tracking, attention mask building,
// advance/reset semantics, and buffer integrity.
//
// Requires CUDA GPU. Skips gracefully without TRT.
// =============================================================================

#include "trtf/runtime/kv_cache.h"

#include <cstdint>
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

static void test_construction()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::KvCache cache(4, 128, 64, stream);
    check(cache.ok(), "cache is ok");
    check(cache.num_layers() == 4, "num_layers = 4");
    check(cache.max_length() == 128, "max_length = 128");
    check(cache.position() == 0, "initial position = 0");

    cudaStreamDestroy(stream);
}

static void test_attention_mask_at_position_0()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::KvCache cache(1, 8, 4, stream);
    std::vector<float> mask;
    cache.build_attention_mask(mask);

    // Mask size = max_length + 1 (extra slot for current token)
    check(mask.size() == 9, "mask size = 9 (max_length + 1)");
    // At position 0: no cached entries visible, only current slot (last)
    check(mask[0] < -1e3f, "mask[0] = masked (no cached entries)");
    check(mask[7] < -1e3f, "mask[7] = masked");
    check(mask[8] == 0.0f, "mask[8] = 0 (current slot)");

    cudaStreamDestroy(stream);
}

static void test_advance_and_position()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::KvCache cache(2, 16, 8, stream);
    check(cache.position() == 0, "start at 0");

    cache.advance();
    check(cache.position() == 1, "after 1 advance: position = 1");

    cache.advance();
    cache.advance();
    check(cache.position() == 3, "after 3 advances: position = 3");

    // Check mask at position 3: 3 cached entries visible + current slot
    std::vector<float> mask;
    cache.build_attention_mask(mask);
    check(mask[0] == 0.0f, "pos3: mask[0] visible");
    check(mask[1] == 0.0f, "pos3: mask[1] visible");
    check(mask[2] == 0.0f, "pos3: mask[2] visible");
    check(mask[3] < -1e3f, "pos3: mask[3] masked");
    check(mask.back() == 0.0f, "pos3: last slot (current) visible");

    cudaStreamDestroy(stream);
}

static void test_reset()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::KvCache cache(2, 16, 8, stream);
    cache.advance();
    cache.advance();
    check(cache.position() == 2, "position = 2 before reset");

    cache.reset();
    check(cache.position() == 0, "position = 0 after reset");

    // Verify mask is back to initial state (no cached entries, only current slot)
    std::vector<float> mask;
    cache.build_attention_mask(mask);
    check(mask[0] < -1e3f, "after reset: mask[0] masked");
    check(mask.back() == 0.0f, "after reset: last slot (current) visible");

    cudaStreamDestroy(stream);
}

static void test_direct_access()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::KvCache cache(3, 8, 4, stream);

    // Verify we can access individual layer buffers
    auto& k0 = cache.cache_k(0);
    auto& v2 = cache.cache_v(2);
    check(k0.ok(), "cache_k(0) is ok");
    check(v2.ok(), "cache_v(2) is ok");
    check(k0.data() != nullptr, "cache_k(0) has device memory");

    cudaStreamDestroy(stream);
}

static void test_max_position_clamp()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::KvCache cache(1, 4, 2, stream);

    // Advance past max_length
    for (int i = 0; i < 10; ++i) cache.advance();

    // With sliding-window shift, position stays at max_length once full
    // (all slots visible, new entries written at tail after shift)
    check(cache.position() == cache.max_length(),
          "position == max_length when cache full (sliding window)");

    // Verify mask: all max_length slots + current slot should be visible
    std::vector<float> mask;
    cache.build_attention_mask(mask);
    for (int i = 0; i < cache.max_length(); ++i)
    {
        check(mask[i] == 0.0f, "full cache: all cached slots visible");
    }
    check(mask.back() == 0.0f, "full cache: current slot visible");

    cudaStreamDestroy(stream);
}

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    test_construction();
    test_attention_mask_at_position_0();
    test_advance_and_position();
    test_reset();
    test_direct_access();
    test_max_position_clamp();
#else
    std::cerr << "TRT not available, skipping KvCache tests\n";
#endif

    if (failures > 0) std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
