// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-KVC-CPP-01
// Architecture:   ARCH-KVC-001
// Unit Design:    UD-KVC-01
// Intent:         Cache construction, mask progression, position clamping, reset
// Preconditions:  CUDA GPU available
// Postconditions: Position advances correctly, mask is causal, reset zeros all
// =============================================================================

// =============================================================================
// Test suite: DeviceKvCache GPU KV cache management
// =============================================================================
//
// Purpose:
//   Validates the DeviceKvCache class from device_kv_cache.h: construction with
//   small dimensions, prepare_step() position/mask progression, clamping at
//   max_cache_length, and reset() behavior.
//
// Dependencies:
//   - runtime/core/device_kv_cache.h (DeviceKvCache)
//   - runtime/core/trt_engine_lifecycle.h (DecoderStepEngine)
//   - runtime/core/trt_common.h (CudaBuffer, CudaStream)
//   - CUDA runtime
//
// Approach:
//   Constructs a DecoderStepEngine with null TRT engine/context but valid
//   scalar config fields (cache_state_size, max_cache_length, num_layers,
//   requires_position_input). The DeviceKvCache constructor only reads these
//   scalar fields, so null engine/context is safe.
//
// Environment:
//   Guarded by TRTF_HAS_TRT. Skips gracefully (exit 0) when TensorRT/CUDA
//   headers are not available. Requires a CUDA-capable GPU at runtime.
// =============================================================================

#include "runtime/core/device_kv_cache.h"
#include "runtime/core/device_kv_cache_update_plan.h"
#include "runtime/core/trt_decode_runtime.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_cache_row_update_plan_append_mode()
{
    constexpr std::size_t row_bytes = 16;
    const trtf::detail::CacheRowUpdatePlan plan = trtf::detail::plan_cache_row_update(2, 4, row_bytes);

    check(!plan.shift_existing_rows, "plan_append: append mode");
    check(plan.append_offset_bytes == 2 * row_bytes, "plan_append: append offset");
    check(plan.next_cache_length == 3, "plan_append: next length");
}

static void test_cache_row_update_plan_shift_mode()
{
    constexpr std::size_t row_bytes = 16;
    const trtf::detail::CacheRowUpdatePlan plan = trtf::detail::plan_cache_row_update(4, 4, row_bytes);

    check(plan.shift_existing_rows, "plan_shift: shift mode");
    check(plan.shift_source_offset_bytes == row_bytes, "plan_shift: shift source offset");
    check(plan.shift_copy_bytes == 3 * row_bytes, "plan_shift: shift copy bytes");
    check(plan.tail_offset_bytes == 3 * row_bytes, "plan_shift: tail offset");
    check(plan.next_cache_length == 4, "plan_shift: next length");
}

static void test_cache_row_update_plan_edge_cases()
{
    constexpr std::size_t row_bytes = 32;

    const trtf::detail::CacheRowUpdatePlan overflow_plan = trtf::detail::plan_cache_row_update(9, 4, row_bytes);
    check(overflow_plan.shift_existing_rows, "plan_edge_overflow: uses shift mode");
    check(overflow_plan.next_cache_length == 4, "plan_edge_overflow: next length clamped");

    const trtf::detail::CacheRowUpdatePlan single_slot_plan = trtf::detail::plan_cache_row_update(1, 1, row_bytes);
    check(single_slot_plan.shift_existing_rows, "plan_edge_single_slot: uses shift mode");
    check(single_slot_plan.shift_copy_bytes == 0, "plan_edge_single_slot: zero shift bytes");
    check(single_slot_plan.tail_offset_bytes == 0, "plan_edge_single_slot: zero tail offset");
    check(single_slot_plan.next_cache_length == 1, "plan_edge_single_slot: next length");
}


// Helper: Create a DecoderStepEngine with test parameters (null TRT engine/context).
static trtf::DecoderStepEngine make_test_engine(
    int32_t max_cache, int32_t num_layers, int32_t cache_state_size,
    bool requires_position)
{
    trtf::DecoderStepEngine engine;
    engine.max_cache_length = max_cache;
    engine.num_layers = num_layers;
    engine.cache_state_size = cache_state_size;
    engine.requires_position_input = requires_position;
    engine.attention_mask_size = requires_position ? (max_cache + 1) : max_cache;
    engine.vocab_size = 10;
    engine.hidden_size = cache_state_size;
    // engine and context remain null (not accessed by DeviceKvCache ctor/prepare/reset)
    return engine;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that DeviceKvCache constructs successfully with small
//             dimensions and reports ok().
// Setup:      max_cache=8, 2 layers, cache_state_size=4.
// Mechanism:  Check ok(), cache pointers are non-null.
// -----------------------------------------------------------------------------
static void test_construction()
{
    auto engine = make_test_engine(8, 2, 4, false);
    trtf::DeviceKvCache cache(engine);
    check(cache.ok(), "construction: ok()=true");
    check(cache.cache_k_device_ptr(0) != nullptr, "construction: cache_k[0] non-null");
    check(cache.cache_v_device_ptr(0) != nullptr, "construction: cache_v[0] non-null");
    check(cache.cache_k_device_ptr(1) != nullptr, "construction: cache_k[1] non-null");
    check(cache.cache_v_device_ptr(1) != nullptr, "construction: cache_v[1] non-null");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that prepare_step() increments position_id and grows the
//             visible mask region on successive steps.
// Setup:      max_cache=4, 1 layer, requires_position=false.
// Mechanism:  Call prepare_step() 3 times, checking position_id increments
//             from 0 to 2 and mask visible region grows.
// Note:       With requires_position=false, include_current_slot=false, so
//             mask width == max_cache (4). Position increments on each call
//             because we call update_after_step() which increments cache_length.
// -----------------------------------------------------------------------------
static void test_prepare_step_progression()
{
    auto engine = make_test_engine(4, 1, 2, false);
    trtf::DeviceKvCache cache(engine);
    trtf::CudaStream stream;
    check(stream.ok(), "step_progression: stream ok");

    // We need present_k/v buffers for update_after_step. Each is cache_state_size floats.
    std::vector<trtf::CudaBuffer> present_k;
    std::vector<trtf::CudaBuffer> present_v;
    present_k.emplace_back(2 * sizeof(float));
    present_v.emplace_back(2 * sizeof(float));
    check(present_k[0].ok(), "step_progression: present_k ok");
    check(present_v[0].ok(), "step_progression: present_v ok");

    // Step 0: position_id=0, mask should have 1 visible slot
    {
        int32_t position_id{};
        std::vector<float> mask;
        cache.prepare_step(position_id, mask);
        check(position_id == 0, "step0: position_id=0");
        check(mask.size() == 4, "step0: mask size=4");
        // With cache_length=0, first slot visible, rest masked
        check(mask[0] == 0.0F, "step0: mask[0]=0 (visible)");
        check(mask[1] < 0.0F, "step0: mask[1]<0 (masked)");

        // Simulate update (increments cache_length)
        cache.update_after_step(present_k, present_v, stream.get());
    }

    // Step 1: position_id=1, cache_length=1 so valid=1 slot visible
    // (include_current_slot=false, so mask width == max_cache == 4)
    {
        int32_t position_id{};
        std::vector<float> mask;
        cache.prepare_step(position_id, mask);
        check(position_id == 1, "step1: position_id=1");
        check(mask[0] == 0.0F, "step1: mask[0]=0 (1 cached slot visible)");
        check(mask[1] < 0.0F, "step1: mask[1]<0 (masked)");

        cache.update_after_step(present_k, present_v, stream.get());
    }

    // Step 2: position_id=2, cache_length=2 so valid=2 slots visible
    {
        int32_t position_id{};
        std::vector<float> mask;
        cache.prepare_step(position_id, mask);
        check(position_id == 2, "step2: position_id=2");
        check(mask[0] == 0.0F, "step2: mask[0]=0");
        check(mask[1] == 0.0F, "step2: mask[1]=0");
        check(mask[2] < 0.0F, "step2: mask[2]<0 (masked)");

        cache.update_after_step(present_k, present_v, stream.get());
    }

    cudaStreamSynchronize(stream.get());
}

// -----------------------------------------------------------------------------
// Intention:  Verify that position_id clamps at max_cache_length-1 when
//             requires_position_input=false (include_current_slot=false).
// Setup:      max_cache=4, fill cache beyond capacity.
// Mechanism:  After 6 steps, position_id should be clamped at 3 (max-1).
// -----------------------------------------------------------------------------
static void test_position_clamping_no_position_input()
{
    auto engine = make_test_engine(4, 1, 2, false);
    trtf::DeviceKvCache cache(engine);
    trtf::CudaStream stream;

    std::vector<trtf::CudaBuffer> present_k;
    std::vector<trtf::CudaBuffer> present_v;
    present_k.emplace_back(2 * sizeof(float));
    present_v.emplace_back(2 * sizeof(float));

    // Run 6 steps (exceeds max_cache=4)
    for (int i = 0; i < 6; ++i)
    {
        int32_t position_id{};
        std::vector<float> mask;
        cache.prepare_step(position_id, mask);
        cache.update_after_step(present_k, present_v, stream.get());
    }

    // 7th prepare_step: cache_length should be clamped at max=4
    int32_t position_id{};
    std::vector<float> mask;
    cache.prepare_step(position_id, mask);

    // position_limit = max(max_cache-1, 0) = 3 when !requires_position_input
    // cache_length is clamped to max_cache=4 by update_after_step
    // position_id = min(cache_length, position_limit) = min(4, 3) = 3
    check(position_id == 3, "clamp_no_pos: position_id=3");

    // All mask slots should be visible (cache full)
    for (int i = 0; i < 4; ++i)
    {
        check(mask[static_cast<std::size_t>(i)] == 0.0F,
              "clamp_no_pos: all mask slots visible");
    }

    cudaStreamSynchronize(stream.get());
}

// -----------------------------------------------------------------------------
// Intention:  Verify that position_id clamps at max_cache_length when
//             requires_position_input=true (include_current_slot=true).
// Setup:      max_cache=4, requires_position=true, fill cache beyond capacity.
// Mechanism:  After 6 steps, position_id should be clamped at 4 (max).
// -----------------------------------------------------------------------------
static void test_position_clamping_with_position_input()
{
    auto engine = make_test_engine(4, 1, 2, true);
    trtf::DeviceKvCache cache(engine);
    trtf::CudaStream stream;

    std::vector<trtf::CudaBuffer> present_k;
    std::vector<trtf::CudaBuffer> present_v;
    present_k.emplace_back(2 * sizeof(float));
    present_v.emplace_back(2 * sizeof(float));

    for (int i = 0; i < 6; ++i)
    {
        int32_t position_id{};
        std::vector<float> mask;
        cache.prepare_step(position_id, mask);
        cache.update_after_step(present_k, present_v, stream.get());
    }

    int32_t position_id{};
    std::vector<float> mask;
    cache.prepare_step(position_id, mask);

    // position_limit = max_cache = 4 when requires_position_input=true
    // position_id = min(cache_length=4, position_limit=4) = 4
    check(position_id == 4, "clamp_with_pos: position_id=4");

    // Mask width = max_cache + 1 = 5 (include current slot)
    check(mask.size() == 5, "clamp_with_pos: mask size=5");

    cudaStreamSynchronize(stream.get());
}

// -----------------------------------------------------------------------------
// Intention:  Verify that reset() zeros cache_length so position/mask restart.
// Setup:      Fill cache for a few steps, then call reset(), then prepare_step().
// Mechanism:  After reset, position_id should be 0 and mask should have only 1
//             visible slot.
// -----------------------------------------------------------------------------
static void test_reset()
{
    auto engine = make_test_engine(8, 2, 4, false);
    trtf::DeviceKvCache cache(engine);
    trtf::CudaStream stream;

    std::vector<trtf::CudaBuffer> present_k;
    std::vector<trtf::CudaBuffer> present_v;
    for (int i = 0; i < 2; ++i)
    {
        present_k.emplace_back(4 * sizeof(float));
        present_v.emplace_back(4 * sizeof(float));
    }

    // Advance 3 steps
    for (int i = 0; i < 3; ++i)
    {
        int32_t pid{};
        std::vector<float> mask;
        cache.prepare_step(pid, mask);
        cache.update_after_step(present_k, present_v, stream.get());
    }

    // Reset
    cache.reset(stream.get());
    cudaStreamSynchronize(stream.get());

    // After reset, should be back to initial state
    int32_t position_id{};
    std::vector<float> mask;
    cache.prepare_step(position_id, mask);

    check(position_id == 0, "reset: position_id=0 after reset");
    check(mask.size() == 8, "reset: mask size=8");
    check(mask[0] == 0.0F, "reset: mask[0]=0 (first slot visible)");
    check(mask[1] < 0.0F, "reset: mask[1]<0 (rest masked)");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that DeviceKvCache with multiple layers allocates distinct
//             cache pointers for each layer.
// Setup:      max_cache=4, 3 layers.
// Mechanism:  Check that cache_k and cache_v pointers differ across layers.
// -----------------------------------------------------------------------------
static void test_multi_layer_distinct_pointers()
{
    auto engine = make_test_engine(4, 3, 8, false);
    trtf::DeviceKvCache cache(engine);
    check(cache.ok(), "multi_layer: ok()");

    // Verify all layer pointers are distinct
    void* k0 = cache.cache_k_device_ptr(0);
    void* k1 = cache.cache_k_device_ptr(1);
    void* k2 = cache.cache_k_device_ptr(2);
    void* v0 = cache.cache_v_device_ptr(0);
    void* v1 = cache.cache_v_device_ptr(1);
    void* v2 = cache.cache_v_device_ptr(2);

    check(k0 != k1, "multi_layer: k0 != k1");
    check(k1 != k2, "multi_layer: k1 != k2");
    check(v0 != v1, "multi_layer: v0 != v1");
    check(v1 != v2, "multi_layer: v1 != v2");
    check(k0 != v0, "multi_layer: k0 != v0");
}


int main()
{
    test_cache_row_update_plan_append_mode();
    test_cache_row_update_plan_shift_mode();
    test_cache_row_update_plan_edge_cases();

    test_construction();
    test_prepare_step_progression();
    test_position_clamping_no_position_input();
    test_position_clamping_with_position_input();
    test_reset();
    test_multi_layer_distinct_pointers();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All DeviceKvCache tests passed.\n";
    return 0;
}
