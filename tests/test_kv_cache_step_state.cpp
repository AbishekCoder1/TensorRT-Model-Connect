// Unit tests for src/runtime/trt/kv_cache_step_state.cpp (IStepState Phase C).
// Tests cache state machine, position tracking, and mask generation.
// Guarded by TRTF_HAS_TRT — skips gracefully when TRT not available.

#include "runtime/trt/kv_cache_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#if TRTF_HAS_TRT

namespace {

// Create a minimal DecoderStepEngine with just metadata (no actual TRT engine).
trtf::DecoderStepEngine make_test_engine(int32_t cache_state_size, int32_t max_cache_length,
    int32_t num_layers, bool requires_position_input)
{
    trtf::DecoderStepEngine engine;
    engine.cache_state_size = cache_state_size;
    engine.max_cache_length = max_cache_length;
    engine.num_layers = num_layers;
    engine.requires_position_input = requires_position_input;
    engine.attention_mask_size = max_cache_length + (requires_position_input ? 1 : 0);
    return engine;
}

bool test_constructor_cache_dimensions()
{
    const auto engine = make_test_engine(4, 3, 2, true);
    trtf::KvCacheStepState state(engine);

    const auto& ck = state.cache_k_by_layer();
    const auto& cv = state.cache_v_by_layer();

    if (static_cast<int32_t>(ck.size()) != 2 || static_cast<int32_t>(cv.size()) != 2)
    {
        std::cerr << "constructor: layer count mismatch k=" << ck.size() << " v=" << cv.size() << std::endl;
        return false;
    }

    const std::size_t expected_elems = 3 * 4; // max_cache * cache_state_size
    if (ck[0].size() != expected_elems || cv[0].size() != expected_elems)
    {
        std::cerr << "constructor: cache size mismatch k=" << ck[0].size()
                  << " v=" << cv[0].size() << " expected=" << expected_elems << std::endl;
        return false;
    }

    // All zeros initially
    for (std::size_t i = 0; i < expected_elems; ++i)
    {
        if (ck[0][i] != 0.0F || cv[0][i] != 0.0F)
        {
            std::cerr << "constructor: cache not zero at " << i << std::endl;
            return false;
        }
    }
    return true;
}

bool test_step0_position_and_mask()
{
    // requires_position_input=true: mask width = max_cache + 1
    const auto engine = make_test_engine(4, 3, 1, true);
    trtf::KvCacheStepState state(engine);

    int32_t position_id = -1;
    std::vector<float> mask;
    state.prepare_step(position_id, mask);

    if (position_id != 0)
    {
        std::cerr << "step0: position_id=" << position_id << std::endl;
        return false;
    }

    // Mask width = 3 + 1 = 4, last slot (current) visible
    if (mask.size() != 4)
    {
        std::cerr << "step0: mask size=" << mask.size() << std::endl;
        return false;
    }

    // Current slot (last) should be 0.0
    if (mask.back() != 0.0F)
    {
        std::cerr << "step0: last mask=" << mask.back() << std::endl;
        return false;
    }

    return true;
}

bool test_step_sequence()
{
    // With position input: position advances with cache length
    const auto engine = make_test_engine(2, 3, 1, true);
    trtf::KvCacheStepState state(engine);

    // Step 0
    {
        int32_t pos = -1;
        std::vector<float> mask;
        state.prepare_step(pos, mask);
        if (pos != 0)
        {
            std::cerr << "seq step0: pos=" << pos << std::endl;
            return false;
        }
    }

    // Simulate update with known present_k/v
    const std::vector<float> present_k = {10.0F, 20.0F};
    const std::vector<float> present_v = {30.0F, 40.0F};
    state.update_after_step({present_k}, {present_v});

    // Step 1
    {
        int32_t pos = -1;
        std::vector<float> mask;
        state.prepare_step(pos, mask);
        if (pos != 1)
        {
            std::cerr << "seq step1: pos=" << pos << std::endl;
            return false;
        }

        // Mask should have 1 visible cache position + current slot
        // mask width = 3 + 1 = 4
        // cache_length=1, so 1 visible + current slot
        int visible_count = 0;
        for (float v : mask)
        {
            if (v == 0.0F)
            {
                ++visible_count;
            }
        }
        if (visible_count != 2)
        {
            std::cerr << "seq step1: visible=" << visible_count << " expected=2" << std::endl;
            return false;
        }
    }

    return true;
}

bool test_update_after_step_writes_cache()
{
    const auto engine = make_test_engine(2, 3, 1, true);
    trtf::KvCacheStepState state(engine);

    const std::vector<float> pk = {1.0F, 2.0F};
    const std::vector<float> pv = {3.0F, 4.0F};
    state.update_after_step({pk}, {pv});

    const auto& ck = state.cache_k_by_layer();
    const auto& cv = state.cache_v_by_layer();

    // Row 0 should now be [1.0, 2.0]
    if (ck[0][0] != 1.0F || ck[0][1] != 2.0F)
    {
        std::cerr << "update: cache_k row0=[" << ck[0][0] << "," << ck[0][1] << "]" << std::endl;
        return false;
    }
    if (cv[0][0] != 3.0F || cv[0][1] != 4.0F)
    {
        std::cerr << "update: cache_v row0=[" << cv[0][0] << "," << cv[0][1] << "]" << std::endl;
        return false;
    }

    return true;
}

bool test_overflow_position_caps()
{
    // max_cache=2, requires_position=true -> position_limit = max_cache = 2
    const auto engine = make_test_engine(2, 2, 1, true);
    trtf::KvCacheStepState state(engine);

    const std::vector<float> pk = {1.0F, 2.0F};
    const std::vector<float> pv = {3.0F, 4.0F};

    // Fill cache: 3 updates (overflow on 3rd)
    state.update_after_step({pk}, {pv});
    state.update_after_step({pk}, {pv});
    state.update_after_step({pk}, {pv});

    int32_t pos = -1;
    std::vector<float> mask;
    state.prepare_step(pos, mask);

    // Position should cap at max_cache_length=2
    if (pos != 2)
    {
        std::cerr << "overflow: pos=" << pos << " expected=2" << std::endl;
        return false;
    }

    return true;
}

bool test_no_position_input()
{
    // requires_position_input=false: position_limit = max_cache - 1
    const auto engine = make_test_engine(2, 3, 1, false);
    trtf::KvCacheStepState state(engine);

    int32_t pos = -1;
    std::vector<float> mask;
    state.prepare_step(pos, mask);

    if (pos != 0)
    {
        std::cerr << "no_pos step0: pos=" << pos << std::endl;
        return false;
    }

    // Mask width = max_cache = 3 (no current slot)
    if (mask.size() != 3)
    {
        std::cerr << "no_pos: mask size=" << mask.size() << std::endl;
        return false;
    }

    return true;
}

bool test_multi_layer()
{
    const auto engine = make_test_engine(2, 3, 2, true);
    trtf::KvCacheStepState state(engine);

    const std::vector<float> pk0 = {1.0F, 2.0F};
    const std::vector<float> pk1 = {5.0F, 6.0F};
    const std::vector<float> pv0 = {3.0F, 4.0F};
    const std::vector<float> pv1 = {7.0F, 8.0F};

    state.update_after_step({pk0, pk1}, {pv0, pv1});

    const auto& ck = state.cache_k_by_layer();
    if (ck[0][0] != 1.0F || ck[1][0] != 5.0F)
    {
        std::cerr << "multi_layer: ck[0][0]=" << ck[0][0] << " ck[1][0]=" << ck[1][0] << std::endl;
        return false;
    }

    return true;
}

} // namespace

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    bool all_passed = true;
    std::cout << "test_kv_cache_step_state:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("constructor_cache_dimensions", test_constructor_cache_dimensions);
    run("step0_position_and_mask", test_step0_position_and_mask);
    run("step_sequence", test_step_sequence);
    run("update_after_step", test_update_after_step_writes_cache);
    run("overflow_position_caps", test_overflow_position_caps);
    run("no_position_input", test_no_position_input);
    run("multi_layer", test_multi_layer);

    if (all_passed)
    {
        std::cout << "test_kv_cache_step_state passed" << std::endl;
        return 0;
    }
    std::cerr << "test_kv_cache_step_state FAILED" << std::endl;
    return 1;
#else
    std::cout << "test_kv_cache_step_state: SKIPPED (TRTF_HAS_TRT=0)" << std::endl;
    return 0;
#endif
}
