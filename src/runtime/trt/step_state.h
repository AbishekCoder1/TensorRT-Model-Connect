#pragma once

#include <cstdint>
#include <vector>

namespace trtf {

// Abstract interface for per-step state management during autoregressive generation.
// KV-cache-based models use KvCacheStepState; Mamba/SSM models can provide
// a recurrent state implementation; hybrid models can combine both.
class IStepState {
public:
    virtual ~IStepState() = default;

    // Prepare inputs for the next decode step.
    // Sets out_position_id and out_attention_mask for the current step.
    // Returns true if the step should proceed (false if generation should stop).
    virtual bool prepare_step(int32_t& out_position_id,
                              std::vector<float>& out_attention_mask) = 0;

    // Access current cache state (read-only). Used as inputs to run_decoder_step().
    virtual const std::vector<std::vector<float>>& cache_k_by_layer() const = 0;
    virtual const std::vector<std::vector<float>>& cache_v_by_layer() const = 0;

    // Update state after a decode step with the new present K/V outputs.
    virtual void update_after_step(
        const std::vector<std::vector<float>>& present_k_by_layer,
        const std::vector<std::vector<float>>& present_v_by_layer) = 0;
};

} // namespace trtf
