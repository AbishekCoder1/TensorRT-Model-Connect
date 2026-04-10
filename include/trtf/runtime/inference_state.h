#pragma once

// IInferenceState: unified interface for autoregressive inference state.
//
// Both KV-cache (attention models) and recurrent state (SSM/RWKV) implement
// this interface. Pipelines and plugins program against it — never against
// concrete state classes.
//
// The interface captures the lifecycle of per-sequence inference state:
//   1. reset()        — prepare for a new sequence
//   2. bind_to()      — bind state tensors to TRT engine I/O
//   3. prepare_step() — write state-related inputs (mask, position) into TensorMap
//   4. advance()      — update state after each decode step
//   5. position()     — current sequence position
//
// Implementations:
//   KvCache          — dense append-only (current default)
//   RecurrentState   — SSM/RWKV conv+ssm state
//   HybridState      — KvCache + RecurrentState composed
//   (future: RingKvCache, PagedKvCache, MlaCache, SlidingWindowCache)

#include "trtf/runtime/tensor.h"

#include <cstddef>
#include <cstdint>

namespace trtf {

class ITrtModule;
using TrtModule = ITrtModule;

class IInferenceState {
public:
    virtual ~IInferenceState() = default;

    // --- Lifecycle ---

    // Reset state for a new sequence (zero buffers, position = 0).
    virtual void reset() = 0;

    // Bind all state tensors to the given TRT module.
    // Called once per sequence after reset(). The module reads/writes
    // state tensors via the bound device pointers.
    virtual void bind_to(TrtModule& module) = 0;

    // Write state-related inputs (mask, position, block table, etc.) into
    // the TensorMap before engine.forward(). The state owns its buffers —
    // Tensor.data pointers remain valid until the next prepare_step() call.
    // Pipelines call this instead of manually constructing mask/position tensors.
    virtual void prepare_step(TensorMap& inputs, int32_t seq_len = 1) = 0;

    // Update state after one decode step. Copies "present" outputs
    // into "cache" inputs, advances position.
    // n_tokens: number of tokens processed in this step (default 1).
    //           >1 for batched prefill / multi-token steps.
    virtual void advance(int32_t n_tokens = 1) = 0;

    // --- Queries ---

    // Current sequence position (0 = empty, increments with advance()).
    virtual int32_t position() const = 0;

    // Maximum sequence length this state can hold.
    // -1 for unbounded (recurrent models with no cache length limit).
    virtual int32_t max_length() const = 0;

    // Number of transformer/SSM layers.
    virtual int32_t num_layers() const = 0;

    // Whether this state type needs an attention mask.
    // KvCache -> true. RecurrentState -> false.
    virtual bool needs_attention_mask() const = 0;

    // Total device memory consumed by this state (bytes).
    virtual std::size_t device_memory_bytes() const = 0;

    // Human-readable state type for diagnostics.
    virtual const char* state_type() const = 0;

    // Whether all allocations succeeded.
    virtual bool ok() const = 0;
};

} // namespace trtf
