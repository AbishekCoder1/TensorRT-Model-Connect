# TASK-02: KvCache + RecurrentState — stateful wrappers over DeviceTensor

## Status: ready (after TASK-01)
## Phase: 1 (Foundation)
## Risk: low — additive, wraps existing DeviceKvCache concepts

## Goal

Create `KvCache` and `RecurrentState` as thin, composable state managers built
on top of `DeviceTensor`. These replace the current monolithic `DeviceKvCache`
+ `DeviceResources` + `MambaStepState` + `RwkvStepState`.

Key insight: all stateful inference is just DeviceTensor vectors with
advance/reset semantics. The differences are only in what tensors exist per layer
and how "advance" works.

## KvCache interface

Replaces: `DeviceKvCache` + `DeviceResources` (currently 22.5KB combined)

```cpp
class KvCache {
public:
    KvCache(int32_t num_layers, int32_t max_length,
            int32_t kv_dim, cudaStream_t stream);

    // Per-step state query
    int32_t position() const;
    int32_t max_length() const;
    void build_attention_mask(std::vector<float>& mask) const;

    // Bind all cache tensors to a TrtModule
    // Binds: cache_k_{i}, cache_v_{i}, present_k_{i}, present_v_{i}
    void bind_to(TrtModule& module);

    // After each decode step: copy present→cache, advance position
    void advance(cudaStream_t stream);

    // Reset for new sequence
    void reset(cudaStream_t stream);

    // Direct access (for cross-attention pre-fill, VL embedding injection)
    DeviceTensor& cache_k(int32_t layer);
    DeviceTensor& cache_v(int32_t layer);

private:
    std::vector<DeviceTensor> cache_k_;     // [num_layers], persistent
    std::vector<DeviceTensor> cache_v_;     // [num_layers], persistent
    std::vector<DeviceTensor> present_k_;   // [num_layers], single-step output
    std::vector<DeviceTensor> present_v_;   // [num_layers], single-step output
    int32_t position_{0};
    int32_t max_length_;
};
```

## RecurrentState interface

Replaces: `MambaStepState` + `RwkvStepState` with a generic pattern.

```cpp
class RecurrentState {
public:
    // Construct from a spec that defines named state tensors per layer.
    // Mamba: {{"conv_state", [d_inner*(conv_kernel-1)]}, {"ssm_state", [state_size*d_inner]}}
    // RWKV:  {{"attn", [hidden]}, {"ff", [hidden]}, {"num", [hidden]}, {"den", [hidden]}, {"max", [hidden]}}
    struct TensorSpec { std::string name; std::vector<int64_t> shape; };
    RecurrentState(int32_t num_layers, std::vector<TensorSpec> specs, cudaStream_t stream);

    // Bind all state tensors to a TrtModule
    // Binds: {name}_{i} (input) and present_{name}_{i} (output) for each spec
    void bind_to(TrtModule& module);

    // After each step: copy present→state for all tensors
    void advance(cudaStream_t stream);

    // Reset all state to zeros
    void reset(cudaStream_t stream);

private:
    // state_[spec_name][layer_idx] → DeviceTensor
    std::unordered_map<std::string, std::vector<DeviceTensor>> state_;
    std::unordered_map<std::string, std::vector<DeviceTensor>> present_;
    int32_t num_layers_;
};
```

## Why this is better than current design

Current `DeviceKvCache` is 22.5KB because it mixes:
- Buffer allocation (→ DeviceTensor handles this)
- Tensor binding (→ `bind_to(TrtModule)` handles this)
- H2D/D2H transfers (→ TrtModule handles this)
- Mask construction (→ stays in KvCache)
- Position tracking (→ stays in KvCache)
- VL embedding injection (→ pipeline handles this via DeviceTensor access)
- DeepStack embedding injection (→ pipeline handles this)

New KvCache: ~200 lines. Clean separation of concerns.

## Files to create

- `include/trtf/runtime/kv_cache.h`
- `include/trtf/runtime/recurrent_state.h`
- `src/runtime/trt/core/kv_cache.cpp`
- `src/runtime/trt/core/recurrent_state.cpp`
- `tests/cpp/test_kv_cache.cpp`
- `tests/cpp/test_recurrent_state.cpp`

## Tests

- KvCache: construct, bind to module, advance 5 steps, verify position
- KvCache: reset, verify position=0 and buffers zeroed
- KvCache: build_attention_mask at various positions
- RecurrentState with Mamba spec: construct, bind, advance, verify state copied
- RecurrentState with RWKV spec (5 tensors): same

## Dependencies

TASK-01 (TrtModule, DeviceTensor)
