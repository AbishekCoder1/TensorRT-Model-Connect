#pragma once

// KvCache: autoregressive KV cache state manager.
// HF equivalent: DynamicCache / past_key_values.
//
// Manages per-layer K/V device tensors, position tracking, and attention mask
// construction. Binds directly to a TrtModule via bind_to().

#include "trtf/runtime/device_tensor.h"

#include <cstdint>
#include <vector>

#if TRTF_HAS_TRT

namespace trtf {

class TrtModule;

class KvCache {
public:
    // Allocate cache buffers for the given configuration.
    // kv_dim = num_kv_heads * head_dim (size of one K or V row per layer).
    KvCache(int32_t num_layers, int32_t max_length,
            int32_t kv_dim, cudaStream_t stream);

    // Per-step state query.
    int32_t position() const { return position_; }
    int32_t max_length() const { return max_length_; }
    int32_t num_layers() const { return num_layers_; }

    // Build the causal attention mask for the current position.
    // Output: mask of size [max_length], with 0.0 for visible positions and
    // -1e9 (or large negative) for masked positions.
    void build_attention_mask(std::vector<float>& mask) const;

    // Bind all cache and present tensors to a TrtModule.
    // Binds: cache_k_{i}, cache_v_{i} (inputs), present_k_{i}, present_v_{i} (outputs).
    void bind_to(TrtModule& module);

    // After each decode step: append present-K/V into cache, advance position.
    void advance();

    // Reset for a new sequence (zero all buffers, position = 0).
    void reset();

    // Direct access for advanced use (cross-attention, VL embedding).
    DeviceTensor& cache_k(int32_t layer) { return cache_k_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& cache_v(int32_t layer) { return cache_v_[static_cast<std::size_t>(layer)]; }

    bool ok() const;

private:
    std::vector<DeviceTensor> cache_k_;    // [num_layers], shape [max_length, kv_dim]
    std::vector<DeviceTensor> cache_v_;    // [num_layers]
    std::vector<DeviceTensor> present_k_;  // [num_layers], shape [1, kv_dim] (single step output)
    std::vector<DeviceTensor> present_v_;  // [num_layers]
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t kv_dim_{0};
    int32_t position_{0};
    cudaStream_t stream_{nullptr};
};

} // namespace trtf

#endif // TRTF_HAS_TRT
