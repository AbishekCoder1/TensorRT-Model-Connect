#pragma once

#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Device-resident KV cache for standard attention-based decoders.
// Keeps the entire KV cache on GPU. Only small inputs (token_id, position_id,
// mask) are transferred H2D per step; cache updates are D2D memcpy.
class DeviceKvCache {
public:
    explicit DeviceKvCache(const DecoderStepEngine& engine);

    // Compute position_id and attention mask for the current step.
    void prepare_step(int32_t& out_position_id, std::vector<float>& out_mask);

    // D2D copy present_k/v outputs into the persistent cache buffers.
    void update_after_step(const std::vector<CudaBuffer>& present_k,
                           const std::vector<CudaBuffer>& present_v,
                           cudaStream_t stream);

    // Zero all cache buffers and reset cache_length.
    void reset(cudaStream_t stream);

    void* cache_k_device_ptr(int32_t layer) const;
    void* cache_v_device_ptr(int32_t layer) const;

    bool ok() const;

private:
    int32_t mCacheStateSize;
    int32_t mMaxCacheLength;
    int32_t mNumLayers;
    bool mIncludeCurrentSlot;
    int32_t mPositionLimit;
    int32_t mCacheLength{0};
    std::vector<CudaBuffer> mCacheK;  // persistent GPU memory [num_layers]
    std::vector<CudaBuffer> mCacheV;
};

// Pre-allocated device buffers for small per-step I/O (token_id, position_id,
// mask, logits, present_k/v outputs). Avoids repeated cudaMalloc per step.
struct DeviceResources {
    CudaStream stream;
    CudaBuffer d_token_id;
    CudaBuffer d_position_id;
    CudaBuffer d_mask;
    CudaBuffer d_logits;
    CudaBuffer d_input_embed;      // VL only (0 bytes otherwise)
    CudaBuffer d_use_input_embed;  // VL only (0 bytes otherwise)
    std::vector<CudaBuffer> d_present_k;  // [num_layers] single-row each
    std::vector<CudaBuffer> d_present_v;

    explicit DeviceResources(const DecoderStepEngine& engine);
    bool ok() const;
};

// Device-resident decode step: only transfers small inputs H2D, binds cache
// device ptrs, executes, D2D cache update, D2H logits.
bool run_decoder_step_device(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    int32_t token_id,
    std::vector<float>& logits,
    std::string& error,
    const float* input_embed_host = nullptr,
    int32_t embed_dim = 0,
    float use_input_embed = 0.0F);

#endif // TRTF_HAS_TRT

} // namespace trtf
