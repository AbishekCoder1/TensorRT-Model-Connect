#pragma once

// IBackend: virtual interface for TRT backend DSOs.
// Each DSO (libtrtf_backend_trt.so, libtrtf_backend_rtx.so) implements
// IBackend and exports C ABI factory functions.

#include "trtf/runtime/trt_module.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

// Options for module creation. RTX-specific fields are silently ignored
// by the standard TRT backend.
struct ModuleCreateOptions {
    cudaStream_t stream{nullptr};       // nullptr = backend creates one
    const char* runtime_cache_path{""}; // RTX: JIT kernel cache file path
    bool cuda_graphs{false};            // RTX: whole-graph CUDA capture
};

// Two modules created from a single engine, one per optimization profile.
// Both share the engine (weights live once on GPU) and the same CUDA stream.
// When the engine has fewer than two profiles, `prefill` is null and `decode`
// holds the only context.
struct BackendDualProfileModules {
    std::unique_ptr<ITrtModule> prefill; // profile 0 — batched-Sq prefill (null if single-profile)
    std::unique_ptr<ITrtModule> decode;  // profile 1, or the only profile if single-profile
};

struct BackendProfileModule {
    int32_t profile_idx{0};
    std::unique_ptr<ITrtModule> module;
};

struct BackendProfileModules {
    std::vector<BackendProfileModule> modules;
};

// Per-DSO backend. Holds shared state (TRT runtime, RTX runtime cache).
// One IBackend creates all ITrtModule instances for a pipeline.
class IBackend {
  public:
    virtual ~IBackend() = default;

    // Deserialize an engine plan and create a module.
    virtual std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                                      const ModuleCreateOptions& options) = 0;

    // Deserialize once, create two execution contexts sharing the engine —
    // profile 0 for prefill, profile 1 for decode. Falls back to single-
    // profile (prefill=null) when the engine only has one profile.
    virtual BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) = 0;

    // Deserialize once and create one execution context per requested profile.
    // Returned modules share engine weights and stream ownership.
    virtual BackendProfileModules
    create_profile_modules(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<int32_t>& profile_indices) = 0;

    // Backend identity: "trt" or "trt_rtx"
    virtual const char* name() const = 0;
};

} // namespace trtf

// C ABI exported by each DSO. The main binary resolves these via dlsym.
extern "C" {
trtf::IBackend* trtf_create_backend();
void trtf_destroy_backend(trtf::IBackend* backend);
}
