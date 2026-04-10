#pragma once

// IBackend: virtual interface for TRT backend DSOs.
// Each DSO (libtrtf_backend_trt.so, libtrtf_backend_rtx.so) implements
// IBackend and exports C ABI factory functions.

#include "trtf/runtime/trt_module.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

namespace trtf {

// Options for module creation. RTX-specific fields are silently ignored
// by the standard TRT backend.
struct ModuleCreateOptions {
    cudaStream_t stream{nullptr};           // nullptr = backend creates one
    const char* runtime_cache_path{""};     // RTX: JIT kernel cache file path
    bool cuda_graphs{false};                // RTX: whole-graph CUDA capture
};

// Per-DSO backend. Holds shared state (TRT runtime, RTX runtime cache).
// One IBackend creates all ITrtModule instances for a pipeline.
class IBackend {
public:
    virtual ~IBackend() = default;

    // Deserialize an engine plan and create a module.
    virtual std::unique_ptr<ITrtModule> create_module(
        const void* plan_data, size_t plan_size,
        const ModuleCreateOptions& options) = 0;

    // Backend identity: "trt" or "trt_rtx"
    virtual const char* name() const = 0;
};

} // namespace trtf

// C ABI exported by each DSO. The main binary resolves these via dlsym.
extern "C" {
    trtf::IBackend* trtf_create_backend();
    void trtf_destroy_backend(trtf::IBackend* backend);
}
