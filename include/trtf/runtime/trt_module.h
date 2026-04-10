#pragma once

// ITrtModule: virtual interface for TRT engine execution.
// Concrete implementations live in backend DSOs (libtrtf_backend_*.so).
// No TRT headers — only CUDA runtime types and our own tensor types.

#include "trtf/runtime/device_tensor.h"
#include "trtf/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

class ITrtModule {
public:
    virtual ~ITrtModule() = default;

    // Forward passes
    virtual TensorMap forward(const TensorMap& inputs) = 0;
    virtual DeviceTensorMap forward_device(const DeviceTensorMap& inputs) = 0;
    virtual void forward_device_async(const DeviceTensorMap& inputs) = 0;
    virtual void forward_async(const TensorMap& inputs) = 0;
    virtual void sync() = 0;

    // Introspection
    virtual cudaStream_t stream() const = 0;
    virtual std::vector<TensorInfo> input_info() const = 0;
    virtual std::vector<TensorInfo> output_info() const = 0;
    virtual bool has_input(const std::string& name) const = 0;
    virtual bool has_output(const std::string& name) const = 0;

    // Direct buffer access (KV cache binding)
    virtual void* device_ptr(const std::string& name) const = 0;
    virtual void bind_external(const std::string& name, void* ptr) = 0;

    virtual bool ok() const = 0;
    virtual void keep_alive(std::shared_ptr<void> resource) = 0;
};

// Backward-compat alias — all existing code references TrtModule.
// This alias lets it compile without changes during migration.
using TrtModule = ITrtModule;

} // namespace trtf
