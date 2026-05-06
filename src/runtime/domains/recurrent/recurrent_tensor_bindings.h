#pragma once

#include "runtime/core/cuda_common.h"
#include "runtime/core/trt_engine_lifecycle.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct PendingCopy {
    void* host_ptr{nullptr};
    void* device_ptr{nullptr};
    std::size_t bytes{0};
};

template <typename Engine, std::size_t N>
bool has_required_tensors(const Engine& trt_engine,
                          const std::array<const std::string*, N>& names) {
    for (const std::string* name : names) {
        if (!has_io_tensor(trt_engine, *name)) {
            return false;
        }
    }
    return true;
}

template <typename StepEngine>
bool bind_input_tensor(const StepEngine& engine, CudaStream& stream,
                       std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
                       const std::string& name, const void* host_ptr, std::size_t bytes) {
    auto buffer = std::make_unique<CudaBuffer>(bytes);
    if (!buffer->ok()) {
        return false;
    }
    if (cudaMemcpyAsync(buffer->data(), host_ptr, bytes, cudaMemcpyHostToDevice, stream.get()) !=
        cudaSuccess) {
        return false;
    }
    engine.module->bind_external(name, buffer->data());

    device_buffers.push_back(std::move(buffer));
    return true;
}

template <typename StepEngine>
bool bind_output_tensor(const StepEngine& engine,
                        std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
                        std::vector<PendingCopy>& output_copies, const std::string& name,
                        void* host_ptr, std::size_t bytes) {
    auto buffer = std::make_unique<CudaBuffer>(bytes);
    if (!buffer->ok()) {
        return false;
    }
    engine.module->bind_external(name, buffer->data());

    output_copies.push_back(PendingCopy{host_ptr, buffer->data(), bytes});
    device_buffers.push_back(std::move(buffer));
    return true;
}

inline bool copy_outputs_to_host(const std::vector<PendingCopy>& output_copies,
                                 CudaStream& stream) {
    for (const PendingCopy& copy : output_copies) {
        if (cudaMemcpyAsync(copy.host_ptr, copy.device_ptr, copy.bytes, cudaMemcpyDeviceToHost,
                            stream.get()) != cudaSuccess) {
            return false;
        }
    }
    return true;
}

template <typename StepEngine>
bool execute_recurrent_step(const StepEngine& engine, CudaStream& stream,
                            const std::vector<PendingCopy>& output_copies, std::string& error) {
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess) {
        error = "cudaStreamSynchronize failed";
        return false;
    }
    engine.module->forward_async({});
    engine.module->sync();
    if (!copy_outputs_to_host(output_copies, stream)) {
        error = "cudaMemcpyAsync output failed";
        return false;
    }
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess) {
        error = "cudaStreamSynchronize failed";
        return false;
    }
    return true;
}

} // namespace trtmc
