#pragma once

// TrtModule: the model.forward() abstraction for TensorRT engines.
//
// Wraps a TRT engine + execution context. All I/O binding, H2D/D2H transfers,
// execution, and synchronization are hidden inside forward().
//
// Usage:
//   auto module = TrtModule(engine, stream);
//   TensorMap outputs = module.forward({{"input_ids", ids}, {"mask", mask}});
//   // Done. No set_tensor_address, no cudaMemcpyAsync, no enqueueV3.
//
// HF equivalent: nn.Module.__call__() / model(input_ids, attention_mask)

#include "trtf/runtime/device_tensor.h"
#include "trtf/runtime/tensor.h"

#include <string>
#include <unordered_map>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace trtf {

#if TRTF_HAS_TRT

class TrtModule {
  public:
    // Construct from a deserialized engine. Pre-allocates all device buffers
    // and binds them to the execution context.
    // The engine must outlive this TrtModule (caller owns it).
    TrtModule(nvinfer1::ICudaEngine* engine, cudaStream_t stream);
    ~TrtModule();

    // Non-copyable, movable.
    TrtModule(const TrtModule&) = delete;
    TrtModule& operator=(const TrtModule&) = delete;
    TrtModule(TrtModule&& other) noexcept;
    TrtModule& operator=(TrtModule&& other) noexcept;

    // === The "forward pass" ===

    // CPU → GPU → execute → GPU → CPU (synchronous).
    // Uploads each input from TensorMap, runs the engine, downloads outputs.
    TensorMap forward(const TensorMap& inputs);

    // GPU → execute → GPU (no H2D/D2H). Caller provides DeviceTensors.
    // Returns pointers to the module's internal output buffers.
    DeviceTensorMap forward_device(const DeviceTensorMap& inputs);

    // GPU → execute (no sync, no D2H). Caller calls sync() later.
    // Enables GPU-resident decode loops: forward_device_async() → device
    // kernels on same stream → periodic sync().
    void forward_device_async(const DeviceTensorMap& inputs);

    // Async: upload + enqueue (no sync). Caller calls sync() later.
    void forward_async(const TensorMap& inputs);
    void sync();

    // Access the CUDA stream used by this module.
    cudaStream_t stream() const { return stream_; }

    // === Introspection ===

    std::vector<TensorInfo> input_info() const;
    std::vector<TensorInfo> output_info() const;
    bool has_input(const std::string& name) const;
    bool has_output(const std::string& name) const;

    // === Direct buffer access (for KvCache binding) ===

    // Get the device pointer for a named tensor (input or output).
    void* device_ptr(const std::string& name) const;

    // Override the pre-allocated buffer for a tensor with an external pointer.
    // Used by KvCache to bind cache_k/v directly.
    // The caller owns the external memory and must keep it alive.
    void bind_external(const std::string& name, void* external_device_ptr);

    bool ok() const;

    // Keep an opaque resource alive for the lifetime of this module.
    // Used by pipeline_factory to transfer ownership of the TRT engine
    // and CUDA stream, ensuring they outlive the execution context.
    void keep_alive(std::shared_ptr<void> resource);

  private:
    struct BufferEntry {
        void* d_ptr{nullptr}; // Device pointer (owned unless external)
        std::vector<int64_t> shape;
        DType dtype{DType::kFloat32};
        std::size_t nbytes{0};
        bool is_input{true};
        bool is_external{false}; // If true, we don't free d_ptr
    };

    nvinfer1::IExecutionContext* ctx_{nullptr};
    cudaStream_t stream_{nullptr};
    bool has_dynamic_shapes_{false};                // True if engine uses optimization profiles
    std::vector<std::shared_ptr<void>> keep_alive_; // opaque resource ownership
    std::unordered_map<std::string, BufferEntry> buffers_;

    // Pre-allocated host staging buffers for output D2H
    std::unordered_map<std::string, std::vector<uint8_t>> host_output_staging_;

    // Internal output DeviceTensors returned by forward_device()
    std::unordered_map<std::string, DeviceTensor> output_device_tensors_;

    void allocate_buffers(nvinfer1::ICudaEngine* engine);
    void free_buffers();

    // Helpers extracted from allocate_buffers to reduce cyclomatic complexity.
    void detect_dynamic_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io);
    void allocate_input_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                int32_t num_profiles);
    void allocate_single_input(nvinfer1::ICudaEngine* engine, const char* name,
                               int32_t num_profiles);
    void allocate_output_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io);
    void set_dynamic_input_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                  nvinfer1::OptProfileSelector selector);
    void update_dynamic_shape(const std::string& name, BufferEntry& entry,
                              const std::vector<int64_t>& new_shape);

    static bool dims_are_dynamic(const nvinfer1::Dims& dims);
    static std::vector<int64_t> dims_to_shape(const nvinfer1::Dims& dims);
    static std::size_t compute_alloc_bytes(const nvinfer1::Dims& dims, DType dtype,
                                           std::vector<int64_t>& shape_out);

    static DType from_trt_dtype(nvinfer1::DataType dt);
};

#endif // TRTF_HAS_TRT

} // namespace trtf
