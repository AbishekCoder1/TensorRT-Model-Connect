#pragma once

// TrtModuleImpl: concrete ITrtModule backed by a TRT engine.
// Compiled inside backend DSOs only (libtrtf_backend_trt.so / _rtx.so).

#include "trtf/runtime/trt_module.h"

#include <NvInfer.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtf {

class CudaGraphExec;

class TrtModuleImpl final : public ITrtModule {
  public:
    // Backend creates engine + context, passes them in.
    // The engine must outlive this module (caller manages lifetime via keep_alive).
    TrtModuleImpl(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* ctx,
                  cudaStream_t stream, int32_t profile_idx = 0);
    ~TrtModuleImpl() override;

    TrtModuleImpl(const TrtModuleImpl&) = delete;
    TrtModuleImpl& operator=(const TrtModuleImpl&) = delete;

    // ITrtModule interface
    TensorMap forward(const TensorMap& inputs) override;
    DeviceTensorMap forward_device(const DeviceTensorMap& inputs) override;
    void forward_device_async(const DeviceTensorMap& inputs) override;
    void forward_async(const TensorMap& inputs) override;
    void sync() override;
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override;
    bool cuda_graph_active() const override { return use_cuda_graph_; }
    int32_t profile_idx() const override { return profile_idx_; }
    std::vector<TensorInfo> input_info() const override;
    std::vector<TensorInfo> output_info() const override;
    bool has_input(const std::string& name) const override;
    bool has_output(const std::string& name) const override;
    void* device_ptr(const std::string& name) const override;
    void bind_external(const std::string& name, void* ptr) override;
    void bind_external(const std::string& name, void* ptr,
                       const std::vector<int64_t>& shape) override;
    int32_t input_rank(const std::string& name) const override;
    bool input_is_dynamic(const std::string& name) const override;
    void reset_execution_context() override;
    bool ok() const override { return ctx_ != nullptr; }
    void keep_alive(std::shared_ptr<void> resource) override;

  private:
    struct BufferEntry {
        void* d_ptr{nullptr};
        std::vector<int64_t> shape;
        DType dtype{DType::kFloat32};
        std::size_t nbytes{0};
        bool is_input{true};
        bool is_external{false};
    };

    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* ctx_{nullptr};
    cudaStream_t stream_{nullptr};
    int32_t profile_idx_{0};
    bool has_dynamic_shapes_{false};
    bool use_cuda_graph_{false};
    std::unique_ptr<CudaGraphExec> cuda_graph_;
    std::vector<std::shared_ptr<void>> keep_alive_;
    std::unordered_map<std::string, BufferEntry> buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>> host_output_staging_;
    std::unordered_map<std::string, DeviceTensor> output_device_tensors_;

    void allocate_buffers(nvinfer1::ICudaEngine* engine);
    void free_buffers();
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
    void execute_enqueue();
    static bool dims_are_dynamic(const nvinfer1::Dims& dims);
    static std::vector<int64_t> dims_to_shape(const nvinfer1::Dims& dims);
    static std::size_t compute_alloc_bytes(const nvinfer1::Dims& dims, DType dtype,
                                           std::vector<int64_t>& shape_out);
    static DType from_trt_dtype(nvinfer1::DataType dt);
};

} // namespace trtf
