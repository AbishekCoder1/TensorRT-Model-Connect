#include "trtf/runtime/trt_module.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtf {

// --- DType conversion ---

DType TrtModule::from_trt_dtype(nvinfer1::DataType dt)
{
    switch (dt)
    {
    case nvinfer1::DataType::kFLOAT: return DType::kFloat32;
    case nvinfer1::DataType::kHALF:  return DType::kFloat16;
    case nvinfer1::DataType::kBF16:  return DType::kBFloat16;
    case nvinfer1::DataType::kINT32: return DType::kInt32;
    case nvinfer1::DataType::kINT8:  return DType::kInt8;
    default: return DType::kFloat32;
    }
}

// --- Construction ---

TrtModule::TrtModule(nvinfer1::ICudaEngine* engine, cudaStream_t stream)
    : stream_(stream)
{
    ctx_ = engine->createExecutionContext();
    if (!ctx_) return;
    allocate_buffers(engine);
}

TrtModule::~TrtModule()
{
    free_buffers();
    delete ctx_;
}

TrtModule::TrtModule(TrtModule&& other) noexcept
    : ctx_(other.ctx_)
    , stream_(other.stream_)
    , keep_alive_(std::move(other.keep_alive_))
    , buffers_(std::move(other.buffers_))
    , host_output_staging_(std::move(other.host_output_staging_))
    , output_device_tensors_(std::move(other.output_device_tensors_))
{
    other.ctx_ = nullptr;
}

TrtModule& TrtModule::operator=(TrtModule&& other) noexcept
{
    if (this != &other)
    {
        free_buffers();
        delete ctx_;
        ctx_ = other.ctx_;
        stream_ = other.stream_;
        keep_alive_ = std::move(other.keep_alive_);
        buffers_ = std::move(other.buffers_);
        host_output_staging_ = std::move(other.host_output_staging_);
        output_device_tensors_ = std::move(other.output_device_tensors_);
        other.ctx_ = nullptr;
    }
    return *this;
}

bool TrtModule::ok() const { return ctx_ != nullptr; }

void TrtModule::keep_alive(std::shared_ptr<void> resource)
{
    keep_alive_.push_back(std::move(resource));
}

// --- Buffer allocation ---

void TrtModule::allocate_buffers(nvinfer1::ICudaEngine* engine)
{
    const int32_t num_io = engine->getNbIOTensors();
    for (int32_t i = 0; i < num_io; ++i)
    {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto trt_shape = engine->getTensorShape(name);
        auto trt_dtype = engine->getTensorDataType(name);
        auto dtype = from_trt_dtype(trt_dtype);

        std::vector<int64_t> shape;
        std::size_t n = 1;
        for (int32_t d = 0; d < trt_shape.nbDims; ++d)
        {
            int64_t dim = std::max(static_cast<int64_t>(trt_shape.d[d]), int64_t{1});
            shape.push_back(dim);
            n *= static_cast<std::size_t>(dim);
        }
        std::size_t nbytes = n * dtype_size(dtype);

        BufferEntry entry;
        entry.shape = shape;
        entry.dtype = dtype;
        entry.nbytes = nbytes;
        entry.is_input = (mode == nvinfer1::TensorIOMode::kINPUT);

        if (nbytes > 0)
        {
            auto err = cudaMalloc(&entry.d_ptr, nbytes);
            if (err != cudaSuccess)
            {
                entry.d_ptr = nullptr;
            }
            else
            {
                cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
            }
        }

        // Set tensor address in execution context
        if (entry.d_ptr)
        {
            ctx_->setTensorAddress(name, entry.d_ptr);
        }

        // Pre-allocate host staging for outputs
        if (!entry.is_input && nbytes > 0)
        {
            host_output_staging_[name].resize(nbytes);
        }

        buffers_[name] = std::move(entry);
    }

    cudaStreamSynchronize(stream_);
}

void TrtModule::free_buffers()
{
    for (auto& [name, entry] : buffers_)
    {
        if (entry.d_ptr && !entry.is_external)
        {
            cudaFree(entry.d_ptr);
        }
        entry.d_ptr = nullptr;
    }
    buffers_.clear();
    host_output_staging_.clear();
    output_device_tensors_.clear();
}

// --- Forward pass (CPU → GPU → CPU) ---

TensorMap TrtModule::forward(const TensorMap& inputs)
{
    forward_async(inputs);
    sync();

    // Download outputs
    TensorMap outputs;
    for (auto& [name, entry] : buffers_)
    {
        if (entry.is_input) continue;

        auto& staging = host_output_staging_[name];
        cudaMemcpy(staging.data(), entry.d_ptr, entry.nbytes,
            cudaMemcpyDeviceToHost);

        Tensor t;
        t.data = staging.data();
        t.shape = entry.shape;
        t.dtype = entry.dtype;
        outputs[name] = t;
    }
    return outputs;
}

// --- Forward async ---

void TrtModule::forward_async(const TensorMap& inputs)
{
    // Upload inputs H2D
    for (const auto& [name, tensor] : inputs)
    {
        auto it = buffers_.find(name);
        if (it == buffers_.end()) continue;
        auto& entry = it->second;
        if (!entry.is_input || !entry.d_ptr) continue;

        auto copy_bytes = std::min(tensor.nbytes(), entry.nbytes);
        if (copy_bytes > 0 && tensor.data)
        {
            cudaMemcpyAsync(entry.d_ptr, tensor.data, copy_bytes,
                cudaMemcpyHostToDevice, stream_);
        }
    }

    // Execute
    ctx_->enqueueV3(stream_);
}

void TrtModule::sync()
{
    cudaStreamSynchronize(stream_);
}

// --- Forward device (GPU → GPU) ---

DeviceTensorMap TrtModule::forward_device(const DeviceTensorMap& inputs)
{
    // Bind or copy input DeviceTensors
    for (const auto& [name, dt_ptr] : inputs)
    {
        auto it = buffers_.find(name);
        if (it == buffers_.end() || !dt_ptr) continue;
        auto& entry = it->second;
        if (!entry.is_input) continue;

        if (dt_ptr->data() != entry.d_ptr)
        {
            // D2D copy from external DeviceTensor to our buffer
            auto copy_bytes = std::min(dt_ptr->nbytes(), entry.nbytes);
            if (copy_bytes > 0)
            {
                cudaMemcpyAsync(entry.d_ptr, dt_ptr->data(), copy_bytes,
                    cudaMemcpyDeviceToDevice, stream_);
            }
        }
    }

    // Execute + sync
    ctx_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);

    // Return pointers to internal output buffers (no copy)
    DeviceTensorMap out;
    for (auto& [name, entry] : buffers_)
    {
        if (entry.is_input) continue;
        // Create or reuse DeviceTensor pointing to our buffer
        // Note: these are "views" — the TrtModule owns the memory
        auto it = output_device_tensors_.find(name);
        if (it == output_device_tensors_.end())
        {
            // First call: create a DeviceTensor wrapper
            // We can't use the normal constructor (it would allocate new memory)
            // So we just store the pointer for the caller
        }
        // Return raw pointer info via the map
        // The caller gets a non-owning reference
        out[name] = nullptr; // placeholder — actual access via device_ptr()
    }
    return out;
}

// --- Introspection ---

std::vector<TensorInfo> TrtModule::input_info() const
{
    std::vector<TensorInfo> result;
    for (const auto& [name, entry] : buffers_)
    {
        if (!entry.is_input) continue;
        TensorInfo ti;
        ti.name = name;
        ti.shape = entry.shape;
        ti.dtype = entry.dtype;
        ti.is_input = true;
        result.push_back(ti);
    }
    return result;
}

std::vector<TensorInfo> TrtModule::output_info() const
{
    std::vector<TensorInfo> result;
    for (const auto& [name, entry] : buffers_)
    {
        if (entry.is_input) continue;
        TensorInfo ti;
        ti.name = name;
        ti.shape = entry.shape;
        ti.dtype = entry.dtype;
        ti.is_input = false;
        result.push_back(ti);
    }
    return result;
}

bool TrtModule::has_input(const std::string& name) const
{
    auto it = buffers_.find(name);
    return it != buffers_.end() && it->second.is_input;
}

bool TrtModule::has_output(const std::string& name) const
{
    auto it = buffers_.find(name);
    return it != buffers_.end() && !it->second.is_input;
}

// --- Direct buffer access ---

void* TrtModule::device_ptr(const std::string& name) const
{
    auto it = buffers_.find(name);
    if (it == buffers_.end()) return nullptr;
    return it->second.d_ptr;
}

void TrtModule::bind_external(const std::string& name, void* external_device_ptr)
{
    auto it = buffers_.find(name);
    if (it == buffers_.end()) return;

    auto& entry = it->second;

    // Free our own buffer if we allocated it
    if (entry.d_ptr && !entry.is_external)
    {
        cudaFree(entry.d_ptr);
    }

    entry.d_ptr = external_device_ptr;
    entry.is_external = true;

    // Update execution context binding
    if (ctx_ && external_device_ptr)
    {
        ctx_->setTensorAddress(name.c_str(), external_device_ptr);
    }
}

} // namespace trtf

#endif // TRTF_HAS_TRT
