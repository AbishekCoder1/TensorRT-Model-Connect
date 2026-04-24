#include "trtf/runtime/trt_module.h"


#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtf {

// --- DType conversion ---

DType TrtModule::from_trt_dtype(nvinfer1::DataType dt) {
    switch (dt) {
    case nvinfer1::DataType::kFLOAT:
        return DType::kFloat32;
    case nvinfer1::DataType::kHALF:
        return DType::kFloat16;
    case nvinfer1::DataType::kBF16:
        return DType::kBFloat16;
    case nvinfer1::DataType::kINT32:
        return DType::kInt32;
    case nvinfer1::DataType::kINT8:
        return DType::kInt8;
    default:
        return DType::kFloat32;
    }
}

// --- Construction ---

TrtModule::TrtModule(nvinfer1::ICudaEngine* engine, cudaStream_t stream, int32_t profile_idx,
                     std::vector<std::string> external_inputs)
    : engine_(engine), stream_(stream), profile_idx_(profile_idx),
      external_inputs_(external_inputs.begin(), external_inputs.end()) {
    ctx_ = engine->createExecutionContext();
    if (!ctx_)
        return;
    const int32_t num_profiles = engine->getNbOptimizationProfiles();
    if (num_profiles > 0) {
        if (!ctx_->setOptimizationProfileAsync(profile_idx_, stream_)) {
            std::cerr << "[trt_module] Failed to set optimization profile " << profile_idx_ << "\n";
            delete ctx_;
            ctx_ = nullptr;
            return;
        }
        cudaStreamSynchronize(stream_);
    }
    allocate_buffers(engine);
}

TrtModule::~TrtModule() {
    free_buffers();
    delete ctx_;
}

TrtModule::TrtModule(TrtModule&& other) noexcept
    : engine_(other.engine_), ctx_(other.ctx_), stream_(other.stream_),
      profile_idx_(other.profile_idx_), has_dynamic_shapes_(other.has_dynamic_shapes_),
      keep_alive_(std::move(other.keep_alive_)),
      external_inputs_(std::move(other.external_inputs_)), buffers_(std::move(other.buffers_)),
      host_output_staging_(std::move(other.host_output_staging_)),
      output_device_tensors_(std::move(other.output_device_tensors_)) {
    other.engine_ = nullptr;
    other.ctx_ = nullptr;
}

TrtModule& TrtModule::operator=(TrtModule&& other) noexcept {
    if (this != &other) {
        free_buffers();
        delete ctx_;
        engine_ = other.engine_;
        ctx_ = other.ctx_;
        stream_ = other.stream_;
        profile_idx_ = other.profile_idx_;
        has_dynamic_shapes_ = other.has_dynamic_shapes_;
        keep_alive_ = std::move(other.keep_alive_);
        external_inputs_ = std::move(other.external_inputs_);
        buffers_ = std::move(other.buffers_);
        host_output_staging_ = std::move(other.host_output_staging_);
        output_device_tensors_ = std::move(other.output_device_tensors_);
        other.engine_ = nullptr;
        other.ctx_ = nullptr;
    }
    return *this;
}

bool TrtModule::ok() const {
    return ctx_ != nullptr;
}

void TrtModule::keep_alive(std::shared_ptr<void> resource) {
    keep_alive_.push_back(std::move(resource));
}

// --- Buffer allocation helpers ---

bool TrtModule::dims_are_dynamic(const nvinfer1::Dims& dims) {
    for (int32_t d = 0; d < dims.nbDims; ++d)
        if (dims.d[d] == -1)
            return true;
    return false;
}

std::vector<int64_t> TrtModule::dims_to_shape(const nvinfer1::Dims& dims) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int32_t d = 0; d < dims.nbDims; ++d)
        shape.push_back(dims.d[d]);
    return shape;
}

void TrtModule::update_dynamic_shape(const std::string& name, BufferEntry& entry,
                                     const std::vector<int64_t>& new_shape) {
    // Skip statically-shaped inputs: TRT rejects setInputShape on them even
    // when the engine as a whole advertises dynamic shapes via optimization
    // profiles.
    if (!has_dynamic_shapes_ || !entry.is_dynamic || new_shape == entry.shape)
        return;
    if (use_cuda_graph_)
        cuda_graph_.reset();
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(new_shape.size());
    for (int32_t d = 0; d < dims.nbDims; ++d)
        dims.d[d] = new_shape[d];
    if (!ctx_->setInputShape(name.c_str(), dims)) {
        throw std::runtime_error("[trt_module] Failed to set dynamic shape for input '" + name +
                                 "'");
    }
    entry.shape = new_shape;
    entry.nbytes = shape_nbytes(new_shape, entry.dtype);
    if (entry.alloc_nbytes > 0 && entry.nbytes > entry.alloc_nbytes) {
        throw std::runtime_error("[trt_module] Dynamic shape for input '" + name +
                                 "' exceeds allocated buffer size");
    }
}

std::size_t TrtModule::compute_alloc_bytes(const nvinfer1::Dims& dims, DType dtype,
                                           std::vector<int64_t>& shape_out) {
    shape_out.clear();
    std::size_t n = 1;
    for (int32_t d = 0; d < dims.nbDims; ++d) {
        int64_t dim = std::max(static_cast<int64_t>(dims.d[d]), int64_t{1});
        shape_out.push_back(dim);
        n *= static_cast<std::size_t>(dim);
    }
    return n * dtype_size(dtype);
}

std::size_t TrtModule::shape_nbytes(const std::vector<int64_t>& shape, DType dtype) {
    std::size_t n = 1;
    for (int64_t dim : shape)
        n *= static_cast<std::size_t>(std::max<int64_t>(dim, 1));
    return n * dtype_size(dtype);
}

void TrtModule::detect_dynamic_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io) {
    has_dynamic_shapes_ = false;
    for (int32_t i = 0; i < num_io && !has_dynamic_shapes_; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        if (dims_are_dynamic(engine->getTensorShape(name)))
            has_dynamic_shapes_ = true;
    }
}

void TrtModule::set_dynamic_input_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                         nvinfer1::OptProfileSelector selector) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        if (dims_are_dynamic(engine->getTensorShape(name))) {
            auto dims = engine->getProfileShape(name, profile_idx_, selector);
            ctx_->setInputShape(name, dims);
        }
    }
}

namespace {
void* allocate_and_zero(std::size_t nbytes, cudaStream_t stream) {
    void* ptr = nullptr;
    if (nbytes == 0)
        return ptr;
    if (cudaMalloc(&ptr, nbytes) != cudaSuccess)
        return nullptr;
    cudaMemsetAsync(ptr, 0, nbytes, stream);
    return ptr;
}
} // namespace

void TrtModule::allocate_single_input(nvinfer1::ICudaEngine* engine, const char* name,
                                      int32_t num_profiles, bool skip_allocation) {
    const auto trt_shape = engine->getTensorShape(name);
    const auto dtype = from_trt_dtype(engine->getTensorDataType(name));
    const bool is_dynamic = has_dynamic_shapes_ && num_profiles > 0 && dims_are_dynamic(trt_shape);

    nvinfer1::Dims alloc_dims = trt_shape;
    nvinfer1::Dims init_dims = trt_shape;
    if (is_dynamic) {
        alloc_dims =
            engine->getProfileShape(name, profile_idx_, nvinfer1::OptProfileSelector::kMAX);
        init_dims = engine->getProfileShape(name, profile_idx_, nvinfer1::OptProfileSelector::kOPT);
    }

    std::vector<int64_t> alloc_shape;
    const std::size_t alloc_nbytes = compute_alloc_bytes(alloc_dims, dtype, alloc_shape);
    std::vector<int64_t> init_shape = alloc_shape;
    std::size_t init_nbytes = alloc_nbytes;
    if (is_dynamic)
        init_nbytes = compute_alloc_bytes(init_dims, dtype, init_shape);

    BufferEntry entry;
    entry.dtype = dtype;
    entry.nbytes = init_nbytes;
    entry.alloc_nbytes = skip_allocation ? 0 : alloc_nbytes;
    entry.is_input = true;
    entry.shape = init_shape;
    entry.is_external = skip_allocation;
    entry.is_dynamic = is_dynamic;
    if (!skip_allocation)
        entry.d_ptr = allocate_and_zero(alloc_nbytes, stream_);
    if (entry.d_ptr)
        ctx_->setTensorAddress(name, entry.d_ptr);
    if (is_dynamic)
        ctx_->setInputShape(name, init_dims);

    buffers_[name] = std::move(entry);
}

void TrtModule::allocate_input_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                       int32_t num_profiles) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        const bool skip_allocation = external_inputs_.find(name) != external_inputs_.end();
        allocate_single_input(engine, name, num_profiles, skip_allocation);
    }
}

void TrtModule::allocate_output_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            continue;

        auto dtype = from_trt_dtype(engine->getTensorDataType(name));

        // For dynamic engines, query the context for inferred output shape
        // (based on the max input shapes set by the caller).
        // For static engines, use the engine shape directly.
        nvinfer1::Dims out_dims =
            has_dynamic_shapes_ ? ctx_->getTensorShape(name) : engine->getTensorShape(name);

        std::vector<int64_t> shape;
        std::size_t nbytes = compute_alloc_bytes(out_dims, dtype, shape);

        BufferEntry entry;
        entry.shape = shape;
        entry.dtype = dtype;
        entry.nbytes = nbytes;
        entry.alloc_nbytes = nbytes;
        entry.is_input = false;

        if (nbytes > 0) {
            auto err = cudaMalloc(&entry.d_ptr, nbytes);
            if (err != cudaSuccess)
                entry.d_ptr = nullptr;
            else
                cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
        }

        if (entry.d_ptr)
            ctx_->setTensorAddress(name, entry.d_ptr);

        if (nbytes > 0)
            host_output_staging_[name].resize(nbytes);

        buffers_[name] = std::move(entry);
    }
}

// --- Buffer allocation ---

void TrtModule::allocate_buffers(nvinfer1::ICudaEngine* engine) {
    const int32_t num_io = engine->getNbIOTensors();
    const int32_t num_profiles = engine->getNbOptimizationProfiles();

    detect_dynamic_shapes(engine, num_io);

    // Pass 1: allocate input buffers (use profile-0 max shape for dynamic inputs).
    allocate_input_buffers(engine, num_io, num_profiles);

    // Pass 2: allocate output buffers. For dynamic shapes, temporarily set
    // inputs to max shapes, query inferred output shapes, then restore opt.
    if (has_dynamic_shapes_ && num_profiles > 0)
        set_dynamic_input_shapes(engine, num_io, nvinfer1::OptProfileSelector::kMAX);

    allocate_output_buffers(engine, num_io);

    if (has_dynamic_shapes_ && num_profiles > 0)
        set_dynamic_input_shapes(engine, num_io, nvinfer1::OptProfileSelector::kOPT);

    cudaStreamSynchronize(stream_);
}

void TrtModule::free_buffers() {
    for (auto& [name, entry] : buffers_) {
        if (entry.d_ptr && !entry.is_external) {
            cudaFree(entry.d_ptr);
        }
        entry.d_ptr = nullptr;
    }
    buffers_.clear();
    host_output_staging_.clear();
    output_device_tensors_.clear();
}

// --- Forward pass (CPU → GPU → CPU) ---

TensorMap TrtModule::forward(const TensorMap& inputs) {
    forward_async(inputs);
    sync();

    // Download outputs — skip externally-bound buffers (they stay on device)
    TensorMap outputs;
    for (auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;
        if (entry.is_external)
            continue;

        if (has_dynamic_shapes_) {
            auto out_dims = ctx_->getTensorShape(name.c_str());
            entry.shape = dims_to_shape(out_dims);
            entry.nbytes = shape_nbytes(entry.shape, entry.dtype);
        }

        auto& staging = host_output_staging_[name];
        cudaMemcpy(staging.data(), entry.d_ptr, entry.nbytes, cudaMemcpyDeviceToHost);

        Tensor t;
        t.data = staging.data();
        t.shape = entry.shape;
        t.dtype = entry.dtype;
        outputs[name] = t;
    }
    return outputs;
}

// --- Forward async ---

void TrtModule::enable_cuda_graph() {
    use_cuda_graph_ = true;
    cuda_graph_.reset(); // Force re-capture on next execution
}

void TrtModule::rebind_buffers_after_reset() {
    for (auto& [name, entry] : buffers_) {
        if (entry.d_ptr)
            ctx_->setTensorAddress(name.c_str(), entry.d_ptr);
        if (entry.is_input && has_dynamic_shapes_ &&
            dims_are_dynamic(engine_->getTensorShape(name.c_str()))) {
            entry.shape.clear();
            entry.nbytes = 0;
        }
    }
}

void TrtModule::reset_execution_context() {
    if (engine_ == nullptr)
        return;
    delete ctx_;
    ctx_ = engine_->createExecutionContext();
    if (!ctx_)
        throw std::runtime_error("[trt_module] Failed to recreate execution context");

    if (engine_->getNbOptimizationProfiles() > 0) {
        if (!ctx_->setOptimizationProfileAsync(profile_idx_, stream_)) {
            delete ctx_;
            ctx_ = nullptr;
            throw std::runtime_error("[trt_module] Failed to reset optimization profile");
        }
        cudaStreamSynchronize(stream_);
    }
    if (use_cuda_graph_)
        cuda_graph_.reset();
    rebind_buffers_after_reset();
}

void TrtModule::forward_async(const TensorMap& inputs) {
    // Upload inputs H2D, updating shapes for dynamic engines
    for (const auto& [name, tensor] : inputs) {
        auto it = buffers_.find(name);
        if (it == buffers_.end())
            continue;
        auto& entry = it->second;
        if (!entry.is_input || !entry.d_ptr)
            continue;

        update_dynamic_shape(name, entry, tensor.shape);

        auto copy_bytes = std::min(tensor.nbytes(), entry.alloc_nbytes);
        if (copy_bytes > 0 && tensor.data) {
            cudaMemcpyAsync(entry.d_ptr, tensor.data, copy_bytes, cudaMemcpyHostToDevice, stream_);
        }
    }

    // Execute — either via CUDA Graph replay or direct enqueue
    execute_enqueue();
}

void TrtModule::execute_enqueue() {
    if (use_cuda_graph_ && cuda_graph_.ready()) {
        // Replay: launch captured graph (no per-kernel launch overhead)
        cuda_graph_.launch(stream_);
        return;
    }
    if (use_cuda_graph_) {
        // First execution: capture the TRT kernel sequence into a graph.
        // During capture, enqueueV3 is RECORDED but NOT executed.
        // After capture, we immediately launch the graph to execute
        // this step's kernels with the current inputs.
        cuda_graph_.begin_capture(stream_);
        ctx_->enqueueV3(stream_);
        if (!cuda_graph_.end_capture(stream_)) {
            // Capture failed — fall back to normal execution permanently
            std::cerr << "[cuda_graph] Capture failed, disabling CUDA Graphs\n";
            use_cuda_graph_ = false;
            ctx_->enqueueV3(stream_);
        } else {
            cuda_graph_.launch(stream_);
        }
        return;
    }
    // Normal execution (no CUDA Graph)
    ctx_->enqueueV3(stream_);
}

void TrtModule::sync() {
    cudaStreamSynchronize(stream_);
}

// --- Forward device async (GPU → GPU, no sync) ---

void TrtModule::forward_device_async(const DeviceTensorMap& inputs) {
    // D2D copy input DeviceTensors into our buffers
    for (const auto& [name, dt_ptr] : inputs) {
        auto it = buffers_.find(name);
        if (it == buffers_.end() || !dt_ptr)
            continue;
        auto& entry = it->second;
        if (!entry.is_input || !entry.d_ptr)
            continue;

        update_dynamic_shape(name, entry, dt_ptr->shape());

        if (dt_ptr->data() != entry.d_ptr) {
            auto copy_bytes = std::min(dt_ptr->nbytes(), entry.alloc_nbytes);
            if (copy_bytes > 0) {
                cudaMemcpyAsync(entry.d_ptr, dt_ptr->data(), copy_bytes, cudaMemcpyDeviceToDevice,
                                stream_);
            }
        }
    }

    // Execute (no sync — caller will sync or run more kernels on same stream)
    ctx_->enqueueV3(stream_);
}

// --- Forward device (GPU → GPU, synchronous) ---

DeviceTensorMap TrtModule::forward_device(const DeviceTensorMap& inputs) {
    forward_device_async(inputs);
    cudaStreamSynchronize(stream_);

    // Return non-owning DeviceTensor* pointers to our internal output buffers.
    // The output_device_tensors_ map is lazily populated on first call.
    DeviceTensorMap out;
    for (auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;

        auto it = output_device_tensors_.find(name);
        if (it == output_device_tensors_.end()) {
            // Create a non-owning view. DeviceTensor constructor allocates memory,
            // so we create a placeholder and overwrite its pointer below.
            // Instead, just map the name to nullptr for now — callers use device_ptr().
        }
        out[name] = nullptr; // callers access via device_ptr(name)
    }
    return out;
}

// --- Introspection ---

std::vector<TensorInfo> TrtModule::input_info() const {
    std::vector<TensorInfo> result;
    for (const auto& [name, entry] : buffers_) {
        if (!entry.is_input)
            continue;
        TensorInfo ti;
        ti.name = name;
        ti.shape = entry.shape;
        ti.dtype = entry.dtype;
        ti.is_input = true;
        result.push_back(ti);
    }
    return result;
}

std::vector<TensorInfo> TrtModule::output_info() const {
    std::vector<TensorInfo> result;
    for (const auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;
        TensorInfo ti;
        ti.name = name;
        ti.shape = entry.shape;
        ti.dtype = entry.dtype;
        ti.is_input = false;
        result.push_back(ti);
    }
    return result;
}

bool TrtModule::has_input(const std::string& name) const {
    auto it = buffers_.find(name);
    return it != buffers_.end() && it->second.is_input;
}

bool TrtModule::has_output(const std::string& name) const {
    auto it = buffers_.find(name);
    return it != buffers_.end() && !it->second.is_input;
}

bool TrtModule::input_is_dynamic(const std::string& name) const {
    auto it = buffers_.find(name);
    return it != buffers_.end() && it->second.is_input && it->second.is_dynamic;
}

int32_t TrtModule::input_rank(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end() || !it->second.is_input)
        return 0;
    return static_cast<int32_t>(it->second.shape.size());
}

// --- Direct buffer access ---

void* TrtModule::device_ptr(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        return nullptr;
    return it->second.d_ptr;
}

void TrtModule::bind_external(const std::string& name, void* external_device_ptr) {
    bind_external(name, external_device_ptr, {});
}

void TrtModule::bind_external(const std::string& name, void* external_device_ptr,
                              const std::vector<int64_t>& shape) {
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        return;

    auto& entry = it->second;

    // Free our own buffer if we allocated it
    if (entry.d_ptr && !entry.is_external) {
        cudaFree(entry.d_ptr);
    }

    entry.d_ptr = external_device_ptr;
    entry.is_external = true;

    // Update execution context binding
    if (ctx_ && external_device_ptr) {
        ctx_->setTensorAddress(name.c_str(), external_device_ptr);
    }

    if (!shape.empty())
        update_dynamic_shape(name, entry, shape);
}

} // namespace trtf

