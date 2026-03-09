# TASK-01: TrtModule + DeviceTensor — the model.forward() abstraction

## Status: ready
## Phase: 1 (Foundation)
## Risk: low — additive only, no existing code changes

## Goal

Create `TrtModule` (the `nn.Module.forward()` equivalent) and `DeviceTensor`
(the `torch.Tensor` on CUDA equivalent). These are the two foundational building
blocks that all pipelines will compose.

After this task, calling a TRT engine should be:
```cpp
TensorMap outputs = module.forward({{"input_ids", ids_tensor}, {"attention_mask", mask_tensor}});
// That's it. No set_tensor_address, no cudaMemcpyAsync, no enqueueV3, no sync.
```

## TrtModule interface

```cpp
class TrtModule {
public:
    explicit TrtModule(nvinfer1::ICudaEngine* engine, cudaStream_t stream);

    // === The "forward pass" ===
    // CPU tensors in → CPU tensors out (H2D + execute + D2H + sync)
    TensorMap forward(const TensorMap& inputs);

    // GPU tensors in → GPU tensors out (no transfers, just execute + sync)
    DeviceTensorMap forward_device(const DeviceTensorMap& inputs);

    // === Async variants (caller calls sync()) ===
    void forward_async(const TensorMap& inputs);
    void sync();

    // === Introspection ===
    std::vector<TensorInfo> inputs() const;
    std::vector<TensorInfo> outputs() const;
    bool has_input(const std::string& name) const;

    // === Direct buffer access (for KvCache/RecurrentState binding) ===
    void* device_ptr(const std::string& name);
    void bind_external(const std::string& name, void* device_ptr);
};
```

## DeviceTensor interface

```cpp
class DeviceTensor {
public:
    DeviceTensor(std::vector<int64_t> shape, DType dtype, cudaStream_t stream);

    // Transfers
    void copy_from_host(const void* data);
    void copy_to_host(void* data) const;
    void copy_from(const DeviceTensor& other);  // D2D

    // Static factory
    static DeviceTensor randn(std::vector<int64_t> shape, cudaStream_t stream, uint64_t seed);
    static DeviceTensor zeros(std::vector<int64_t> shape, DType dtype, cudaStream_t stream);

    // Access
    void* data();
    const void* data() const;
    const std::vector<int64_t>& shape() const;
    DType dtype() const;
    size_t nbytes() const;
    int64_t numel() const;
};
```

## Implementation notes

- `TrtModule` constructor: iterate `engine->num_io_tensors()`, pre-allocate one
  `CudaBuffer` per tensor, call `ctx->set_tensor_address()` once. All reused across calls.
- `forward()`: loop inputs, `cudaMemcpyAsync` H2D → `enqueueV3` → loop outputs,
  `cudaMemcpyAsync` D2H → `cudaStreamSynchronize`. Return TensorMap of host arrays.
- `forward_device()`: for each input DeviceTensor, either `bind_external` or D2D copy
  → `enqueueV3` → sync. Return refs to pre-allocated output DeviceTensors.
- `bind_external()`: replaces pre-allocated buffer with external pointer. Used by KvCache
  to bind cache_k[layer] directly into the module's context.
- `DeviceTensor` wraps existing `CudaBuffer` with shape metadata.

## Files to create

- `include/trtf/runtime/tensor.h` — Tensor, TensorMap, TensorInfo, DType
- `include/trtf/runtime/device_tensor.h` — DeviceTensor
- `include/trtf/runtime/trt_module.h` — TrtModule
- `src/runtime/trt/core/trt_module.cpp`
- `src/runtime/trt/core/device_tensor.cpp`
- `tests/cpp/test_trt_module.cpp` — build tiny identity engine, test forward/forward_device
- `tests/cpp/test_device_tensor.cpp` — test alloc, H2D, D2H, D2D, randn

## Acceptance criteria

- [ ] `TrtModule::forward()` works with a real TRT engine in the container
- [ ] `forward_device()` keeps data on GPU (verify via D2H after)
- [ ] `bind_external()` allows KvCache-style external buffer binding
- [ ] `DeviceTensor::randn()` produces GPU random data
- [ ] All tests pass in container with `ctest --test-dir build -R test_trt_module`

## Dependencies

None — this is the foundation.
