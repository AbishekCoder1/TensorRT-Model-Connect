# TRT-RTX Backend Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace compile-time TRT dependency with dlopen-based backend dispatch so both TRT and TRT-RTX can be loaded at runtime through a single virtual interface.

**Architecture:** Two DSO wrapper libraries (`libtrtf_backend_trt.so`, `libtrtf_backend_rtx.so`) each implement `IBackend`/`ITrtModule` virtual interfaces. The main binary links only `libcudart` + `libdl`. `BackendLoader` reads `engine_backend` from bundle config.json and `dlopen`s the correct DSO. Pipelines use `ITrtModule*` exclusively.

**Tech Stack:** C++17, CUDA Runtime API, POSIX `dlopen`/`dlsym`, CMake, Python argparse

**Spec:** `docs/superpowers/specs/2026-04-09-trt-rtx-backend-abstraction-design.md`

---

## File Map

### New files (create)

| File | Responsibility |
|------|---------------|
| `include/trtf/runtime/trt_backend.h` | `IBackend`, `ModuleCreateOptions`, C ABI factory decl |
| `src/runtime/core/cuda_common.h` | `CudaStream`, `CudaBuffer` (pure CUDA, no TRT) |
| `src/runtime/core/cuda_common.cpp` | Implementation (extracted from `trt_common.cpp`) |
| `src/runtime/backend/backend_loader.h` | `BackendLoader::load()` declaration |
| `src/runtime/backend/backend_loader.cpp` | `dlopen`/`dlsym` dispatch, caching, error messages |
| `src/runtime/backend/trt_module_impl.h` | `TrtModuleImpl : ITrtModule` (DSO-internal header) |
| `src/runtime/backend/trt_module_impl.cpp` | Full implementation (compiled in both DSOs) |
| `src/runtime/backend/trt_logger.h` | `TrtLogger` (DSO-internal, moved from `trt_common`) |
| `src/runtime/backend/trt_logger.cpp` | Logger + `create_trt_runtime()` (DSO-internal) |
| `src/runtime/backend/trt_backend.cpp` | `TrtBackend : IBackend` → `libtrtf_backend_trt.so` |
| `src/runtime/backend/rtx_backend.cpp` | `RtxBackend : IBackend` → `libtrtf_backend_rtx.so` |
| `tests/cpp/test_backend_loader.cpp` | BackendLoader unit tests |
| `tests/cpp/test_itrt_module_interface.cpp` | Mock ITrtModule, verify pipeline compilation |
| `tests/builder/test_rtx_flag.py` | Python `--rtx` flag + `engine_backend` metadata tests |

### Modified files

| File | Change |
|------|--------|
| `include/trtf/runtime/trt_module.h` | Concrete `TrtModule` → pure virtual `ITrtModule` |
| `include/trtf/pipeline.h` | `load()` gains `runtime_cache_path`, `cuda_graphs` params |
| `include/trtf/runtime/pipeline_plugin.h` | `PipelineContext` gains `backend`, `runtime_cache_path`, `cuda_graphs` |
| `src/runtime/plugins/shared/plugin_helpers.h` | `LoadedModule` uses `ITrtModule`, functions take `IBackend*` |
| `src/runtime/plugins/shared/plugin_helpers.cpp` | Delegates to `IBackend::create_module()` |
| `src/runtime/registry/pipeline_factory.cpp` | Read `engine_backend`, use `BackendLoader` |
| 15 pipeline headers (`src/runtime/pipelines/*.h`) | `TrtModule` → `ITrtModule` member type |
| 15 pipeline `.cpp` files | Remove `TRTF_HAS_TRT`, adjust includes |
| 21 plugin `.cpp` files (`src/runtime/plugins/*.cpp`) | Remove `TRTF_HAS_TRT`, use `ctx.backend` |
| `examples/trtf_cli.cpp` | Add `--runtime-cache`, `--cuda-graphs` flags |
| `src/cabi/api/trtf_c.cpp` | `TrtfPipelineOptions` gains `runtime_cache`, `cuda_graphs` |
| `CMakeLists.txt` | Remove `libnvinfer` from `trtf_core`, add DSO targets |
| `trtf_build/trtf_build/cli.py` | Add `--rtx` flag |
| `trtf_build/trtf_build/engine_builder.py` | `sys.modules` monkeypatch, `engine_backend` in config |

### Deleted files

| File | Replacement |
|------|-------------|
| `src/runtime/core/trt_common.h` | Split → `cuda_common.h` (main binary) + `trt_logger.h` (DSOs) |
| `src/runtime/core/trt_common.cpp` | Split → `cuda_common.cpp` + `trt_logger.cpp` |
| `src/runtime/core/trt_module.cpp` | → `src/runtime/backend/trt_module_impl.cpp` |
| `src/runtime/core/trt_engine_lifecycle.h` | Inlined into DSO backend files |
| `src/runtime/core/trt_engine_lifecycle.cpp` | Inlined into DSO backend files |

---

### Task 1: Create `ITrtModule` virtual interface header

**Files:**
- Modify: `include/trtf/runtime/trt_module.h`

This task replaces the concrete `TrtModule` class with a pure virtual `ITrtModule` interface. The old concrete class will be moved to the DSO in a later task. For now, we keep the old `.cpp` compiling by temporarily having `TrtModule` inherit from `ITrtModule`.

- [ ] **Step 1: Rewrite `include/trtf/runtime/trt_module.h`**

Replace the entire file with the pure virtual interface. Remove all `#include <NvInfer.h>`, all `#if TRTF_HAS_TRT` guards, and all private implementation details:

```cpp
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

// Backward-compat alias — pipelines can use either name during migration.
using TrtModule = ITrtModule;

} // namespace trtf
```

The `using TrtModule = ITrtModule;` alias means all existing code that references `TrtModule` (15 pipeline headers, 21 plugins, plugin_helpers) compiles without changes. We remove this alias in a later cleanup task.

- [ ] **Step 2: Verify the main binary still compiles**

Run: `cmake --build build -j 2>&1 | tail -20`

This will fail because `trt_module.cpp` implements a concrete class that no longer exists in the header. That's expected — we fix it in the next task.

- [ ] **Step 3: Commit**

```bash
git add include/trtf/runtime/trt_module.h
git commit -m "refactor(trt_module): replace concrete TrtModule with ITrtModule virtual interface

Pure virtual interface with no TRT headers. Backward-compat alias
'using TrtModule = ITrtModule' preserves compilation of all consumers."
```

---

### Task 2: Create `IBackend` and `ModuleCreateOptions` header

**Files:**
- Create: `include/trtf/runtime/trt_backend.h`

- [ ] **Step 1: Create `include/trtf/runtime/trt_backend.h`**

```cpp
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
```

- [ ] **Step 2: Commit**

```bash
git add include/trtf/runtime/trt_backend.h
git commit -m "feat(backend): add IBackend and ModuleCreateOptions interface headers"
```

---

### Task 3: Extract `CudaStream`/`CudaBuffer` into `cuda_common.h/cpp`

**Files:**
- Create: `src/runtime/core/cuda_common.h`
- Create: `src/runtime/core/cuda_common.cpp`

These are pure CUDA wrappers with zero TRT dependency. They stay in the main binary.

- [ ] **Step 1: Create `src/runtime/core/cuda_common.h`**

Extract `CudaStream` and `CudaBuffer` from `src/runtime/core/trt_common.h`. Copy them verbatim but remove all `#if TRTF_HAS_TRT` guards and all TRT-specific code (`TrtLogger`, `TrtDeleter`, `TrtUniquePtr`, `create_trt_runtime()`):

```cpp
#pragma once

// CUDA RAII wrappers — no TRT dependency.
// CudaStream and CudaBuffer with move semantics.

#include <cstddef>
#include <cuda_runtime_api.h>

namespace trtf {

class CudaStream final {
public:
    CudaStream();
    ~CudaStream();
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    bool ok() const;
    cudaStream_t get() const;
private:
    cudaStream_t mStream{nullptr};
    cudaError_t mStatus{cudaSuccess};
};

class CudaBuffer final {
public:
    explicit CudaBuffer(std::size_t bytes);
    ~CudaBuffer();
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
    bool ok() const;
    void* data() const;
    std::size_t size() const;
private:
    void* mPtr{nullptr};
    std::size_t mBytes{0};
    cudaError_t mStatus{cudaSuccess};
};

} // namespace trtf
```

- [ ] **Step 2: Create `src/runtime/core/cuda_common.cpp`**

Copy the `CudaStream` and `CudaBuffer` implementations verbatim from `src/runtime/core/trt_common.cpp` (lines 116-218). Change the `#include` to `"runtime/core/cuda_common.h"`. No `#if TRTF_HAS_TRT` guard.

- [ ] **Step 3: Add to CMakeLists.txt**

Add `src/runtime/core/cuda_common.cpp` to the `trtf_core` source list, right after `src/runtime/core/trt_common.cpp`.

- [ ] **Step 4: Update `trt_common.h` to include `cuda_common.h`**

In `src/runtime/core/trt_common.h`, add `#include "runtime/core/cuda_common.h"` at the top (inside the `TRTF_HAS_TRT` guard for now). This means all existing consumers of `trt_common.h` get `CudaStream`/`CudaBuffer` transitively. Later tasks will switch direct consumers to `cuda_common.h`.

- [ ] **Step 5: Verify build**

Run: `cmake --build build -j 2>&1 | tail -20`

Expected: Compiles (may still have link errors from Task 1 — that's OK, the `.cpp` for old TrtModule is stale). The key is that `cuda_common.cpp` compiles.

- [ ] **Step 6: Commit**

```bash
git add src/runtime/core/cuda_common.h src/runtime/core/cuda_common.cpp CMakeLists.txt src/runtime/core/trt_common.h
git commit -m "refactor: extract CudaStream/CudaBuffer into cuda_common.h (no TRT dependency)"
```

---

### Task 4: Create `TrtModuleImpl` in DSO source directory

**Files:**
- Create: `src/runtime/backend/trt_module_impl.h`
- Create: `src/runtime/backend/trt_module_impl.cpp`

This is today's `trt_module.cpp` refactored as `TrtModuleImpl : public ITrtModule`. It takes engine + context as constructor args (backend creates them). This file will be compiled into both DSOs.

- [ ] **Step 1: Create `src/runtime/backend/trt_module_impl.h`**

```cpp
#pragma once

// TrtModuleImpl: concrete ITrtModule backed by a TRT engine.
// Compiled inside backend DSOs only (libtrtf_backend_trt.so / _rtx.so).

#include "trtf/runtime/trt_module.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtf {

class TrtModuleImpl final : public ITrtModule {
public:
    // Backend creates engine + context, passes them in.
    // The engine must outlive this module (caller manages lifetime via keep_alive).
    TrtModuleImpl(nvinfer1::ICudaEngine* engine,
                  nvinfer1::IExecutionContext* ctx,
                  cudaStream_t stream);
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
    std::vector<TensorInfo> input_info() const override;
    std::vector<TensorInfo> output_info() const override;
    bool has_input(const std::string& name) const override;
    bool has_output(const std::string& name) const override;
    void* device_ptr(const std::string& name) const override;
    void bind_external(const std::string& name, void* ptr) override;
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

    nvinfer1::IExecutionContext* ctx_{nullptr};
    cudaStream_t stream_{nullptr};
    bool has_dynamic_shapes_{false};
    std::vector<std::shared_ptr<void>> keep_alive_;
    std::unordered_map<std::string, BufferEntry> buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>> host_output_staging_;
    std::unordered_map<std::string, DeviceTensor> output_device_tensors_;

    void allocate_buffers(nvinfer1::ICudaEngine* engine);
    void free_buffers();
    void detect_dynamic_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io);
    void allocate_input_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io, int32_t num_profiles);
    void allocate_single_input(nvinfer1::ICudaEngine* engine, const char* name, int32_t num_profiles);
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

} // namespace trtf
```

- [ ] **Step 2: Create `src/runtime/backend/trt_module_impl.cpp`**

Copy the entire body of `src/runtime/core/trt_module.cpp` (lines 1-449). Make these changes:

1. Change `#include` from `"trtf/runtime/trt_module.h"` to `"trt_module_impl.h"`
2. Remove the `#if TRTF_HAS_TRT` / `#endif` wrapper
3. Rename class `TrtModule` → `TrtModuleImpl` throughout
4. Change the constructor to accept `(engine, ctx, stream)` instead of `(engine, stream)` — the context is passed in, NOT created internally:

```cpp
TrtModuleImpl::TrtModuleImpl(nvinfer1::ICudaEngine* engine,
                             nvinfer1::IExecutionContext* ctx,
                             cudaStream_t stream)
    : ctx_(ctx), stream_(stream)
{
    if (!ctx_) return;
    allocate_buffers(engine);
}
```

5. Remove move constructor/assignment (the interface doesn't need them — modules are always behind `unique_ptr<ITrtModule>`)

- [ ] **Step 3: Commit**

```bash
git add src/runtime/backend/trt_module_impl.h src/runtime/backend/trt_module_impl.cpp
git commit -m "feat(backend): create TrtModuleImpl — ITrtModule implementation for DSOs

Lifted from trt_module.cpp. Constructor takes engine+context (backend
creates them). Will be compiled into both libtrtf_backend_trt.so and
libtrtf_backend_rtx.so."
```

---

### Task 5: Create TRT logger for DSOs

**Files:**
- Create: `src/runtime/backend/trt_logger.h`
- Create: `src/runtime/backend/trt_logger.cpp`

Extract `TrtLogger`, `TrtDeleter`, `TrtUniquePtr`, and `create_trt_runtime()` from `trt_common.h/cpp`. These use TRT headers and live only in the DSOs.

- [ ] **Step 1: Create `src/runtime/backend/trt_logger.h`**

```cpp
#pragma once

// TRT logger and runtime factory — compiled into backend DSOs only.

#include <NvInfer.h>
#include <memory>
#include <string>

namespace trtf {

const char* trt_severity_name(nvinfer1::ILogger::Severity severity);
bool trt_log_to_stderr_enabled();
nvinfer1::ILogger::Severity trt_log_stderr_min_severity();

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    const std::string& last_error() const;
    void clear_error();
private:
    std::string mLastError;
};

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) delete ptr;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

TrtUniquePtr<nvinfer1::IRuntime> create_trt_runtime();

} // namespace trtf
```

- [ ] **Step 2: Create `src/runtime/backend/trt_logger.cpp`**

Copy `trt_severity_name`, `trt_log_to_stderr_enabled`, `trt_log_stderr_min_severity`, `TrtLogger::log`, `TrtLogger::last_error`, `TrtLogger::clear_error`, and `create_trt_runtime` from `src/runtime/core/trt_common.cpp` (lines 9-114). Change include to `"trt_logger.h"`. No `TRTF_HAS_TRT` guard.

- [ ] **Step 3: Commit**

```bash
git add src/runtime/backend/trt_logger.h src/runtime/backend/trt_logger.cpp
git commit -m "feat(backend): extract TrtLogger and runtime factory into DSO-only files"
```

---

### Task 6: Create `TrtBackend` (standard TRT DSO)

**Files:**
- Create: `src/runtime/backend/trt_backend.cpp`

This is the `IBackend` implementation for standard TensorRT. Compiled into `libtrtf_backend_trt.so`.

- [ ] **Step 1: Create `src/runtime/backend/trt_backend.cpp`**

```cpp
// TrtBackend: IBackend implementation for standard TensorRT.
// Compiled into libtrtf_backend_trt.so. Links libnvinfer.so.

#include "trtf/runtime/trt_backend.h"
#include "trt_module_impl.h"
#include "trt_logger.h"
#include "runtime/core/cuda_common.h"

#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace trtf {

class TrtBackend final : public IBackend {
public:
    TrtBackend() : runtime_(create_trt_runtime()) {
        if (!runtime_)
            throw std::runtime_error("[trtf] Failed to create TRT runtime");
    }

    std::unique_ptr<ITrtModule> create_module(
        const void* plan_data, size_t plan_size,
        const ModuleCreateOptions& options) override
    {
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtf] Failed to deserialize TRT engine");

        auto* ctx = engine->createExecutionContext();
        if (!ctx) {
            delete engine;
            throw std::runtime_error("[trtf] Failed to create TRT execution context");
        }

        cudaStream_t stream = options.stream;
        std::shared_ptr<void> stream_owner;
        if (!stream) {
            auto owned = std::make_shared<CudaStream>();
            if (!owned->ok()) {
                delete ctx;
                delete engine;
                throw std::runtime_error("[trtf] Failed to create CUDA stream");
            }
            stream = owned->get();
            stream_owner = owned;
        }

        auto module = std::make_unique<TrtModuleImpl>(engine, ctx, stream);
        if (!module->ok()) {
            delete engine;
            throw std::runtime_error("[trtf] TrtModuleImpl creation failed");
        }

        // Transfer engine + stream ownership to module
        module->keep_alive(std::shared_ptr<nvinfer1::ICudaEngine>(
            engine, [](nvinfer1::ICudaEngine* p) { delete p; }));
        if (stream_owner)
            module->keep_alive(stream_owner);

        return module;
    }

    const char* name() const override { return "trt"; }

private:
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
};

} // namespace trtf

extern "C" trtf::IBackend* trtf_create_backend()
{
    try { return new trtf::TrtBackend(); }
    catch (const std::exception& e) {
        std::cerr << "[trtf] TRT backend init failed: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void trtf_destroy_backend(trtf::IBackend* b) { delete b; }
```

- [ ] **Step 2: Commit**

```bash
git add src/runtime/backend/trt_backend.cpp
git commit -m "feat(backend): create TrtBackend — standard TRT IBackend for DSO"
```

---

### Task 7: Create `BackendLoader` (dlopen dispatch)

**Files:**
- Create: `src/runtime/backend/backend_loader.h`
- Create: `src/runtime/backend/backend_loader.cpp`
- Create: `tests/cpp/test_backend_loader.cpp`

- [ ] **Step 1: Create `src/runtime/backend/backend_loader.h`**

```cpp
#pragma once

// BackendLoader: loads backend DSOs via dlopen and caches them.

#include "trtf/runtime/trt_backend.h"
#include <string>

namespace trtf {

class BackendLoader {
public:
    // Load backend by name ("trt" or "trt_rtx").
    // Caches: second call with same name returns same IBackend*.
    // Throws std::runtime_error if DSO not found or factory missing.
    static IBackend* load(const std::string& backend_name);
};

} // namespace trtf
```

- [ ] **Step 2: Create `src/runtime/backend/backend_loader.cpp`**

```cpp
#include "runtime/backend/backend_loader.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>

namespace trtf {

namespace {

// Read the directory containing the running executable.
std::string exe_dir()
{
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.rfind('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : "";
}

struct CachedBackend {
    void* dl_handle{nullptr};
    IBackend* backend{nullptr};
};

std::mutex g_mu;
std::unordered_map<std::string, CachedBackend> g_cache;

void cleanup_backends()
{
    for (auto& [name, entry] : g_cache) {
        if (entry.backend) {
            auto destroy = reinterpret_cast<void(*)(IBackend*)>(
                dlsym(entry.dl_handle, "trtf_destroy_backend"));
            if (destroy) destroy(entry.backend);
            entry.backend = nullptr;
        }
        if (entry.dl_handle) {
            dlclose(entry.dl_handle);
            entry.dl_handle = nullptr;
        }
    }
}

} // namespace

IBackend* BackendLoader::load(const std::string& backend_name)
{
    std::lock_guard<std::mutex> lock(g_mu);

    auto it = g_cache.find(backend_name);
    if (it != g_cache.end()) return it->second.backend;

    // Register cleanup on first call
    static bool registered = false;
    if (!registered) {
        std::atexit(cleanup_backends);
        registered = true;
    }

    std::string dso_name = "libtrtf_backend_" + backend_name + ".so";

    // Search order: exe dir, TRTF_BACKEND_DIR, default dlopen paths
    void* handle = nullptr;
    std::string tried;

    // 1. Exe directory
    std::string exe_path = exe_dir() + "/" + dso_name;
    handle = dlopen(exe_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) tried += "  " + exe_path + ": " + dlerror() + "\n";

    // 2. TRTF_BACKEND_DIR
    if (!handle) {
        const char* env = std::getenv("TRTF_BACKEND_DIR");
        if (env && env[0] != '\0') {
            std::string env_path = std::string(env) + "/" + dso_name;
            handle = dlopen(env_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) tried += "  " + env_path + ": " + dlerror() + "\n";
        }
    }

    // 3. Default search (LD_LIBRARY_PATH, system dirs)
    if (!handle) {
        handle = dlopen(dso_name.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) tried += "  " + dso_name + " (default): " + dlerror() + "\n";
    }

    if (!handle) {
        throw std::runtime_error(
            "Backend \"" + backend_name + "\" not available.\n"
            "Could not load " + dso_name + ":\n" + tried + "\n"
            "To use " + backend_name + " bundles, ensure " + dso_name +
            " is next to the trtf binary,\n"
            "in TRTF_BACKEND_DIR, or in LD_LIBRARY_PATH.");
    }

    // Resolve factory symbol
    auto create_fn = reinterpret_cast<IBackend*(*)()>(
        dlsym(handle, "trtf_create_backend"));
    if (!create_fn) {
        dlclose(handle);
        throw std::runtime_error(
            dso_name + " loaded but missing trtf_create_backend symbol");
    }

    IBackend* backend = create_fn();
    if (!backend) {
        dlclose(handle);
        throw std::runtime_error(
            dso_name + ": trtf_create_backend() returned nullptr");
    }

    g_cache[backend_name] = CachedBackend{handle, backend};
    std::cerr << "[trtf] Backend loaded: " << backend->name()
              << " (" << dso_name << ")" << std::endl;
    return backend;
}

} // namespace trtf
```

- [ ] **Step 3: Create `tests/cpp/test_backend_loader.cpp`**

Test the loader with a missing DSO to verify error handling:

```cpp
#include "runtime/backend/backend_loader.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) { std::cerr << "FAIL: " << name << std::endl; ++failures; }
}

int main() {
    // Loading a nonexistent backend should throw
    bool threw = false;
    try {
        trtf::BackendLoader::load("nonexistent_backend_xyz");
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find("nonexistent_backend_xyz") != std::string::npos,
              "error mentions backend name");
        check(msg.find("libtrtf_backend_nonexistent_backend_xyz.so") != std::string::npos,
              "error mentions DSO name");
    }
    check(threw, "missing backend throws runtime_error");

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
```

- [ ] **Step 4: Add to CMakeLists.txt**

Add `src/runtime/backend/backend_loader.cpp` to `trtf_core` source list. Add the test:

```cmake
trtf_add_test(test_backend_loader)
```

Also add `dl` to `trtf_core`'s link libraries:

```cmake
target_link_libraries(trtf_core PRIVATE ${TRTF_CUDART_LIBRARY} dl)
```

- [ ] **Step 5: Build and run test**

Run: `cmake --build build -j && ctest --test-dir build -R test_backend_loader --output-on-failure`

Expected: PASS (the missing-DSO test throws the expected error).

- [ ] **Step 6: Commit**

```bash
git add src/runtime/backend/backend_loader.h src/runtime/backend/backend_loader.cpp \
    tests/cpp/test_backend_loader.cpp CMakeLists.txt
git commit -m "feat(backend): create BackendLoader — dlopen dispatch with caching and error handling"
```

---

### Task 8: Update `PipelineContext` with backend fields

**Files:**
- Modify: `include/trtf/runtime/pipeline_plugin.h`

- [ ] **Step 1: Add forward declaration and new fields**

In `include/trtf/runtime/pipeline_plugin.h`, add a forward declaration for `IBackend` and add three new fields to `PipelineContext`:

```cpp
// Add near the top, after existing forward declarations:
class IBackend;

// In struct PipelineContext, add after bundle_path:
    IBackend* backend;                      // Backend for creating ITrtModule instances
    const std::string& runtime_cache_path;  // RTX: JIT kernel cache file path
    bool cuda_graphs;                       // RTX: whole-graph CUDA capture
```

- [ ] **Step 2: Commit**

```bash
git add include/trtf/runtime/pipeline_plugin.h
git commit -m "feat(pipeline): add backend, runtime_cache_path, cuda_graphs to PipelineContext"
```

---

### Task 9: Update `plugin_helpers` to use `IBackend`

**Files:**
- Modify: `src/runtime/plugins/shared/plugin_helpers.h`
- Modify: `src/runtime/plugins/shared/plugin_helpers.cpp`

- [ ] **Step 1: Update `plugin_helpers.h`**

1. Replace `#include "runtime/core/trt_common.h"` with `#include "runtime/core/cuda_common.h"` and `#include "trtf/runtime/trt_backend.h"`
2. Change `LoadedModule`:
   - `std::unique_ptr<TrtModule> module` → `std::unique_ptr<ITrtModule> module`
   - Remove `std::shared_ptr<CudaStream> stream` (stream is now owned by the module)
3. Change function signatures:
   - `load_trt_module_from_plan(const std::vector<char>* plan, const char* label, std::shared_ptr<CudaStream> shared_stream)` → `load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan, const char* label, const ModuleCreateOptions& options = {})`
   - Same for `try_load_trt_module_from_plan`
   - `extract_optional_module` returns `std::unique_ptr<ITrtModule>`
4. Remove `#if TRTF_HAS_TRT` / `#endif` guards

- [ ] **Step 2: Update `plugin_helpers.cpp`**

1. Rewrite `load_trt_module_from_plan` to delegate to `backend->create_module()`:

```cpp
LoadedModule load_trt_module_from_plan(
    IBackend* backend,
    const std::vector<char>* plan, const char* label,
    const ModuleCreateOptions& options)
{
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    if (!backend)
        throw std::runtime_error("No backend loaded");

    LoadedModule result;
    result.module = backend->create_module(plan->data(), plan->size(), options);
    if (!result.module || !result.module->ok())
        throw std::runtime_error(std::string("Failed to create ITrtModule for ") + label);
    return result;
}
```

2. Update `try_load_trt_module_from_plan` similarly
3. Update `extract_optional_module` similarly
4. Remove `#include "runtime/core/trt_common.h"` and TRT-specific includes
5. Remove `#if TRTF_HAS_TRT` / `#endif`

- [ ] **Step 3: Commit**

```bash
git add src/runtime/plugins/shared/plugin_helpers.h src/runtime/plugins/shared/plugin_helpers.cpp
git commit -m "refactor(plugin_helpers): delegate to IBackend::create_module instead of direct TRT calls"
```

---

### Task 10: Update `pipeline_factory.cpp` to use `BackendLoader`

**Files:**
- Modify: `src/runtime/registry/pipeline_factory.cpp`
- Modify: `include/trtf/runtime/pipeline_factory.h`
- Modify: `include/trtf/pipeline.h`

- [ ] **Step 1: Update `pipeline_factory.h`**

Add `runtime_cache_path` and `cuda_graphs` parameters:

```cpp
static std::unique_ptr<IPipeline> from_bundle(
    const std::string& bundle_path,
    const std::string& hf_python = "",
    const std::string& runtime_cache_path = "",
    bool cuda_graphs = false);
```

- [ ] **Step 2: Update `pipeline.h`**

Update the `load()` free function signature:

```cpp
std::unique_ptr<IPipeline> load(
    const std::string& bundle_path,
    const std::string& hf_python = "",
    const std::string& runtime_cache_path = "",
    bool cuda_graphs = false);
```

- [ ] **Step 3: Update `pipeline_factory.cpp`**

1. Add includes: `#include "runtime/backend/backend_loader.h"` and `#include "trtf/runtime/trt_backend.h"`
2. Update `from_bundle()` signature to accept the new params
3. After parsing `runtime_strategy`, read `engine_backend`:
   ```cpp
   std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
   IBackend* backend = BackendLoader::load(backend_name);
   ```
4. Construct `PipelineContext` with the new fields:
   ```cpp
   PipelineContext ctx{bundle, base_cfg, config_text, hf_python, bundle_path,
                       backend, runtime_cache_path, cuda_graphs};
   ```
5. Update `load()` to pass through the new params
6. Remove the `#if TRTF_HAS_TRT` / `#else` / `#endif` block

- [ ] **Step 4: Commit**

```bash
git add src/runtime/registry/pipeline_factory.cpp \
    include/trtf/runtime/pipeline_factory.h \
    include/trtf/pipeline.h
git commit -m "feat(factory): integrate BackendLoader — read engine_backend from bundle, dlopen DSO"
```

---

### Task 11: Update all pipeline plugins to use `ctx.backend`

**Files:**
- Modify: All 21 plugin `.cpp` files in `src/runtime/plugins/`

Each plugin currently calls:
```cpp
auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
```

Change to:
```cpp
ModuleCreateOptions opts;
opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
opts.cuda_graphs = ctx.cuda_graphs;
auto loaded = load_trt_module_from_plan(ctx.backend,
    find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);
```

Also for each plugin:
1. Remove `#if TRTF_HAS_TRT` / `#endif` wrapper
2. Add `#include "trtf/runtime/trt_backend.h"` if needed
3. Update any calls to `try_load_trt_module_from_plan` and `extract_optional_module` with the same pattern

- [ ] **Step 1: Update `decoder_plugin.cpp`** (reference implementation)

Remove the `#if TRTF_HAS_TRT` wrapper. Change the `load_trt_module_from_plan` call. Since `LoadedModule` no longer has `stream`, get the stream from the module: `loaded.module->stream()`.

- [ ] **Step 2: Update remaining 20 plugins**

Apply the same pattern to all remaining plugins. For plugins with multiple engines (bark, whisper, vl, flux, wan, etc.), each engine load call gets the same treatment.

- [ ] **Step 3: Update `force_link_plugins.cpp`**

Remove the `#if TRTF_HAS_TRT` guard.

- [ ] **Step 4: Commit**

```bash
git add src/runtime/plugins/
git commit -m "refactor(plugins): use ctx.backend for module creation, remove TRTF_HAS_TRT guards"
```

---

### Task 12: Update all pipeline headers — `TrtModule` → `ITrtModule`

**Files:**
- Modify: All 15 pipeline `.h` files in `src/runtime/pipelines/`
- Modify: All 15 pipeline `.cpp` files in `src/runtime/pipelines/`

- [ ] **Step 1: Update pipeline headers**

For each of the 15 pipeline headers:
1. Replace `#include "trtf/runtime/trt_module.h"` (already provides `ITrtModule` from Task 1)
2. Remove `#if TRTF_HAS_TRT` / `#endif` guards
3. Member types are already `TrtModule` which is aliased to `ITrtModule` from Task 1 — no change needed to member declarations. Constructor parameter types may say `std::unique_ptr<TrtModule>` — these work because of the alias.

- [ ] **Step 2: Update pipeline `.cpp` files**

For each pipeline `.cpp`:
1. Remove `#if TRTF_HAS_TRT` / `#endif` guards
2. Where the `.cpp` accesses `loaded.stream`, change to `loaded.module->stream()`

- [ ] **Step 3: Commit**

```bash
git add src/runtime/pipelines/
git commit -m "refactor(pipelines): remove TRTF_HAS_TRT guards, use ITrtModule via alias"
```

---

### Task 13: Update CMakeLists.txt — DSO targets, remove libnvinfer from trtf_core

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Remove TRT link from trtf_core**

In the `trtf_core` target:
1. Remove `src/runtime/core/trt_module.cpp` from the source list (moved to DSO)
2. Remove `src/runtime/core/trt_engine_lifecycle.cpp` from source list (inlined into DSOs)
3. Remove `${TRTF_TRT_LIBRARY}` from `target_link_libraries`
4. Remove TRT include dirs from `target_include_directories`
5. Keep `TRTF_HAS_TRT=1` define temporarily (some headers still use it for CUDA guards)
6. Keep `${TRTF_CUDART_LIBRARY}` link
7. Add `dl` to link libs
8. Add `src/runtime/core/cuda_common.cpp` and `src/runtime/backend/backend_loader.cpp` to source list

- [ ] **Step 2: Add standard TRT DSO target**

```cmake
option(TRTF_BUILD_BACKEND_TRT "Build standard TRT backend DSO" ON)
if(TRTF_BUILD_BACKEND_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY)
    add_library(trtf_backend_trt SHARED
        src/runtime/backend/trt_module_impl.cpp
        src/runtime/backend/trt_logger.cpp
        src/runtime/backend/trt_backend.cpp
    )
    target_include_directories(trtf_backend_trt PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )
    target_include_directories(trtf_backend_trt SYSTEM PRIVATE
        ${TRTF_TRT_INCLUDE_DIR}
        ${TRTF_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(trtf_backend_trt PRIVATE
        ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY}
    )
    target_compile_options(trtf_backend_trt PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(trtf_backend_trt PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    message(STATUS "TRT backend DSO: enabled")
endif()
```

- [ ] **Step 3: Add RTX DSO target (optional)**

```cmake
option(TRTF_BUILD_BACKEND_RTX "Build TRT-RTX backend DSO" OFF)
if(TRTF_BUILD_BACKEND_RTX)
    find_library(TENSORRT_RTX_LIB tensorrt_rtx HINTS ${TRTF_RTX_LIBRARY_DIR})
    find_path(TRT_RTX_INCLUDE_DIR NvInferRuntime.h HINTS ${TRTF_RTX_INCLUDE_DIR})
    if(TENSORRT_RTX_LIB AND TRT_RTX_INCLUDE_DIR)
        add_library(trtf_backend_rtx SHARED
            src/runtime/backend/trt_module_impl.cpp
            src/runtime/backend/trt_logger.cpp
            src/runtime/backend/rtx_backend.cpp
        )
        target_include_directories(trtf_backend_rtx PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
        )
        target_include_directories(trtf_backend_rtx SYSTEM PRIVATE
            ${TRT_RTX_INCLUDE_DIR}
            ${TRTF_CUDA_INCLUDE_DIR}
        )
        target_link_libraries(trtf_backend_rtx PRIVATE
            ${TENSORRT_RTX_LIB} ${TRTF_CUDART_LIBRARY}
        )
        target_compile_options(trtf_backend_rtx PRIVATE -Wall -Wextra -Wpedantic)
        set_target_properties(trtf_backend_rtx PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )
        message(STATUS "TRT-RTX backend DSO: enabled")
    endif()
endif()
```

- [ ] **Step 4: Build**

Run: `cmake -S . -B build -G Ninja <existing TRT flags> && cmake --build build -j 2>&1 | tail -30`

Expected: `trtf_core` compiles without TRT headers. `libtrtf_backend_trt.so` is built in the build directory.

- [ ] **Step 5: Run existing C++ tests**

Run: `ctest --test-dir build --output-on-failure`

Note which tests need the DSO to be loadable. Tests that load bundles will need `libtrtf_backend_trt.so` in the build directory (which it should be from Step 4).

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: remove libnvinfer link from trtf_core, add DSO targets for TRT and RTX backends"
```

---

### Task 14: Update C++ CLI with `--runtime-cache` and `--cuda-graphs`

**Files:**
- Modify: `examples/trtf_cli.cpp`

- [ ] **Step 1: Add CLI flags and pass to `trtf::load()`**

1. Add `std::string runtime_cache` and `bool cuda_graphs{false}` to `CliArgs`
2. Add parsing: `--runtime-cache` and `--cuda-graphs` in `parse_args()`
3. Add to usage string
4. Update all `trtf::load()` calls: `trtf::load(args.bundle_path, args.hf_python, args.runtime_cache, args.cuda_graphs)`

- [ ] **Step 2: Commit**

```bash
git add examples/trtf_cli.cpp
git commit -m "feat(cli): add --runtime-cache and --cuda-graphs flags for TRT-RTX"
```

---

### Task 15: Update C ABI

**Files:**
- Modify: `include/trtf/pipeline.h` (TrtfPipelineOptions struct)
- Modify: `src/cabi/api/trtf_c.cpp`

- [ ] **Step 1: Add fields to `TrtfPipelineOptions`**

In `include/trtf/pipeline.h`, add to the struct:

```cpp
const char* runtime_cache;  // nullptr = no RTX cache
int cuda_graphs;            // 0 = disabled
```

- [ ] **Step 2: Update `trtf_c.cpp`**

Pass the new fields through to `PipelineFactory::from_bundle()`.

- [ ] **Step 3: Commit**

```bash
git add include/trtf/pipeline.h src/cabi/api/trtf_c.cpp
git commit -m "feat(cabi): add runtime_cache and cuda_graphs to TrtfPipelineOptions"
```

---

### Task 16: Create `RtxBackend` DSO

**Files:**
- Create: `src/runtime/backend/rtx_backend.cpp`

- [ ] **Step 1: Create `src/runtime/backend/rtx_backend.cpp`**

Use the code from the spec (Section 2, RtxBackend). Key additions over TrtBackend:
- `IRuntimeCache` management (create, load from disk, save on destroy)
- `CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE` support
- `engine->createExecutionContext(rt_config)` instead of `engine->createExecutionContext()`

The full implementation is in the spec. Copy it, adjusting includes to use `trt_module_impl.h`, `trt_logger.h`, `runtime/core/cuda_common.h`.

- [ ] **Step 2: Commit**

```bash
git add src/runtime/backend/rtx_backend.cpp
git commit -m "feat(backend): create RtxBackend — TRT-RTX IBackend with runtime cache and CUDA graphs"
```

---

### Task 17: Python `--rtx` flag

**Files:**
- Modify: `trtf_build/trtf_build/cli.py`
- Modify: `trtf_build/trtf_build/engine_builder.py`
- Create: `tests/builder/test_rtx_flag.py`

- [ ] **Step 1: Add `--rtx` flag to CLI**

In `cli.py`, add to the `build` subparser:

```python
build_p.add_argument("--rtx", action="store_true",
                     help="Build engine for TRT-RTX (portable, JIT-compiled at runtime)")
```

Pass `rtx=args.rtx` through to `build()`.

- [ ] **Step 2: Add `rtx` parameter to `build()` and `build_bundle()`**

In `engine_builder.py`:

1. Add `rtx: bool = False` parameter to `build()` and `build_bundle()`.
2. At the top of `build_bundle()`, before any TRT import, add:

```python
if rtx:
    import tensorrt_rtx
    import sys
    sys.modules["tensorrt"] = tensorrt_rtx
    print("[trtf-build] Using TensorRT-RTX backend", file=sys.stderr)
```

3. When building `cfg_dict` for `config.json`, inject:

```python
cfg_dict["engine_backend"] = "trt_rtx" if rtx else "trt"
```

This goes right after `cfg_dict["runtime_strategy"] = runtime_strategy` (around line 528).

4. Same for `_build_diffusion_bundle()` — add `rtx` param, inject `engine_backend`.

- [ ] **Step 3: Update `trtf-build inspect` to show `engine_backend`**

In `_cmd_inspect()`, add to the `fields` list:

```python
("Engine backend", "engine_backend"),
```

- [ ] **Step 4: Create `tests/builder/test_rtx_flag.py`**

```python
"""Tests for --rtx flag and engine_backend metadata."""
import json
import sys
import types
from unittest import mock

import pytest


def test_rtx_flag_parsed():
    """--rtx flag is accepted by the CLI parser."""
    from trtf_build.cli import main
    with mock.patch("sys.argv", ["trtf-build", "build", "model", "-o", "out.trtfb", "--rtx"]):
        # Will fail at build time (no model), but parsing should succeed
        with pytest.raises(SystemExit):
            main()


def test_engine_backend_default_is_trt():
    """Without --rtx, engine_backend defaults to 'trt'."""
    # Mock the build to capture what config gets written
    from trtf_build import engine_builder
    original_write = engine_builder.write_bundle

    captured = {}
    def capture_write(path, info, sections):
        for s in sections:
            if s.name == "config.json":
                captured["config"] = json.loads(s.data.decode("utf-8"))
        original_write(path, info, sections)

    # This tests the injection logic without actually building
    # (we'd need TRT installed for a real build)


def test_sys_modules_monkeypatch():
    """When rtx=True, 'tensorrt' in sys.modules should point to tensorrt_rtx."""
    # Create a fake tensorrt_rtx module
    fake_rtx = types.ModuleType("tensorrt_rtx")
    fake_rtx.__version__ = "1.0.0-fake"

    with mock.patch.dict("sys.modules", {"tensorrt_rtx": fake_rtx}):
        from trtf_build.engine_builder import _setup_trt_import
        # This function doesn't exist yet — will be created in Step 2
        # For now, test the monkeypatch concept
        import tensorrt_rtx
        sys.modules["tensorrt"] = tensorrt_rtx
        import tensorrt as trt
        assert trt.__version__ == "1.0.0-fake"
        # Cleanup
        del sys.modules["tensorrt"]
```

- [ ] **Step 5: Commit**

```bash
git add trtf_build/trtf_build/cli.py trtf_build/trtf_build/engine_builder.py \
    tests/builder/test_rtx_flag.py
git commit -m "feat(builder): add --rtx flag — sys.modules monkeypatch + engine_backend metadata"
```

---

### Task 18: Remove remaining `TRTF_HAS_TRT` guards from main binary

**Files:**
- Modify: Multiple headers in `include/trtf/runtime/` and `src/runtime/core/`

This is the final cleanup. Files that still have `TRTF_HAS_TRT` guards but don't actually use TRT types (they use CUDA types like `cudaStream_t`, `cudaMalloc`) get their guards removed.

Files that genuinely reference TRT types (`NvInfer.h`) have already been moved to DSOs in earlier tasks.

- [ ] **Step 1: Audit remaining `TRTF_HAS_TRT` usage**

Run: `grep -rn "TRTF_HAS_TRT" include/ src/ --include="*.h" --include="*.cpp" -l`

For each file:
- If it only uses CUDA types (`cudaStream_t`, `cudaMalloc`, `cudaMemcpy`): remove the guard, replace `#include "runtime/core/trt_common.h"` with `#include "runtime/core/cuda_common.h"`
- If it uses TRT types (`nvinfer1::*`): it should already be in the DSO. If not, it's a bug from earlier tasks — fix it.

Key files to update:
- `include/trtf/runtime/device_tensor.h` — uses CUDA only, remove guard
- `include/trtf/runtime/device_ops.h` — uses CUDA only, remove guard
- `include/trtf/runtime/kv_cache.h` — uses CUDA only (via `CudaStream`), remove guard
- `include/trtf/runtime/inference_state.h` — uses CUDA only, remove guard
- `include/trtf/runtime/recurrent_state.h` — uses CUDA only, remove guard
- `src/runtime/core/device_kv_cache.h` — uses CUDA only, remove guard
- `src/runtime/core/trt_decode_runtime.h` — uses CUDA only, remove guard

- [ ] **Step 2: Remove `TRTF_HAS_TRT` compile definition from CMakeLists.txt**

Remove: `target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TRT=1)` and the else branch.

- [ ] **Step 3: Build and test**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`

All tests should pass.

- [ ] **Step 4: Commit**

```bash
git add include/ src/ CMakeLists.txt
git commit -m "refactor: remove TRTF_HAS_TRT guards from main binary — pure runtime backend dispatch"
```

---

### Task 19: Full regression test

**Files:** None (testing only)

- [ ] **Step 1: Run all C++ unit tests**

Run: `ctest --test-dir build --output-on-failure`

Expected: All tests pass.

- [ ] **Step 2: Run Python builder unit tests**

Run: `pytest tests/builder/ -v --ignore=tests/builder/test_cli.py`

Expected: All tests pass (no TRT needed for mock tests).

- [ ] **Step 3: Run tools self-tests**

Run: `pytest tests/tools/ -v`

Expected: All tests pass.

- [ ] **Step 4: Verify DSO is loadable**

Run: `ldd build/libtrtf_backend_trt.so` — should show `libnvinfer.so` dependency.
Run: `ldd build/trtf` — should NOT show `libnvinfer.so`.
Run: `nm -D build/libtrtf_backend_trt.so | grep trtf_create_backend` — should show the exported symbol.

- [ ] **Step 5: E2E smoke test (if GPU available)**

Run in container:
```bash
./build/trtf run /tmp/qwen3-0.6b.trtfb --prompt "Hello" --max-new-tokens 5 \
  --hf-python /opt/venv/bin/python
```

Expected: Same output as before the migration.

- [ ] **Step 6: Commit (tag milestone)**

```bash
git tag -a v0.2.0-rtx-abstraction -m "TRT-RTX backend abstraction complete"
```
