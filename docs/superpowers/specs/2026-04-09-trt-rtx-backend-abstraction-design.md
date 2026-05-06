# TRT-RTX Backend Abstraction Design

**Date:** 2026-04-09  
**Status:** Draft  
**Author:** Yifei Fang + Claude  
**Reference:** GitLab MR !49 (rajerao — "Enable TensorRT-RTX and its features")

## Problem

The codebase links directly against `libnvinfer.so` at compile time. Adding TRT-RTX
support requires a second TRT SDK (`tensorrt_rtx`) with incompatible engine plan
formats but a nearly identical C++ API. We need both backends available without
compile-time macros (`#ifdef`) and without making either a link-time dependency.

## Goals

1. **Zero link-time TRT dependency** — the main binary and `trtmc_core` link only `libcudart`
2. **Runtime backend selection** — bundle metadata (`engine_backend` field) drives which DSO is loaded via `dlopen`
3. **Single virtual interface** — `ITrtModule` replaces concrete `TrtModule`; all pipeline code is backend-agnostic
4. **Python `--rtx` flag** — builder dynamically imports `tensorrt_rtx` instead of `tensorrt`, records backend in bundle
5. **No macros** — no `#ifdef TRT_MAJOR_RTX`, no `#if TRTMC_HAS_TRT` gating in the main binary's runtime code
6. **RTX-specific features** — runtime cache (JIT kernel persistence) and CUDA graph capture exposed via `ModuleCreateOptions`

## Non-Goals

- Simultaneously loading both backends in one process
- Hot-swapping backends after pipeline creation
- Python-side RTX runtime features (Python builder only builds; C++ runtime infers)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                 trtmc main binary                      │
│                                                       │
│  Links: libcudart ONLY (no libnvinfer, no TRT headers)│
│                                                       │
│  pipeline_factory reads config.json from bundle:      │
│    "engine_backend": "trt" | "trt_rtx"                │
│    → BackendLoader::load("trt")                       │
│      → dlopen("libtrtmc_backend_trt.so")               │
│      → trtmc_create_backend() → IBackend*              │
│    → backend->create_module(plan, size, opts)          │
│      → ITrtModule*                                    │
│                                                       │
│  All pipeline code uses ITrtModule* only              │
└────────────┬──────────────────┬───────────────────────┘
             │                  │
        dlopen()           dlopen()
             │                  │
┌────────────▼───────┐  ┌──────▼────────────────────────┐
│ libtrtmc_backend_   │  │ libtrtmc_backend_              │
│   trt.so           │  │   rtx.so                      │
│                    │  │                                │
│ Links: libnvinfer  │  │ Links: libtensorrt_rtx        │
│ Headers: NvInfer.h │  │ Headers: NvInfer.h (RTX copy) │
│                    │  │                                │
│ Compiles:          │  │ Compiles:                     │
│  TrtModuleImpl     │  │  TrtModuleImpl (same source)  │
│  TrtBackend        │  │  RtxBackend                   │
│   (IBackend impl)  │  │   (IBackend impl)             │
│                    │  │   + IRuntimeCache mgmt        │
│ Exports:           │  │   + CudaGraphStrategy         │
│  trtmc_create_      │  │                                │
│    backend()       │  │ Exports:                      │
│  trtmc_destroy_     │  │  trtmc_create_backend()        │
│    backend()       │  │  trtmc_destroy_backend()       │
└────────────────────┘  └────────────────────────────────┘
```

---

## Section 1: Interface — `ITrtModule` and `IBackend`

### Header: `include/trtmc/runtime/trt_module.h`

Replaces the current concrete `TrtModule` class. No TRT headers — only CUDA
runtime types and our own tensor types.

```cpp
#pragma once

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// Per-engine module interface. Wraps engine deserialization, buffer allocation,
// and execution. All methods match today's TrtModule API exactly.
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

    // Stash opaque resources for lifetime management.
    virtual void keep_alive(std::shared_ptr<void> resource) = 0;
};

} // namespace trtmc
```

### Header: `include/trtmc/runtime/trt_backend.h`

```cpp
#pragma once

#include "trtmc/runtime/trt_module.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

namespace trtmc {

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

} // namespace trtmc

// C ABI exported by each DSO. The main binary resolves these via dlsym.
extern "C" {
    trtmc::IBackend* trtmc_create_backend();
    void trtmc_destroy_backend(trtmc::IBackend* backend);
}
```

### What pipelines change

Every pipeline that currently stores `std::unique_ptr<TrtModule>` changes to
`std::unique_ptr<ITrtModule>`. The method calls (`forward()`, `has_input()`,
`device_ptr()`, `bind_external()`, etc.) stay identical — the interface is a
1:1 virtual projection of today's concrete API.

Affected pipeline headers (member type change only, no logic change):
- `text_generation_pipeline.h` — `decoder_` member
- `recurrent_pipeline.h` — `module_` member
- `encoder_pipeline.h` — `module_` member
- `segment_pipeline.h` — `encoder_` member
- `sam_pipeline.h` — `encoder_`, `mask_decoder_` members
- `vl_pipeline.h` — `decoder_`, `vision_module_` members
- `whisper_pipeline.h` — `encoder_`, `decoder_` members
- `bark_pipeline.h` — `semantic_`, `coarse_`, `fine_`, `codec_` members
- `magpie_pipeline.h` — `text_`, `codec_` members
- `speech_pipeline.h` — `depth_`, `temporal_`, `mimi_decode_` members
- `omni_pipeline.h` — multiple modules
- `flux_pipeline.h` — `text_encoder_`, `denoiser_`, `vae_` members
- `wan_pipeline.h` — `text_encoder_`, `denoiser_`, `vae_` members
- `z_image_pipeline.h` — similar
- `torchtrt_diffusion_pipeline.h` — `module_` member

---

## Section 2: DSO Implementation

### Shared source: `src/runtime/backend/trt_module_impl.h` / `.cpp`

This is today's `trt_module.cpp` refactored as `TrtModuleImpl : public ITrtModule`.
Both DSOs compile this same source against their respective TRT SDK headers.
The code is identical — both standard TRT and TRT-RTX use the same `nvinfer1`
C++ API (`ICudaEngine`, `IExecutionContext`, `enqueueV3`, `setTensorAddress`, etc.).

Key difference in construction: `TrtModuleImpl` takes the engine and context
as constructor args (created by the backend), rather than creating the context itself.

```cpp
// src/runtime/backend/trt_module_impl.h (compiled inside DSOs only)
#pragma once
#include "trtmc/runtime/trt_module.h"
#include <NvInfer.h>

namespace trtmc {

class TrtModuleImpl final : public ITrtModule {
public:
    // Backend creates engine + context, passes them in.
    TrtModuleImpl(nvinfer1::ICudaEngine* engine,
                  nvinfer1::IExecutionContext* ctx,
                  cudaStream_t stream);
    ~TrtModuleImpl() override;

    // ITrtModule interface — same implementation as today's TrtModule
    TensorMap forward(const TensorMap& inputs) override;
    DeviceTensorMap forward_device(const DeviceTensorMap& inputs) override;
    void forward_device_async(const DeviceTensorMap& inputs) override;
    void forward_async(const TensorMap& inputs) override;
    void sync() override;
    cudaStream_t stream() const override;
    std::vector<TensorInfo> input_info() const override;
    std::vector<TensorInfo> output_info() const override;
    bool has_input(const std::string& name) const override;
    bool has_output(const std::string& name) const override;
    void* device_ptr(const std::string& name) const override;
    void bind_external(const std::string& name, void* ptr) override;
    bool ok() const override;
    void keep_alive(std::shared_ptr<void> resource) override;

private:
    // Identical internals to today's TrtModule (BufferEntry, etc.)
    // ...
};

} // namespace trtmc
```

### Standard TRT backend: `src/runtime/backend/trt_backend.cpp`

```cpp
// Compiled into libtrtmc_backend_trt.so, links libnvinfer.so
#include "trtmc/runtime/trt_backend.h"
#include "trt_module_impl.h"
#include <NvInfer.h>

namespace trtmc {

class TrtBackend final : public IBackend {
public:
    TrtBackend() {
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
    }

    std::unique_ptr<ITrtModule> create_module(
        const void* plan_data, size_t plan_size,
        const ModuleCreateOptions& options) override
    {
        // Deserialize engine
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine) throw std::runtime_error("Failed to deserialize engine");

        // Create execution context (standard — no IRuntimeConfig)
        auto* ctx = engine->createExecutionContext();
        if (!ctx) { delete engine; throw std::runtime_error("Failed to create context"); }

        // Create or reuse stream
        cudaStream_t stream = options.stream;
        std::shared_ptr<void> stream_owner;
        if (!stream) {
            auto owned = std::make_shared<CudaStream>();
            stream = owned->get();
            stream_owner = owned;
        }

        auto module = std::make_unique<TrtModuleImpl>(engine, ctx, stream);
        // Transfer ownership of engine and stream to module
        module->keep_alive(std::shared_ptr<void>(engine,
            [](void* p) { delete static_cast<nvinfer1::ICudaEngine*>(p); }));
        if (stream_owner) module->keep_alive(stream_owner);
        return module;
    }

    const char* name() const override { return "trt"; }

private:
    TrtLogger logger_;  // process-lifetime logger
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
};

} // namespace trtmc

extern "C" trtmc::IBackend* trtmc_create_backend() { return new trtmc::TrtBackend(); }
extern "C" void trtmc_destroy_backend(trtmc::IBackend* b) { delete b; }
```

### TRT-RTX backend: `src/runtime/backend/rtx_backend.cpp`

```cpp
// Compiled into libtrtmc_backend_rtx.so, links libtensorrt_rtx.so
#include "trtmc/runtime/trt_backend.h"
#include "trt_module_impl.h"
#include <NvInfer.h>  // RTX copy — same nvinfer1 namespace
#include <fstream>
#include <vector>

namespace trtmc {

class RtxBackend final : public IBackend {
public:
    RtxBackend() {
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
    }

    ~RtxBackend() override {
        flush_runtime_cache();
        delete runtime_cache_;
    }

    std::unique_ptr<ITrtModule> create_module(
        const void* plan_data, size_t plan_size,
        const ModuleCreateOptions& options) override
    {
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine) throw std::runtime_error("Failed to deserialize RTX engine");

        // Create IRuntimeConfig with RTX-specific features
        auto* rt_config = engine->createRuntimeConfig();

        // JIT kernel cache
        if (options.runtime_cache_path && options.runtime_cache_path[0] != '\0') {
            ensure_runtime_cache(rt_config, options.runtime_cache_path);
        }

        // CUDA graph capture
        if (options.cuda_graphs) {
            rt_config->setCudaGraphStrategy(
                nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
        }

        auto* ctx = engine->createExecutionContext(rt_config);
        delete rt_config;
        if (!ctx) { delete engine; throw std::runtime_error("Failed to create RTX context"); }

        cudaStream_t stream = options.stream;
        std::shared_ptr<void> stream_owner;
        if (!stream) {
            auto owned = std::make_shared<CudaStream>();
            stream = owned->get();
            stream_owner = owned;
        }

        auto module = std::make_unique<TrtModuleImpl>(engine, ctx, stream);
        module->keep_alive(std::shared_ptr<void>(engine,
            [](void* p) { delete static_cast<nvinfer1::ICudaEngine*>(p); }));
        if (stream_owner) module->keep_alive(stream_owner);
        return module;
    }

    const char* name() const override { return "trt_rtx"; }

private:
    TrtLogger logger_;
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    nvinfer1::IRuntimeCache* runtime_cache_{nullptr};
    std::string cache_path_;

    void ensure_runtime_cache(nvinfer1::IRuntimeConfig* cfg, const char* path) {
        if (!runtime_cache_) {
            runtime_cache_ = cfg->createRuntimeCache();
            cache_path_ = path;
            // Load existing cache from disk
            std::ifstream ifs(path, std::ios::binary | std::ios::ate);
            if (ifs) {
                auto sz = ifs.tellg();
                if (sz > 0) {
                    std::vector<char> buf(static_cast<size_t>(sz));
                    ifs.seekg(0);
                    ifs.read(buf.data(), sz);
                    runtime_cache_->deserialize(buf.data(), buf.size());
                    std::cerr << "[trtmc] RTX runtime cache loaded: "
                              << path << " (" << sz << " bytes)\n";
                }
            }
        }
        cfg->setRuntimeCache(*runtime_cache_);
    }

    void flush_runtime_cache() {
        if (!runtime_cache_ || cache_path_.empty()) return;
        auto* mem = runtime_cache_->serialize();
        if (mem && mem->size() > 0) {
            std::ofstream ofs(cache_path_, std::ios::binary | std::ios::trunc);
            if (ofs) {
                ofs.write(static_cast<const char*>(mem->data()),
                          static_cast<std::streamsize>(mem->size()));
                std::cerr << "[trtmc] RTX runtime cache saved: "
                          << cache_path_ << " (" << mem->size() << " bytes)\n";
            }
            delete mem;
        }
    }
};

} // namespace trtmc

extern "C" trtmc::IBackend* trtmc_create_backend() { return new trtmc::RtxBackend(); }
extern "C" void trtmc_destroy_backend(trtmc::IBackend* b) { delete b; }
```

---

## Section 3: Backend Loader

### `src/runtime/backend/backend_loader.h` / `.cpp`

Lives in the main binary. Uses `dlopen` / `dlsym` / `dlclose` (POSIX). No TRT headers.

```cpp
#pragma once
#include "trtmc/runtime/trt_backend.h"
#include <string>

namespace trtmc {

class BackendLoader {
public:
    // Load backend by name ("trt" or "trt_rtx").
    // Caches: second call with same name returns same IBackend*.
    // Throws std::runtime_error if DSO not found or factory missing.
    static IBackend* load(const std::string& backend_name);
};

} // namespace trtmc
```

**DSO search order:**
1. Directory of the running binary (`/proc/self/exe` → dirname)
2. `TRTMC_BACKEND_DIR` environment variable
3. Default `dlopen` search path (`LD_LIBRARY_PATH`, system dirs)

**DSO naming:** `libtrtmc_backend_{name}.so` — e.g., `libtrtmc_backend_trt.so`.

**Lifetime:** `IBackend*` is process-lifetime (stored in a static map). `dlclose`
is called via `atexit`. The backend must outlive all modules it created.

**Error message on missing DSO:**
```
Error: backend "trt_rtx" not available.
  Could not load libtrtmc_backend_rtx.so: libtensorrt_rtx.so: cannot open shared object file
  
  To use TRT-RTX bundles:
    1. Install TensorRT-RTX: pip install tensorrt_rtx
    2. Build the RTX backend: cmake -DTRTMC_RTX_LIBRARY_DIR=<path> ...
    3. Ensure libtrtmc_backend_rtx.so is next to the trtmc binary or in LD_LIBRARY_PATH
  
  To use standard TRT bundles, build with engine_backend="trt" (the default).
```

---

## Section 4: Pipeline Factory Changes

### `src/runtime/registry/pipeline_factory.cpp`

The factory reads `engine_backend` from config.json and loads the right DSO.
It creates an `IBackend*` and passes it into `PipelineContext` so plugins can
call `backend->create_module(...)`.

```cpp
std::unique_ptr<IPipeline> PipelineFactory::from_bundle(
    const std::string& bundle_path, const std::string& hf_python,
    const std::string& runtime_cache_path, bool cuda_graphs)
{
    BundleFile bundle = ReadBundleFile(bundle_path);
    // ... extract config_text, parse strategy (same as today) ...

    // NEW: read engine_backend, default "trt" for backward compat
    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = BackendLoader::load(backend_name);

    // Store in PipelineContext so plugins can create modules
    PipelineContext ctx{bundle, base_cfg, config_text, hf_python, bundle_path,
                        backend, runtime_cache_path, cuda_graphs};
    auto pipeline = plugin->create(ctx);
    // ...
}
```

### `PipelineContext` gains new fields

```cpp
struct PipelineContext {
    const BundleFile& bundle;
    const BaseConfig& config;
    const std::string& config_json;
    const std::string& hf_python;
    const std::string& bundle_path;
    IBackend* backend;                   // NEW
    const std::string& runtime_cache_path; // NEW (for RTX JIT cache)
    bool cuda_graphs;                     // NEW (for RTX CUDA graphs)
};
```

### Plugin helpers change

`load_trt_module_from_plan()` and its variants now take `IBackend*` and
`ModuleCreateOptions` instead of doing their own deserialization:

```cpp
// Before:
LoadedModule load_trt_module_from_plan(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream = nullptr);

// After:
struct LoadedModule {
    std::unique_ptr<ITrtModule> module;
    // stream is now owned by the module internally
};

LoadedModule load_trt_module_from_plan(
    IBackend* backend,
    const std::vector<char>* plan, const char* label,
    const ModuleCreateOptions& options = {});
```

### Plugin code changes (minimal)

Each plugin's `create()` changes from:
```cpp
auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
```
to:
```cpp
ModuleCreateOptions opts;
opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
opts.cuda_graphs = ctx.cuda_graphs;
auto loaded = load_trt_module_from_plan(ctx.backend,
    find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);
```

---

## Section 5: `TRTMC_HAS_TRT` Removal Strategy

Today, `TRTMC_HAS_TRT` gates ALL TRT-touching code. After this change:

| Code location | Before | After |
|---|---|---|
| `trt_module.h` | `#if TRTMC_HAS_TRT` around concrete class | Pure virtual interface, no guard needed |
| `trt_module.cpp` | `#if TRTMC_HAS_TRT` around implementation | Moves to DSO (`trt_module_impl.cpp`), guarded internally |
| `trt_common.h/cpp` | `#if TRTMC_HAS_TRT` around CudaStream/Buffer/Logger | Split: CudaStream/CudaBuffer stay in main binary (pure CUDA, no guard). Logger moves to DSOs. |
| `trt_engine_lifecycle.h/cpp` | TRT engine helpers | Moves to DSOs |
| Pipeline headers | `#if TRTMC_HAS_TRT` around entire class | No guard — uses `ITrtModule*` (always defined) |
| Plugin `.cpp` files | `#if TRTMC_HAS_TRT` around entire file | No guard — uses `IBackend*` and `ITrtModule*` |
| `pipeline_factory.cpp` | `#if TRTMC_HAS_TRT` block | No guard — uses `BackendLoader::load()` which throws at runtime if no DSO |

The `TRTMC_HAS_TRT` define is **removed entirely** from the main binary. If no backend
DSO is available, `BackendLoader::load()` throws at runtime with a clear error —
exactly like Python's `ImportError` for missing packages.

---

## Section 6: CMake Build System

```cmake
# ─── CUDA runtime (always linked) ───
find_package(CUDAToolkit REQUIRED)

# ─── Main library (NO TRT link) ───
add_library(trtmc_core
    # Everything currently listed MINUS:
    #   src/runtime/core/trt_common.cpp  (logger portion → DSOs)
    #   src/runtime/core/trt_module.cpp  (→ DSOs as trt_module_impl.cpp)
    #   src/runtime/core/trt_engine_lifecycle.cpp (→ DSOs)
    # PLUS:
    #   src/runtime/core/cuda_common.cpp  (CudaStream, CudaBuffer — pure CUDA)
    #   src/runtime/backend/backend_loader.cpp
    ...all current sources except the 3 TRT-touching files above...
    src/runtime/core/cuda_common.cpp
    src/runtime/backend/backend_loader.cpp
)
target_link_libraries(trtmc_core PRIVATE CUDA::cudart dl)  # dl for dlopen
# NO libnvinfer, NO TRT include dirs

# ─── Shared DSO sources (compiled into both backends) ───
set(BACKEND_COMMON_SOURCES
    src/runtime/backend/trt_module_impl.cpp
    src/runtime/backend/trt_logger.cpp
    src/runtime/backend/trt_engine_helpers.cpp  # deserialization helpers
)

# ─── Standard TRT backend (optional) ───
option(TRTMC_BUILD_BACKEND_TRT "Build standard TRT backend DSO" ON)
if(TRTMC_BUILD_BACKEND_TRT)
    find_library(NVINFER_LIB nvinfer HINTS ${TRTMC_TRT_LIBRARY_DIR})
    find_path(TRT_INCLUDE_DIR NvInferRuntime.h HINTS ${TRTMC_TRT_INCLUDE_DIR})
    if(NVINFER_LIB AND TRT_INCLUDE_DIR)
        add_library(trtmc_backend_trt SHARED
            ${BACKEND_COMMON_SOURCES}
            src/runtime/backend/trt_backend.cpp
        )
        target_include_directories(trtmc_backend_trt PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${TRT_INCLUDE_DIR}
        )
        target_link_libraries(trtmc_backend_trt PRIVATE
            ${NVINFER_LIB} CUDA::cudart
        )
        # Install next to the trtmc binary
        install(TARGETS trtmc_backend_trt LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()

# ─── TRT-RTX backend (optional) ───
option(TRTMC_BUILD_BACKEND_RTX "Build TRT-RTX backend DSO" OFF)
if(TRTMC_BUILD_BACKEND_RTX)
    find_library(TENSORRT_RTX_LIB tensorrt_rtx HINTS ${TRTMC_RTX_LIBRARY_DIR})
    find_path(TRT_RTX_INCLUDE_DIR NvInferRuntime.h HINTS ${TRTMC_RTX_INCLUDE_DIR})
    if(TENSORRT_RTX_LIB AND TRT_RTX_INCLUDE_DIR)
        add_library(trtmc_backend_rtx SHARED
            ${BACKEND_COMMON_SOURCES}
            src/runtime/backend/rtx_backend.cpp
        )
        target_include_directories(trtmc_backend_rtx PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${TRT_RTX_INCLUDE_DIR}
        )
        target_link_libraries(trtmc_backend_rtx PRIVATE
            ${TENSORRT_RTX_LIB} CUDA::cudart
        )
        install(TARGETS trtmc_backend_rtx LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()
```

---

## Section 7: Python Builder `--rtx` Flag

### CLI change (`cli.py`)

```python
build_p.add_argument("--rtx", action="store_true",
                     help="Build engine for TRT-RTX (portable, JIT-compiled at runtime)")
```

### Import abstraction (`trt_import.py` — new file)

```python
"""Centralized TRT import. All builder modules import `trt` from here."""

_trt_module = None

def get_trt(rtx: bool = False):
    """Return the tensorrt module (standard or RTX)."""
    global _trt_module
    if _trt_module is not None:
        return _trt_module
    if rtx:
        import tensorrt_rtx as trt
    else:
        import tensorrt as trt
    _trt_module = trt
    return trt

def trt():
    """Return the cached trt module. Must call get_trt() first."""
    if _trt_module is None:
        raise RuntimeError("Call get_trt() before using trt()")
    return _trt_module
```

### Module-level import changes

Today, `graph_ops.py`, `graph_blocks.py`, `standard_decoder_builder.py`, and
family plugins all do `import tensorrt as trt` at module level.

**Change:** Replace with lazy import via the abstraction:

```python
# graph_ops.py — before:
import tensorrt as trt

# graph_ops.py — after:
from .trt_import import trt
# Usage unchanged: trt().float32, trt().INetworkDefinition, etc.
```

Alternatively, since all builder entry points go through `engine_builder.py`,
we can call `get_trt(rtx=args.rtx)` once at the top of `build_bundle()` before
any TRT-using module is imported, and let modules continue using
`import tensorrt as trt` since `tensorrt_rtx` installs as a separate name.

**Simpler approach (recommended):** Monkeypatch `sys.modules`:

```python
# In engine_builder.py, before any TRT import:
def _setup_trt_import(rtx: bool):
    if rtx:
        import tensorrt_rtx
        import sys
        sys.modules["tensorrt"] = tensorrt_rtx
```

This means all existing `import tensorrt as trt` statements throughout the
codebase automatically resolve to `tensorrt_rtx` when `--rtx` is active.
Zero changes to `graph_ops.py`, `graph_blocks.py`, family plugins, etc.

### Bundle metadata injection

In `engine_builder.py`, when writing `config.json` into the bundle:

```python
if rtx:
    cfg_dict["engine_backend"] = "trt_rtx"
else:
    cfg_dict["engine_backend"] = "trt"
```

The C++ runtime reads this field to decide which DSO to `dlopen`.

### `trtmc-build inspect` output

Add `engine_backend` to the inspect display:

```
Engine backend:  trt_rtx
```

---

## Section 8: CLI Changes (C++ Runtime)

### `trtmc run` gains two flags

```
--runtime-cache PATH   TRT-RTX JIT kernel cache file (ignored for standard TRT bundles)
--cuda-graphs          TRT-RTX CUDA graph capture (ignored for standard TRT bundles)
```

These are passed through `trtmc::load()` → `PipelineFactory::from_bundle()` →
`PipelineContext` → plugin → `ModuleCreateOptions` → `IBackend::create_module()`.

### Public API change

```cpp
// include/trtmc/pipeline.h
std::unique_ptr<IPipeline> load(
    const std::string& bundle_path,
    const std::string& hf_python = "",
    const std::string& runtime_cache_path = "",
    bool cuda_graphs = false);
```

### C ABI change

```cpp
struct TrtmcPipelineOptions {
    int max_new_tokens;
    const char* hf_python;
    const char* image_path;
    const char* runtime_cache;   // NEW
    int cuda_graphs;             // NEW (0/1)
};
```

---

## Section 9: File Layout After Migration

### New files

```
include/trtmc/runtime/trt_backend.h           # IBackend interface + ModuleCreateOptions
src/runtime/core/cuda_common.h                # CudaStream, CudaBuffer (extracted from trt_common)
src/runtime/core/cuda_common.cpp
src/runtime/backend/backend_loader.h          # BackendLoader (dlopen dispatch)
src/runtime/backend/backend_loader.cpp
src/runtime/backend/trt_module_impl.h         # TrtModuleImpl : ITrtModule (DSO-only)
src/runtime/backend/trt_module_impl.cpp       # Implementation (compiled in both DSOs)
src/runtime/backend/trt_logger.h              # TrtLogger (DSO-only, moved from trt_common)
src/runtime/backend/trt_logger.cpp
src/runtime/backend/trt_engine_helpers.h      # Engine deserialization helpers (DSO-only)
src/runtime/backend/trt_engine_helpers.cpp
src/runtime/backend/trt_backend.cpp           # TrtBackend : IBackend (standard TRT DSO)
src/runtime/backend/rtx_backend.cpp           # RtxBackend : IBackend (TRT-RTX DSO)
tensorrt_model_connect/tensorrt_model_connect/trt_import.py           # (optional if using sys.modules approach)
```

### Modified files

```
include/trtmc/runtime/trt_module.h             # Concrete class → ITrtModule interface
include/trtmc/pipeline.h                       # load() signature gains 2 params
include/trtmc/runtime/pipeline_plugin.h        # PipelineContext gains backend/cache/graphs fields
CMakeLists.txt                                # Remove libnvinfer link from trtmc_core, add DSO targets
src/runtime/registry/pipeline_factory.cpp     # BackendLoader integration
src/runtime/plugins/shared/plugin_helpers.h   # LoadedModule uses ITrtModule, takes IBackend*
src/runtime/plugins/shared/plugin_helpers.cpp # Delegates to IBackend::create_module()
src/runtime/plugins/*.cpp                     # Remove TRTMC_HAS_TRT guards, use ctx.backend
src/runtime/pipelines/*.h                     # TrtModule → ITrtModule member type
examples/trtmc_cli.cpp                         # --runtime-cache, --cuda-graphs flags
tensorrt_model_connect/tensorrt_model_connect/cli.py                  # --rtx flag
tensorrt_model_connect/tensorrt_model_connect/engine_builder.py       # sys.modules monkeypatch, engine_backend metadata
```

### Deleted files

```
src/runtime/core/trt_common.h                 # Split into cuda_common.h + DSO logger
src/runtime/core/trt_common.cpp               # Split into cuda_common.cpp + DSO logger
src/runtime/core/trt_module.cpp               # → backend/trt_module_impl.cpp
src/runtime/core/trt_engine_lifecycle.h/cpp    # → backend/trt_engine_helpers.h/cpp
```

---

## Section 10: Backward Compatibility

- **Old bundles (no `engine_backend` field):** Default to `"trt"` — fully backward compatible
- **Old C ABI callers (no `runtime_cache`/`cuda_graphs` fields):** New fields are at the end of the struct, initialized to zero/null — ABI compatible if callers use `{0}` initialization
- **Python builder without `--rtx`:** Produces standard TRT bundles — no change in behavior
- **Build without either TRT SDK:** Main binary compiles fine. `trtmc inspect` works. `trtmc run` fails at runtime with a clear message about missing backend DSO
- **Build with only standard TRT:** Only `libtrtmc_backend_trt.so` is produced. RTX bundles fail to load with a clear error

---

## Section 11: Testing Strategy

### Unit tests (no GPU, no TRT)

- `test_backend_loader.cpp` — mock DSO loading, error messages, caching
- `test_itrt_module_interface.cpp` — mock ITrtModule, verify pipeline code compiles and calls correctly
- `test_bundle_metadata.cpp` — verify `engine_backend` field parsing, defaulting to `"trt"`

### Integration tests (needs TRT SDK)

- Build a standard TRT bundle, verify `engine_backend: "trt"` in config.json
- Load via `BackendLoader`, verify `libtrtmc_backend_trt.so` is dlopen'd
- Run inference, verify identical results to pre-migration

### Python builder tests

- `test_cli.py` — verify `--rtx` flag is parsed
- `test_engine_builder.py` — verify `sys.modules` monkeypatch works, `engine_backend` in bundle config
- Mock-based: verify `import tensorrt_rtx` is attempted when `--rtx` is active

### E2E (needs GPU + TRT SDK)

- Standard E2E suite passes with no changes (all existing bundles are `engine_backend: "trt"`)
- If TRT-RTX is available: build an RTX bundle, run inference, compare against HF reference

---

## Implementation Order

1. **Create `ITrtModule` interface** and `IBackend` + `ModuleCreateOptions` headers
2. **Extract `CudaStream`/`CudaBuffer`** from `trt_common` into `cuda_common.h/cpp`
3. **Create `TrtModuleImpl`** in `src/runtime/backend/` — lift today's `TrtModule` implementation
4. **Create `TrtBackend`** (`trt_backend.cpp`) — standard TRT IBackend
5. **Create `BackendLoader`** — dlopen + dlsym + caching
6. **Update `PipelineContext`** with `IBackend*`, cache path, CUDA graphs
7. **Update `plugin_helpers`** — `load_trt_module_from_plan()` takes `IBackend*`
8. **Update all pipeline headers** — `TrtModule` → `ITrtModule`
9. **Update all plugins** — use `ctx.backend`, remove `TRTMC_HAS_TRT` guards
10. **Update `pipeline_factory.cpp`** — read `engine_backend`, use `BackendLoader`
11. **Update CMakeLists.txt** — DSO targets, remove libnvinfer from trtmc_core
12. **Update CLI** — `--runtime-cache`, `--cuda-graphs` flags
13. **Update public API** — `load()` signature
14. **Create `RtxBackend`** (`rtx_backend.cpp`) — TRT-RTX IBackend with runtime cache
15. **Python `--rtx` flag** — `sys.modules` monkeypatch, `engine_backend` metadata
16. **Remove `TRTMC_HAS_TRT`** guards from all main binary code
17. **Tests** — backend loader, interface, E2E regression
