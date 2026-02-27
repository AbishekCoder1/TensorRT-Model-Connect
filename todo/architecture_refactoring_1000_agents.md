# Architecture Refactoring Plan: Scaling to 100-1000 Parallel Agents

## Context

The C++ runtime (`src/`) and Python builder (`trtf_build/`) have accumulated significant
code duplication as the project grew from a handful of models to 50+. The biggest bottleneck
for parallel development is **`trtf_c.cpp` (2,715 lines)** — a single monolithic file that
every new backend must edit. With 100-1000 agents working simultaneously, this file alone
would cause >90% merge conflict rate. The audit found **~4,500 lines of duplication** across
both codebases.

**Goal:** Restructure the project so that adding a new backend/model requires **zero edits
to shared files** — only adding new files in self-contained directories.

---

## Audit Summary

### C++ Runtime Duplication

| Issue | Severity | Files | Lines Affected |
|-------|----------|-------|----------------|
| Generate loop duplication | CRITICAL | 5 backends | 250-350 |
| TRT deserialization duplication | HIGH | trtf_c.cpp | 120+ |
| trtf_c.cpp monolith (2,715 lines) | CRITICAL | 1 file | 2,715 |
| Missing backend registry | HIGH | trtf_c.cpp | 150+ |
| Inconsistent state updates | MEDIUM | 4 backends | 40-80 |
| No step engine abstraction | MEDIUM | 3 backends | — |
| Duplicate engine validation | LOW-MEDIUM | 4 backends | 40 |
| Encoder backend duplication | MEDIUM | 5 backends | 250 |
| Speech backend overload | MEDIUM | 1 file (1,503 lines) | 1,503 |

### Python Builder Duplication

| Category | Files | Duplicated Lines |
|----------|-------|------------------|
| Standard Decoder Plugins | 18+ | 500+ (30 per file) |
| Diffusion Preprocessor Serialization | 3 | 120+ (40 per file) |
| DiT Builders | 3 | 1200+ boilerplate |
| Text Encoder Builders | 3 | 800+ boilerplate |
| Debug Runner CUDA/TRT init | 7 classes | 400+ |
| Vision Builders | 5 | 200+ (ViT patterns) |
| Safetensors Loading | 3+ | 150+ |
| **TOTAL** | **50+** | **~4,500+ lines** |

### Merge Conflict Hotspots (for 100+ agents)

1. **`trtf_c.cpp`** — EXTREME risk (>90% conflict probability)
   - Lines 2509-2627: 28-way strategy dispatch
   - Lines 1-31: 20+ backend `#include`s
   - Lines 130-1100: PipelineImpl methods
   - Lines 2000-2427: 35+ `create_*_pipeline` functions
2. **`src/runtime/trt/` backend files** — HIGH risk (~60%)
3. **`debug_runner.py`** — MEDIUM risk (single 69KB file)

---

## Phase 1: C++ Runtime — Break the Monolith (CRITICAL)

### 1.1 Backend Registry Pattern (eliminates trtf_c.cpp bottleneck)

**Problem:** `trtf_c.cpp:2429-2627` has a 28-way if-else dispatch. Every new backend edits this.

**Solution:** Self-registering backend factory pattern (like Python `families/` auto-discovery).

**Files to create:**
- `src/cabi/backend_registry.h` — `IBackendFactory` interface + global registry
- `src/cabi/backend_registry.cpp` — registry implementation

```cpp
// backend_registry.h
namespace trtf {

class IBackendFactory {
public:
    virtual ~IBackendFactory() = default;
    virtual const char* strategy_name() const = 0;
    virtual std::unique_ptr<IGenerationBackend> create(
        const BundleSections& sections,
        const FastPathConfig& config,
        std::unique_ptr<ITokenizer> tokenizer) = 0;
};

// Self-registration macro — backends call this at file scope
#define TRTF_REGISTER_BACKEND(ClassName) \
    static bool _registered_##ClassName = \
        ::trtf::BackendRegistry::instance().register_factory( \
            std::make_unique<ClassName>())

class BackendRegistry {
public:
    static BackendRegistry& instance();
    bool register_factory(std::unique_ptr<IBackendFactory> factory);
    std::unique_ptr<IGenerationBackend> create(
        const std::string& strategy,
        const BundleSections& sections,
        const FastPathConfig& config,
        std::unique_ptr<ITokenizer> tokenizer);
};

} // namespace trtf
```

**Files to modify:**
- `src/cabi/trtf_c.cpp` — Replace 28-way if-else with `BackendRegistry::instance().create(strategy, ...)`
- Each backend `.cpp` — Add `TRTF_REGISTER_BACKEND(MyFactory)` at file scope

**Result:** New backends register themselves. Zero edits to `trtf_c.cpp`.

### 1.2 Extract Autoregressive Loop Template (eliminates 250-350 lines duplication)

**Problem:** 5 backends (`trt_backend_shared`, `mamba_backend`, `rwkv_backend`, `vl_backend`,
`hybrid_backend`) duplicate the same prefill + decode loop.

**Solution:** Template Method with NVI (Non-Virtual Interface):

**File to create:**
- `src/runtime/trt/autoregressive_loop.h`

```cpp
// The invariant skeleton. Subclasses override only the hooks.
template <typename StepState>
std::vector<int32_t> run_autoregressive(
    const std::vector<int32_t>& input_ids,
    const GenerationConfig& config,
    // Hooks (callables):
    std::function<void(StepState&, int32_t token_id)> run_step,
    std::function<int32_t(const StepState&)> select_token,
    int32_t eos_token_id);
```

**Files to simplify:**
- `src/runtime/trt/trt_backend_shared.cpp` — generate() calls `run_autoregressive()`
- `src/runtime/trt/mamba_backend.cpp` — same
- `src/runtime/trt/rwkv_backend.cpp` — same
- `src/runtime/trt/hybrid_backend.cpp` — same
- `src/runtime/trt/vl_backend.cpp` — same (with pre-hook for vision embedding)

### 1.3 Reorganize into Per-Backend Subdirectories (merge conflict isolation)

**Current (flat):**
```
src/runtime/trt/
    trt_backend_shared.h/cpp    # KV-cache decoder
    mamba_backend.h/cpp         # Mamba SSM
    rwkv_backend.h/cpp          # RWKV
    vl_backend.h/cpp            # Vision-language
    hybrid_backend.h/cpp        # Hybrid Mamba+Attention
    speech_backend.h/cpp        # Whisper/Bark/PersonaPlex
    diffusion_backend.h/cpp     # Wan/FLUX/Z-Image
    encoder_backend.h/cpp       # BERT
    embedding_backend.h/cpp     # Eagle-embed
    segmentation_backend.h/cpp  # SegFormer
    sam_backend.h/cpp           # SAM
    reranking_backend.h/cpp     # Eagle-rerank
    device_kv_cache.h/cpp       # Shared KV cache
    trt_common.h/cpp            # Shared TRT helpers
    trt_engine_lifecycle.h/cpp  # Shared engine lifecycle
    ...
```

**Proposed (per-backend directories):**
```
src/
  core/                              # Stable abstractions (rarely modified)
    backend_registry.h/cpp           # NEW: self-registering factory
    autoregressive_loop.h            # NEW: shared generate template
    engine_helpers.h/cpp             # NEW: extracted from trtf_c.cpp (deserialize, validate)
  runtime/
    common/                          # Shared TRT plumbing
      trt_common.h/cpp
      trt_engine_lifecycle.h/cpp
      cuda_buffer.h  (in trt_common.h currently)
      cuda_stream.h  (in trt_common.h currently)
    kv_cache/                        # Standard decoder backend
      trt_backend_shared.h/cpp
      device_kv_cache.h/cpp
      trt_decode_runtime.h/cpp
      step_state.h
    mamba/                           # Mamba SSM backend
      mamba_backend.h/cpp
      mamba_step_state.h/cpp
      mamba_decode_runtime.h/cpp
    rwkv/                            # RWKV backend
      rwkv_backend.h/cpp
      rwkv_step_state.h/cpp
      rwkv_decode_runtime.h/cpp
    hybrid/                          # Hybrid Mamba+Attention
      hybrid_backend.h/cpp
      hybrid_step_state.h/cpp
    vl/                              # Vision-language
      vl_backend.h/cpp
      image_preprocessor.h/cpp
      vision_engine.h/cpp
    diffusion/                       # Wan, FLUX, Z-Image
      diffusion_backend.h/cpp
      wan_backend.h/cpp              # Wan-specific if needed
    audio/                           # Whisper, Bark, PersonaPlex
      speech_backend.h/cpp
    segmentation/                    # SegFormer, SAM
      segmentation_backend.h/cpp
      sam_backend.h/cpp
    encoder/                         # BERT, embedding, reranking
      encoder_backend.h/cpp
      embedding_backend.h/cpp
      reranking_backend.h/cpp
  bundle/                            # Bundle format (unchanged)
  tokenizer/                         # Tokenizers (unchanged)
  cabi/                              # C ABI (simplified)
    trtf_c.cpp                       # Slimmed: uses BackendRegistry
    fast_path_config.h/cpp
    bundle_helpers.h/cpp
  utils/                             # Shared utilities (unchanged)
```

### 1.4 Extract Engine Deserialization Helpers (eliminates ~120 lines duplication)

**Problem:** 8+ engine types in `trtf_c.cpp` repeat the same deserialize+validate pattern.

**File to create:**
- `src/core/engine_helpers.h/cpp`

```cpp
struct EngineComponent {
    std::string name;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;
};

// Single function replaces 8 copies of deserialize+validate
EngineComponent deserialize_engine(
    nvinfer1::IRuntime* runtime,
    const uint8_t* plan_data, size_t plan_size,
    const std::string& name);
```

### 1.5 Split trtf_c.cpp (reduce from 2,715 lines)

**Problem:** Monolithic C ABI file is the #1 merge conflict hotspot.

**Split into:**
| New file | Contents | Lines (approx) |
|----------|----------|----------------|
| `trtf_c.cpp` | C ABI entry points only (create_pipeline, generate, etc.) | ~400 |
| `pipeline_impl.h/cpp` | PipelineImpl class | ~500 |
| `pipeline_dispatch.cpp` | Uses BackendRegistry (replaces 28-way if-else) | ~50 |
| `pipeline_audio.cpp` | Audio-specific pipeline methods | ~300 |
| `pipeline_diffusion.cpp` | Diffusion-specific pipeline methods | ~300 |
| `pipeline_vl.cpp` | VL-specific pipeline methods | ~200 |

---

## Phase 2: Python Builder — Reduce Plugin Boilerplate

### 2.1 StandardDecoderPlugin Base Class (eliminates 500+ lines across 18 plugins)

**File to create:**
- `trtf_build/trtf_build/families/standard_decoder_base.py`

```python
class StandardDecoderPlugin:
    """Base for families that use standard decoder with no weight customization."""
    runtime_strategy = "decoder_kv_cache"

    def load_weights(self, model_dir, config):
        weights = load_standard_weights(model_dir, config)
        return self.transform_weights(weights, config)  # Hook

    def transform_weights(self, weights, config):
        return weights  # Override for Gemma +1.0, Phi QKV split, etc.

    def build_engine(self, config, weights, max_cache_length, *, verbose=False):
        return build_standard_decoder_engine(config, weights, max_cache_length, verbose=verbose)
```

**Files to simplify:** `qwen.py`, `llama.py`, `mistral.py`, `opt.py`, `gpt2.py`, `internlm.py`,
`starcoder2.py`, `xglm.py`, `gpt_neo.py`, `gpt_neox.py`, `codegen.py`, `bloom.py`, `olmo.py`,
`stablelm.py`, `granite.py`, `falcon.py`

Each becomes ~10-20 lines (just `matches()` + optional `transform_weights()`).

### 2.2 Shared DiT Preprocessor Serialization (eliminates 120 lines across 3 files)

**File to create:**
- `trtf_build/trtf_build/dit_utils.py`

```python
def serialize_dit_preprocessor_weights(dit_weights: dict, key_map: dict) -> bytes:
    """Shared serialization for Wan, FLUX, Z-Image preprocessor weights."""
    ...
```

**Files to simplify:** `wan_t2v.py`, `flux.py`, `z_image.py`

### 2.3 BaseTrtRunner for Debug Runners (eliminates 400+ lines)

**Modify:** `trtf_build/trtf_build/debug_runner.py`

Extract shared CUDA/TRT initialization into `BaseTrtRunner.__init__()`. Subclasses
(`TrtRunner`, `MambaTrtRunner`, `RwkvTrtRunner`, etc.) only override `step()`.

---

## Phase 3: CMake Build System — Per-Backend Targets

### 3.1 One CMake Target Per Backend Subdirectory

```cmake
# src/runtime/kv_cache/CMakeLists.txt
add_library(trtf_kv_cache
    trt_backend_shared.cpp
    device_kv_cache.cpp
    trt_decode_runtime.cpp)
target_link_libraries(trtf_kv_cache PRIVATE trtf_core trtf_common)

# src/runtime/mamba/CMakeLists.txt
add_library(trtf_mamba
    mamba_backend.cpp
    mamba_step_state.cpp
    mamba_decode_runtime.cpp)
target_link_libraries(trtf_mamba PRIVATE trtf_core trtf_common)
```

**Result:** Changing a file in `runtime/mamba/` only recompiles `trtf_mamba`. Other backends
untouched. Parallel agents working on different backends never conflict.

### 3.2 Optional Backend Compilation

```cmake
option(TRTF_ENABLE_MAMBA "Build Mamba SSM backend" ON)
option(TRTF_ENABLE_DIFFUSION "Build diffusion backends" ON)
option(TRTF_ENABLE_VL "Build vision-language backend" ON)

if(TRTF_ENABLE_MAMBA)
    add_subdirectory(runtime/mamba)
    target_link_libraries(trtf_runtime PRIVATE trtf_mamba)
endif()
```

---

## Phase 4: Parallel Agent Enablement

### 4.1 CODEOWNERS-Style Ownership

```
# .github/CODEOWNERS
src/core/                    @core-team
src/runtime/kv_cache/        @decoder-team
src/runtime/mamba/           @ssm-team
src/runtime/diffusion/       @diffusion-team
src/runtime/vl/              @vl-team
src/runtime/audio/           @audio-team
trtf_build/families/         @builder-team
tests/e2e/models/            @any-contributor
```

### 4.2 "Add a Backend" Checklist (Zero Shared-File Edits)

After refactoring, adding a new backend requires ONLY:

1. **Create** `src/runtime/mybackend/` with backend files
2. **Add** `TRTF_REGISTER_BACKEND(MyBackendFactory)` in the `.cpp`
3. **Create** `trtf_build/trtf_build/families/myfamily.py` (auto-discovered)
4. **Create** `tests/e2e/models/mymodel.json` manifest
5. **Add** `add_subdirectory(runtime/mybackend)` in parent CMakeLists.txt

Items 1-4 touch zero existing files. Item 5 is a single line in a CMake file
(lowest conflict risk — append-only).

---

## Design Patterns Applied

Based on research of TensorRT-LLM, vLLM, llama.cpp, Chromium, and LLVM:

| Pattern | Where Applied | Rationale |
|---------|---------------|-----------|
| **Strategy** (virtual dispatch) | `IBackendFactory` + `BackendRegistry` | Runtime-selected backends based on bundle config |
| **Template Method** (NVI) | `run_autoregressive()` loop | Shared prefill+decode skeleton, customizable step hooks |
| **Self-Registration** (static init) | `TRTF_REGISTER_BACKEND` macro | Zero shared-file edits for new backends (llama.cpp style) |
| **Plugin auto-discovery** | Python `families/` (existing) | Already working; extend pattern to C++ |
| **Component composition** | Split `IPipeline` by modality | Each modality is an independent interface |
| **LLVM layering** | Per-backend CMake libraries | Strict dependency DAG, no circular deps |
| **Chromium CODEOWNERS** | Per-directory ownership | Review routing, responsibility isolation |
| **Microkernel** | `src/core/` stable kernel + backend plugins | Core infra stable; backends evolve independently |

---

## Execution Order

| Step | Phase | Risk | Merge Conflicts During | Est. Lines Changed |
|------|-------|------|------------------------|-------------------|
| 1 | 1.1 Backend Registry | Low | None (additive) | +200 new, ~100 modified |
| 2 | 1.4 Engine Helpers | Low | None (extract) | +80 new, -120 in trtf_c.cpp |
| 3 | 1.2 Autoregressive Loop | Medium | Backend files | +100 new, -250 across 5 files |
| 4 | 1.5 Split trtf_c.cpp | Medium | trtf_c.cpp | +0 net (reorganization) |
| 5 | 1.3 Directory Reorganization | High | CMakeLists, includes | +0 net (moves) |
| 6 | 2.1 StandardDecoderPlugin | Low | Family plugins | +50 new, -500 across 18 files |
| 7 | 2.2 DiT Utils | Low | 3 diffusion plugins | +40 new, -120 across 3 files |
| 8 | 2.3 BaseTrtRunner | Medium | debug_runner.py | -400 lines |
| 9 | 3.1-3.2 CMake Restructure | Medium | CMakeLists.txt | CMake only |
| 10 | 4.1-4.2 CODEOWNERS + docs | Low | None | Docs only |

**Total estimated savings:** ~1,500 lines removed, ~500 lines added = **~1,000 net reduction**
plus elimination of the #1 merge conflict hotspot.

---

## Verification

After each step, run the regression tiers in order:

```bash
# Tier 1: Unit tests (must pass after every step)
.venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py
ctest --test-dir build --output-on-failure

# Tier 2: Graph-op GPU tests (after Phase 2 changes)
.venv/bin/python -m pytest tests/builder/test_graph_ops.py tests/builder/test_graph_ops_extended.py -v

# Tier 3: E2E smoke (after Phase 1 changes)
.venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python

# Tier 4: Full E2E (after directory reorganization)
.venv/bin/python -m pytest tests/test_e2e.py -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python
```
