# Infrastructure & Testing Modularity Improvement Plan

## Context

The project supports 23+ model families via Python plugins and 4 C++ runtime strategies. The **Python builder** is well-factored (zero-edit extensibility via auto-discovered plugins), but the **C++ runtime** and **test infrastructure** have scalability bottlenecks that prevent multiple agents from working in parallel on new model support. Specifically:

- Adding a new C++ runtime strategy requires modifying 3 existing files (OCP violations)
- VL-specific logic leaks into generic decoder paths
- Autoregressive loop is copy-pasted across 3 backends
- C++ tests use an inconsistent custom framework with duplicated helpers
- Python test parametrization uses a single hardcoded list (merge conflict magnet)

**Goal**: Make the system fully modular so adding a new runtime strategy = adding new files only, and adding a new model family = adding one plugin file + one test manifest. No edits to shared code.

---

## Work Stream 1: C++ Backend Registry Pattern [M]

**Problem**: `src/cabi/trtf_c.cpp:409-428` — if/else chain dispatches on `runtime_strategy` string. Adding a strategy means editing this file + writing a factory function in the same translation unit.

**Solution**: Self-registering backend factory with a map-based registry.

### New files
| File | Purpose |
|------|---------|
| `src/cabi/backend_registry.h` | `BackendRegistry` singleton + `BackendFactoryArgs/Result` structs + `TRTF_REGISTER_BACKEND` macro |
| `src/cabi/backend_registry.cpp` | Registry implementation (map insert/lookup) |
| `src/cabi/decoder_factory.cpp` | Extract `create_decoder_pipeline()` from trtf_c.cpp, self-registers for `"decoder_kv_cache"` and `"decoder_moe"` |
| `src/cabi/mamba_factory.cpp` | Extract `create_mamba_pipeline()`, self-registers for `"ssm_recurrent"` |
| `src/cabi/vl_factory.cpp` | Extract `create_vl_pipeline()`, self-registers for `"vision_language"` |

### Modify
- **`src/cabi/trtf_c.cpp`** — Replace lines 193-428 (3 factory functions + if/else dispatch) with:
  ```cpp
  const auto* factory = BackendRegistry::instance().find(strategy);
  if (!factory) throw std::runtime_error("Unknown strategy: " + strategy);
  auto result = (*factory)(std::move(args));
  ```
- **`CMakeLists.txt`** — Add new `.cpp` files to `trtf_core` sources. May need `-Wl,--whole-archive` to prevent linker dead-stripping of static registrations.

### Registry pattern
```cpp
// In any new_factory.cpp:
static BackendFactoryResult create_new_pipeline(BackendFactoryArgs args) { ... }
TRTF_REGISTER_BACKEND("new_strategy", create_new_pipeline)
// No edits to trtf_c.cpp needed!
```

### Parallel-agent benefit
Adding a new runtime strategy = one new `*_factory.cpp` file. Zero edits to dispatch code. No merge conflicts.

---

## Work Stream 2: Decouple VL from Generic Paths [M]

**Problem**: Three coupling issues:
1. `trtf_c.cpp:106` — `dynamic_cast<VLBackendFastPath*>` breaks `IGenerationBackend` abstraction
2. `fast_path_config.h` — god-struct with decoder + Mamba + VL fields
3. `device_kv_cache.cpp:199` — `run_decoder_step_device()` has 11 params, 5 VL-specific

### Solution A: IVisionCapable interface
Add to `include/trtf/backend.h`:
```cpp
class IVisionCapable {
public:
    virtual ~IVisionCapable() = default;
    virtual bool prepare_image(const std::string& path, ...) = 0;
};
```
- `VLBackendFastPath` implements `IGenerationBackend` + `IVisionCapable`
- `PipelineImpl` casts to `IVisionCapable*` (interface, not concrete type)
- Or: store optional `IVisionCapable*` at construction time (no cast needed)

### Solution B: Split FastPathModelConfig
```cpp
struct BaseModelConfig { vocab_size, hidden_size, num_layers, ... };
struct MambaConfig : BaseModelConfig { d_inner, state_size, conv_kernel };
struct VLConfig : BaseModelConfig { image_token_id, vision_output_dim, ... };
```
- Shared `parse_fast_path_config()` parses `BaseModelConfig` only
- Strategy-specific parsing moves to each factory file (from WS1)

### Solution C: Non-VL overload for run_decoder_step_device
Add a simplified overload that omits VL params (calls through with defaults). This documents intent and allows future split.

### Modify
- `include/trtf/backend.h` — Add `IVisionCapable`
- `src/cabi/fast_path_config.h/.cpp` — Split struct, move strategy parsing to factories
- `src/runtime/trt/vl_backend.h` — Implement `IVisionCapable`
- `src/cabi/trtf_c.cpp` — Remove `dynamic_cast<VLBackendFastPath*>`
- `src/runtime/trt/device_kv_cache.h` — Add non-VL overload

---

## Work Stream 3: Extract Shared Autoregressive Loop [S]

**Problem**: Prefill+decode+argmax+EOS pattern copy-pasted in:
- `trt_backend_shared.cpp:28-77`
- `mamba_backend.cpp:27-76`
- `vl_backend.cpp:45-97` (text-only path)

### New file
`src/runtime/trt/autoregressive_loop.h` — Template function:
```cpp
using StepFn = std::function<bool(int32_t token, std::vector<float>& logits, std::string& err)>;

std::vector<int32_t> autoregressive_generate(
    const std::vector<int32_t>& input_ids,
    const GenerationConfig& config,
    int32_t id_eos,
    StepFn step);
```

### Modify
- `trt_backend_shared.cpp` — Replace loop body with `autoregressive_generate()` call + lambda capture
- `mamba_backend.cpp` — Same
- `vl_backend.cpp` — Text-only path uses it; VL image path stays custom (fundamentally different prefill)

### Parallel-agent benefit
New backends get the loop for free. Just provide a step function.

---

## Work Stream 4: C++ Test Infrastructure Modernization [M]

**Problem**:
- Two inconsistent assertion patterns across 11 test files
- `check()` copy-pasted in 8 files
- `test_helpers.h` exists but is unused
- CMake: 44 lines of boilerplate (no foreach)
- `test_decode_runtime.cpp` silently passes when `TRTF_HAS_TRT=0`

### Part A: Unified harness in test_helpers.h
Add to `tests/cpp/test_helpers.h`:
- `test_failure_count()` — static counter
- `check(bool, const char*)` — shared assertion
- `run_test(name, fn)` — runs bool-returning test function
- `test_main_result(suite)` — standard main epilogue
- `TRTF_SKIP_IF_NO_TRT(suite)` — proper skip reporting

Migrate existing tests incrementally: `#include "test_helpers.h"`, remove local `check()`.

### Part B: CMake foreach
Replace 44 lines with:
```cmake
set(TRTF_TESTS test_text_parsers test_json_helpers test_data_dir ...)
foreach(t IN LISTS TRTF_TESTS)
    add_executable(${t} tests/cpp/${t}.cpp)
    target_link_libraries(${t} PRIVATE trtf_core)
    target_include_directories(${t} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
```
Adding a new test = append one name to the list.

### Part C: New tests
| File | Tests | GPU |
|------|-------|-----|
| `tests/cpp/test_backend_registry.cpp` | Registry register/find/unknown | No |
| `tests/cpp/test_autoregressive_loop.cpp` | Loop with mock step fn, EOS early-exit, empty input | No |

### Part D: Dead code cleanup
- Remove `select_topk_tokens()` from `trt_decode_runtime.cpp` (unused) or add a test for it
- Remove `trt_graph_builder.cpp` (vestigial 2-line stub)

---

## Work Stream 5: Python Builder Deduplication [S]

### Part A: Move shared functions to graph_blocks.py
- `_mark_debug_output` → `graph_blocks.mark_debug_output()` (used by 5 files, duplicated in 2)
- `_add_swiglu_expert` → `graph_blocks.add_swiglu_expert()` (duplicated in phi_moe.py + mixtral.py)

### Part B: Engine build boilerplate context manager
Add to `graph_blocks.py`:
```python
@contextmanager
def trt_builder_context(*, verbose=False):
    builder = trt.Builder(...)
    network = builder.create_network(...)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    yield builder, network, config

def build_and_serialize(builder, network, config) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None: raise RuntimeError("TRT build failed")
    return bytes(plan)
```
Reduces 20-30 lines of boilerplate per custom builder to 3-4 lines.

### Modify
- `trtf_build/trtf_build/graph_blocks.py` — Add `mark_debug_output`, `add_swiglu_expert`, builder utilities
- `trtf_build/trtf_build/standard_decoder_builder.py` — Import from graph_blocks
- `trtf_build/trtf_build/families/mamba.py` — Remove local copy, import
- `trtf_build/trtf_build/families/mixtral.py` — Import `add_swiglu_expert`
- `trtf_build/trtf_build/families/phi_moe.py` — Same

---

## Work Stream 6: Test Scalability for Parallel Agents [S]

### Part A: Discovery-based plugin test parametrization
**Problem**: `tests/builder/test_families.py` `_POSITIVE_MATCH_CASES` (lines 68-128) is a single list — merge conflict when two agents add families.

**Solution**: Each plugin declares `test_model_types` in its own file:
```python
# In qwen.py
plugin = QwenPlugin()
plugin.test_model_types = ["qwen", "qwen2", "qwen3", "qwq"]
```

`test_families.py` discovers them:
```python
def _discover_match_cases():
    return [(mt, p.name) for p in _ALL_PLUGINS
            for mt in getattr(p, "test_model_types", [])]
```

### Part B: Lightweight builder-to-runtime integration test
Create `tests/builder/test_bundle_roundtrip.py`:
- Python writes a minimal bundle via `bundle_writer.py`
- Subprocess runs C++ `test_bundle_format` binary to verify readability
- No GPU needed, catches format incompatibilities early

### Parallel-agent benefit
Adding a family plugin = add `test_model_types` to your plugin file + add a JSON manifest in `tests/e2e/models/`. Zero edits to shared test files.

---

## Dependency Graph & Parallel Assignment

```
WS1 (Registry) ───→ WS2 (VL decouple) [WS2 depends on WS1 for factory extraction]
WS3 (Loop extract)   independent
WS4 (Test infra)     independent
WS5 (Python dedup)   independent
WS6 (Test scale)     depends lightly on WS5 (plugin protocol extension)
```

**Recommended parallel assignment**:
| Agent | Work Streams | Primary Files |
|-------|-------------|---------------|
| A | WS1 + WS2 | `src/cabi/*.{h,cpp}`, `include/trtf/backend.h` |
| B | WS3 + WS4 | `src/runtime/trt/*.{h,cpp}`, `tests/cpp/*`, `CMakeLists.txt` |
| C | WS5 + WS6 | `trtf_build/trtf_build/**/*.py`, `tests/builder/*` |

All three agents touch disjoint file sets. Only shared point: `CMakeLists.txt` (WS1 adds sources, WS4 refactors test section) — WS4 goes first.

---

## Verification

After all work streams, run the standard regression tiers:

```bash
# Tier 1: Unit tests (no GPU)
pytest tests/builder/ -v --ignore=tests/builder/test_cli.py
pytest tests/tools/ -v
ctest --test-dir build --output-on-failure

# Tier 2: Graph-op GPU tests
pytest tests/builder/test_graph_ops.py -v -m trt

# Tier 3: E2E smoke test (one model, rebuild)
pytest tests/e2e/test_full_pipeline.py -v -k qwen3-0.6b \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python --rebuild-engines

# Tier 4: Full E2E suite
pytest tests/e2e/ -v --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python --rebuild-engines
```

Key invariant: **every existing test passes with identical output**. These are pure refactorings — no behavioral changes.

## Bonus: Dead code to remove
- `select_topk_tokens()` in `trt_decode_runtime.cpp` (unused anywhere)
- `trt_graph_builder.cpp` (vestigial 2-line comment-only stub)
- Per-file `check()` functions after migration to `test_helpers.h`
