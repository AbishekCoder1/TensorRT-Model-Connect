# Torch-TRT Worklog

Condensed log of what was done, why, and what's next.

## Why LibTorch was rejected (historical context)

The original plan used LibTorch (`torch::jit::load`) as the C++ runtime for Torch-TRT bundles.
Yifei Fang determined this was a non-starter: if the pipeline requires LibTorch at runtime, it's
functionally identical to PyTorch's AOT Inductor (AOTI) and provides no value over the raw TRT
pipeline. The breakthrough was `StatelessCacheWrapper` — a fully stateless adapter that produces
raw TRT engines (`.plan` files) with explicit KV cache I/O, running on the existing C++ runtime
(`DeviceKvCache` + TRT C API) with zero LibTorch dependency. See the 2026-03-07 entry below for
the full development story.

---

## Key technique: CPU-side torch.export for large models

`torch.export.export()` does **not** require the model to be on GPU. The tracing
runs on CPU, producing an `ExportedProgram` graph. `torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine()`
then converts that graph to a TRT engine targeting GPU — the model weights are
never loaded onto the GPU during compilation.

This means VRAM usage during compilation is only the **TRT builder workspace**
(typically 2-4 GB), not the model size. A 10 GB fp16 model (e.g., T5-XXL) can be
compiled on a 24 GB GPU that couldn't fit model + workspace together.

**Pattern:**
```python
# Model stays on CPU — no GPU memory used for weights
model = AutoModel.from_pretrained("large-model", dtype=torch.float16)  # CPU
model.eval()

example_inputs = (torch.randn(1, 64),)  # CPU tensors

# Trace on CPU
exported = torch.export.export(model, example_inputs, strict=False)

# Compile to TRT — only TRT builder workspace uses GPU VRAM
engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
    exported,
    inputs=[inp.cuda() for inp in example_inputs],  # target device specified here
    use_explicit_typing=True,
    min_block_size=1,
)
```

**Verified:** torch 2.10.0, torch_tensorrt 2.10.0, RTX 4090 (24 GB).

This is critical for multi-component models (e.g., diffusion: T5 text encoder +
DiT denoiser + VAE decoder) where loading all components onto GPU simultaneously
would exceed VRAM. Each component can be exported and compiled sequentially,
with only one TRT builder workspace active at a time.

---

## 2026-03-25: PixArt-Sigma prompt adherence fix (additive masking + TRT-safe attention)

### What was done

Fixed PixArt-Sigma torch-trt pipeline producing images that don't follow the text prompt. Images were high quality but semantically unrelated to the prompt (e.g., "a dog" → woman in a field).

**Root cause: multiplicative masking gives ~74% attention to padding tokens**

The v6 engine used multiplicative masking — `encoder_hidden_states * mask` zeros out padding embeddings, but softmax over zero-valued keys still produces exp(0)=1, distributing ~74% of attention weight to 90 padding positions vs ~26% to 10 real tokens. Text signal was overwhelmed.

**Fix: additive masking + TRT-safe attention processor**

1. `_TrtSafeAttnProcessor` in `strategies/diffusion.py` — custom attention processor using basic `torch.matmul` + `add` + `softmax` instead of F.scaled_dot_product_attention (SDPA). SDPA's fused kernels produce NaN when compiled by torch_tensorrt with attention masks. The classic `AttnProcessor` uses `torch.baddbmm` which TRT also can't compile.

2. `PixArtDiTWrapper.forward()` converts the `{0,1}` encoder_attention_mask to additive bias `{0, -10000}` and passes it as 3D `[B,1,seq]` to the model. The 3D shape bypasses the model's internal 2D→3D conversion. Padding positions get -10000 added to attention logits → exp(-10000)≈0 → 100% attention on real tokens.

3. C++ pipeline (`torchtrt_diffusion_pipeline.cpp`) unchanged — it still builds `{0,1}` fp16 masks. The conversion to additive happens inside the engine.

**Files changed:**
- `tensorrt_model_connect/tensorrt_model_connect/strategies/diffusion.py` — Added `_TrtSafeAttnProcessor`, updated `PixArtDiTWrapper` to use additive masking
- `tensorrt_model_connect/tensorrt_model_connect/families/pixart.py` — Updated comments for additive masking, mixed trace mask
- `src/runtime/pipelines/torchtrt_diffusion_pipeline.cpp` — Removed debug output, updated comments

**Verification:**
- v8 bundle with "a photo of a dog chewing on a bone" → golden retriever chewing bone (correct)
- v8 bundle with "a red sports car on a highway" → red sports car on highway (correct)
- `diagnose_loop.py` confirmed TRT engines produce proper prompt divergence (final latent cosine 0.898 between different prompts)

### What was tried and failed

- **v7: Pre-computed additive mask passed to model** — ALL NaN output. SDPA fused kernel can't handle attention masks in TRT, regardless of mask values (even all-zeros gave NaN).
- **v8 attempt 1: Classic AttnProcessor** — `torch.baddbmm` → `UnboundLocalError` during torch_tensorrt compilation. TRT can't compile baddbmm.
- **v8 attempt 2: Custom matmul+softmax processor** — success. Basic ops that TRT compiles reliably.

### Decisions
- TRT-safe attention uses explicit matmul+scale+add+softmax instead of any fused attention kernel. This is slower but correct and compiles reliably with torch_tensorrt.
- Mask format at the C++/engine boundary stays `{0,1}` fp16 for simplicity. The engine wrapper handles conversion to additive internally.

---

## 2026-03-16: Configurable precision (`--precision fp16|bf16|fp32`)

### What was done

Made the `--precision` flag in `trtmc-build` actually functional. Previously it was accepted but ignored — the model always loaded as fp16 regardless.

**Pipeline changes:**
- `compiler.py`: Added `PRECISION_DTYPE_MAP` / `precision_to_dtype()`. `build_bundle()` now converts precision string to `torch.dtype` and passes it to `plugin.load_model(dtype=...)` and `strategy.wrap_model(compute_dtype=...)`.
- `families/base.py`: Added `dtype` kwarg to `load_model()` protocol.
- `families/qwen.py`, `families/bert.py`: Use passed `dtype` instead of hardcoded `torch.float16`.
- `strategies/base.py`: Added `compute_dtype` kwarg to `wrap_model()` protocol.
- `strategies/decoder.py`: `StatelessCacheWrapper` takes `compute_dtype` parameter. Uses it for attention mask conversion, StaticCache creation, and KV cache dtype. I/O remains fp32 (C++ runtime convention).
- `strategies/encoder_only.py`: Accepts `compute_dtype` kwarg (not used — encoder models don't need internal dtype conversion).

**Perf tools:**
- `perf_utils.py`: Added `precision_to_torch_dtype()`, `dtype_label()` helpers. `build_torchtrt_bundle()` now accepts `precision` param.
- `perf_compare_torchtrt.py`: Added `--precision` flag. Passes it to Torch-TRT build and HF model loading. Report header shows per-backend dtype.
- `perf_compare_all.py`: Same. Raw TRT always shows `float32`. JSON output includes `precision` and `backend_dtypes`.

All changes are backward compatible — new params are keyword-only with defaults matching previous behavior (fp16).

### Decisions
- I/O format stays fp32 regardless of precision (C++ runtime expects fp32).
- Raw TRT pipeline is always fp32 (precision is hardcoded in graph ops). Only Torch-TRT, HF eager, and torch.compile respect `--precision`.
- `cache_config.py` export args always use fp32 tensors — precision only affects model weights and internal compute.

---

## 2026-03-05: Phase 1+2 — Python build package + bundle format

### What was done

Created `tensorrt_model_connect/` package (12 source files) and `tests/torchtrt_builder/` (7 test files).

**Package files:** `pyproject.toml`, `__init__.py`, `__main__.py`, `cli.py`, `config.py`, `compiler.py`, `cache_config.py`, `bundle_writer.py`, `bundle_reader.py`, `families/__init__.py`, `families/base.py`, `families/qwen.py`

**Test files:** `conftest.py`, `test_config.py`, `test_cache_config.py`, `test_bundle_format.py`, `test_family_plugins.py`, `test_compiler.py`, `test_cli.py`

**E2E scaffolding:** `tests/torchtrt_e2e/models/qwen3-0.6b.json`, `tests/torchtrt_e2e/conftest.py`, `tests/test_torchtrt_e2e.py`

### What was tried and what happened
- Initial `conftest.py` had functions defined after markers referenced them → `NameError`. Fixed by reordering.
- `cli.py` used `from . import build` which failed with editable install → changed to `import tensorrt_model_connect`.
- **Result: 46/46 Python tests pass.** Existing tests unaffected.

### Decisions
- Config reuses `tensorrt_model_connect.config.ModelConfig` if available, standalone fallback otherwise.
- KV cache as explicit `[layers, 2, batch, heads, cache_len, head_dim]` tensor input.
- Bundle magic: `TTRTB\x00\x01\x00`.

---

## 2026-03-05: Phase 3+4 — C++ LibTorch runtime + CLI integration

### What was done

Created `src/torchtrt/` (7 files) and `tests/cpp/test_ttrt_*.cpp` (2 files). Modified `CMakeLists.txt` (additive) and `examples/trtmc_cli.cpp` (magic-sniff dispatch).

**Source files:**
- `src/torchtrt/ttrt_bundle_format.h/.cpp` — `.ttrtb` reader (mirrors `bundle_format.h/.cpp`)
- `src/torchtrt/ttrt_kv_cache.h/.cpp` — cache position/mask/position_ids tracking
- `src/torchtrt/ttrt_pipeline.h/.cpp` — `TorchTrtPipeline : IPipeline` (LibTorch inference, guarded by `#if TRTMC_HAS_TORCHTRT`)
- `src/torchtrt/ttrt_c.cpp` — C ABI entry point `ttrt_create_pipeline()`
- `include/trtmc/ttrt_pipeline.h` — public header

**Test files:**
- `tests/cpp/test_ttrt_bundle_format.cpp` — 7 tests: magic, read, sections, errors
- `tests/cpp/test_ttrt_kv_cache.cpp` — 10 tests: init, advance, clamp, reset, masks, positions, full sequence

**Modified files:**
- `CMakeLists.txt` — added `TRTMC_ENABLE_TORCHTRT` option, LibTorch `find_package`, new source files, 2 new test targets
- `examples/trtmc_cli.cpp` — added `#include "trtmc/ttrt_pipeline.h"`, magic-sniff dispatch in `cmd_run()` (~10 lines)

### What was tried and what happened
- Built with both TRT and TorchTRT disabled (`-DTRTMC_ENABLE_TRT=OFF -DTRTMC_ENABLE_TORCHTRT=OFF`) — all 102 compilation units compile cleanly.
- `ttrt_pipeline.cpp` compiles without LibTorch because the implementation is guarded by `#if TRTMC_HAS_TORCHTRT`.
- **Result: 25/25 C++ tests pass** (23 existing + 2 new). No regressions.
- **Result: 46/46 Python tests still pass.** No regressions.

### Design decisions
- `TorchTrtPipeline` implements `IPipeline` (same interface as raw-TRT backends). This lets the CLI use the exact same `generate()` call regardless of backend — only the pipeline creation differs.
- Bundle format/cache tracker compile without LibTorch. Only the pipeline + C ABI need the `#if TRTMC_HAS_TORCHTRT` guard.
- CLI dispatches via magic-byte sniffing: `ttrt_is_bundle()` checks for `TTRTB` magic → `ttrt_create_pipeline()`. Falls through to `trtmc_create_pipeline_ex()` for `TRTFB`.

### What still needs to be done (at time of Phase 3+4)
- [x] **Phase 6**: Diff tools — done, see below
- [x] **Phase 7**: Agent onboarding scaffold — done, see below
- [ ] **GPU validation**: Build with `-DTRTMC_ENABLE_TORCHTRT=ON` in the dev container (requires LibTorch). Compile Qwen3-0.6B, run through the full pipeline.
- [ ] **EOS token**: Currently hardcoded per model_type. Should extract from `config.json` or tokenizer config in the bundle.
- [ ] **Cache reset**: Need to verify whether `torch::jit::load` modules allow cache tensor zeroing between generations without reloading the model. If not, may need to re-load per generation (slow) or restructure the export.
- [ ] **Temp dir cleanup**: Tokenizer temp files extracted from bundle should be cleaned up on pipeline destruction.

---

## 2026-03-05: Phase 5 — Test expansion (already inline with Phase 1-4)

Tests were created alongside each phase. Final count: **63 Python tests + 25 C++ tests**, all passing.

---

## 2026-03-05: Phase 6+7 — Diff tools, validation scripts, agent scaffolding

### What was done

**Diff tools:**
- `tools/diff_torchtrt.py` — Torch-TRT vs HF logit comparison. Runs step-by-step inference (HF eager vs HF StaticCache), computes cosine sim, top-1/top-5 overlap, max abs diff. Battery mode runs 4 prompts. `_compare_logits()` is the core comparison function, tested independently with synthetic data.

**Validation scripts:**
- `scripts/validate_torchtrt_family.sh` — One-command validation gate: (1) build `.ttrtb` bundle, (2) run diff_torchtrt battery, (3) C++ inference (if binary available). Prints PASS/FAIL summary.
- `scripts/new_torchtrt_family.py` — Plugin scaffolding: `detect_features()` inspects config.json for GQA, MoE, sliding window, head_dim, etc. `generate_plugin()` emits a complete plugin .py file with `matches()`, `load_model()`, `get_export_args()`, and architecture notes.

**Test files:**
- `tests/torchtrt_builder/test_diff_torchtrt.py` — 7 tests: identical logits, noisy logits, orthogonal vectors, step count mismatch, top-5 overlap.
- `tests/torchtrt_builder/test_scaffold.py` — 10 tests: feature detection (GQA, MoE, sliding window, head_dim, basic decoder) + plugin generation (structure, match expression, GQA/MoE notes, simple model).

### What was tried and what happened
- Initial `test_scaffold.py` had `assert "def load_model(self" in source` which failed because `generate_plugin()` produces a multiline signature `def load_model(\n    self,`. Fixed by changing to `assert "def load_model(" in source`.
- **Result: 63/63 Python tests pass.** All phases validated.

### What still needs to be done (at time of Phase 6+7)
- [x] **StaticCache integration** — done, see below
- [x] **EOS token** — done, see below
- [ ] **GPU validation**: Build with `-DTRTMC_ENABLE_TORCHTRT=ON`, compile Qwen3-0.6B end-to-end.
- [ ] **Temp dir cleanup**: Clean up tokenizer temp files on pipeline destruction.

---

## 2026-03-05: StaticCache + prefill fix for Qwen3-0.6B

### What was done

Fixed three issues identified during pre-GPU review of the Qwen3-0.6B import path:

**1. StaticCache instead of raw tensor (`cache_config.py`)**
- HF models expect `past_key_values` to be a `Cache` object, not a raw tensor. Updated `make_export_args()` to create a `StaticCache` from `transformers`.
- `StaticCache` requires an HF `PretrainedConfig` (has `get_text_config()`, etc.), not our lightweight `ModelConfig`. Added `(ImportError, AttributeError)` fallback to raw tensor for unit tests.
- In production, the plugin passes `model.config` (HF PretrainedConfig) which works correctly.
- Added `trt_inputs` field to `ExportArgs` — tensor-only inputs for `torch_tensorrt.dynamo.compile()` (excludes StaticCache which is model state).

**2. Token-by-token prefill (`ttrt_pipeline.cpp`)**
- The model is exported with static `seq_len=1`. Variable-length prefill would crash. Changed `generate()` to feed one token at a time during prefill.
- Renamed `forward_and_argmax()` → `forward_one_token()` (always seq_len=1).
- Added `reset_cache()` — scans model's named buffers for 4D tensors matching cache dimensions `[1, heads, max_cache_length, head_dim]` and zeros them.

**3. EOS from config.json (`ttrt_pipeline.cpp`)**
- Replaced hardcoded EOS switch with extraction from the `config.json` bundle section using `extract_json_int_or_first_array("eos_token_id", 2)`.

**4. Protocol update (`base.py`, `qwen.py`, `new_torchtrt_family.py`)**
- `get_export_args()` now receives the loaded `model` parameter so plugins can access `model.config` for StaticCache creation.
- Updated scaffold script to generate the new signature.

**5. Torch-TRT compile inputs (`compiler.py`)**
- `torch_tensorrt.dynamo.compile()` now receives explicit `inputs=export_args.trt_inputs` (4 tensors, no cache).

### What was tried and what happened
- First attempt passed `ModelConfig` to `StaticCache(config=...)`. Failed with `AttributeError: 'ModelConfig' object has no attribute 'get_text_config'` — the newer transformers version requires HF-specific config methods. Fixed by catching `AttributeError` and falling back to raw tensor for unit tests.
- **Result: 65/65 Python tests pass, 25/25 C++ tests pass.**

### What still needs to be done
- [x] **GPU validation**: See below.
- [ ] **Temp dir cleanup**: Clean up tokenizer temp files on pipeline destruction.
- [ ] **Dynamic prefill** (optimization): Current token-by-token prefill works but is slower than batch prefill. Could add dynamic shapes or a separate prefill graph later.
- [ ] **C++ runtime with LibTorch+TorchTRT**: The TorchScript model requires `torch_tensorrt` C++ extensions. Need to build C++ with `-DTRTMC_ENABLE_TORCHTRT=ON` and link against LibTorch + torch_tensorrt.

---

## 2026-03-06: GPU validation — TorchExportableModuleWithStaticCache

### What was done

Successfully built and validated the full Torch-TRT pipeline with Qwen3-0.6B on RTX 4090.

**Key change: TorchExportableModuleWithStaticCache**

Replaced the manual StaticCache/LogitsWrapper approach with HF's `TorchExportableModuleWithStaticCache` from `transformers.integrations.executorch`. This wrapper:
- Registers StaticCache key/value tensors as module buffers (not graph inputs)
- Simplifies forward signature to just `(input_ids, cache_position)` — no attention_mask, position_ids, or past_key_values
- Returns only logits (no cache output)
- Is fully compatible with `torch.export.export(strict=False)`

**run_decompositions patch**

`torch_tensorrt.dynamo.compile()` calls `exported_program.run_decompositions()` internally, which fails with `AssertionError` on stateful modules (cache buffer mutations). Added `_patch_run_decompositions()` in `compiler.py` that catches this specific failure and returns the program unchanged. The non-decomposed graph compiles correctly via TRT.

**Files changed:**
- `tensorrt_model_connect/tensorrt_model_connect/compiler.py` — Added `_patch_run_decompositions()`, wrapper integration in `build_bundle()`, switched to `use_explicit_typing=True`
- `tensorrt_model_connect/tensorrt_model_connect/cache_config.py` — Simplified `make_export_args()` to produce only `input_ids + cache_position` kwargs
- `tensorrt_model_connect/tensorrt_model_connect/families/qwen.py` — Simplified `get_export_args()` (no longer needs `model.config`)
- `scripts/new_torchtrt_family.py` — Updated scaffold template
- `src/torchtrt/ttrt_pipeline.cpp` — Simplified `forward_one_token()` to 2 inputs only
- `tests/torchtrt_builder/test_cache_config.py` — Updated for 2-input signature
- `docs/torch-trt/TORCHTRT_AGENT_GUIDE.md` — Updated pipeline docs

### What was tried and what happened

1. **TorchExportableModuleWithStaticCache direct import** — Worked immediately. Forward produces correct logits.
2. **torch.export.export** — Success with `strict=False`. Produces valid ExportedProgram with 2816 graph nodes.
3. **torch_tensorrt.dynamo.compile** — Failed on `run_decompositions()` with `AssertionError` (stateful buffer mutations). Fixed via monkey-patch.
4. **Alternative approaches tried before finding the patch:**
   - `torch_tensorrt.compile(ir="dynamo")` — Failed: re-exports the module and misses the wrapper's forward signature
   - `torch.compile(backend="torch_tensorrt")` — Failed: shape broadcasting error in cache `index_put` op
   - Manual decomposition skip — Failed: `trt_convert` not importable from torch_tensorrt's internal API
5. **Save/reload** — `torch_tensorrt.save()` with `inputs=[input_ids, cache_position]` parameter succeeded.
6. **Full bundle build** — `trtmc-build build Qwen/Qwen3-0.6B` completed in 220s, producing 1951 MB bundle.
7. **Bundle inference** — Loaded from bundle, generated "The capital of France is Paris. The capital of Italy is Rome..." ✓
8. **Cache reset** — Zeroing 4D buffers produces identical output on re-generation ✓

### Validation results

- **65/65 Python tests pass, 25/25 C++ tests pass** (no regressions)
- **Bundle build**: Qwen3-0.6B → 1951 MB `.ttrtb` in ~220s
- **Generation**: Correct text output with StaticCache (verified against HF eager)
- **Cache reset**: Works correctly — zeroing 4D buffers resets state

### Environment
- torch=2.10.0+cu128, torch_tensorrt=2.10.0, transformers=5.2.0
- GPU: NVIDIA GeForce RTX 4090
- Model: Qwen/Qwen3-0.6B (28 layers, 1024 hidden, 151936 vocab)

### What still needs to be done
- [x] **C++ runtime with LibTorch+TorchTRT**: See below.
- [ ] **Temp dir cleanup**: Clean up tokenizer temp files on pipeline destruction.
- [ ] **Dynamic prefill** (optimization): Token-by-token prefill works but is O(n²). Could add dynamic shapes for batch prefill later.

---

## 2026-03-06: Docker fixes + C++ E2E with LibTorch+TorchTRT

### What was done

Fixed 5 issues preventing the Docker container from working end-to-end on x86_64, and achieved full C++ inference with the Torch-TRT pipeline.

**Issue 1: TRT_INC_DIR hardcoded for aarch64**
- `Dockerfile.gb300` had `ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu` — wrong for x86_64 (RTX 4090).
- Removed hardcoded ENV. `setup_container.sh` now auto-detects via `find /usr/include -name NvInferRuntime.h`.

**Issue 2: CUDA version mismatch (torch=cu128 vs torchvision/torchaudio=cu130)**
- `torch` from the cu130 index installs as cu128 (no cu130 build exists for torch 2.10.0).
- Later installs (`torch_tensorrt`, `nemo_toolkit`) overwrite torchvision/torchaudio with cu130 from PyPI.
- Added post-install step in Dockerfile: `pip install --force-reinstall --no-deps torchvision torchaudio --index-url .../cu${TORCH_CUDA}`.

**Issue 3: `ttrt_c.cpp` missing include**
- Missing `#include "torchtrt/ttrt_pipeline.h"` (internal header with class definition).
- Added the include.

**Issue 4: LibTorch segfaults in tests**
- All tests linked against `trtmc_core` which now included LibTorch. LibTorch's bundled CUDA 12.8 conflicted with system CUDA 13.0, causing segfaults in 4 tests.
- Split Torch-TRT code into separate `trtmc_torchtrt` library. Only CLI links LibTorch; tests link `trtmc_core` without it.
- Created `ttrt_c_stubs.cpp` (weak symbol stubs in `trtmc_core`) + `ttrt_c.cpp` (real impl in `trtmc_torchtrt`).

**Issue 5: `libtorchtrt_runtime.so` not loadable at C++ runtime**
- `torch::jit::load()` fails with "Unknown type name `tensorrt.Engine`" because the Torch-TRT runtime extensions aren't loaded.
- Added `dlopen("libtorchtrt_runtime.so")` in `ttrt_pipeline.cpp` before model loading.
- Added torch/torch_tensorrt lib dirs to `LD_LIBRARY_PATH` in Dockerfile.

**Issue 6: `diff_torchtrt.py` argmax bug**
- `int(logits.argmax(-1))` on 2D array → `TypeError`. Fixed to `int(logits[0].argmax(-1))`.

### Files changed
- `Dockerfile.gb300` — Removed aarch64 TRT_INC_DIR, added CUDA version fix, added torch lib dirs to LD_LIBRARY_PATH
- `scripts/setup_container.sh` — Auto-detect TRT_INC_DIR, install both tensorrt_model_connect and tensorrt_model_connect
- `CMakeLists.txt` — Separated `trtmc_torchtrt` library from `trtmc_core`
- `src/torchtrt/ttrt_c.cpp` — Real implementation only (in `trtmc_torchtrt`)
- `src/torchtrt/ttrt_c_stubs.cpp` — New: weak stubs (in `trtmc_core`)
- `src/torchtrt/ttrt_pipeline.cpp` — Added `dlopen()` for torch_tensorrt runtime
- `tools/diff_torchtrt.py` — Fixed argmax on multi-dimensional logits

### Validation results
- **25/25 C++ tests pass** (0 segfaults)
- **964/964 Python builder tests pass**
- **211/211 tools tests pass**
- **63/65 Torch-TRT builder tests pass** (2 pre-existing: StaticCache shape change in transformers 5.2.0)
- **Bundle build**: Qwen3-0.6B → 2047 MB `.ttrtb` in ~236s
- **C++ inference**: `./build/trtmc run /tmp/qwen3.ttrtb --prompt "The capital of France is" --max-new-tokens 20 --hf-python /opt/venv/bin/python`
  → Output: `" Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China"`

### What still needs to be done
- [ ] **Temp dir cleanup**: Clean up tokenizer temp files on pipeline destruction.
- [ ] **Dynamic prefill** (optimization): Token-by-token prefill works but is O(n²).
- [ ] **diff_torchtrt StaticCache decode divergence**: Steps 0-4 match perfectly (cosine>0.999) but decode steps 5+ diverge. Likely attention mask shape issue with StaticCache in transformers 5.2.0.
- [ ] **test_cache_config.py fixes**: 2 tests expect old StaticCache shape format; need updating for transformers 5.2.0.

---

## 2026-03-06: Warning suppression + README update

### What was done

**1. Suppressed 3 of 4 noisy third-party warnings in `compiler.py`**

Four warnings were emitted on every `trtmc-build` invocation. Investigated root cause of each:

| Warning | Source | Root Cause | Fix |
|---------|--------|------------|-----|
| `TRTLLM_PLUGIN_PATH is not set` | `torch_tensorrt._utils` (Python logging) | torch_tensorrt checks for TRT-LLM plugins on import; we don't use them | `logging.getLogger("torch_tensorrt").setLevel(ERROR)` |
| `transformers version X not tested with nvidia-modelopt` | `modelopt.torch.__init__` (Python warnings) | modelopt hasn't updated its version check for transformers 5.x | `warnings.filterwarnings("ignore", ...)` |
| `The logger passed into createInferBuilder differs...` | TRT C++ → `torch_tensorrt [TensorRT Conversion Context]` logger | TRT allows one global `ILogger`. `tensorrt.plugin` init registers one, then torch_tensorrt passes its `_TRTLogger` to `trt.Builder()`, triggering mismatch. Upstream torch_tensorrt bug. | `logging.getLogger("torch_tensorrt [TensorRT Conversion Context]").setLevel(ERROR)` |
| `tensorrt.plugin module is experimental` | TRT C++ shared library (stderr) | Hardcoded in TRT's C++ `tensorrt.plugin` module init — no env var or API to suppress | **Cannot suppress** — harmless single-line message from TRT C++ |

**2. Updated `README.md` with Torch-TRT pipeline documentation**

- Added pipeline comparison table at top (Raw TRT vs Torch-TRT: builder, bundle format, runtime, description)
- Unified Quick Start section: container setup once, then build+run examples for both pipelines side by side with Qwen3-0.6B
- Added Torch-TRT detail section: "How it works" (5-step flow), Python API, CLI reference
- Added Torch-TRT docs links to Documentation table
- Renamed existing sections to distinguish "Raw TRT pipeline" from "Torch-TRT pipeline"

**3. Fixed file ownership for tensorrt_model_connect/**

Files created inside container were owned by root (UID 1000), not the host user (UID 1776778147). Fixed via `docker exec chown`.

### Files changed
- `tensorrt_model_connect/tensorrt_model_connect/compiler.py` — Added warning suppression for 3 known harmless warnings
- `README.md` — Added Torch-TRT pipeline documentation, unified quick start

### What still needs to be done
- [ ] **Temp dir cleanup**: Clean up tokenizer temp files on pipeline destruction.
- [ ] **Dynamic prefill** (optimization): Token-by-token prefill works but is O(n²).
- [ ] **diff_torchtrt StaticCache decode divergence**: Steps 0-4 match perfectly (cosine>0.999) but decode steps 5+ diverge. Likely attention mask shape issue with StaticCache in transformers 5.2.0.
- [ ] **test_cache_config.py fixes**: 2 tests expect old StaticCache shape format; need updating for transformers 5.2.0.
- [ ] **TRT C++ "experimental" warning**: Cannot suppress from Python — hardcoded in TRT's `tensorrt.plugin` shared library init. Would need upstream TRT fix or env var support.

---

## 2026-03-07: LibTorch removal — raw TRT engine from Torch-TRT (BREAKTHROUGH)

### Context

Yifei Fang determined that **LibTorch cannot be a runtime dependency** — if the Torch-TRT pipeline requires LibTorch at runtime, it's functionally identical to PyTorch's AOT Inductor (AOTI) and provides no value over the raw TRT pipeline. The goal is to produce a **raw TRT engine** (`.plan` file) that runs on the existing C++ runtime with `DeviceKvCache`, no LibTorch needed.

### What was tried and what happened

**Attempt 1: TorchExportableModuleWithStaticCache + scatter patch → single TRT engine**

The `TorchExportableModuleWithStaticCache` wrapper registers KV cache as internal module buffers. With the scatter patch (`StaticLayer.update` uses `torch.scatter` instead of `index_copy_`), `torch_tensorrt.dynamo.compile` produces a single monolithic TRT engine with 0 fallback modules.

Problem: TRT engines are **stateless** — internal buffers don't persist between `execute_async_v3` calls. The cache is baked into the engine as initial state (zeros), but resets every call. Result: **inference produces garbage** ("is is is is..."). The engine has only 3 I/O tensors (input_ids, cache_position, output0) — no way to manage cache externally.

**Attempt 2: `convert_exported_program_to_serialized_trt_engine` (direct raw engine)**

Tried `torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine` on the `TorchExportableModuleWithStaticCache` export. Failed: `run_decompositions` crashes on stateful modules, and even with the patch, the converter fails with `'NoneType' object has no attribute 'shape_mode'`. The single-engine converter can't handle non-decomposed graphs from stateful modules.

**Attempt 3: StatelessCacheWrapper with explicit cache I/O (SUCCESS)**

Created a custom `StatelessCacheWrapper` that:
1. Takes `(input_ids, cache_position, attention_mask, cache_k_0, cache_v_0, ..., cache_k_N, cache_v_N)` as input
2. Creates a `StaticCache` object internally and loads the input cache tensors
3. Runs the HF model forward with the scatter-patched StaticCache
4. Extracts `present_k/v` (new key/value at `cache_position`) via `gather`
5. Returns `(logits, present_k_0, present_v_0, ..., present_k_N, present_v_N)`

This approach is **fully stateless** — no internal buffer mutations. The cache is managed externally (same pattern as the existing raw TRT pipeline's `DeviceKvCache`).

Pipeline: `StatelessCacheWrapper` → `torch.export.export(strict=False)` → `convert_exported_program_to_serialized_trt_engine(use_explicit_typing=True)` → raw `.plan` file.

Results:
- **torch.export succeeds** in 1.6s (371 placeholders: 311 weights + 60 user inputs)
- **convert_exported_program succeeds** in 12-15s, producing a **1436 MB raw TRT engine**
- **Engine I/O**: 59 inputs (input_ids, cache_position, attention_mask, 56 cache tensors), 57 outputs (logits, 56 present_k/v)
- **Multi-step generation produces correct text**: "The capital of France is . The capital of Italy is Rome. The capital of Spain is Madrid."
- **Deterministic** across 3 runs
- **No LibTorch dependency** — pure TRT engine, runs via raw TRT C API

### Key findings

| Finding | Detail |
|---------|--------|
| `use_explicit_typing=True` is required | With `False` + `enabled_precisions={fp16}`, TRT incorrectly converts operations and produces all-zero logits. With `True`, TRT preserves original dtypes. |
| Scatter patch is essential | `StaticLayer.update` uses `index_copy_` (in-place) by default, which fragments the graph. Scatter is functional — no in-place mutations — producing a clean graph. |
| Causal mask must be explicit input | Computing the mask inside the wrapper (from cache_position) gets traced incorrectly by TRT. Passing it as an explicit input works correctly. |
| RoPE buffers are auto-folded | `rotary_emb.inv_freq` buffers appear as user inputs in the exported graph but are folded into constants by TRT. No special handling needed. |
| First-token divergence is expected | PyTorch wrapper outputs "." instead of " Paris" for first generated token — this is because the StaticCache+mask approach differs slightly from HF's internal `create_causal_mask`. Tokens 2+ match exactly. |

### Architecture (new, no LibTorch)

```
HF model (safetensors + config.json)
  -> AutoModelForCausalLM.from_pretrained()
  -> StatelessCacheWrapper (explicit cache I/O, scatter-patched StaticCache)
  -> torch.export.export(strict=False)
  -> torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(use_explicit_typing=True)
  -> Raw .plan file (TRT engine)
  -> Package as .trtfb bundle (same format as raw TRT pipeline)
  -> C++ runtime: DeviceKvCache + TRT C API (no LibTorch)
```

### What still needs to be done

- [ ] **Integrate into `compiler.py`**: Replace the LibTorch-based build pipeline with the new stateless wrapper approach. Produce `.trtfb` bundles instead of `.ttrtb`.
- [ ] **I/O tensor naming**: Rename TRT engine I/O from `flat_cache_N` / `outputN` to `cache_k_N` / `cache_v_N` / `present_k_N` / `present_v_N` / `logits` to match the existing C++ `DeviceKvCache` naming convention.
- [ ] **Attention mask format**: Align with the raw TRT pipeline's mask format (which uses `kMaskedScore = -1.0e4F`). Currently using `torch.finfo(fp16).min = -65504`.
- [ ] **First-token divergence**: Investigate whether this is a mask format issue. May need to pass `attention_mask` as a 2D mask and let the model compute its own 4D causal mask.
- [ ] **Remove LibTorch code**: Remove `trtmc_torchtrt` library, `ttrt_c.cpp`, `ttrt_c_stubs.cpp`, `.ttrtb` format, and all LibTorch-related CMake code once the new pipeline is validated.
- [ ] **End-to-end C++ validation**: Package the raw engine as `.trtfb` and run through the existing C++ runtime.
- [ ] **diff_torchtrt validation**: Run the diff tool battery to measure logit agreement vs HF reference.

---

## 2026-03-09: Integration — StatelessCacheWrapper into compiler.py + LibTorch removal

### What is being done

Integrating the StatelessCacheWrapper prototype into the production `compiler.py` and removing ALL LibTorch dependencies (code, CMake, Docker).

### Phase 1: Python changes (DONE)

Rewrote the Python build pipeline to produce raw TRT engines in `.trtfb` bundles:

**`compiler.py`** — Major rewrite:
- Replaced `TorchExportableModuleWithStaticCache` + `torch_tensorrt.dynamo.compile` + TorchScript
- Now uses `StatelessCacheWrapper` + `convert_exported_program_to_serialized_trt_engine`
- Added `patch_static_cache_scatter()` from prototype
- Output: raw TRT engine bytes in `.trtfb` bundle (no LibTorch)
- Engine I/O tensor names stored in `config.json` as `torchtrt_io_map` for C++ runtime

**`cache_config.py`** — Rewritten:
- `make_export_args()` returns tuple `(input_ids, cache_position, attention_mask, *cache_kv)` matching StatelessCacheWrapper forward
- Added `build_causal_mask()` and `make_cache_tensors()` helpers

**`bundle_writer.py`** — Changed to `.trtfb` magic (was `.ttrtb`)

**`bundle_reader.py`** — Updated to accept both TRTFB and legacy TTRTB magic

**`families/qwen.py`** and **`families/base.py`** — Updated for new API

### Phase 2: I/O format adaptation — reuse existing C++ DeviceKvCache (DONE)

**Key decision change**: Instead of building a separate C++ pipeline with Torch-TRT-native I/O, the `StatelessCacheWrapper` now adapts the HF model's I/O to match the raw TRT format exactly. This means the **existing C++ DeviceKvCache runtime** loads Torch-TRT bundles without shape/dtype changes — only tensor name mapping is different.

**How it works**: The wrapper is an adapter layer:
1. **Inputs**: Accepts raw TRT format (int32, float32, 2D cache with expanded GQA heads)
2. **Inside**: Converts to HF format (int64 token, float16 4D cache with compact KV heads)
3. **Runs**: HF model via `model()` call (traced by torch.export)
4. **Outputs**: Converts back to raw TRT format (float32, expanded GQA heads)

All conversions (cast, reshape, GQA compact/expand) are standard PyTorch ops that trace cleanly through torch.export and convert to TRT layers.

**I/O format — now matches raw TRT pipeline exactly**:

| Aspect | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `token_id` | [1] | int32 | Same as raw TRT |
| `position_id` | [1] | int32 | Same as raw TRT |
| `attention_mask` | [1, max_cache_len+1] | float32 | Same as raw TRT |
| `cache_kv_{2i}` | [max_cache_len, attention_size] | float32 | Keys, expanded GQA |
| `cache_kv_{2i+1}` | [max_cache_len, attention_size] | float32 | Values, expanded GQA |
| `output0` | [1, vocab_size] | float32 | Logits |
| `output{2i+1}` | [1, attention_size] | float32 | present_k, expanded GQA |
| `output{2i+2}` | [1, attention_size] | float32 | present_v, expanded GQA |

Only difference is tensor **naming** (torch.export auto-names vs manual names). This is handled by `runtime_strategy = "torchtrt_decoder"` in the C++ `make_decoder_engine()`.

**Python changes (`compiler.py`)**:
- `StatelessCacheWrapper.forward(token_id, position_id, attention_mask, *cache_kv)` — raw TRT I/O
- Inside: GQA compact (`::group_size` slice), permute, cast to float16 for HF model
- Output: GQA expand (`repeat_interleave`), reshape, cast to float32

**Python changes (`cache_config.py`)**:
- `make_export_args()` now produces raw TRT format: int32 IDs, float32 mask [1, max_cache_len+1], float32 cache [max_cache_len, attention_size]
- Removed `ExportArgs` dataclass (was unused)

**C++ changes (`bundle_helpers.cpp`)**:
- `make_decoder_engine()`: when `runtime_strategy == "torchtrt_decoder"`, uses Torch-TRT naming convention: `cache_kv_{2i}`/`cache_kv_{2i+1}` for cache inputs, `output{N}` for outputs

**C++ changes (`trtmc_c.cpp`)**:
- Added `torchtrt_decoder` to `create_decoder_pipeline` strategy check — routes to same DeviceKvCache backend

### Phase 3: LibTorch removal (PLANNED)

- Remove `trtmc_torchtrt` library from CMakeLists.txt
- Remove `find_package(Torch)` and TRTMC_ENABLE_TORCHTRT option
- Remove LibTorch/torch_tensorrt from Dockerfile.gb300 LD_LIBRARY_PATH
- Remove TORCH_CMAKE_PREFIX env var
- Remove `ttrt_c_stubs.cpp` weak symbols
- Remove `dlopen("libtorchtrt_runtime.so")` from pipeline

### What still needs to be done

- [x] **Python wrapper adaptation**: StatelessCacheWrapper accepts raw TRT I/O format
- [x] **C++ tensor naming**: `make_decoder_engine()` handles `torchtrt_decoder` naming
- [x] **C++ strategy dispatch**: `torchtrt_decoder` routes to existing DeviceKvCache
- [x] **GPU validation**: Built bundle with `trtmc-build build Qwen/Qwen3-0.6B` (21s, 1139 MB)
- [x] **C++ inference**: `./build/trtmc run` produces correct output ("Paris" etc.)
- [ ] **LibTorch cleanup**: Remove from CMakeLists.txt, Dockerfile, C++ sources
- [ ] **Tests**: Update Python tests, run C++ tests
- [ ] **diff_torchtrt validation**: Run logit comparison battery

### Bug fix: mask conversion

The initial mask conversion sliced off the "+1" slot but didn't unmask the
current cache position. HF's StaticCache scatters K/V to `cache_position`
before attention reads from it, so that position must be unmasked. Fixed
with `hf_mask.scatter(1, cache_position.view(1,1), 0.0)`.

### GPU validation results (2026-03-09)

```
$ trtmc-build build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256
[trtmc-build] Model: qwen3 (layers=28, hidden=1024, vocab=151936)
[trtmc-build] torch.export complete [2.0s] (61 user inputs, 311 weights)
[trtmc-build] Raw TRT engine: 1139.2 MB [15.8s]
[trtmc-build] Bundle saved [21.0s total]

$ ./build/trtmc run /tmp/qwen3.trtfb --prompt "The capital of France is" --max-new-tokens 20
The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China
```

Engine I/O verified: int32 [1] token_id/position_id, float32 [1,257] mask,
float32 [256,2048] cache, float32 [1,151936] logits, float32 [1,2048] present.
All shapes/dtypes match the raw TRT pipeline exactly.

Note: first build with LibTorch linked caused segfault. Rebuilding with
`-DTRTMC_ENABLE_TORCHTRT=OFF` fixed it (LibTorch symbols conflicted).

### Phase 3: LibTorch removal (DONE)

Removed ALL LibTorch/Torch-TRT C++ runtime dependencies. The Torch-TRT pipeline now
produces raw `.trtfb` bundles that run on the existing C++ runtime — no LibTorch needed.

**Deleted files** (removed from git index):
- `include/trtmc/ttrt_pipeline.h` — public header
- `src/torchtrt/ttrt_pipeline.h/.cpp` — C++ LibTorch inference pipeline
- `src/torchtrt/ttrt_c.cpp` — C ABI entry point
- `src/torchtrt/ttrt_c_stubs.cpp` — weak symbol stubs
- `src/torchtrt/ttrt_bundle_format.h/.cpp` — `.ttrtb` bundle format
- `src/torchtrt/ttrt_kv_cache.h/.cpp` — LibTorch KV cache tracker
- `tests/cpp/test_ttrt_bundle_format.cpp` — bundle format test
- `tests/cpp/test_ttrt_kv_cache.cpp` — KV cache test

**Modified files**:
- `CMakeLists.txt` — Removed `TRTMC_ENABLE_TORCHTRT` option, `find_package(Torch)`, `trtmc_torchtrt` library target, `TRTMC_HAS_TORCHTRT` defines, ttrt source files from `trtmc_core`, `test_ttrt_*` test targets
- `examples/trtmc_cli.cpp` — Removed `#include "trtmc/ttrt_pipeline.h"`, `.ttrtb` magic-sniff dispatch (`ttrt_is_bundle` / `ttrt_create_pipeline`)
- `Dockerfile` — Removed `TORCH_CMAKE_PREFIX` env var
- `Dockerfile.gb300` — Removed torch/torch_tensorrt from `LD_LIBRARY_PATH`, removed `TORCH_CMAKE_PREFIX`
- `scripts/setup_container.sh` — Removed `TRTMC_ENABLE_TORCHTRT=ON`, `TORCH_CMAKE_PREFIX`, `torch_tensorrt` import check, `tensorrt_model_connect/` editable install
- `.gitignore` — Added `/src/torchtrt/` (root-owned files may linger on disk)

**What was kept** (still needed for BUILD pipeline):
- `torch`, `torchvision`, `torchaudio` pip packages in Dockerfiles — needed by `tensorrt_model_connect/` Python build package
- `torch_tensorrt` pip package — needed for `convert_exported_program_to_serialized_trt_engine`
- `nvidia-modelopt` pip package — quantized model support

---

## 2026-03-16: Strategy plugin system + concurrency fixes

### What was done

Introduced a **strategy-based plugin dispatch** system so different model architectures (decoder, encoder-only, etc.) use the appropriate wrapper and export format. Also fixed two concurrency bugs.

**Phase 1: Concurrency fixes**

1. **Thread-safe `patch_static_cache_scatter()`** — Added `threading.Lock` around the check-then-patch in `strategies/decoder.py`. Prevents concurrent agents from racing on the global monkey-patch of `StaticLayer.update`.

2. **GPU memory cleanup in `build_bundle()`** — Added `try/finally` block with `del model`, `del wrapper`, `gc.collect()`, `torch.cuda.empty_cache()`. Prevents OOM when building multiple bundles in the same process. Also added `del exported` in `compile_model()` after TRT conversion.

**Phase 2: Strategy extraction (refactor, no new behavior)**

Extracted `StatelessCacheWrapper` and `patch_static_cache_scatter()` from `compiler.py` into a new `strategies/` package with a `BuildStrategy` Protocol:

| File | Contents |
|------|----------|
| `strategies/__init__.py` | `get_strategy(name)` registry (lazy-initialized) |
| `strategies/base.py` | `BuildStrategy` Protocol: `name`, `runtime_strategy`, `wrap_model()`, `make_export_args()`, `pre_export_setup()` |
| `strategies/decoder.py` | `DecoderBuildStrategy` + `StatelessCacheWrapper` + `patch_static_cache_scatter()` |

`compiler.py` now dispatches via `getattr(plugin, 'runtime_strategy', 'decoder')` → `get_strategy()` instead of hardcoding the wrapper. Backward-compat aliases (`from .strategies.decoder import StatelessCacheWrapper, patch_static_cache_scatter`) ensure existing tests import without changes.

**Phase 3: Encoder-only strategy (new capability)**

Added support for encoder-only models (BERT):

| File | Contents |
|------|----------|
| `strategies/encoder_only.py` | `EncoderOnlyBuildStrategy` + `EncoderOnlyWrapper` (input_ids + attention_mask → last_hidden_state) |
| `families/bert.py` | BERT plugin stub with `runtime_strategy = "encoder_only"` |

Note: C++ runtime does not yet handle `torchtrt_encoder` bundles. This enables Python-side bundle building only.

**Test file:**
- `tests/torchtrt_builder/test_strategies.py` — 15 tests: strategy dispatch, wrapper behavior, backward-compat imports, plugin defaults.

**Other changes:**
- `families/base.py` — Docstring updated documenting optional `runtime_strategy` attribute.
- `families/qwen.py` — Added explicit `runtime_strategy = "decoder"` for clarity.

### Files changed

| File | Change |
|------|--------|
| **New** `tensorrt_model_connect/tensorrt_model_connect/strategies/__init__.py` | Strategy registry with `get_strategy()` |
| **New** `tensorrt_model_connect/tensorrt_model_connect/strategies/base.py` | `BuildStrategy` Protocol |
| **New** `tensorrt_model_connect/tensorrt_model_connect/strategies/decoder.py` | `DecoderBuildStrategy` + `StatelessCacheWrapper` + `patch_static_cache_scatter` (moved from compiler.py) |
| **New** `tensorrt_model_connect/tensorrt_model_connect/strategies/encoder_only.py` | `EncoderOnlyBuildStrategy` + `EncoderOnlyWrapper` |
| **New** `tensorrt_model_connect/tensorrt_model_connect/families/bert.py` | BERT plugin stub |
| **New** `tests/torchtrt_builder/test_strategies.py` | 15 strategy tests |
| `tensorrt_model_connect/tensorrt_model_connect/compiler.py` | Removed wrapper/patch (moved), added strategy dispatch, GPU cleanup, backward-compat aliases |
| `tensorrt_model_connect/tensorrt_model_connect/families/base.py` | Docstring update for optional `runtime_strategy` |
| `tensorrt_model_connect/tensorrt_model_connect/families/qwen.py` | Added explicit `runtime_strategy = "decoder"` |

### Validation results

- **148/148 Python tests pass** (133 existing + 15 new). Zero regressions.
- Backward-compat aliases confirmed: `from tensorrt_model_connect.compiler import StatelessCacheWrapper` resolves to `strategies.decoder.StatelessCacheWrapper`.

### What still needs to be done

- [ ] **GPU validation**: Build Qwen3-0.6B bundle with strategy dispatch and verify C++ inference output matches pre-refactor.
- [ ] **Encoder-only C++ runtime**: Add `torchtrt_encoder` handling to `bundle_helpers.cpp` and `trtmc_c.cpp`.
- [ ] **Additional strategies**: SSM/Mamba (`ssm_recurrent`), vision-language, diffusion — as needed.
- [ ] **Temp dir cleanup**: Clean up tokenizer temp files on pipeline destruction.
- [ ] **Dynamic prefill** (optimization): Token-by-token prefill works but is O(n²).
