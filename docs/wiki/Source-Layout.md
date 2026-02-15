# Source Layout

File-by-file guide to the codebase. The system is split: Python (`trtf_build/`) handles engine building, C++ handles runtime inference.

## Python: Build Package (`trtf_build/`)

### Core

| File/Dir | Purpose |
|----------|---------|
| `trtf_build/__main__.py` | CLI entry point: `trtf-build build|inspect|version` |
| `trtf_build/builder.py` | Main build pipeline: load config, match family, build engine, create bundle |
| `trtf_build/bundle.py` | `.trtfb` bundle writing: package engine plan + tokenizer files |
| `trtf_build/config.py` | `config.json` parsing: extract architecture params (family, dims, heads, etc.) |
| `trtf_build/graph_ops.py` | Shared TRT graph ops (Python): RMSNorm, matmul, RoPE, SwiGLU, attention |
| `trtf_build/registry.py` | Family plugin registry: match `model_type` to family plugin |

### Family Plugins (`trtf_build/families/`)

Each family provides a checkpoint mapper (HF tensor keys to canonical) and optionally a custom graph builder.

#### Qwen (`trtf_build/families/qwen/`)

| File | Purpose |
|------|---------|
| `plugin.py` | Family registration: matches `model_type` starting with `qwen`/`qwq`. Handles Qwen, Qwen2, Qwen3, QWQ. |
| `checkpoint.py` | Checkpoint mapper: standard HF tensor naming, handles q_norm/k_norm for Qwen3. |

#### LLaMA (`trtf_build/families/llama/`)

| File | Purpose |
|------|---------|
| `plugin.py` | Family registration: matches `model_type` `llama`. |
| `checkpoint.py` | Checkpoint mapper: standard HF tensor naming, no q_norm/k_norm. |

#### Mistral (`trtf_build/families/mistral/`)

| File | Purpose |
|------|---------|
| `plugin.py` | Family registration: matches `model_type` `mistral`. |
| `checkpoint.py` | Checkpoint mapper: standard HF tensor naming. |

#### Gemma (`trtf_build/families/gemma/`)

| File | Purpose |
|------|---------|
| `plugin.py` | Family registration: matches `model_type` `gemma`. |
| `checkpoint.py` | Checkpoint mapper: adds +1.0 to RMSNorm gamma, scales embedding by `sqrt(hidden_size)`. |

---

## C++: Runtime (~13 source files)

### Public API (`include/trtf/`)

| File | Purpose |
|------|---------|
| `pipeline.h` | **C ABI entry point**: `IPipeline` virtual interface, `TrtfPipelineOptions`, `trtf_create_pipeline()`, `trtf_create_pipeline_ex()`, `trtf_last_error()`, `trtf_version()`, `trtf_has_trt()`. Accepts `.trtfb` bundle paths. |
| `bundle.h` | Bundle API: `InspectBundle()`, `IsBundle()`, `BundleInfo`. |
| `backend.h` | `IGenerationBackend` interface, `CreateHfPythonBackend()` |
| `tokenizer.h` | `ITokenizer` interface, `CreateVocabTokenizer()`, `CreateHfPythonTokenizer()` |
| `generation.h` | `GenerationConfig`, `GenerationResult` |

### Shared Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `data_dir.h/cpp` | Centralized source-dir resolution: `source_dir()`, `scripts_dir()`, `models_dir()`. Supports `TRTF_DATA_DIR` env override. |
| `text_parsers.h/cpp` | `starts_with()`, `to_lower_ascii()`, `read_file()`, etc. |
| `json_helpers.h/cpp` | `extract_json_string()`, `extract_json_int()`, etc. |

### Bundle Format (`src/bundle/`)

| File | Purpose |
|------|---------|
| `bundle_format.h/cpp` | `.trtfb` binary format: `WriteBundleFile()`, `ReadBundleFile()`, `HasBundleMagic()`, JSON header serialization |
| `bundle_api.cpp` | Public API implementation: `InspectBundle()`, `IsBundle()`. Loading a bundle deserializes the engine plan and extracts tokenizer files. |

### C ABI Layer (`src/cabi/`)

| File | Purpose |
|------|---------|
| `trtf_c.cpp` | `extern "C"` factory: `trtf_create_pipeline()`, `trtf_create_pipeline_ex()`, `trtf_last_error()`, `trtf_version()`, `trtf_has_trt()`. Contains `PipelineImpl`. Detects `.trtfb` bundles and loads them directly. |

### Runtime / Backends (`src/runtime/`)

| File | Purpose |
|------|---------|
| `hf_python_backend.cpp` | `HfPythonBackend`: spawns Python subprocess with HF transformers (fallback) |

### TRT Infrastructure (`src/runtime/trt/`)

| File | Purpose |
|------|---------|
| `trt_common.h/cpp` | RAII wrappers: `TrtLogger`, `TrtUniquePtr`, `CudaStream`, `CudaBuffer` |
| `trt_engine_lifecycle.h/cpp` | `DecoderStepEngine` struct, `finalize_decoder_step_engine()` |
| `trt_decode_runtime.h/cpp` | `run_decoder_step()`, `select_argmax_token()`, `build_attention_mask()`, `append_cache_state()` |
| `step_state.h` | `IStepState` interface: opaque base for per-step state during autoregressive generation |
| `kv_cache_step_state.h/cpp` | `KvCacheStepState`: KV-cache implementation of `IStepState` |
| `trt_backend_shared.h/cpp` | `TrtBackendFastPath`: autoregressive generate loop with prefill + decode phases. `CreateTrtBackendFromEngine()`. |

### Tokenizers (`src/tokenizer/`)

| File | Purpose |
|------|---------|
| `vocab_tokenizer.cpp` | Word-to-id lookup from vocabulary list. |
| `hf_python_tokenizer.cpp` | Bridges to HF `tokenizers` library via Python subprocess. |

---

## Removed C++ Code (migrated to Python)

The following C++ files were removed as part of the Python build migration (~3500 lines):

| Former File | What It Did | Python Replacement |
|-------------|-------------|-------------------|
| `src/model/model_loader.cpp` | Generic `LoadDecoderModel()` | `trtf_build/builder.py` |
| `src/model/safetensors_loader.h/cpp` | `SafetensorReader`, `TensorSource` | `safetensors` Python library |
| `src/model/model_resolver.cpp` | Multi-stage model resolution | Python model resolution |
| `src/model/hf_family_registry.cpp` | Family registry, alias resolution | `trtf_build/registry.py` |
| `src/model/checkpoint_mapper.h/cpp` | `ICheckpointMapper` interface + registry | Per-family Python mappers |
| `src/model/standard_checkpoint_mapper.h/cpp` | Base class for standard HF naming | Shared Python mapper base |
| `src/model/standard_decoder_graph_builder.h/cpp` | TRT network construction in C++ | `trtf_build/graph_ops.py` |
| `src/model/trt_model_definition.h/cpp` | `TrtDecoderDefinition` conversion | Direct Python conversion |
| `src/models/qwen/` | Qwen C++ registration + mapper | `trtf_build/families/qwen/` |
| `src/models/llama/` | LLaMA C++ registration + mapper | `trtf_build/families/llama/` |
| `src/models/mistral/` | Mistral C++ registration + mapper | `trtf_build/families/mistral/` |
| `src/models/gemma/` | Gemma C++ registration + mapper | `trtf_build/families/gemma/` |
| `src/utils/tensor_math.h/cpp` | `transpose_2d`, `expand_kv_projection` | NumPy operations |
| `src/utils/trt/engine_cache.h/cpp` | On-disk TRT engine plan cache | Bundles replace caching |
| `src/cabi/fast_path_config.h/cpp` | Zero-weight fast path config | Not needed (bundle-only) |
| `src/runtime/trt_backend.cpp` | `CreateTrtBackend()` with model runtime dispatch | Not needed (bundle-only) |
| `src/runtime/runtime_factory.cpp` | Runtime assembly from resolved model | Not needed (bundle-only) |
| `src/runtime/trt/model_runtime.h/cpp` | `IModelRuntime` + registry + factories | Python family dispatch |
| `src/runtime/trt/trt_graph_ops.h/cpp` | Reusable TRT graph ops (C++) | `trtf_build/graph_ops.py` |
| `src/runtime/trt/trt_graph_builder.h` | `ITrtGraphBuilder` interface | Python graph builder |
| `cmake/family_dispatch.cpp.in` | Auto-generated family dispatch | Python plugin discovery |

---

## Tests (`tests/`)

11 tests remain (down from 26). Removed tests covered C++ build infrastructure that has migrated to Python.

| File | What it tests |
|------|--------------|
| `test_helpers.h` | Shared utilities: temp dirs |
| `test_tokenizer.cpp` | VocabTokenizer encode/decode round-trip |
| `test_bundle_format.cpp` | `.trtfb` format: write/read roundtrip, magic validation, JSON header |
| `test_c_abi_entry.cpp` | C ABI factory: create/destroy pipeline, error handling, flags |
| `test_pipeline_api.cpp` | IPipeline interface: generate, max_tokens, pointer lifetime |
| `test_bundle_e2e.cpp` | Bundle E2E: save + load roundtrip (TRT-guarded) |
| `test_cli_args.cpp` | CLI arg parsing: subcommands, flags, error handling |
| `test_data_dir.cpp` | data_dir resolution, env override |
| `test_text_parsers.cpp` | starts_with, ends_with, trim, split_words |
| `test_json_helpers.cpp` | extract_json_string/int/float/array |
| `test_decode_runtime.cpp` | select_argmax_token, build_attention_mask (TRT-guarded) |

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `diff_logits.py` | E2E logit comparison: trtf binary vs HF transformers |
| `diff_layers.py` | Per-layer hidden state comparison between trtf and HF |
| `eval_mmlu.py` | MMLU benchmark evaluation |
| `docker_build.sh` | Build the TRT dev container image |
| `docker_run.sh` | Launch the TRT dev container |
| `test_qwen3_trt_e2e.sh` | All-in-one Qwen3 TRT E2E diagnostic script |
