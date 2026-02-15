# Source Layout

File-by-file guide to the codebase.

## Public API (`include/trtf/`)

| File | Purpose |
|------|---------|
| `pipeline.h` | **C ABI entry point**: `IPipeline` virtual interface, `TrtfPipelineOptions`, `trtf_create_pipeline()`, `trtf_create_pipeline_ex()`, `trtf_last_error()`, `trtf_version()`, `trtf_has_trt()` |
| `bundle.h` | Bundle API: `BuildBundle()`, `InspectBundle()`, `IsBundle()`, `BundleInfo` |
| `model.h` | `DecoderModel`, `DecoderCheckpoint`, `DecoderLayerCheckpoint`, `DecoderArchitectureConfig`, `LoadDecoderModel()` |
| `backend.h` | `IGenerationBackend` interface, `CreateTrtBackend()`, `CreateHfPythonBackend()` |
| `tokenizer.h` | `ITokenizer` interface, `CreateVocabTokenizer()`, `CreateHfPythonTokenizer()` |
| `generation.h` | `GenerationConfig`, `GenerationResult` |
| `model_resolver.h` | `ResolvedModelSpec`, `ResolveTextGenerationModel()` |
| `runtime_factory.h` | `RuntimeAssembly`, `BackendSelection`, `BuildRuntimeForTextGeneration()` |
| `hf_family_registry.h` | `HfModelFamilyRegistration`, `RegisterHfModelFamily()`, `ResolveHfModelViaFamilyRegistry()` |

## Shared Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `data_dir.h/cpp` | Centralized source-dir resolution: `source_dir()`, `scripts_dir()`, `models_dir()`, `script_path()`, `model_path()`. Supports `TRTF_DATA_DIR` env override. |
| `text_parsers.h/cpp` | `starts_with()`, `to_lower_ascii()`, `iequals_ascii()`, `read_file()`, `parse_positive_env_int()` |
| `json_helpers.h/cpp` | `extract_json_string()`, `extract_json_int()`, `extract_json_float()`, `extract_json_string_array()` |
| `tensor_math.h/cpp` | `transpose_2d()`, `expand_kv_projection()`, `repeat_head_norm()` — used by checkpoint mappers |
| `trt/engine_cache.h/cpp` | On-disk TRT engine plan cache (mmap-based read, serialize/deserialize). Includes model-dir index (`BuildModelDirIndexKey`, `SaveModelDirIndex`, `LookupModelDirIndex`) for zero-weight fast-path startup. |

## Model Loading (`src/model/`)

| File | Purpose |
|------|---------|
| `model_loader.cpp` | Generic `LoadDecoderModel()`: parses config.json, loads vocab, dispatches to checkpoint mapper |
| `safetensors_loader.h/cpp` | `SafetensorReader` (single file), `TensorSource` (single or sharded), F32/F16/BF16 support |
| `model_resolver.cpp` | Multi-stage model resolution: custom → HF family → raw HF dir → decoder definition |
| `hf_family_registry.cpp` | Family registry, alias resolution (QWEN3), `RegisterBuiltinHfModelFamilies()` |
| `checkpoint_mapper.h/cpp` | `ICheckpointMapper` interface + priority-sorted registry |
| `standard_checkpoint_mapper.h/cpp` | Base class for standard HF tensor naming (transpose, GQA expand, tied lm_head) |
| `trt_model_definition.h/cpp` | `TrtDecoderDefinition`, `BuildTrtDecoderWeights()` (inlined conversion, no registry) |

## Model Families (`src/models/`)

### Qwen (`src/models/qwen/`)

| File | Purpose |
|------|---------|
| `registration.h/cpp` | `RegisterQwenFamily()`: registers into all 3 registries. Handles Qwen, Qwen2, Qwen3, QWQ model types. Contains inline `load_qwen_decoder_model()` with cache-length cap. |
| `checkpoint_mapper.h/cpp` | `QwenCheckpointMapper`: subclasses `StandardCheckpointMapper`, overrides `can_map()` for qwen/qwq family detection |

### LLaMA (`src/models/llama/`)

| File | Purpose |
|------|---------|
| `registration.h/cpp` | `RegisterLlamaFamily()`: registers checkpoint mapper + TRT graph builder + HF family matcher |
| `checkpoint_mapper.h/cpp` | `LlamaCheckpointMapper`: subclasses `StandardCheckpointMapper`, overrides `can_map()` for llama family detection |

### Mistral (`src/models/mistral/`)

| File | Purpose |
|------|---------|
| `registration.h/cpp` | `RegisterMistralFamily()`: registers checkpoint mapper + TRT graph builder + HF family matcher for `model_type: mistral` |
| `checkpoint_mapper.h/cpp` | `MistralCheckpointMapper`: subclasses `StandardCheckpointMapper`, overrides `can_map()` for mistral family detection |

### Gemma (`src/models/gemma/`)

| File | Purpose |
|------|---------|
| `registration.h/cpp` | `RegisterGemmaFamily()`: registers checkpoint mapper + TRT graph builder + HF family matcher for `model_type: gemma` |
| `checkpoint_mapper.h/cpp` | `GemmaCheckpointMapper`: overrides both `can_map()` and `map_checkpoint()`. Adds +1.0 to RMSNorm gamma (Gemma `(1+gamma)` convention) and scales embedding by `sqrt(hidden_size)`. |

### Template (`src/models/template/`)

| File | Purpose |
|------|---------|
| `registration.cpp` | Documented skeleton for adding a new family. Copy and customize. |

## Runtime / Backends (`src/runtime/`)

| File | Purpose |
|------|---------|
| `runtime_factory.cpp` | `BuildRuntimeForTextGeneration()`: tokenizer selection + backend cascade |
| `trt_backend.cpp` | `CreateTrtBackend()`: finds graph builder, dispatches to `CreateTrtBackendWithBuilder()` |
| `hf_python_backend.cpp` | `HfPythonBackend`: spawns Python subprocess with HF transformers |

## TRT Infrastructure (`src/runtime/trt/`)

| File | Purpose |
|------|---------|
| `trt_common.h/cpp` | RAII wrappers: `TrtLogger`, `TrtUniquePtr`, `CudaStream`, `CudaBuffer` |
| `trt_graph_ops.h/cpp` | Reusable TRT graph operations: RMSNorm, matmul, RoPE, bias, dimension helpers |
| `trt_graph_builder.h/cpp` | `ITrtGraphBuilder` interface + name-based registry |
| `standard_decoder_graph_builder.h/cpp` | Builds Pre-RMSNorm+GQA+RoPE+SwiGLU decoder network |
| `trt_engine_lifecycle.h/cpp` | `DecoderStepEngine` struct (with `TensorBinding` and `extra_bindings`), `finalize_decoder_step_engine()`, `find_extra_bindings()` |
| `trt_decode_runtime.h/cpp` | `run_decoder_step()`, `select_argmax_token()`, `build_attention_mask()`, `append_cache_state()` |
| `step_state.h` | `IStepState` interface: abstract per-step state management for autoregressive generation |
| `kv_cache_step_state.h/cpp` | `KvCacheStepState`: KV-cache implementation of `IStepState` for attention-based decoders |
| `trt_backend_shared.h/cpp` | `TrtBackendShared`: autoregressive generate loop using `IStepState` with prefill + decode phases |

## Tokenizers (`src/tokenizer/`)

| File | Purpose |
|------|---------|
| `vocab_tokenizer.cpp` | Word-to-id lookup from vocabulary list. |
| `hf_python_tokenizer.cpp` | Bridges to HF `tokenizers` library via Python subprocess. Exact parity with HF. |

## Bundle Format (`src/bundle/`)

| File | Purpose |
|------|---------|
| `bundle_format.h/cpp` | `.trtfb` binary format: `WriteBundleFile()`, `ReadBundleFile()`, `HasBundleMagic()`, JSON header serialization |
| `bundle_api.cpp` | Public API implementation: `BuildBundle()`, `InspectBundle()`, `IsBundle()` |

## C ABI Layer (`src/cabi/`)

| File | Purpose |
|------|---------|
| `trtf_c.cpp` | `extern "C"` factory: `trtf_create_pipeline()`, `trtf_create_pipeline_ex()`, `trtf_last_error()`, `trtf_version()`, `trtf_has_trt()`. Contains `PipelineImpl` (concrete `IPipeline`). Fast path via `try_create_from_cached_engine()`. |
| `fast_path_config.h/cpp` | `FastPathModelConfig` struct + `parse_fast_path_config()` — extracts model metadata from config.json text for the zero-weight fast path. |

## Tests (`tests/`)

| File | What it tests |
|------|--------------|
| `test_helpers.h` | Shared utilities: temp dirs, safetensors writing, `write_standard_decoder_checkpoint()` |
| `test_tokenizer.cpp` | VocabTokenizer encode/decode round-trip, unknown token handling |
| `test_model_loader.cpp` | Single safetensors, sharded safetensors |
| `test_model_resolver.cpp` | Resolution pipeline: HF → unknown |
| `test_runtime_factory.cpp` | Runtime assembly, backend selection, force_trt errors |
| `test_hf_family_registry.cpp` | Family priority ordering, metadata parsing, mock family |
| `test_qwen_family.cpp` | Qwen family detection, checkpoint with q_norm/k_norm, QWEN3 alias |
| `test_llama_family.cpp` | LLaMA family detection, checkpoint without q_norm/k_norm, GQA |
| `test_tensor_math.cpp` | transpose_2d, repeat_head_norm, expand_kv_projection (CPU-only) |
| `test_json_helpers.cpp` | extract_json_string/int/float/array, int_or_first_array (CPU-only) |
| `test_text_parsers.cpp` | starts_with, ends_with, trim, split_words, iequals_ascii (CPU-only) |
| `test_decode_runtime.cpp` | select_argmax_token, select_topk_tokens, build_attention_mask, append_cache_state (TRT-guarded) |
| `test_engine_cache_key.cpp` | BuildTrtEngineCacheKey determinism, extra_params sensitivity (CPU-only) |
| `test_engine_cache_index.cpp` | BuildModelDirIndexKey, SaveModelDirIndex, LookupModelDirIndex: roundtrip, staleness, cache disable (CPU-only) |
| `test_engine_cache_io.cpp` | SaveTrtEnginePlanToCache, LoadTrtEnginePlanFromCache: roundtrip, mmap, 10MB large file, cache disable (CPU-only) |
| `test_fast_path_config.cpp` | parse_fast_path_config: head_dim explicit/computed, GQA attention_size, cache length capping, eos/bos from array/scalar (CPU-only) |
| `test_kv_cache_step_state.cpp` | KvCacheStepState position tracking, mask generation, cache overflow (TRT-guarded) |
| `test_extra_fields.cpp` | Phase A extensibility: extra_tensors round-trip, extra_params, find_extra_bindings |
| `test_trt_graph_ops_gold.cpp` | Per-op gold tensor tests: rms_norm, matmul, swiglu, rope, rms_norm_per_head, bias_sum (GPU-only) |
| `test_data_dir.cpp` | data_dir resolution, env override, script/model path construction |
| `test_bundle_format.cpp` | .trtfb format: write/read roundtrip, magic validation, JSON header, inspect |
| `test_c_abi_entry.cpp` | C ABI factory: create/destroy pipeline, error handling, flags, version/has_trt |
| `test_pipeline_api.cpp` | IPipeline interface: generate, max_tokens, pointer lifetime, save_bundle, vtable ABI |
| `test_bundle_e2e.cpp` | Bundle E2E: save + load roundtrip, inspect (TRT-guarded, auto-skip without GPU) |
| `test_cli_args.cpp` | CLI arg parsing: subcommands, flags, error handling for unknown flags/commands |

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `diff_logits.py` | E2E logit comparison: trtf binary vs HF transformers |
| `diff_layers.py` | Per-layer hidden state comparison between trtf and HF |
| `generate_op_gold_tensors.py` | Generate gold `.safetensors` fixtures for per-op TRT tests |
| `eval_mmlu.py` | MMLU benchmark evaluation |
| `docker_build.sh` | Build the TRT dev container image |
| `docker_run.sh` | Launch the TRT dev container |
| `test_qwen3_trt_e2e.sh` | All-in-one Qwen3 TRT E2E diagnostic script |
