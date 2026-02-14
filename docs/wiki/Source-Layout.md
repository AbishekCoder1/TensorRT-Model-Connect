# Source Layout

File-by-file guide to the codebase.

## Public API (`include/trtf/`)

| File | Purpose |
|------|---------|
| `pipeline.h` | Main entry point: `Pipeline::CreateTextGeneration()`, `pipeline.generate()` |
| `model.h` | `DecoderModel`, `DecoderCheckpoint`, `DecoderLayerCheckpoint`, `DecoderArchitectureConfig`, `LoadDecoderModel()` |
| `backend.h` | `IGenerationBackend` interface, `CreateCpuReferenceBackend()`, `CreateTrtBackend()`, `CreateHfPythonBackend()` |
| `tokenizer.h` | `ITokenizer` interface, `CreateVocabTokenizer()`, `CreateHfPythonTokenizer()` |
| `generation.h` | `GenerationConfig`, `GenerationResult` |
| `model_resolver.h` | `ResolvedModelSpec`, `ResolveTextGenerationModel()`, `RegisterTextGenerationModelResolver()` |
| `runtime_factory.h` | `RuntimeAssembly`, `BackendSelection`, `BuildRuntimeForTextGeneration()` |
| `hf_family_registry.h` | `HfModelFamilyRegistration`, `RegisterHfModelFamily()`, `ResolveHfModelViaFamilyRegistry()` |

## Shared Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `text_parsers.h/cpp` | `starts_with()`, `to_lower_ascii()`, `iequals_ascii()`, `read_file()`, `parse_positive_env_int()` |
| `json_helpers.h/cpp` | `extract_json_string()`, `extract_json_int()`, `extract_json_float()`, `extract_json_string_array()` |
| `tensor_math.h/cpp` | `transpose_2d()`, `expand_kv_projection()`, `repeat_head_norm()` — used by checkpoint mappers |
| `trt/engine_cache.h/cpp` | On-disk TRT engine plan cache (serialize/deserialize `ICudaEngine`) |

## Model Loading (`src/model/`)

| File | Purpose |
|------|---------|
| `model_loader.cpp` | Generic `LoadDecoderModel()`: parses config.json, loads vocab, dispatches to checkpoint mapper |
| `safetensors_loader.h/cpp` | `SafetensorReader` (single file), `TensorSource` (single or sharded), F32/F16/BF16 support |
| `model_resolver.cpp` | Multi-stage model resolution: custom → HF family → raw HF dir → decoder definition |
| `hf_family_registry.cpp` | Family registry, alias resolution (QWEN3), `RegisterBuiltinHfModelFamilies()` |
| `checkpoint_mapper.h/cpp` | `ICheckpointMapper` interface + priority-sorted registry |
| `standard_checkpoint_mapper.h/cpp` | Base class for standard HF tensor naming (transpose, GQA expand, tied lm_head) |
| `trt_model_definition.h/cpp` | `TrtDecoderDefinition`, `BuildTrtDecoderWeights()` |
| `trt_model_definition_populator.h/cpp` | `ITrtModelDefinitionPopulator` interface + registry |
| `standard_trt_model_definition_populator.h/cpp` | Family-agnostic populator for any model with `has_decoder_layers` |

## Model Families (`src/models/`)

### Qwen (`src/models/qwen/`)

| File | Purpose |
|------|---------|
| `registration.h/cpp` | `RegisterQwenFamily()`: registers into all 4 registries. Handles Qwen, Qwen2, Qwen3, QWQ model types. Contains inline `load_qwen_decoder_model()` with cache-length cap. |
| `checkpoint_mapper.h/cpp` | `QwenCheckpointMapper`: subclasses `StandardCheckpointMapper`, overrides `can_map()` for qwen/qwq family detection |
| `trt_model_populator.h/cpp` | Type alias for `StandardTrtModelDefinitionPopulator` (placeholder, no custom logic) |

### LLaMA (`src/models/llama/`)

| File | Purpose |
|------|---------|
| `registration.h/cpp` | `RegisterLlamaFamily()`: registers checkpoint mapper + TRT graph builder + HF family matcher |
| `checkpoint_mapper.h/cpp` | `LlamaCheckpointMapper`: subclasses `StandardCheckpointMapper`, overrides `can_map()` for llama family detection |

### Template (`src/models/template/`)

| File | Purpose |
|------|---------|
| `registration.cpp` | Documented skeleton for adding a new family. Copy and customize. |

## Runtime / Backends (`src/runtime/`)

| File | Purpose |
|------|---------|
| `runtime_factory.cpp` | `BuildRuntimeForTextGeneration()`: tokenizer selection + backend cascade |
| `trt_backend.cpp` | `CreateTrtBackend()`: finds graph builder, dispatches to `CreateTrtBackendWithBuilder()` |
| `cpu_reference_backend.cpp` | `CpuReferenceBackend`: transition-table-based deterministic generation |
| `hf_python_backend.cpp` | `HfPythonBackend`: spawns Python subprocess with HF transformers |

## TRT Infrastructure (`src/runtime/trt/`)

| File | Purpose |
|------|---------|
| `trt_common.h/cpp` | RAII wrappers: `TrtLogger`, `TrtUniquePtr`, `CudaStream`, `CudaBuffer` |
| `trt_graph_ops.h/cpp` | Reusable TRT graph operations: RMSNorm, matmul, RoPE, bias, dimension helpers |
| `trt_graph_builder.h/cpp` | `ITrtGraphBuilder` interface + name-based registry |
| `standard_decoder_graph_builder.h/cpp` | Builds Pre-RMSNorm+GQA+RoPE+SwiGLU decoder network. Two paths: multi-layer (real models) and legacy (tiny built-in) |
| `trt_engine_lifecycle.h/cpp` | `DecoderStepEngine` struct (with `TensorBinding` and `extra_bindings`), `finalize_decoder_step_engine()`, `find_extra_bindings()` |
| `trt_decode_runtime.h/cpp` | `run_decoder_step()`, `select_argmax_token()`, `build_attention_mask()`, `append_cache_state()` |
| `step_state.h` | `IStepState` interface: abstract per-step state management for autoregressive generation |
| `kv_cache_step_state.h/cpp` | `KvCacheStepState`: KV-cache implementation of `IStepState` for attention-based decoders |
| `trt_backend_shared.h/cpp` | `TrtBackendShared`: autoregressive generate loop using `IStepState` with prefill + decode phases |

## Tokenizers (`src/tokenizer/`)

| File | Purpose |
|------|---------|
| `toy_tokenizer.cpp` | Word-to-id lookup from `vocab.txt`. Used for built-in models. |
| `hf_python_tokenizer.cpp` | Bridges to HF `tokenizers` library via Python subprocess. Exact parity with HF. |

## Pipeline (`src/pipeline/`)

| File | Purpose |
|------|---------|
| `pipeline.cpp` | `Pipeline::CreateTextGeneration()`, `operator()()`, `generate()`. Orchestrates all stages. |

## Tests (`tests/`)

| File | What it tests |
|------|--------------|
| `test_helpers.h` | Shared utilities: temp dirs, safetensors writing, `write_standard_decoder_checkpoint()` |
| `test_tokenizer.cpp` | ToyTokenizer encode/decode round-trip, unknown token handling |
| `test_pipeline.cpp` | Pipeline E2E with tiny-cake-v1, error handling for invalid backends |
| `test_trt_smoke.cpp` | TRT backend force + graceful fallback when TRT unavailable |
| `test_model_loader.cpp` | Built-in loading, single safetensors, sharded safetensors |
| `test_model_resolver.cpp` | Resolution pipeline: built-in → HF → unknown |
| `test_runtime_factory.cpp` | Runtime assembly, backend selection, force_trt errors |
| `test_extension_registry.cpp` | Custom resolver + assembler with mock backend |
| `test_hf_family_registry.cpp` | Family priority ordering, metadata parsing, mock family |
| `test_qwen_family.cpp` | Qwen family detection, checkpoint with q_norm/k_norm, QWEN3 alias |
| `test_llama_family.cpp` | LLaMA family detection, checkpoint without q_norm/k_norm, GQA |
| `test_tensor_math.cpp` | transpose_2d, repeat_head_norm, expand_kv_projection (CPU-only) |
| `test_json_helpers.cpp` | extract_json_string/int/float/array, int_or_first_array (CPU-only) |
| `test_text_parsers.cpp` | starts_with, ends_with, trim, split_words, iequals_ascii (CPU-only) |
| `test_decode_runtime.cpp` | select_argmax_token, select_topk_tokens, build_attention_mask, append_cache_state (TRT-guarded) |
| `test_engine_cache_key.cpp` | BuildTrtEngineCacheKey determinism, extra_params sensitivity (CPU-only) |
| `test_kv_cache_step_state.cpp` | KvCacheStepState position tracking, mask generation, cache overflow (TRT-guarded) |
| `test_extra_fields.cpp` | Phase A extensibility: extra_tensors round-trip, extra_params, find_extra_bindings |
| `test_trt_graph_ops_gold.cpp` | Per-op gold tensor tests: rms_norm, matmul, swiglu, rope, rms_norm_per_head, bias_sum (GPU-only) |

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
