# Source Layout

File-by-file guide to the codebase. The system is split: Python (`trtf_build/`) handles engine building, C++ handles runtime inference.

## Python: Build Package (`trtf_build/`)

### Core

| File/Dir | Purpose |
|----------|---------|
| `trtf_build/__init__.py` | Package init: exports `build`, `build_bundle`, `ModelConfig` |
| `trtf_build/__main__.py` | CLI entry point: `python -m trtf_build` |
| `trtf_build/cli.py` | CLI implementation: `trtf-build build|inspect|version` |
| `trtf_build/engine_builder.py` | Orchestrator: `build()` (HF ID or local) and `build_bundle()` (local only) |
| `trtf_build/bundle_writer.py` | `.trtfb` bundle writing: package engine plan + tokenizer files |
| `trtf_build/config.py` | `config.json` parsing: extract architecture params (family, dims, heads, etc.) |
| `trtf_build/graph_ops.py` | Shared TRT graph ops (Python): RMSNorm, matmul, RoPE, SwiGLU, attention |
| `trtf_build/standard_decoder_builder.py` | Standard decoder TRT engine builder |
| `trtf_build/checkpoint_mapper.py` | HF safetensors → weight dict (transpose, GQA expansion) |
| `trtf_build/debug_runner.py` | `TrtRunner` (KV cache), `MambaTrtRunner` (SSM), `VisionTrtRunner`, `VLTrtRunner` for pure-Python TRT inference. VL image preprocessing with 4 strategies + configurable interpolation. |
| `trtf_build/qwen_vl_vision_builder.py` | Qwen2.5-VL vision encoder TRT engine builder: 3D patch embedding, 2D RoPE with spatial merge, ViT blocks, spatial merge MLP. |
| `trtf_build/vision_encoder_builder.py` | Deprecated shim: re-exports from `qwen_vl_vision_builder.py`. |

### Family Plugins (`trtf_build/trtf_build/families/`)

Each family is a single Python file implementing the `FamilyPlugin` protocol (see `base.py`). Auto-discovered by `__init__.py`.

| File | Purpose |
|------|---------|
| `__init__.py` | Auto-discovers plugin files via `pkgutil.iter_modules()`, exports `find_plugin()` |
| `base.py` | `FamilyPlugin` protocol definition |
| `qwen.py` | Qwen family: matches `qwen`/`qwen2`/`qwen3`/`qwq`. Handles q_norm/k_norm for Qwen3. |
| `llama.py` | LLaMA family: matches `llama`. Standard HF tensor naming. |
| `mistral.py` | Mistral family: matches `mistral`. Standard HF tensor naming. |
| `gemma.py` | Gemma family: matches `gemma`/`gemma2`. Adds +1.0 to RMSNorm gamma, scales embedding by `sqrt(hidden_size)`. |
| `phi.py` | Phi family: matches `phi3`/`phi` (not `phimoe`). Splits fused QKV and gate_up projections. |
| `phi_moe.py` | Phi-MoE family: matches `phimoe`. SparseMixer routing, per-expert SwiGLU MLPs. Custom graph builder. |
| `granite.py` | Granite family: matches `granite`. Absorbs scaling multipliers into weights. |
| `internlm.py` | InternLM2 family: matches `internlm`/`internlm2`. Fused wqkv, non-standard key names. |
| `starcoder2.py` | StarCoder2 family: matches `starcoder2`. LayerNorm + GELU FC + RoPE. |
| `gpt2.py` | GPT-2 family: matches `gpt2`. Learned positions, fused QKV via Conv1D weights. |
| `opt.py` | OPT family: matches `opt`. Learned positions with offset=2, ReLU MLP. |
| `falcon.py` | Falcon family: matches `falcon`. LayerNorm + GELU FC + RoPE + GQA. |
| `stablelm.py` | StableLM family: matches `stablelm`. LayerNorm + SwiGLU + RoPE. |
| `mamba.py` | Mamba family: matches `mamba`. SSM with selective scan, custom graph builder. Runtime strategy: `ssm_recurrent`. |
| `qwen_vl.py` | Qwen-VL family: matches `qwen*vl`. Vision encoder (ViT + 3D RoPE + spatial merge) + text decoder with embed_input mode. |

---

## C++: Runtime (~18 source files)

### Public API (`include/trtf/`)

| File | Purpose |
|------|---------|
| `pipeline.h` | **C ABI entry point**: `IPipeline` virtual interface, `TrtfPipelineOptions`, `trtf_create_pipeline()`, `trtf_create_pipeline_ex()`, `trtf_last_error()`, `trtf_version()`, `trtf_has_trt()`. Accepts `.trtfb` bundle paths. |
| `bundle.h` | Bundle API: `InspectBundle()`, `IsBundle()`, `BundleInfo`. |
| `backend.h` | `IGenerationBackend` interface |
| `tokenizer.h` | `ITokenizer` interface, `CreateVocabTokenizer()`, `CreateHfPythonTokenizer()` |
| `generation.h` | `GenerationConfig`, `GenerationResult` |

### Shared Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `data_dir.h/cpp` | Centralized source-dir resolution: `source_dir()`, `scripts_dir()`. |
| `text_parsers.h/cpp` | `starts_with()`, `to_lower_ascii()`, `read_file()`, etc. |
| `json_helpers.h/cpp` | `extract_json_string()`, `extract_json_int()`, etc. |

### Bundle Format (`src/bundle/`)

| File | Purpose |
|------|---------|
| `bundle_format.h/cpp` | `.trtfb` binary format: `WriteBundleFile()`, `ReadBundleFile()`, `HasBundleMagic()`, JSON header serialization |

### C ABI Layer (`src/cabi/`)

| File | Purpose |
|------|---------|
| `trtf_c.cpp` | `extern "C"` factory: `trtf_create_pipeline()`, `trtf_create_pipeline_ex()`, `trtf_last_error()`, `trtf_version()`, `trtf_has_trt()`. Contains `PipelineImpl`. Dispatches to `TrtBackendFastPath` or `MambaBackend` based on `runtime_strategy`. |
| `fast_path_config.h/cpp` | `FastPathModelConfig`: parses config.json for runtime parameters including `runtime_strategy`, SSM dimensions. |

### TRT Infrastructure (`src/runtime/trt/`)

| File | Purpose |
|------|---------|
| `trt_common.h/cpp` | RAII wrappers: `TrtLogger`, `TrtUniquePtr`, `CudaStream`, `CudaBuffer` |
| `trt_engine_lifecycle.h/cpp` | `DecoderStepEngine` struct, `finalize_decoder_step_engine()` |
| `trt_decode_runtime.h/cpp` | `run_decoder_step()`, `select_argmax_token()`, `build_attention_mask()`, `append_cache_state()` |
| `step_state.h` | `IStepState` interface: opaque base for per-step state during autoregressive generation |
| `kv_cache_step_state.h/cpp` | `KvCacheStepState`: KV-cache implementation of `IStepState` |
| `trt_backend_shared.h/cpp` | `TrtBackendFastPath`: autoregressive generate loop with prefill + decode phases. `CreateTrtBackendFromEngine()`. |
| `mamba_backend.h/cpp` | `MambaBackend`: SSM autoregressive loop with recurrent state. `CreateMambaBackendFromEngine()`. |
| `mamba_decode_runtime.h/cpp` | `MambaStepEngine` struct, `run_mamba_step()`, `has_all_required_mamba_tensors()`. |
| `mamba_step_state.h/cpp` | `MambaStepState`: conv_state + ssm_state per layer (constant memory, no growth). |
| `image_preprocessor.h/cpp` | VL image preprocessing: `VLPreprocessConfig`, `PreprocessedImage`, `load_and_preprocess_image()`, `format_vl_prompt()`, `parse_vl_preprocess_config()`. Supports 4 strategies: `qwen_merge_group`, `simple_chw`, `center_crop_chw`, `aspect_preserve_chw`. Configurable interpolation (`bicubic`/`bilinear`/`nearest`). |

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
| `src/models/qwen/` | Qwen C++ registration + mapper | `trtf_build/trtf_build/families/qwen.py` |
| `src/models/llama/` | LLaMA C++ registration + mapper | `trtf_build/trtf_build/families/llama.py` |
| `src/models/mistral/` | Mistral C++ registration + mapper | `trtf_build/trtf_build/families/mistral.py` |
| `src/models/gemma/` | Gemma C++ registration + mapper | `trtf_build/trtf_build/families/gemma.py` |
| `src/utils/tensor_math.h/cpp` | `transpose_2d`, `expand_kv_projection` | NumPy operations |
| `src/utils/trt/engine_cache.h/cpp` | On-disk TRT engine plan cache | Bundles replace caching |
| `src/runtime/trt_backend.cpp` | `CreateTrtBackend()` with model runtime dispatch | Not needed (bundle-only) |
| `src/runtime/runtime_factory.cpp` | Runtime assembly from resolved model | Not needed (bundle-only) |
| `src/runtime/trt/model_runtime.h/cpp` | `IModelRuntime` + registry + factories | Python family dispatch |
| `src/runtime/trt/trt_graph_ops.h/cpp` | Reusable TRT graph ops (C++) | `trtf_build/graph_ops.py` |
| `src/runtime/trt/trt_graph_builder.h` | `ITrtGraphBuilder` interface | Python graph builder |
| `cmake/family_dispatch.cpp.in` | Auto-generated family dispatch | Python plugin discovery |

---

## Tests (`tests/`)

12 test executables (down from 26). Removed tests covered C++ build infrastructure that migrated to Python.

| File | What it tests |
|------|--------------|
| `test_helpers.h` | Shared utilities: temp dirs |
| `test_bundle_format.cpp` | `.trtfb` format: write/read roundtrip, magic validation, JSON header |
| `test_bundle_e2e.cpp` | Bundle E2E: save + load roundtrip (TRT-guarded) |
| `test_c_abi_entry.cpp` | C ABI factory: create/destroy pipeline, error handling, flags |
| `test_pipeline_api.cpp` | IPipeline interface: generate, max_tokens, pointer lifetime |
| `test_cli_args.cpp` | CLI arg parsing: subcommands, flags, error handling |
| `test_data_dir.cpp` | data_dir resolution, env override |
| `test_text_parsers.cpp` | starts_with, ends_with, trim, split_words |
| `test_json_helpers.cpp` | extract_json_string/int/float/array |
| `test_fast_path_config.cpp` | FastPathModelConfig parsing, runtime_strategy detection |
| `test_kv_cache_step_state.cpp` | KvCacheStepState: cache append, mask building, state management |
| `test_decode_runtime.cpp` | select_argmax_token, build_attention_mask (TRT-guarded) |
| `test_image_preprocessor.cpp` | VL image preprocessing: qwen_merge_group, simple_chw, center_crop_chw, aspect_preserve_chw, interpolation parsing, unknown type fallback, prompt formatting, config parsing (13 tests) |

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `setup_container.sh` | One-shot container setup: venv, pip deps, cmake build, tests |
| `docker_build.sh` | Build the self-contained dev container image |
| `docker_run.sh` | Launch the dev container |
| `eval_mmlu.py` | MMLU benchmark evaluation |
| `new_family.py` | Scaffold a new family plugin from HF repo |
| `validate_family.sh` | One-command validation gate (build + diff + parity) |
| `hf_tokenizer.py` | HuggingFace tokenizer bridge for C++ subprocess calls |
| `hf_generate.py` | HuggingFace reference generation for parity testing |
| `test_qwen3_trt_e2e.sh` | All-in-one Qwen3 TRT E2E diagnostic script |

## Tools (`tools/`)

Diff-test and performance comparison tools (TRT vs HF).

| File | Purpose |
|------|---------|
| `diff_logits.py` | E2E logit comparison: TRT vs HF transformers (per-step logits, text match) |
| `diff_layers.py` | Per-layer hidden state comparison between TRT and HF |
| `diff_vl.py` | VL diff testing: vision features, embed_input, full VL generation, C++ binary parity. Supports `--preprocessor-type` override. |
| `test_runner_parity.py` | Python-vs-C++ runner parity cross-validation |
| `test_graph_ops.py` | TRT graph operation unit testing |
| `perf_compare.py` | TRT vs HF performance comparison — serial GPU execution to support large models on 24GB GPUs |
