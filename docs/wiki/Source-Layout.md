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
| `trtf_build/graph_ops.py` | Layer 1: Atomic TRT graph ops (tensor-in/tensor-out, no weight naming) |
| `trtf_build/graph_blocks.py` | Layer 2: Composable building blocks (weight-aware attention, SwiGLU MLP, GELU MLP, norm dispatch). Callers compose residual patterns. |
| `trtf_build/standard_decoder_builder.py` | Layer 3: Standard decoder TRT engine builder (uses `graph_blocks`) |
| `trtf_build/checkpoint_mapper.py` | HF safetensors → weight dict (transpose, GQA expansion) |
| `trtf_build/debug_runner.py` | `TrtRunner` (device-resident KV cache), `MambaTrtRunner` (device-resident SSM), `VisionTrtRunner`, `VLTrtRunner` for pure-Python TRT inference. VL image preprocessing with 4 strategies + configurable interpolation. |
| `trtf_build/qwen_vl_vision_builder.py` | Qwen2.5-VL vision encoder TRT engine builder: 3D patch embedding, 2D RoPE with spatial merge, ViT blocks, spatial merge MLP. |
| `trtf_build/vision_encoder_builder.py` | Deprecated shim: re-exports from `qwen_vl_vision_builder.py`. |
| `trtf_build/onnx_vision_builder.py` | ONNX-based vision encoder builder: trace HF vision model to ONNX, convert to TRT via `trt.OnnxParser`. |
| `trtf_build/diffusion_runner.py` | `DiffusionRunner`: pure-Python TRT diffusion pipeline (T5 encode, denoising loop with CFG, frame-by-frame VAE decode). |
| `trtf_build/standard_dit_builder.py` | Shared DiT (Diffusion Transformer) TRT engine builder: self-attention with AdaLN, cross-attention, FFN, 3D RoPE. |
| `trtf_build/causal_vae_3d_builder.py` | Shared Causal 3D VAE decoder TRT engine builder: per-frame decoding with temporal caches, causal convolutions, spatial upsampling. |
| `trtf_build/t5_encoder_builder.py` | Shared T5 encoder TRT engine builder: UMT5/mT5/T5 with relative position bias, gated GELU FFN, RMSNorm. |
| `trtf_build/pipeline.py` | Thin Python wrapper around the C++ `trtf` CLI: `Pipeline` class for Pythonic inference via subprocess. |
| `trtf_build/schedulers/` | Noise scheduler implementations for diffusion models. |
| `trtf_build/schedulers/base.py` | Scheduler protocol (pure numpy interface). |
| `trtf_build/schedulers/flow_match_euler.py` | Flow matching Euler discrete scheduler (Wan2.1, FLUX, SD3). Implements `z_t = (1-t)*x + t*noise` with configurable shift. |

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
| `qwen_vl.py` | Qwen-VL family: matches `qwen*vl`. Qwen2.5-VL (3D RoPE ViT) + Qwen3-VL (learned pos + RoPE + DeepStack). Text decoder with embed_input mode; Qwen3-VL adds deepstack injection via `graph_blocks`. |
| `nemotron.py` | Nemotron-4 family: matches `nemotron`. LayerNorm1P (+1 gamma) + squared ReLU MLP + GQA + partial RoPE. |
| `olmo.py` | OLMo family: matches `olmo`. Standard decoder. |
| `xglm.py` | XGLM family: matches `xglm`. Standard decoder. |
| `gpt_neox.py` | GPT-NeoX family: matches `gpt_neox`. Standard decoder. |
| `gpt_neo.py` | GPT-Neo family: matches `gpt_neo`. Standard decoder. |
| `codegen.py` | CodeGen family: matches `codegen`. Standard decoder. |
| `bloom.py` | BLOOM family: matches `bloom`. ALiBi attention, GELU MLP. |
| `mixtral.py` | Mixtral family: matches `mixtral`. Standard top-2 softmax MoE routing. Runtime strategy: `decoder_moe`. |
| `wan_t2v.py` | Wan2.1 T2V family: matches `wan`. Composes T5 encoder + DiT denoiser + Causal 3D VAE. Runtime strategy: `diffusion`. |

---

## C++: Runtime (~26 source files)

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
| `trtf_c.cpp` | Thin dispatch: `PipelineImpl`, per-strategy factory functions, `extern "C"` API. ~200 lines. |
| `fast_path_config.h/cpp` | `FastPathModelConfig`: parses config.json for runtime parameters including `runtime_strategy`, SSM dimensions. |
| `bundle_helpers.h/cpp` | Shared plumbing: `BundleSections`, `extract_tokenizer_from_bundle()`, `make_decoder_engine()`. Eliminates tokenizer/engine-init duplication across strategies. |

### TRT Infrastructure (`src/runtime/trt/`)

| File | Purpose |
|------|---------|
| `trt_common.h/cpp` | RAII wrappers: `TrtLogger`, `TrtUniquePtr`, `CudaStream`, `CudaBuffer` |
| `trt_engine_lifecycle.h/cpp` | `DecoderStepEngine` struct, `finalize_decoder_step_engine()` |
| `trt_decode_runtime.h/cpp` | `select_argmax_token()`, `select_topk_tokens()`, `build_attention_mask()` |
| `step_state.h` | `IStepState` interface: opaque base for per-step state during autoregressive generation |
| `device_kv_cache.h/cpp` | `DeviceKvCache`: device-resident KV cache (persistent GPU buffers, D2D updates). `DeviceResources`: pre-allocated per-step I/O buffers. `run_decoder_step_device()`: device-resident decode step. |
| `trt_backend_shared.h/cpp` | `TrtBackendFastPath`: autoregressive generate loop with device-resident cache. `CreateTrtBackendFromEngine()`. |
| `mamba_backend.h/cpp` | `MambaBackend`: SSM autoregressive loop with recurrent state. `CreateMambaBackendFromEngine()`. |
| `mamba_decode_runtime.h/cpp` | `MambaStepEngine` struct, `run_mamba_step()`, `has_all_required_mamba_tensors()`. |
| `mamba_step_state.h/cpp` | `MambaStepState`: conv_state + ssm_state per layer (constant memory, no growth). |
| `image_preprocessor.h/cpp` | VL image preprocessing: `VLPreprocessConfig`, `PreprocessedImage`, `load_and_preprocess_image()`, `format_vl_prompt()`, `parse_vl_preprocess_config()`. Supports 4 strategies: `qwen_merge_group`, `simple_chw`, `center_crop_chw`, `aspect_preserve_chw`. Configurable interpolation (`bicubic`/`bilinear`/`nearest`). |
| `vision_engine.h/cpp` | `VisionStepEngine`: TRT engine wrapper for vision encoders. `run_vision_encoder()`, `run_vision_encoder_with_deepstack()` (multi-level features for Qwen3-VL). |
| `vl_backend.h/cpp` | `VLBackendFastPath`: vision-language pipeline backend. Owns decoder engine + vision encoder. `generate_vl()` with image features, `prepare_image()` for full image pipeline. DeepStack support for multi-scale injection. |
| `diffusion_backend.h` | Core diffusion abstractions: `DiffusionConfig`, `PreprocessorWeights`, `DiffusionEngine`, `VideoResult`, `IDiffusionBackend` protocol, `DiffusionBackendBase` shared base. |
| `diffusion_backend_base.cpp` | Shared diffusion utilities: CPU math helpers (`cpu_matmul_bias`, `cpu_silu_inplace`, `cpu_gelu_tanh_inplace`), preprocessor weight parsing, `run_t5_encoder()`, `run_denoiser()`, `decode_vae_subprocess()`. |
| `wan_diffusion_backend.h/cpp` | `WanDiffusionBackend`: Wan2.1-specific diffusion backend. Flow-match Euler scheduler, 3D RoPE, patchify/unpatchify, causal VAE decode with cache management, `generate_video()`. |
| `stb_impl.cpp` | STB library implementation file: defines `STB_IMAGE_IMPLEMENTATION`, `STB_IMAGE_RESIZE_IMPLEMENTATION`, `STB_IMAGE_WRITE_IMPLEMENTATION` for single-compilation-unit linking. |

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

### C++ Unit Tests (`tests/cpp/`)

11 test executables. Removed tests covered C++ build infrastructure that migrated to Python.

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
| `test_decode_runtime.cpp` | select_argmax_token, select_topk_tokens, build_attention_mask (TRT-guarded) |
| `test_image_preprocessor.cpp` | VL image preprocessing: qwen_merge_group, simple_chw, center_crop_chw, aspect_preserve_chw, interpolation parsing, unknown type fallback, prompt formatting, config parsing (13 tests) |

### Python Builder Tests (`tests/builder/`)

11 test modules covering config parsing, checkpoint mapping, bundle writing, graph ops, and CLI.

### Tools Self-Tests (`tests/tools/`)

| File | What it tests |
|------|--------------|
| `test_diff_framework.py` | Diff framework: DiffResult serialization, registry lookup, runner orchestration, CLI parsing |
| `test_diff_logits.py` | diff_logits.py: battery list sanity, numpy logit comparison, argmax matching, top-k overlap |
| `test_diff_layers.py` | diff_layers.py: per-layer comparison, tolerance checks, std/mean computation |
| `test_diff_vl.py` | diff_vl.py: model name matching, cosine similarity, sanity checks, preprocessor defaults |
| `test_parity.py` | test_runner_parity.py: text matching logic, token ID comparison |
| `test_perf_compare.py` | perf_compare.py: timing stats, formatting, JSON output, serial GPU execution order |
| `test_perf_parity.py` | Performance parity: C++ binary vs Python TrtRunner output comparison |

### E2E Tests (`tests/e2e/`)

| File | What it tests |
|------|--------------|
| `conftest.py` | Fixtures: engine_dir, trtf_binary, hf_python, model parametrization, bundle building |
| `test_full_pipeline.py` | Full pipeline: build + C++ inference + diff_logits + perf_compare (standard + VL models) |
| `test_inference.py` | Basic inference: non-empty output, deterministic generation |
| `test_bundle_inspect.py` | `trtf inspect` output validation, runtime_strategy presence |
| `test_logit_parity.py` | `diff_logits.py --battery` wrapper |
| `test_runner_parity.py` | `test_runner_parity.py` wrapper (Python vs C++) |
| `test_vl_pipeline.py` | VL: vision-only smoke test + VL generation via C++ binary |
| `test_diffusion_pipeline.py` | Diffusion: build + 9-step debug pipeline + C++ generate-video + frame quality checks |
| `models/*.json` | Per-model JSON manifests (28 models across 5 runtime strategies) |

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `setup_container.sh` | One-shot repo setup in container: editable install, cmake build, C++ tests |
| `docker_build_gb300.sh` | Build the GB300 (aarch64) container image |
| `docker_run_gb300.sh` | Launch the GB300 container |
| `bootstrap_workspace.sh` | Create per-team isolated workspace clone + container on shared GB300 hosts |
| `run_e2e_parallel.sh` | Multi-GPU E2E dispatcher (used by CI) |
| `schedule_e2e.py` | Scheduler helper for `run_e2e_parallel.sh` |
| `check_family_coverage.py` | CI guard for family/test coverage alignment |
| `eval_mmlu.py` | MMLU benchmark evaluation |
| `new_family.py` | Scaffold a new family plugin from HF repo |
| `validate_family.sh` | One-command validation gate (build + diff + parity) |
| `hf_tokenizer.py` | HuggingFace tokenizer bridge for C++ subprocess calls |
| `hf_mel_extract.py` | HF mel extraction subprocess for audio pipelines |
| `hf_generate.py` | HuggingFace reference generation for parity testing |
| `magpie_tokenizer.py` | Magpie tokenizer subprocess bridge for TRT runtime |
| `magpie_codec_bridge.py` | Magpie codec decode bridge for audio generation |
| `build_wan14b.py` | Build Wan2.1-T2V-14B bundle with configurable frame count and video dimensions |

## Tools (`tools/`)

Diff-test, performance comparison, and validation tools (TRT vs HF).

### Standalone Diff Tools

| File | Purpose |
|------|---------|
| `diff_logits.py` | E2E logit comparison: TRT vs HF transformers (per-step logits, text match) |
| `diff_layers.py` | Per-layer hidden state comparison between TRT and HF |
| `diff_vl.py` | VL diff testing: vision features, embed_input, full VL generation, C++ binary parity. Supports `--preprocessor-type` override. |
| `test_runner_parity.py` | Python-vs-C++ runner parity cross-validation |
| `test_graph_ops.py` | TRT graph operation unit testing |
| `perf_compare.py` | TRT vs HF performance comparison -- serial GPU execution to support large models on 24GB GPUs |

### Unified Diff Framework

| File | Purpose |
|------|---------|
| `diff.py` | Unified CLI: `list` (show checks) and `run` (execute applicable checks). Auto-detects `runtime_strategy`. |
| `diff_framework/__init__.py` | Framework public API, auto-discovers check modules |
| `diff_framework/protocol.py` | Core types: `DiffResult`, `TestContext`, `DiffTest` protocol |
| `diff_framework/registry.py` | Check registry: `register()` decorator, `get_all_tests()`, `get_tests_for_strategy()` |
| `diff_framework/runner.py` | Runner: `detect_runtime_strategy()`, `list_tests()`, `run_tests()` |
| `diff_framework/checks/logit_diff.py` | `LogitDiffTest`: delegates to `diff_logits.py` |
| `diff_framework/checks/layer_diff.py` | `LayerDiffTest`: delegates to `diff_layers.py` |
| `diff_framework/checks/runner_parity.py` | `RunnerParityTest`: delegates to `test_runner_parity.py` |
| `diff_framework/checks/vl_pipeline.py` | `VLPipelineTest`: delegates to `diff_vl.py` |
| `diff_framework/checks/perf_benchmark.py` | `PerfBenchmarkTest`: delegates to `perf_compare.py` |
| `diff_framework/checks/diffusion_components.py` | `DiffusionComponentsTest`: delegates to `debug_diffusion_pipeline.py` |

### Diffusion Validation Tools

| File | Purpose |
|------|---------|
| `debug_diffusion_pipeline.py` | 9-step component-by-component TRT-vs-HF diffusion validation (config, T5, DiT, scheduler, full pipeline) |
| `diffusion_helpers.py` | Shared diffusion utilities: activation functions, weight loading, timestep embedding, TRT engine execution |
| `diff_t5.py` | T5 encoder TRT vs HF validation with configurable tolerance |
| `validate_dit.py` | DiT denoiser single-step validation with cosine similarity check |
| `validate_t5.py` | T5 encoder per-token validation with attention mask handling |
| `tool_helpers.py` | Shared utilities for standard tools: `build_trt_engine()`, `load_hf_model()`, `cosine_sim()`, `compare_arrays()` |
