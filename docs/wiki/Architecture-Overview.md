# Architecture Overview

## Python Builds, C++ Runs

The system is split into two phases across two languages:

| Phase | Language | Tool | Input | Output |
|-------|----------|------|-------|--------|
| **Build** | Python | `trtf-build` / `trtf_build.build()` | HF repo ID or local directory | `.trtfb` bundle |
| **Run** | C++ | `trtf` / C ABI | `.trtfb` bundle | Task-specific outputs (text, masks, detections, embeddings, audio, video frames, etc.) |

**Build phase** (Python `trtf_build/` package — accepts HF repo IDs or local directories):
1. Resolve model: auto-download from HuggingFace Hub if needed
2. Parse `config.json` to identify model family and architecture
3. Load safetensors weights via the `safetensors` Python library
4. Map HF tensor keys to canonical format (per-family checkpoint mapper)
5. Build TRT `INetworkDefinition` via the TensorRT Python API
6. Compile to `ICudaEngine` (GPU kernel compilation)
7. Package engine plan + tokenizer files into a `.trtfb` bundle

**Run phase** (C++ runtime):
1. Load `.trtfb` bundle, deserialize TRT engine plan
2. Create tokenizer (HfPythonTokenizer or VocabTokenizer)
3. Parse `runtime_strategy` from bundle config and dispatch to the matching backend
4. Run task-specific GPU execution loops (autoregressive decode, recurrent state updates, diffusion denoising, segmentation/detection heads, speech pipelines, etc.)
5. Return task-specific outputs via `IPipeline`

---

## Build Phase: Python CLI + API

### CLI (`trtf-build`)

```
trtf-build build Qwen/Qwen3-0.6B -o output.trtfb [--max-cache-length N]
trtf-build build <local-model-dir> -o output.trtfb [--max-cache-length N]
trtf-build inspect <bundle.trtfb>
trtf-build version
```

### Python API

```python
import trtf_build
trtf_build.build("Qwen/Qwen3-0.6B", "qwen3.trtfb")            # auto-downloads
trtf_build.build("models/hf/Qwen__Qwen3-0.6B", "qwen3.trtfb")  # local directory
```

The build phase uses the `trtf_build/` Python package, which provides:

### Family Plugin System

Each model family is a single Python file in `trtf_build/trtf_build/families/` (list below is illustrative, not exhaustive):

```
trtf_build/trtf_build/families/
  base.py         # FamilyPlugin protocol
  qwen.py         # Qwen, Qwen2, Qwen3, QWQ
  llama.py        # LLaMA, TinyLlama
  mistral.py      # Mistral
  gemma.py        # Gemma (adds +1.0 to RMSNorm gamma, scales embedding)
  phi.py          # Phi-3 (fused QKV and gate_up weight splitting)
  phi_moe.py      # Phi-MoE (SparseMixer routing, expert MLPs)
  granite.py      # Granite (multiplier absorption)
  internlm.py     # InternLM2 (fused wqkv, custom key names)
  starcoder2.py   # StarCoder2 (LayerNorm + GELU FC + RoPE)
  gpt2.py         # GPT-2 (learned positions, fused QKV, Conv1D)
  opt.py          # OPT (learned positions, ReLU, position offset)
  falcon.py       # Falcon (LayerNorm + GELU FC + RoPE)
  stablelm.py     # StableLM (LayerNorm + SwiGLU + RoPE)
  mamba.py        # Mamba (SSM, custom graph, no KV cache)
  qwen_vl.py      # Qwen-VL (vision encoder + text decoder with embed_input)
```

List all currently available family modules in your checkout:

```bash
ls trtf_build/trtf_build/families/*.py \
  | sed 's|.*/||; s|\.py$||' \
  | rg -v '^(__init__|base)$' \
  | sort
```

A family plugin provides:
- **Matcher**: Checks `model_type` from `config.json` to claim the model
- **Checkpoint mapper**: Translates HF safetensors tensor keys to canonical format (transpose, GQA KV expansion, etc.)
- **Graph builder**: Constructs the TRT network definition (reusing shared ops for RMSNorm, RoPE, GQA attention, SwiGLU, etc.)

### Builder Stack: Three-Layer Abstraction

The Python builder uses a three-layer abstraction for TRT engine construction:

```
graph_ops.py            Layer 1: Atomic TRT operations
    |
graph_blocks.py         Layer 2: Composable architectural blocks
    |
builders / plugins      Layer 3: Full engine assembly
```

**Layer 1 — `graph_ops.py` (atomic ops)**: 1:1 port of TRT graph operations. Each function adds a single logical op to the TRT network (one RMSNorm, one matmul, one RoPE application). Functions take raw `trt.ITensor` inputs and `np.ndarray` weights — they know nothing about weight naming conventions, layer prefixes, or model architecture. This file is a stable foundation that rarely changes.

**Layer 2 — `graph_blocks.py` (composable blocks)**: Weight-aware, architecture-aware building blocks that compose multiple `graph_ops` into reusable sub-structures. Functions accept a `weights` dict + `prefix` string to resolve weight names. Blocks do NOT apply residual connections — callers compose the residual pattern, which is what varies across architectures.

Available blocks:
- `add_attention_block()` — pre-norm → QKV → optional biases/norms → optional RoPE/ALiBi → cache concat → multi-head attention → output projection
- `add_swiglu_mlp()` — gate/up/down SwiGLU MLP
- `add_gelu_fc_mlp()` — fc1 → activation → fc2 MLP
- `apply_norm()` — dispatch to RMSNorm or LayerNorm

**Layer 3 — Builders + family plugins (full engines)**: `standard_decoder_builder.py` is the default recipe (sequential residual, standard MLP). Family plugins that need non-standard topology (MoE, DeepStack, hybrid SSM) compose their own layer loop from Layer 2 blocks directly.

| When you need to... | Change... |
|---------------------|-----------|
| Add a new TRT primitive | `graph_ops.py` |
| Reuse attention/MLP with different wiring | `graph_blocks.py` |
| Standard decoder with different weights/norms | Family plugin using `standard_decoder_builder` |
| Non-standard topology (MoE, DeepStack, hybrid) | Family plugin composing `graph_blocks` directly |

---

### C++ Runtime: Modular C ABI Layout

The runtime path is now split across focused `src/cabi/*.cpp` units instead of one monolithic file:

| File(s) | Responsibility |
|---------|----------------|
| `src/cabi/api/trtf_c.cpp` | C ABI entrypoints (`trtf_create_pipeline*`, `trtf_last_error`, version/capability), bundle-load orchestration, early `diffusion` branch, registry dispatch call, decoder fallback, and top-level error handling. |
| `src/cabi/pipeline/pipeline_impl.cpp` | Concrete `PipelineImpl` implementation of `IPipeline`: lifecycle ownership, temp-dir cleanup, and method dispatch across generation/vision/audio/encoder/diffusion capabilities. |
| `src/cabi/bundle/bundle_helpers.{h,cpp}` | Shared bundle plumbing (`find_bundle_sections`, tokenizer extraction, `make_decoder_engine`, mel/CLIP helpers). |
| `src/cabi/registry/backend_registry.{h,cpp}` | Thread-safe strategy registry API: `register_backend_factory`, `find_backend_factory`, `try_create_pipeline_from_registry`, unregister/count helpers. |
| `src/cabi/registry/backend_registry_dispatch.{h,cpp}` | Built-in strategy registration orchestration (`register_builtin_backend_factories_once`) and `BackendRegistryDispatchContext` contract. |
| `src/cabi/registry/backend_registry_strategy_plugins_*.cpp` | Per-domain built-in registration plugins (`register_backend_factory("...", &wrapper)`) for text/vision/encoder/audio/misc strategies. |
| `src/cabi/registry/backend_registry_strategy_wrappers.{h,cpp}` | Typed `*_via_registry(void*)` wrappers and dispatch-context validation for registry construction. |
| `src/cabi/factories/factories_*.{h,cpp}` | Strategy-specific pipeline assembly modules (text, multimodal, audio, vision, encoder, diffusion). |

Factory modules are split by domain:

| Factory module | Strategies assembled there |
|----------------|----------------------------|
| `factories_text.cpp` | `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention` |
| `factories_multimodal.cpp` | `speech_to_text`, `vision_language` |
| `factories_audio.cpp` | `text_to_audio`, `omni_multimodal`, `speech_to_speech` |
| `factories_vision.cpp` | `segmentation`, `object_detection`, `prompted_segmentation`, `neural_operator` |
| `factories_encoder.cpp` | `encoder_only`, `embedding`, `reranking` |
| `factories_diffusion.cpp` | `diffusion` |

Built-in strategy plugin registration is modular:
1. `backend_registry_dispatch.cpp` runs one-time orchestration (`std::call_once`) and invokes per-domain plugin registration functions.
2. `backend_registry_strategy_plugins_*.cpp` binds each `runtime_strategy` with literal `register_backend_factory("...", &wrapper)` calls.
3. `backend_registry_strategy_wrappers.cpp` validates `BackendRegistryDispatchContext` and forwards to the relevant modular factory.

Current dispatch behavior in `try_create_from_bundle()`:
1. Parse config early and route `runtime_strategy == "diffusion"` directly to `create_diffusion_pipeline(...)` (diffusion bundles do not require primary `engine_plan`).
2. For non-diffusion paths, deserialize primary TRT engine and execution context.
3. Register built-ins, build `BackendRegistryDispatchContext`, and call `try_create_pipeline_from_registry(...)`.
4. If registry lookup misses, fall back to `create_decoder_pipeline(...)` (which accepts decoder strategies only).

**How to add a new runtime strategy:**
1. Add backend implementation(s) under `src/runtime/trt/` and a matching factory in the appropriate `src/cabi/factories/factories_*.cpp` module.
2. Add a `*_via_registry` wrapper in `src/cabi/registry/backend_registry_strategy_wrappers.cpp` and register it via `register_backend_factory("new_strategy", &wrapper)` in the appropriate `src/cabi/registry/backend_registry_strategy_plugins_*.cpp` module.
3. Extend `FastPathModelConfig` parsing in `src/cabi/config/fast_path_config.{h,cpp}` when new config fields are required.
4. Update strategy governance files (`tests/runtime_strategy_matrix.yaml`, `tests/e2e_harness/contracts.py`, and related runner/comparator/diff-check coverage).

### Runtime Strategy Governance Matrix

Strategy governance is enforced by `tools/check_runtime_strategy_matrix.py` across all `src/cabi/*.cpp` files:

1. Discovers strategy keys from:
   - `register_backend_factory("...")` registrations
   - direct strategy comparisons (`strategy == "..."`, `strategy != "..."`, and early `fp_cfg_early.runtime_strategy == "..."`)
2. Validates parity between:
   - discovered `src/cabi/*.cpp` strategy keys
   - `tests/runtime_strategy_matrix.yaml`
   - `tests/e2e_harness/contracts.py` (`RUNTIME_TO_TASK_STRATEGY`)
3. Enforces matrix coverage requirements:
   - non-empty CLI command list
   - runner/comparator class mapping consistency
   - diff-framework check-class parity or explicit exemption
4. Validates registry wrapper hygiene:
   - every `register_backend_factory` wrapper symbol has a definition
   - no orphan `*_via_registry` wrappers
   - extraction-boundary check for lingering wrappers in `trtf_c.cpp`

Unit tests for the checker are in `tests/tools/test_runtime_strategy_matrix_checker.py`.

### Cyclomatic Complexity Governance

In addition to strategy-governance checks, the repository enforces a C++ complexity budget:

1. Tooling: `tools/check_cyclomatic_complexity.py` (wrapper around `lizard`).
2. Scope: scans C++ runtime sources under `src/`.
3. CI policy: `check-cyclomatic-complexity` in `.gitlab-ci.yml` fails if any function exceeds `CCN > 10` (default `CCM_MAX_CCN=10`).
4. Local parity command:

```bash
python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

---

### What Python replaced in C++

The following C++ code (~3500 lines) was removed and replaced by Python equivalents:

| Former C++ Component | Python Replacement |
|---|---|
| `SafetensorReader`, `TensorSource` | `safetensors` Python library |
| `ICheckpointMapper`, `StandardCheckpointMapper` | Per-family Python checkpoint mappers |
| `LoadDecoderModel`, `model_loader.cpp` | Python config.json parsing + weight loading |
| `StandardDecoderGraphBuilder`, `trt_graph_ops` | Python TRT network construction via `tensorrt` API |
| `TrtDecoderDefinition`, `BuildTrtDecoderWeights` | Direct Python weight-to-TRT conversion |
| `IModelRuntime`, `RegisterModelRuntime` | Python family plugin dispatch |
| `ResolveTextGenerationModel`, `ResolveHfModelViaFamilyRegistry` | Python model resolution |
| `RegisterHfModelFamily`, `RegisterBuiltinHfModelFamilies` | Python family registry |
| Engine cache system | Bundle-based caching (build once, run many times) |
| `tensor_math.h/cpp` | NumPy operations in Python |

`FastPathModelConfig` was not removed. It is parsed from bundle `config.json` in `src/cabi/config/fast_path_config.cpp` and consumed by `src/cabi/api/trtf_c.cpp`, `src/cabi/bundle/bundle_helpers.cpp`, and multiple runtime backends under `src/runtime/trt/`.

---

## Run Phase: C++ Runtime

### C ABI Entry Point (`trtf_c.cpp`)

Users access the library through `extern "C"` factory functions `trtf_create_pipeline()` and `trtf_create_pipeline_ex()` which return an `IPipeline*`. `trtf_c.cpp` is the orchestration layer (input validation, bundle loading, dispatch context creation, error handling), while `pipeline_impl.cpp` provides the concrete `IPipeline` behavior after a strategy factory assembles backend components.

`trtf_create_pipeline_ex()` accepts a `TrtfPipelineOptions` struct:

```cpp
struct TrtfPipelineOptions {
    int max_new_tokens;           // 0 = use model default (20)
    const char* hf_python;        // nullptr = auto-detect
    const char* image_path;       // nullptr = text-only
};
```

### Bundle Loading Flow

```
trtf_create_pipeline_ex("model.trtfb", &options)
  |
  +-- IsBundle() validation
  +-- try_create_from_bundle():
  |     +-- ReadBundleFile() + find_bundle_sections()
  |     +-- Parse config early for runtime_strategy
  |     +-- if runtime_strategy == "diffusion":
  |     |     create_diffusion_pipeline(...)
  |     +-- else:
  |           deserialize primary engine_plan + create execution context
  |           parse FastPathModelConfig
  |           register_builtin_backend_factories_once()
  |           build BackendRegistryDispatchContext
  |           try_create_pipeline_from_registry(strategy, &ctx)
  |           fallback -> create_decoder_pipeline(...)
  +-- apply options.max_new_tokens via set_default_max_new_tokens()
  +-- return assembled PipelineImpl-backed IPipeline*
```

### Runtime Strategy Dispatch

For non-diffusion strategies, dispatch is registry-driven (`backend_registry.cpp`, `backend_registry_dispatch.cpp`, `backend_registry_strategy_plugins_*.cpp`, and `backend_registry_strategy_wrappers.cpp`). Decoder-style text backends implement `IGenerationBackend`; non-text strategies are exposed through task-specific `IPipeline` methods (`segment`, `detect`, `transcribe`, `generate_audio`, `embed`, `rerank`, `solve`, `generate_video`, etc.).

```cpp
class IGenerationBackend {
    virtual bool is_available() const = 0;
    virtual const char* name() const = 0;
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& input_ids,
                                           const GenerationConfig& config) = 0;
};
```

| Runtime strategy | Factory module | Backend name(s) |
|------------------|----------------|-----------------|
| `decoder_kv_cache`, `decoder_moe` | `factories_text.cpp` | `trt` |
| `ssm_recurrent` | `factories_text.cpp` | `trt_mamba` |
| `rwkv_recurrent` | `factories_text.cpp` | `trt_rwkv` |
| `hybrid_mamba_attention` | `factories_text.cpp` | `trt_hybrid` |
| `vision_language` | `factories_multimodal.cpp` | `trt_vl` |
| `speech_to_text` | `factories_multimodal.cpp` | `trt_whisper` |
| `text_to_audio` | `factories_audio.cpp` | `trt_bark` or `trt_magpie_tts` |
| `speech_to_speech` | `factories_audio.cpp` | `trt_speech` |
| `omni_multimodal` | `factories_audio.cpp` | `trt_omni` |
| `segmentation`, `object_detection`, `prompted_segmentation`, `neural_operator` | `factories_vision.cpp` | `trt_segmentation`, `trt_detection`, `trt_sam`, `trt_neural_operator` |
| `encoder_only`, `embedding`, `reranking` | `factories_encoder.cpp` | `trt_encoder`, `trt_embedding`, `trt_reranking` |
| `diffusion` | `factories_diffusion.cpp` | `trt_diffusion` |

`diffusion` is handled as a direct early branch before primary engine-plan deserialization; the rest go through registry dispatch.

### VL Image Preprocessing (`image_preprocessor.cpp`)

For vision-language models, the C++ runtime preprocesses images before passing pixel values to the vision TRT engine. The preprocessing strategy is configurable via `preprocessor_type` in the bundle's config.json:

| Strategy | Output Layout | Description |
|----------|--------------|-------------|
| `qwen_merge_group` | `[C*T, H, W]` | Merge-group patch permutation + temporal duplication (Qwen2.5-VL) |
| `simple_chw` | `[C, H, W]` | Standard resize + normalize (LLaVA, InternVL, Phi-3-Vision) |
| `center_crop_chw` | `[C, H, W]` | Center-crop to square, then resize + normalize (CLIP, DINOv2-based) |
| `aspect_preserve_chw` | `[C, H, W]` | Aspect-ratio-preserving resize + zero-pad (InternVL v2) |

Interpolation mode is configurable via the `interpolation` field: `"bicubic"` (default, matches PIL BICUBIC), `"bilinear"`, or `"nearest"`. The C++ implementation uses stb_image_resize2 filters; the Python debug runner uses PIL constants. The interpolation mode is read from `config.json` (set by the engine builder), with a fallback to the HF `preprocessor_config.json` `resample` integer (PIL enum: 0=NEAREST, 2=BILINEAR, 3=BICUBIC).

Unknown `preprocessor_type` values emit a warning and fall back to `qwen_merge_group`.

Only single-image input is currently supported.

---

## CLI Split

### Python CLI: `trtf-build`

```bash
trtf-build build Qwen/Qwen3-0.6B -o output.trtfb [--max-cache-length N]  # HF repo ID
trtf-build build <local-model-dir> -o output.trtfb [--max-cache-length N]  # local dir
trtf-build inspect <bundle.trtfb>
trtf-build version
```

### C++ CLI: `trtf`

```bash
trtf run <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--hf-python PATH]
trtf encode <bundle.trtfb> --prompt "text" [--hf-python PATH]
trtf segment <bundle.trtfb> --image INPUT --output MASK.png [--hf-python PATH]
trtf segment-sam <bundle.trtfb> --image INPUT --output DIR [--point-x 0.5] [--point-y 0.5] [--background]
trtf detect <bundle.trtfb> --image INPUT --output DETECTIONS.json [--threshold 0.5] [--hf-python PATH]
trtf embed <bundle.trtfb> --prompt "text" [--hf-python PATH]
trtf rerank <bundle.trtfb> --prompt "query" --document "text" [--hf-python PATH]
trtf transcribe <bundle.trtfb> --audio INPUT.wav [--max-new-tokens N] [--hf-python PATH]
trtf generate-audio <bundle.trtfb> --prompt "text" --output OUTPUT.wav [--max-new-tokens N] [--hf-python PATH]
trtf speak <bundle.trtfb> --audio-in INPUT.wav --audio-out OUTPUT.wav [--max-new-tokens N] [--tail-frames N]
trtf solve <bundle.trtfb> --branch-input "..." --trunk-input "..."   # DeepONet
trtf solve <bundle.trtfb> --field-input "..."                         # FNO
trtf generate-video <bundle.trtfb> --prompt "text" --output DIR [--num-steps N] [--guidance-scale S] [--hf-python PATH]
trtf inspect <bundle.trtfb>
trtf version
```

---

## Key Design Decisions

### Why split Python/C++?

The TensorRT C++ API for network construction is verbose and error-prone. The Python API is more ergonomic for graph building, checkpoint loading (via the `safetensors` library and NumPy), and iterating on new model families. Meanwhile, C++ excels at the runtime: engine deserialization, CUDA memory management, and tight GPU execution loops across strategies (autoregressive, diffusion, vision, and speech). The bundle format bridges the two cleanly.

### Why no ONNX?

ONNX export introduces an intermediate representation that limits control over the TensorRT graph. By building `INetworkDefinition` directly via the TensorRT Python API, we get:
- Exact control over layer fusion and memory layout
- No ONNX parser dependency
- Reusable op primitives shared across all families
- Easier debugging of the graph structure

### Why bundles?

`.trtfb` bundles are self-contained artifacts: compiled TRT engine plan + tokenizer config. Benefits:
- **Build once, run anywhere** (same GPU architecture): no Python needed at runtime
- **Instant startup**: engine deserialization (~5s) vs. full build (~60-300s)
- **Deployment simplicity**: single file, no model directory needed
- **Self-describing**: all runtime behavior is determined by the bundle's JSON header — the C++ runtime needs no external configuration

### Bundle Config: Self-Describing Runtime Behavior

The `.trtfb` bundle embeds a JSON header that captures all build-time decisions
the C++ runtime needs. This ensures the runtime always behaves consistently
with how the engine was built, without guessing or relying on external state.

**Format**: 8-byte magic (`TRTFB\x00\x01\x00`) + 8-byte header length + JSON + binary sections.

**Key config fields** (written at build time, consumed at runtime):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `runtime_strategy` | string | `"decoder_kv_cache"` | Selects C++ backend (decoder, mamba, VL, diffusion, etc.) |
| `max_cache_length` | int | 256 | KV cache size baked into TRT engine tensor shapes |
| `vocab_size` | int | — | Output logits dimension |
| `hidden_size` | int | — | Attention hidden dimension |
| `num_layers` | int | — | Number of decoder/encoder layers |
| `num_attention_heads` | int | — | Number of query heads |
| `num_key_value_heads` | int | — | Number of KV heads (GQA) |
| `tokenizer_add_special_tokens` | int (0/1) | absent | Whether the tokenizer should add BOS/EOS when encoding prompts |

**Tokenizer special tokens**: At build time, the Python builder detects
whether the HF tokenizer adds special tokens by default (checks
`tokenizer_config.json` for `add_bos_token`, falls back to comparing
`encode()` output with and without special tokens). The result is stored as
`tokenizer_add_special_tokens: 0` or `1` in the bundle. At runtime, the C++
binary reads this field and passes it to the HF tokenizer bridge. If the field
is absent (old bundles), the runtime defaults to `true` to match HF's
`tokenizer.encode()` default behavior.

This design ensures:
- **Parity**: C++ binary tokenizes prompts the same way as HF Transformers
- **Per-model control**: Models that explicitly don't want BOS (e.g., GPT-2) set `0`
- **Backward compat**: Old bundles without the field use the safe default (`true`)

**Strategy-specific fields** are parsed conditionally based on `runtime_strategy`:
- SSM: `d_inner`, `conv_kernel`, `state_size`, `n_ssm_layers`
- VL: `vision_output_dim`, `fixed_image_size`, `preprocessor_type`, `vl_prompt_template`
- Speech: `num_mel_bins`, `max_source_positions`, `max_target_positions`
- Bark: `semantic_vocab_size`, `coarse_max_cache_length`, `sample_rate`
- Diffusion: `scheduler`, `num_inference_steps`, `guidance_scale`, `video_*` dimensions
- Segmentation: `num_classes`, `input_image_h/w`, `seg_image_mean/std`
- **Clean separation**: Python handles the complex build, C++ handles the fast runtime
