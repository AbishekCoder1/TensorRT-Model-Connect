# Architecture Overview

## Python Builds, C++ Runs

The system is split into two phases across two languages:

| Phase | Language | Tool | Input | Output |
|-------|----------|------|-------|--------|
| **Build** | Python | `trtf-build` / `trtf_build.build()` | HF repo ID or local directory | `.trtfb` bundle |
| **Run** | C++ | `trtf` / C ABI | `.trtfb` bundle | Generated text |

**Build phase** (Python `trtf_build/` package — accepts HF repo IDs or local directories):
1. Resolve model: auto-download from HuggingFace Hub if needed
2. Parse `config.json` to identify model family and architecture
3. Load safetensors weights via the `safetensors` Python library
4. Map HF tensor keys to canonical format (per-family checkpoint mapper)
5. Build TRT `INetworkDefinition` via the TensorRT Python API
6. Compile to `ICudaEngine` (GPU kernel compilation)
7. Package engine plan + tokenizer files into a `.trtfb` bundle

**Run phase** (C++ runtime, ~18 source files):
1. Load `.trtfb` bundle, deserialize TRT engine plan
2. Create tokenizer (HfPythonTokenizer or VocabTokenizer)
3. For VL bundles: load and preprocess image via `image_preprocessor` (4 strategies + configurable interpolation)
4. Run autoregressive generation loop on GPU with KV-cache management
5. Return generated text via C ABI

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

Each model family is a single Python file in `trtf_build/trtf_build/families/`:

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

### C++ Runtime: Dispatch Architecture

The C++ runtime uses a thin dispatch pattern with shared helpers and per-strategy factory functions:

```
trtf_c.cpp              Thin dispatch on runtime_strategy
    |
bundle_helpers.{h,cpp}  Shared plumbing (tokenizer extraction, engine init)
    |
Per-backend factories    create_decoder_pipeline / create_mamba_pipeline / create_vl_pipeline
    |
Backend implementations  TrtBackendFastPath / MambaBackend / VLBackendFastPath
```

**`bundle_helpers.{h,cpp}`**: Shared plumbing — bundle section discovery (`find_bundle_sections`), tokenizer file extraction to temp dir (`extract_tokenizer_from_bundle`), and `DecoderStepEngine` initialization from config (`make_decoder_engine`). Backend-agnostic.

**Per-strategy factory functions**: Live in `trtf_c.cpp`. Each factory takes the shared components (TRT engine, config, sections) and returns a fully assembled `PipelineImpl`. The strategy-specific setup (Mamba state tensors, VL vision engine, etc.) is encapsulated in each factory.

**`trtf_c.cpp`**: Thin dispatch on `runtime_strategy` string (~200 lines). Adding a new strategy = one new factory function + one new `if` line in the dispatch.

**How to add a new runtime strategy:**
1. Create `new_backend.{h,cpp}` implementing `IGenerationBackend`
2. Add a `create_new_pipeline()` factory in `trtf_c.cpp`
3. Add one `if (strategy == "new_strategy")` line in `try_create_from_bundle()`
4. Add config fields to `FastPathModelConfig` if needed

Design rationale: factory functions instead of a full registry (simpler for 3-5 backends, no self-registration macro overhead, explicit > implicit).

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
| `FastPathModelConfig` | Not needed (Python builds, C++ only loads bundles) |
| `tensor_math.h/cpp` | NumPy operations in Python |

---

## Run Phase: C++ Runtime

### C ABI Entry Point (`trtf_c.cpp`)

Users access the library through `extern "C"` factory functions `trtf_create_pipeline()` and `trtf_create_pipeline_ex()` which return an `IPipeline*`. These functions accept `.trtfb` bundle paths. The factory loads the pre-compiled engine plan, creates a tokenizer, and returns a ready-to-use pipeline.

`trtf_create_pipeline_ex()` accepts a `TrtfPipelineOptions` struct:

```cpp
struct TrtfPipelineOptions {
    int max_new_tokens;           // 0 = use model default (20)
    const char* hf_python;        // nullptr = auto-detect
};
```

### Bundle Loading Flow

```
trtf_create_pipeline("model.trtfb", TRTF_FORCE_TRT)
  |
  +-- Detect bundle magic bytes
  +-- ReadBundleFile() -> engine plan bytes + tokenizer files
  +-- TRT deserializeCudaEngine(plan)
  +-- Parse runtime_strategy from config.json
  +-- Dispatch based on strategy:
  |     "decoder_kv_cache" / "decoder_moe" -> TrtBackendFastPath
  |     "ssm_recurrent" -> MambaBackend
  |     "vision_language" -> VL pipeline (vision + text decoders)
  +-- For VL bundles: parse VLPreprocessConfig (preprocessor_type, interpolation, etc.)
  +-- Extract tokenizer files to temp dir
  +-- Create HfPythonTokenizer or VocabTokenizer
  +-- Return PipelineImpl
```

### Runtime Strategy Dispatch

The C++ runtime dispatches to the correct backend based on the `runtime_strategy` field in the bundle's config.json. All backends implement `IGenerationBackend`:

```cpp
class IGenerationBackend {
    virtual bool is_available() const = 0;
    virtual const char* name() const = 0;
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& input_ids,
                                           const GenerationConfig& config) = 0;
};
```

**TRT Backend (KV Cache)** (`trt_backend_shared.cpp`):
- **Strategies**: `decoder_kv_cache`, `decoder_moe`
- **Name**: `"trt"`
- **How it works**: Deserializes a pre-compiled TRT engine from a `.trtfb` bundle, then runs an autoregressive generation loop on GPU with KV-cache management, CUDA memory allocation, and greedy argmax sampling. MoE routing is handled entirely within the TRT graph, so both strategies use the same C++ backend.

**Mamba Backend (SSM)** (`mamba_backend.cpp`):
- **Strategy**: `ssm_recurrent`
- **Name**: `"trt-mamba"`
- **How it works**: Loads a Mamba TRT engine from a `.trtfb` bundle, then runs an autoregressive loop with conv_state and ssm_state per layer (constant memory, no cache growth). Uses `MambaStepState` instead of `DeviceKvCache`.

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
trtf inspect <bundle.trtfb>
trtf version
```

---

## Key Design Decisions

### Why split Python/C++?

The TensorRT C++ API for network construction is verbose and error-prone. The Python API is more ergonomic for graph building, checkpoint loading (via the `safetensors` library and NumPy), and iterating on new model families. Meanwhile, C++ excels at the runtime: engine deserialization, CUDA memory management, and the tight autoregressive loop. The bundle format bridges the two cleanly.

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
- **Clean separation**: Python handles the complex build, C++ handles the fast runtime
