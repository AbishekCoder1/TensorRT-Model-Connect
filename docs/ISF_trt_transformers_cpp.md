# ISF Document

**Title**: "An Agentic, Infinitely-Scalable System for Bridging HuggingFace Models to TensorRT Inference at Speed of Light"

---

## 4. What is the problem you are trying to solve? If others have tried to solve this problem previously, please summarize what you know of those attempted solutions and their shortcomings.

### The Core Problem: The HuggingFace-to-TensorRT Gap

The HuggingFace (HF) Hub hosts thousands of large language models (LLMs) and vision-language models (VLMs) across hundreds of model families. TensorRT (TRT) delivers the fastest GPU inference for these models — often 2-5x faster than PyTorch. However, **bridging any given HF model to TRT inference today requires significant manual engineering**, creating a bottleneck that prevents organizations from deploying the best available models at optimal speed.

The ideal system would allow an AI agent (or human engineer) to take any HF model and produce a production-ready TRT inference pipeline with minimal effort — ideally just a single command — while achieving the theoretical maximum inference throughput on the hardware. Furthermore, the system should scale to accommodate new model families as they appear on the Hub, without requiring changes to any shared infrastructure.

### Problem 1: Manual, Per-Family TRT Integration is O(N) and Unscalable

Each HF model family (Qwen, LLaMA, Mistral, Gemma, Phi, Mamba, etc.) has its own unique combination of:
- Normalization type (RMSNorm vs LayerNorm)
- MLP structure (SwiGLU vs GELU FC vs MoE routing)
- Positional encoding (RoPE vs learned vs ALiBi, with variants: standard vs interleaved, full vs partial rotary)
- Attention pattern (MHA vs GQA vs MQA, sequential vs parallel residuals)
- Weight storage format (separate Q/K/V vs fused QKV, separate gate/up vs fused gate_up)
- Architectural class (standard decoder, MoE, SSM/Mamba, vision-language)

Prior approaches to bridging HF models to TRT:

- **TensorRT-LLM**: Provides TRT support for a curated set of models, but each model requires hand-written C++ graph construction code. Adding a new family requires deep TRT expertise, takes weeks, and changes are tightly coupled to a specific TRT version. The codebase uses a monolithic builder with deeply nested conditionals. An AI agent cannot productively work in this environment because the blast radius of any change is uncontrollable.

- **ONNX export + TRT conversion**: HF models can be exported to ONNX and converted to TRT engines. However, the ONNX intermediate representation is lossy (dynamic shapes are poorly supported, operator coverage is incomplete), the conversion adds a fragile step, and the resulting engines are often suboptimal because TRT cannot fuse operations across ONNX boundaries.

- **torch.compile + TRT backend**: PyTorch 2.x's `torch.compile` with the TensorRT backend (`torch_tensorrt`) can JIT-compile PyTorch models. However, compilation is slow (minutes per model), not all operations are supported (frequent graph breaks), and the resulting artifacts are not portable across machines — they depend on the exact PyTorch and CUDA version.

- **Manual per-model scripts**: Many teams write bespoke conversion scripts per model. This is O(N) in the number of model families, requires deep expertise in both HF internals and TRT APIs, and produces unmaintainable one-off artifacts.

None of these approaches are **agentic-ready**: they cannot be driven by an AI agent that reads a model's `config.json`, generates the bridge code, validates correctness, and produces a deployable artifact — all in a single automated pipeline.

### Problem 2: Inference Runtime is Not Separated from Build Concerns

Existing TRT inference systems (TensorRT-LLM, vLLM with TRT backend, Triton Inference Server) typically combine model weight loading, TRT engine construction, and autoregressive inference into a single monolithic runtime. This means:

- The runtime must link against safetensors/PyTorch/HuggingFace libraries, creating a large, fragile deployment footprint.
- Cold start requires re-parsing model checkpoints and potentially rebuilding TRT engines.
- The runtime must handle all model families' weight formats, making it impossible to keep the runtime minimal and fast.
- Security exposure: loading models from untrusted sources may require executing arbitrary Python code (`trust_remote_code`).

A clean separation between build-time (Python, with full HF/TRT ecosystem) and runtime (minimal C++, no dependencies) is needed so that the agentic build system can iterate freely without affecting the deployed runtime.

### Problem 3: KV Cache PCIe Bottleneck Prevents Speed-of-Light Inference

Standard KV cache implementations transfer cache state between host (CPU) and device (GPU) memory at every decode step. For a 32-layer model with 4096-dimensional attention at `max_cache_length=2048`, this is approximately 2 GB of PCIe round-trip transfer per step — the dominant bottleneck that prevents achieving speed-of-light throughput. Even with PagedAttention (vLLM) or inflight batching (TensorRT-LLM), the host-managed cache paradigm requires PCIe traffic for cache state.

### Problem 4: No Automated Correctness Guarantee Across the HF-to-TRT Bridge

When an agent (human or AI) creates a new model family bridge, how do you know the TRT engine produces the same tokens as the original HF model? Prior approaches use:
- **Golden file comparison**: Brittle — any change invalidates all golden files.
- **Integration tests only**: Catches end-to-end failures but cannot localize the bug.
- **Manual inspection**: Does not scale to hundreds of models.

What's needed is a **multi-tier automated validation pipeline** that an agent can invoke after generating a new plugin to get immediate, actionable feedback on correctness at every level — from individual TRT operations through per-layer hidden states to full autoregressive generation.

### Problem 5: Vision-Language Models Require Architectural Innovations for TRT

VL models (Qwen2.5-VL, Qwen3-VL, LLaVA, InternVL) inject visual features into the text decoder, but TRT compiles a single static graph with no conditional branching. The Qwen3-VL "DeepStack" architecture introduces multi-level vision features from intermediate ViT layers that must be injected at specific early decoder layers. No prior system handles this within a single static TRT engine.

---

## 5. How does your invention solve the problem described in the previous question? Be specific.

This invention introduces an **agentic, infinitely-scalable system** that bridges any HuggingFace model to speed-of-light TensorRT inference. The system is designed so that adding support for a new model family is a single-file operation that an AI agent can perform autonomously, with automated validation providing immediate correctness guarantees. The system comprises six interconnected innovations.

### Innovation 1: Zero-Registration Auto-Discovery Plugin Architecture for Infinite Scalability

The core of the system's scalability is a plugin architecture designed for agentic extension. Adding a new HF model family to TRT requires **creating a single Python file** — no edits to any shared code, no registration, no configuration changes, no C++ modifications.

**Auto-discovery mechanism**: At import time, `pkgutil.iter_modules()` scans the `families/` directory for `.py` files with a module-level `plugin` attribute:

```python
for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    if _name.startswith("_") or _name == "base":
        continue
    _mod = importlib.import_module(f"{__name__}.{_name}")
    _plugin = getattr(_mod, "plugin", None)
    if _plugin is not None:
        _ALL_PLUGINS.append(_plugin)
```

**Structural typing protocol**: Plugins implement the `FamilyPlugin` protocol via Python's `typing.Protocol` — no inheritance required, no base class import needed. A minimal plugin that bridges a standard-architecture HF model to TRT is ~30 lines:

```python
class QwenPlugin:
    name = "qwen"
    def matches(self, model_type): return mt.startswith("qwen") and "vl" not in mt
    def load_weights(self, model_dir, config): return load_standard_weights(model_dir, config)
    def build_engine(self, config, weights, max_cache_length, *, verbose=False):
        return build_standard_decoder_engine(config, weights, max_cache_length, verbose=verbose)
plugin = QwenPlugin()
```

**Four levels of customization depth** — an agent selects the minimal level needed:

1. **Pure delegation** (~30 lines): Plugin maps model_type and delegates to shared functions. Covers architecturally-standard models (Qwen, LLaMA, Mistral, OLMo). An AI agent can generate these from `config.json` alone.
2. **Post-load weight transformation** (~50 lines): Standard loading + family-specific math on weights. Covers models with unusual normalization conventions (Gemma: +1.0 to RMSNorm gamma, sqrt(hidden) embed scale).
3. **Custom weight loading, standard graph** (~100 lines): Handles non-standard weight tensor layouts (Phi-3: fused QKV and gate_up projections that must be split) while reusing the standard graph builder.
4. **Fully custom graph** (~300 lines): Complete TRT graph construction for fundamentally different architectures (Mamba/SSM, PhiMoE SparseMixer routing). Even at this level, the plugin reuses Layer 1 and Layer 2 building blocks.

**Agentic scaffolding**: The `scripts/new_family.py` tool downloads a model's `config.json`, detects architectural features (GQA, MoE, LayerNorm, tied embeddings, fused weights), and generates a working plugin with appropriate warnings:

```bash
python3 scripts/new_family.py --hf-repo microsoft/Phi-3-mini-4k-instruct --family-name phi
```

**Agentic validation gate**: After creating a plugin, the agent runs a single command:

```bash
./scripts/validate_family.sh Qwen/Qwen3-0.6B
```

This executes four validation stages in sequence — build bundle, diff_logits (battery of 4 prompts), diff_layers (per-layer hidden state comparison), and test_runner_parity (Python vs C++ exact token match) — and prints a structured pass/fail summary. All four stages run even if earlier ones fail, giving complete diagnostic information in a single invocation.

**Security boundary**: The build pipeline never downloads or executes `*.py` files from HF repos. `snapshot_download` uses an explicit allowlist that excludes executable code. No `trust_remote_code` path exists in the engine builder.

**Current coverage**: 24+ model families bridged to TRT via this system: Qwen, Qwen VL (Qwen2.5-VL + Qwen3-VL DeepStack), LLaMA, Mistral, Gemma, Phi-3, Phi-MoE, Mamba, Mixtral, GPT-2, GPT-Neo, GPT-NeoX, OPT, BLOOM, Falcon, StableLM, StarCoder2, InternLM, Granite, OLMo, Nemotron, XGLM, CodeGen.

### Innovation 2: Three-Layer Composable Graph Builder Stack

The plugin system's power comes from a three-layer architecture that decomposes the combinatorial space of LLM architectures into orthogonal, reusable components. This is what makes agentic extension tractable — an agent working on a new family can compose from existing building blocks rather than constructing TRT graphs from scratch.

**Layer 1 — `graph_ops.py` (Atomic TRT Operations)**: Pure tensor-in/tensor-out functions. No model configuration, no weight naming, no side effects. Every function takes a TRT network + input tensors + NumPy weight arrays and returns an output tensor. Operations include:

- `add_rms_norm`: Decomposed into 7 standard TRT layers (no native RMSNorm — full TRT graph optimization)
- `add_apply_rope`: RoPE via sparse matrix multiply (`rotate_half(x) = x @ M`), with pre-computed cos/sin tables baked as TRT constants. Supports both standard and interleaved modes, partial rotary factor.
- `add_self_attention_block` / `add_windowed_self_attention_with_rope`: Vision encoder attention variants
- `add_patch_embed_3d`: 3D patch embedding as Conv2D over flattened temporal-channel dimension

**Layer 2 — `graph_blocks.py` (Weight-Aware Composable Blocks)**: Each function accepts a `weights: WeightDict` and a `prefix: str` for weight lookups. The critical design principle: **blocks do not apply residual connections**. `add_attention_block` returns `{"normed", "attn_out", "present_k", "present_v"}` — the caller composes the residual pattern. This single design decision enables the same attention block to handle:
- Sequential residuals (standard decoders)
- Parallel residuals (GPT-J/Falcon)
- DeepStack injection points (Qwen3-VL, where features are injected between attention and MLP)

Optional features are weight-driven: if `q_bias` weights exist, bias is added; if `q_norm` weights exist, per-head RMSNorm is applied. No configuration flags needed — the weights ARE the configuration.

**Layer 3 — `standard_decoder_builder.py` + Family Plugins**: Full engine assembly parameterized by: `norm_type`, `mlp_type`, `position_type`, `activation`, `partial_rotary_factor`, `interleaved_rope`, `parallel_residual`, `scale_attn_weights`, `embed_input`, `debug_layer_outputs`. The parameter space covers the full combinatorial space of known decoder architectures.

**Why this enables agentic scaling**: An AI agent examining a new model's `config.json` can determine which Layer 3 parameters to set. For standard decoders, the agent writes a ~30-line plugin. For models with unusual weight layouts, the agent writes a custom `load_weights()` (~100 lines) while reusing the standard builder. Only for fundamentally new architectures (like Mamba) does the agent need to construct a custom graph — and even then, it composes from Layer 1 and Layer 2 blocks. The validation pipeline gives the agent immediate feedback on correctness.

### Innovation 3: Two-Stage Bundle Architecture with Engine-Introspection-Driven Feature Detection

The system cleanly separates build-time concerns (Python, with full HF/TRT ecosystem) from runtime concerns (minimal C++, no dependencies) via a portable binary artifact: the `.trtfb` bundle.

**Stage 1 — Python Build** (`trtf_build/`): One-command bridge from HF to TRT:

```bash
trtf-build build Qwen/Qwen3-0.6B -o qwen3.trtfb
```

This auto-downloads from HF Hub, parses config, selects the matching plugin, loads and transforms weights, builds TRT engine(s), packages everything into a self-contained `.trtfb` bundle. The bundle contains: serialized TRT engine plan(s), config.json augmented with runtime metadata, tokenizer artifacts, and preprocessor configuration.

**Bundle format**: Binary layout with 8-byte magic, uint64 JSON header with section index (name -> {offset, size}), and concatenated binary sections. Any section is located with a single file seek. int64 section offsets support engine plans exceeding 2 GB.

**Stage 2 — C++ Runtime**: Loads the `.trtfb` bundle and runs autoregressive inference. Zero dependency on Python, PyTorch, safetensors, or HuggingFace. The runtime binary is minimal (~13 source files) and deployment-ready.

**Engine-Introspection-Driven Feature Detection**: The C++ runtime discovers model capabilities by probing the TRT engine's IO tensor registry — not by reading configuration fields:

```cpp
// Auto-detect VL embedding support
d_input_embed(has_io_tensor(*engine.engine, "input_embed") ? embed_bytes : 0)

// Auto-detect DeepStack levels (0, 1, 2, 3, ...)
for (int32_t i = 0; ; ++i) {
    std::string name = "deepstack_embed_" + std::to_string(i);
    if (!has_io_tensor(*engine.engine, name)) break;
    d_deepstack_embeds.emplace_back(embed_bytes);
}
```

A text-only decoder, a standard VL model, and a Qwen3-VL with 3 DeepStack levels execute through identical C++ code. The engine IS the interface contract. This means **an agent can add new capabilities to the Python builder without ever touching the C++ runtime** — as long as the new capabilities are expressed through named engine tensors, the runtime adapts automatically.

### Innovation 4: Device-Resident KV Cache for Speed-of-Light Inference

The `DeviceKvCache` achieves theoretical maximum inference throughput by eliminating PCIe transfers for the KV cache — the largest data structure in autoregressive inference.

**Architecture**: `2 * num_layers` persistent GPU memory allocations, each sized to `max_cache_length * attention_size * sizeof(float)`. Allocated once at generation start. No per-step `cudaMalloc` or `cudaFree`.

**Per-step operation** (`run_decoder_step_device`) — all on a single CUDA stream:
1. CPU: compute position_id (one integer min) and attention mask (~1 KB) — the only CPU work
2. H2D async: token_id (4B) + position_id (4B) + mask (~1KB) — the only PCIe transfers
3. TRT engine bind: persistent cache device pointers bound directly (zero-copy)
4. `enqueueV3(stream)`: TRT forward pass
5. D2D async: cache update — append during fill, shift-and-append during eviction
6. D2H async: logits (vocab_size * 4B) — the only result back
7. Single `cudaStreamSynchronize`

The KV cache (potentially hundreds of MB) **never crosses the PCIe bus**. Only ~2 KB of control inputs go H2D per step.

**Sliding-window eviction**:
```
if (cache_length < max_cache_length):
    cudaMemcpyAsync(cache + offset, present, row_bytes, D2D, stream)  // append
else:
    cudaMemcpyAsync(cache, cache + row_bytes, (max-1) * row_bytes, D2D, stream)  // shift left
    cudaMemcpyAsync(cache + tail, present, row_bytes, D2D, stream)  // append at end
```

**Pre-allocated per-step I/O** (`DeviceResources`): Every tensor touched per step is allocated once, driven by engine introspection. This includes auto-detected DeepStack and VL buffers.

**Position clamping**: Once the cache fills, position_id is clamped to `max_cache_length` to prevent out-of-bounds RoPE table indexing. The RoPE tables are baked into the TRT engine as constants at build time.

### Innovation 5: Branchless Soft-Switch for Vision-Language Models

The system bridges VL models to TRT without conditional branching, using scalar-multiplication gating that produces exact arithmetic zeros:

**VL embedding blend** (single static TRT graph, no branching):
```python
hidden = (1 - use_input_embed) * token_embed + use_input_embed * input_embed
```
When `flag=0.0`: `hidden = token_embed`. When `flag=1.0`: `hidden = input_embed`.

**DeepStack multi-level injection** at early decoder layers:
```python
post_attn = post_attn + deepstack_embed[layer_idx] * deepstack_active
```
When `deepstack_active=0.0`, the runtime `cudaMemset`s embed buffers to zero — product is exactly 0.0, sum is a no-op. One TRT engine for all inference phases.

**Merge-group image preprocessing**: Rather than adding a GPU-side gather/scatter to reorder patches, the preprocessor physically reorders input pixels on the CPU so the Conv2D naturally produces patches in merge-group order. Three-way coordinate system agreement (preprocessor layout, Conv2D output order, position embedding order) eliminates any in-engine reordering.

### Innovation 6: Multi-Tier Automated Validation Pipeline for Agentic Correctness

The system provides **immediate, multi-level correctness feedback** that an AI agent can use to iterate on a new model family bridge. The validation architecture has five independent planes, each catching a different class of error:

**The Mirror Runner**: A pure-Python TRT inference runner (`TrtRunner`) maintains exact algorithmic parity with the C++ runtime — identical mask construction, position clamping, cache eviction, VL gating, DeepStack injection. This is not an approximation — it is a specification document written as executable code.

**Plane 1 — Atomic op unit tests** (`test_graph_ops.py`): Each `graph_ops.py` function tested against HF/PyTorch reference implementations at `atol=1e-5`. An agent modifying Layer 1 gets immediate feedback on numerical correctness.

**Plane 2 — Per-layer hidden state comparison** (`diff_layers.py`): Debug engine with intermediate outputs compared against HF `output_hidden_states=True` at `atol=0.05`. An agent whose plugin has a weight mapping bug sees exactly which layer diverges.

**Plane 3 — Full autoregressive logit trajectory** (`diff_logits.py`): Four "battery" prompts (factual, reasoning, code, multi-turn) stress different token distributions. Per-step logit comparison at `atol=1e-3`. An agent whose plugin produces correct single-step outputs but has a cache bug sees divergence at the step where it manifests.

**Plane 4 — Token-level Python vs C++ parity** (`test_runner_parity.py`): Exact string equality of generated text between Python mirror and C++ binary. Because autoregressive generation amplifies any single-token divergence, this catches every component simultaneously.

**Plane 5 — VL pipeline validation** (`diff_vl.py`): Vision feature cosine similarity, embed_input gate test, full VL generation, and C++ binary parity — with GPU memory isolation (TRT freed before HF loads to prevent OOM).

**The agentic workflow**: An AI agent:
1. Reads a model's `config.json` and determines the architectural delta from standard
2. Runs `new_family.py` to scaffold a plugin (or writes one from scratch)
3. Runs `validate_family.sh` — gets immediate pass/fail with localized diagnostics
4. Iterates on the plugin based on which validation plane failed
5. Upon all-pass, the model is permanently bridged to TRT inference

The entire cycle — from "new model appears on HF Hub" to "production-ready TRT inference" — can be completed by an AI agent in a single session.

---

## 6. What are the specific technical differences (not benefits) between the solution described in the previous question and prior solutions?

### Difference 1: Single-File Agentic Extension vs. Multi-File Manual Integration

**Prior art** (TensorRT-LLM): Adding a new model family requires editing multiple C++ files across the codebase: model definition, weight loader, graph builder, config parser, and registration code. Typical new-model PRs touch 10-20 files.

**This invention**: Adding a new model family requires creating a single `.py` file in the `families/` directory with a module-level `plugin` attribute. Zero shared files are edited. The plugin is discovered at import time via `pkgutil.iter_modules()`. The `FamilyPlugin` protocol uses structural typing (`typing.Protocol`) — no inheritance, no base class import. Optional capabilities (VL engine building, runtime strategy override) are detected via `getattr(plugin, 'method', None)` probing. The scaffolding script auto-generates a working plugin from a model's `config.json`. The validation script provides a one-command correctness gate. This pipeline is designed for an AI agent to execute autonomously.

### Difference 2: Three-Layer Composable Stack vs. Monolithic Builders or Per-Model Code

**Prior art** (TensorRT-LLM): Single builder with deeply nested conditionals for architectural variations. (HuggingFace): Per-model `modeling_*.py` files with O(N) code duplication.

**This invention**: Three composable layers with strict separation. Layer 1 (graph_ops) is purely tensor-in/tensor-out — no weight naming, no model awareness. Layer 2 (graph_blocks) provides weight-aware blocks that explicitly do NOT apply residual connections — the caller composes the residual pattern, enabling one attention block implementation across sequential residuals, parallel residuals, and DeepStack injection points. Layer 3 (standard_decoder_builder + plugins) assembles full engines through parameterization. An agent working at Layer 3 composes from Layers 1 and 2 without understanding TRT internals. An agent at Layer 4 (fully custom graph) still reuses Layer 1 and 2 blocks.

### Difference 3: Automated Multi-Tier Validation Pipeline vs. Manual Testing

**Prior art**: Correctness validation uses golden-file comparison (brittle) or end-to-end tests (cannot localize bugs). Adding a new model requires manual test authoring.

**This invention**: Five independent validation planes (atomic ops, per-layer hidden states, autoregressive logit trajectories across 4 prompt types, token-level Python/C++ parity, VL pipeline) are invoked by a single `validate_family.sh` command. A pure-Python TRT runner maintains exact algorithmic parity with the C++ runtime, serving as a specification document. The debug engine uses TRT identity layers to expose intermediate hidden states as outputs without affecting fusion. Per-model JSON manifests in `tests/e2e/models/` enable adding a new model to the E2E test suite by creating one JSON file. The validation pipeline is designed to be invoked by an AI agent after each iteration.

### Difference 4: Engine-Introspection Feature Detection vs. Configuration-Driven Dispatch

**Prior art** (TensorRT-LLM, vLLM, Triton): Runtime reads explicit configuration fields or plugin registries to determine model capabilities. Adding a new capability requires modifying the runtime's dispatch logic.

**This invention**: The C++ runtime discovers capabilities by probing TRT engine tensor names (`has_io_tensor(*engine, "deepstack_embed_0")`). The engine's IO tensor table IS the interface contract. A text-only model, a standard VL model, and a Qwen3-VL with 3 DeepStack levels execute through identical C++ code paths. An agent can add new Python-side capabilities (new tensor outputs) without any C++ changes — the runtime adapts automatically via tensor name probing.

### Difference 5: Fully Device-Resident KV Cache vs. Host-Managed Cache

**Prior art** (standard TRT inference): KV cache transferred H2D/D2H each step (~2 GB/step for large models). PagedAttention manages fragmentation but not PCIe elimination.

**This invention**: KV cache allocated once, remains on GPU for entire generation. Per-step PCIe: ~2 KB control inputs H2D, vocab_size*4 bytes logits D2H. Cache update is purely D2D. Complete decode loop on a single CUDA stream with a single `cudaStreamSynchronize`. Pre-allocated I/O buffers driven by engine introspection — including auto-detected DeepStack and VL buffers.

### Difference 6: Bundle-Only Runtime vs. Monolithic Load+Build+Infer

**Prior art**: Runtime links against safetensors/PyTorch/HF libraries and can rebuild engines on the fly.

**This invention**: Complete stage separation. Python build (`trtf_build/`) has full HF/TRT ecosystem access. C++ runtime is bundle-only — cannot load weights, cannot build engines, has zero dependency on Python/PyTorch/safetensors. Connected by a portable `.trtfb` binary artifact with int64 section offsets. The build stage is where the agent operates; the runtime is a fixed, minimal, deploy-anywhere binary.

### Difference 7: Branchless Scalar-Gate VL Integration vs. Separate Engines or Dynamic Shapes

**Prior art**: Separate engine compilation for text-only and VL-prefill modes, or dynamic shape support with optimization limitations.

**This invention**: Single static TRT engine for all phases. Mode selection via scalar multiplication: `hidden = (1 - flag) * token_embed + flag * input_embed`. DeepStack injection: `post_attn = post_attn + embed * active_gate`. When gate is 0.0, `cudaMemset` zeros the embed buffers — product is exactly 0.0. TRT compiles and optimizes the full graph. No engine switching, no dynamic shapes. Image preprocessing reorders pixels on CPU so Conv2D naturally produces merge-group-ordered patches, eliminating in-engine gather/scatter.

### Difference 8: Build-Time Weight and Positional Encoding Baking vs. Runtime Computation

**Prior art**: GQA head expansion and RoPE positional encoding computed at inference time.

**This invention**: GQA K/V weights expanded at build time by replicating each KV head's columns. RoPE cos/sin tables pre-computed as NumPy arrays and baked as TRT constants. Rotate-half matrix pre-computed as sparse `[attention_size, attention_size]` matrix. ALiBi cached index tensors baked. Only scalar `position_id` is dynamic at runtime. All positional encoding computation is shifted to build time; the runtime does only trivial integer arithmetic per step.

---

## Additional Notes

### The Agentic Workflow in Practice

The system has been used to bridge 24+ model families to TRT. The typical workflow for an AI agent adding a new family:

| Step | Agent Action | Time |
|------|-------------|------|
| 1 | Read model's `config.json` from HF Hub | Seconds |
| 2 | Run `new_family.py --hf-repo <repo>` to scaffold plugin | Seconds |
| 3 | Examine scaffolded plugin, customize if needed | Minutes |
| 4 | Run `validate_family.sh <repo>` | ~5 min |
| 5 | If validation fails, iterate on plugin based on diagnostic output | Minutes per iteration |
| 6 | All planes pass — model is permanently bridged to TRT | — |

For architecturally-standard models (Level 1 plugins), the scaffolded plugin works without modification. For models with unusual weight layouts (Level 3), the agent typically needs 1-2 iterations. The validation pipeline pinpoints exactly where the issue is — the agent never has to guess.

### Scaling Properties

| Dimension | Scaling Behavior |
|-----------|-----------------|
| New model family | O(1): single file, no shared code changes |
| New architectural feature (e.g., DeepStack) | O(1) in C++ runtime via tensor-name auto-detection |
| New TRT operation | O(1): add to Layer 1, compose in Layer 2/3 |
| New runtime strategy (e.g., Mamba) | O(1): one factory function + one backend class |
| Validation of new model | O(1): one JSON manifest file for E2E suite |
| Agent iterations per family | Typically 1-3 for standard, 3-5 for custom |

### Key Implementation Files

| Component | Files |
|---|---|
| Plugin auto-discovery | `trtf_build/trtf_build/families/__init__.py` |
| Plugin protocol | `trtf_build/trtf_build/families/base.py` |
| Family plugins (24+) | `trtf_build/trtf_build/families/*.py` |
| Plugin scaffolding | `scripts/new_family.py` |
| Validation gate | `scripts/validate_family.sh` |
| Layer 1: atomic ops | `trtf_build/trtf_build/graph_ops.py` |
| Layer 2: composable blocks | `trtf_build/trtf_build/graph_blocks.py` |
| Layer 3: decoder builder | `trtf_build/trtf_build/standard_decoder_builder.py` |
| Engine builder orchestrator | `trtf_build/trtf_build/engine_builder.py` |
| Bundle format (C++ reader) | `src/bundle/bundle_format.{h,cpp}` |
| Bundle format (Python writer) | `trtf_build/trtf_build/bundle_writer.py` |
| Device-resident KV cache | `src/runtime/trt/device_kv_cache.{h,cpp}` |
| Decode step execution | `src/runtime/trt/device_kv_cache.cpp` |
| Engine-introspection dispatch | `src/cabi/trtf_c.cpp`, `src/cabi/bundle_helpers.{h,cpp}` |
| VL vision encoder builder | `trtf_build/trtf_build/qwen_vl_vision_builder.py` |
| Image preprocessor (4 strategies) | `src/runtime/trt/image_preprocessor.{h,cpp}` |
| Mirror runner (Python) | `trtf_build/trtf_build/debug_runner.py` |
| Diff tools | `tools/diff_logits.py`, `tools/diff_layers.py`, `tools/diff_vl.py` |
| Runner parity test | `tools/test_runner_parity.py` |
| Graph op unit tests | `tools/test_graph_ops.py` |
| E2E test framework | `tests/e2e/test_full_pipeline.py`, `tests/e2e/conftest.py` |
| Per-model manifests | `tests/e2e/models/*.json` |
