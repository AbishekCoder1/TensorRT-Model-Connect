# ISF Document

**Title**: "An Agent-Guided Differential Testing Framework for Automated Bridging of HuggingFace Models to TensorRT Inference"

---

## 4. What is the problem you are trying to solve? If others have tried to solve this problem previously, please summarize what you know of those attempted solutions and their shortcomings.

### The Core Problem: No Automated Oracle Exists to Guide an Agent From HF Model to Correct TRT Engine

The HuggingFace (HF) Hub hosts thousands of large language models (LLMs) and vision-language models (VLMs) across hundreds of model families. TensorRT (TRT) delivers the fastest GPU inference — often 2-5x faster than PyTorch. Bridging a new HF model to TRT requires constructing a TRT computation graph that is **numerically equivalent** to the original HF model's forward pass, mapping HF weight tensors into the graph, and verifying that the resulting engine produces identical token-by-token output across diverse inputs.

The fundamental bottleneck is not writing the bridge code — it is **knowing whether the bridge code is correct**. An AI agent (or human engineer) writing a TRT graph for a new model family faces a debugging problem with no automated oracle: if the engine produces wrong tokens, where is the bug? Is it a weight mapping error (wrong tensor transposition, missing GQA expansion)? A graph construction error (wrong normalization, wrong activation function)? A positional encoding error (wrong RoPE mode, wrong frequency computation)? A runtime error (wrong mask, wrong cache update)? Without an oracle that can localize the error to a specific layer, operation, or component, the agent cannot efficiently iterate to correctness.

**This is the problem we solve: we invent a multi-tier differential testing framework that serves as an automated oracle, guiding an AI agent (or human) through the implementation of a correct HF-to-TRT bridge for any model family.**

### Problem 1: Existing Validation Approaches Cannot Localize Errors Across the HF-to-TRT Boundary

When a TRT engine produces different tokens from the HF model, the error could originate at any point in a deep pipeline:

| Error class | Example | Where it hides |
|---|---|---|
| Weight mapping | K/V not transposed, GQA not expanded, fused QKV not split correctly | Load-time, invisible until inference |
| Atomic op | Wrong RMSNorm epsilon, wrong GELU approximation variant | Single TRT layer, propagates through all downstream layers |
| Graph structure | Wrong residual pattern, wrong attention mask shape, missing bias | Compound effect across layers |
| Positional encoding | Wrong RoPE mode (interleaved vs standard), wrong theta, wrong partial rotary | Subtle — first tokens may be correct, errors grow with position |
| Cache management | Wrong eviction order, wrong position clamping | Only manifests after max_cache_length tokens |
| Cross-language parity | Python build and C++ runtime disagree on mask construction | Only manifests in deployed binary, not in Python testing |

Prior validation approaches:

- **Golden file comparison**: Record expected outputs and diff against them. Brittle — any change to the model, TRT version, or GPU hardware invalidates all golden files. Cannot localize errors within the pipeline. Cannot guide an agent to the fix.

- **End-to-end integration tests**: Run the full pipeline and check final output. If the test fails, the engineer must manually bisect the pipeline to find the error. An AI agent cannot perform this bisection because it requires understanding the full stack.

- **ONNX-based validation**: Export to ONNX and compare intermediate values. But ONNX export itself introduces errors (incomplete operator coverage, dynamic shape limitations), so the ONNX intermediate does not serve as a faithful reference.

- **Manual per-layer inspection**: An expert manually inserts debug prints at each layer and compares against HF. This is O(layers * models) manual work and does not scale.

None of these approaches provide what an AI agent needs: **immediate, automatically-localized feedback at the right abstraction level for the specific class of error encountered**.

### Problem 2: No Dual-Implementation Mirror Exists for Cross-Language Parity

The HF-to-TRT system has a two-language architecture: Python builds the TRT engine, C++ runs it. The C++ runtime implements mask construction, KV cache management, position tracking, and VL gating — all of which must be algorithmically identical to the assumptions baked into the TRT engine at build time. If the C++ runtime computes the attention mask differently from what the engine expects, the tokens are wrong.

No existing system provides a **behaviorally-identical mirror implementation** — a second implementation in a different language that can be directly compared at the token level to detect cross-language divergence. Without this mirror, cross-language bugs can only be found by end-to-end testing, which cannot localize the error.

### Problem 3: The Combinatorial Space of LLM Architectures Makes Manual Bridging O(N) and Unscalable

Each HF model family has its own combination of normalization (RMSNorm vs LayerNorm), MLP (SwiGLU vs GELU FC vs MoE), positional encoding (RoPE vs learned vs ALiBi, standard vs interleaved, full vs partial rotary), attention (MHA vs GQA vs MQA, sequential vs parallel residuals), weight format (separate vs fused Q/K/V, separate vs fused gate/up), and architectural class (decoder, MoE, SSM/Mamba, vision-language).

Prior bridging approaches:

- **TensorRT-LLM**: Each model requires hand-written C++ code across 10-20 files. Deep TRT expertise needed. Monolithic builder with deeply nested conditionals. An AI agent cannot work productively — blast radius of any change is uncontrollable.
- **ONNX export**: Lossy intermediate representation, suboptimal engines, fragile conversion.
- **torch.compile + TRT**: Slow compilation, frequent graph breaks, non-portable artifacts.
- **Manual scripts**: O(N), unmaintainable.

None of these are **agentic-ready**: they cannot be driven by an AI agent that reads `config.json`, generates bridge code, invokes an oracle for correctness feedback, and iterates to a correct solution.

### Problem 4: Speed-of-Light Inference Requires Eliminating PCIe Bottleneck

Standard KV cache implementations transfer cache state H2D/D2H each step (~2 GB for a 32-layer model at 2048 cache length). This PCIe traffic is the dominant bottleneck preventing theoretical maximum throughput.

### Problem 5: VL Models Require Architectural Innovations for Static TRT Graphs

Vision-language models inject visual features into the text decoder, but TRT compiles a single static graph with no conditional branching. Qwen3-VL "DeepStack" adds multi-level vision features from intermediate ViT layers. No prior system handles this in a single static engine.

---

## 5. How does your invention solve the problem described in the previous question? Be specific.

This invention introduces an **agent-guided differential testing framework** that serves as an automated oracle for bridging HuggingFace models to TensorRT inference. The framework provides immediate, localized correctness feedback at five independent abstraction levels, enabling an AI agent to iterate from a scaffolded plugin to a correct, production-ready TRT bridge without human intervention. The framework is supported by a composable builder stack, a device-resident runtime, and an auto-discovery plugin architecture that together make the agent's task tractable.

### Innovation 1 (Core Invention): Multi-Tier Differential Testing Framework as an Automated Agent Oracle

The central invention is a **five-plane differential testing architecture** where each plane serves as an oracle at a different abstraction level. An agent encountering a failure at one plane receives diagnostic output that is specific enough to guide the fix — not just "the output is wrong" but "layer 7's hidden state diverges by 0.3 after attention, suggesting a Q/K weight mapping error."

The five planes form a correctness pyramid. Lower planes catch lower-level errors with higher precision; higher planes catch integration errors that span multiple components:

#### Plane 1 — Atomic Operation Oracle (`test_graph_ops.py`)

**What it validates**: Every function in `graph_ops.py` (Layer 1 of the builder stack) against HF/PyTorch reference implementations.

**How it works**: For each operation (RMSNorm, LayerNorm, RoPE, GELU, ALiBi slopes, etc.), the framework:
1. Builds a minimal TRT engine containing **only** the operation under test
2. Generates randomized input tensors matching the operation's expected shapes
3. Executes the TRT engine on GPU
4. Executes the HF/PyTorch reference implementation on the same inputs
5. Compares outputs at `atol=1e-5` (fp32 TRT precision)

Each operation variant is tested across multiple parameter combinations — RoPE covers standard/interleaved mode, full/partial rotary factor, and positions 0/3/7/max_cache_length. The tolerance `1e-5` is tight enough to catch any mathematical error while accommodating fp32 non-determinism.

**Agent guidance**: If an agent modifies or adds a Layer 1 operation, this plane provides immediate feedback on numerical correctness in isolation. A failure here means the math is wrong, independent of any model or weight mapping.

**Prior art distinction**: No prior TRT inference system provides isolated, per-operation unit tests against reference implementations. TensorRT-LLM tests are integration-level, testing entire models.

#### Plane 2 — Per-Layer Hidden State Oracle (`diff_layers.py`)

**What it validates**: The hidden state tensor after every transformer block in the TRT engine against the corresponding HF model layer output.

**How it works**:
1. The engine builder is invoked with `debug_layer_outputs=True`, which causes the standard decoder builder to mark intermediate hidden states as additional network outputs. Each intermediate tensor is wrapped in a TRT identity layer to prevent buffer aliasing:
```python
def _mark_debug_output(network, tensor, name):
    identity = network.add_identity(tensor)  # prevents fusion affecting debug values
    out = identity.get_output(0)
    out.name = name
    network.mark_output(out)
    out.dtype = trt.float32
```
2. The resulting engine has outputs: `debug_embed`, `debug_hidden_0` through `debug_hidden_N`, and `logits` — in addition to the normal `present_k/v` outputs
3. The HF model is run with `output_hidden_states=True` on the same input
4. Each layer's hidden state is compared at `atol=0.05`

The comparison output is a diagnostic table designed for agent consumption:
```
Layer                Shape    MaxDiff   MeanDiff  TRT_std   HF_std    Status
embed              (4096,)  0.000003  0.000001   1.2341    1.2341       OK
layer.0.hidden     (4096,)  0.004812  0.000231   0.8732    0.8742       OK
layer.7.hidden     (4096,)  0.312451  0.089123   0.5432    0.8901     FAIL
```

**Agent guidance**: The table tells the agent **exactly which layer** first diverges significantly. The `TRT_std` vs `HF_std` columns distinguish between:
- **Scaling errors** (stds match but max_diff is high): weight magnitude issue, e.g., missing Gemma +1.0 gamma correction
- **Permutation/rotation errors** (stds differ): wrong RoPE mode, wrong head dimension mapping
- **Accumulation errors** (divergence grows linearly layer by layer): small per-layer error, likely an epsilon or activation variant issue

The tolerance `0.05` reflects error accumulation physics: fp32 TRT operations introduce ~1e-5 noise per op; over ~28 transformer layers with multiple ops per layer, this accumulates to ~0.05 at the hidden state level.

**Prior art distinction**: No prior system provides per-layer differential comparison with diagnostic std/mean/max columns. ONNX-based validation compares at the ONNX graph boundary, not at the original model's layer boundaries.

#### Plane 3 — Autoregressive Logit Trajectory Oracle (`diff_logits.py`)

**What it validates**: The complete autoregressive decode loop — not just a single forward pass, but the full multi-step generation trajectory including cache accumulation.

**How it works**:
1. Builds a standard TRT engine (no debug outputs)
2. Runs the engine through the Python mirror runner (`TrtRunner`) for N steps, recording logit vectors at each step
3. Runs the HF model for N steps, recording logit vectors at each step
4. Compares per-step: `max_diff`, `argmax_match` (same predicted token?), `top_K_overlap` (how many top-10 candidates agree?)

**The "battery" design**: The test runs across four canonical prompts simultaneously:
- `"factual"`: "The capital of France is" — short, high-confidence distributions
- `"reasoning"`: "Explain why water boils at 100 degrees Celsius." — multi-sentence, tests sustained coherence
- `"code"`: "Write a Python function that checks if a number is prime:" — syntax tokens, low-frequency in general text
- `"multi-turn"`: "User: What is 2+2?\nAssistant:" — chat template, tests tokenization edge cases

**Agent guidance**: Different failure patterns in the battery point to different error classes:
- All prompts fail at step 0 → weight mapping or graph structure error (Plane 2 would also fail)
- All prompts diverge gradually → epsilon, activation variant, or normalization error
- Only `code` prompt fails → tokenizer issue or embedding edge case for rare tokens
- Prompts pass for first K steps then diverge → cache management or position tracking bug (cache fills at step `max_cache_length` and eviction begins)

**Prior art distinction**: The battery concept — testing across multiple prompt types simultaneously to triangulate error classes — is novel. Prior systems test with a single prompt.

#### Plane 4 — Cross-Language Parity Oracle (`test_runner_parity.py`)

**What it validates**: Token-level exact equality between the Python mirror runner (`TrtRunner`) and the C++ binary (`trtf run`), both executing the same TRT engine plan.

**How it works**:
1. Reads the `.trtfb` bundle directly (parses magic, header, section table)
2. Extracts tokenizer files from the bundle to a temp directory
3. Runs the C++ binary as a subprocess: `trtf run bundle.trtfb --prompt "text"`
4. Runs the Python `TrtRunner` with identical input
5. Asserts character-for-character string equality of generated text

**Why this is the hardest-to-pass plane**: The Python `TrtRunner` and C++ `TrtBackendFastPath` both execute the same TRT engine plan. Any divergence must come from differences in:
- Attention mask construction
- Position ID computation and clamping
- Cache write ordering (especially sliding window eviction)
- EOS token detection
- Tokenizer encoding/decoding (`add_special_tokens=False` must match)

Because autoregressive generation amplifies any single-token divergence — one wrong token changes all subsequent predictions — this test is exquisitely sensitive to every runtime component simultaneously.

**The Mirror Runner (`debug_runner.py`) — the key enabling invention**: `TrtRunner` is not a convenience wrapper. It is a **behaviorally-identical reimplementation** of the C++ runtime in Python, using `cuda-python` for GPU memory management. Every aspect is mirrored:

```python
# Attention mask (matches C++ build_attention_mask exactly)
position_id = min(self.cache_length, self.max_cache_length)
self._h_mask[:] = -1e9
valid = min(self.cache_length, self.max_cache_length)
self._h_mask[0, :valid] = 0.0
self._h_mask[0, -1] = 0.0
```

```python
# Sliding-window cache eviction (matches C++ DeviceKvCache::update_after_step exactly)
if self.cache_length < self.max_cache_length:
    offset = self.cache_length * row_bytes
    cudart.cudaMemcpyAsync(cache_buf + offset, present_buf, row_bytes, D2D, stream)
else:
    cudart.cudaMemcpyAsync(cache_buf, cache_buf + row_bytes,
                           (self.max_cache_length - 1) * row_bytes, D2D, stream)
    offset = (self.max_cache_length - 1) * row_bytes
    cudart.cudaMemcpyAsync(cache_buf + offset, present_buf, row_bytes, D2D, stream)
```

The mirror runner auto-detects engine capabilities (VL embeddings, DeepStack levels) by probing tensor names, identical to the C++ `DeviceResources` constructor. This means the mirror stays synchronized with any new capabilities added to the Python builder without code changes.

**Maintenance invariant**: The project enforces: "If you change the C++ mask/cache/position logic, you MUST also update `debug_runner.py` and verify with `test_runner_parity.py`." The mirror is a specification document — divergence is a bug.

**Agent guidance**: If Planes 1-3 all pass but Plane 4 fails, the error is in the C++ runtime, not the plugin or engine. If Planes 1-3 fail AND Plane 4 fails, the error is in the Python-side build.

**Prior art distinction**: No prior TRT inference system maintains a behaviorally-identical dual-language implementation for cross-language parity testing. The mirror runner pattern — where a Python reimplementation of the C++ runtime serves as both a testing oracle and a living specification — is novel.

#### Plane 5 — Vision-Language Pipeline Oracle (`diff_vl.py`)

**What it validates**: The complete VL pipeline: image preprocessing, vision encoder, feature injection, text generation, and C++ binary parity.

**How it works**: Four independent subtests, each with GPU memory isolation (TRT runner freed before HF model loads to prevent OOM on single-GPU machines):

1. **Vision feature comparison**: Cosine similarity of TRT vs HF vision encoder outputs. Hard gate at `cosine_similarity >= 0.5` catches structural errors (wrong patch ordering, wrong channel layout) that element-wise tolerance alone might miss when feature magnitudes are small.

2. **embed_input gate test**: Verifies the scalar-gate VL embedding injection mechanism — calls `step()` with `use_input_embed=0.0` (normal) and `1.0` (bypass), asserting both produce valid logits.

3. **Full VL generation**: `VLTrtRunner` runs the complete image-token substitution pipeline: encode image → format prompt → tokenize → prefill with vision features → decode.

4. **C++ binary parity**: `trtf run --image` subprocess, verifying the C++ VL pipeline produces non-empty output.

**Debug-layers mode**: When initial feature comparison fails, `--debug-layers` attaches PyTorch forward hooks to every ViT block, printing per-block statistics and inter-block cosine similarity. This pinpoints the exact vision encoder block where divergence begins.

**Agent guidance**: Vision feature comparison failure → preprocessor or vision encoder builder error. embed_input failure → TRT graph gating error. Generation failure but features pass → text decoder injection error. C++ failure but Python passes → C++ VL backend error.

**Prior art distinction**: The structured decomposition of VL validation into four independent subtests with GPU memory isolation is novel. Prior VL validation is typically monolithic end-to-end testing.

#### The Validation Gate: One-Command Agent Oracle

All five planes are orchestrated by `validate_family.sh`:

```bash
./scripts/validate_family.sh Qwen/Qwen3-0.6B
```

This executes in sequence:
1. Build bundle (`trtf-build build`)
2. Logit trajectory comparison (`diff_logits.py --battery`)
3. Per-layer hidden state comparison (`diff_layers.py`)
4. Cross-language parity (`test_runner_parity.py`)

All stages run even if earlier ones fail, providing complete diagnostics in a single invocation. The output is a structured summary:

```
========================================
  Validation Summary: Qwen/Qwen3-0.6B
========================================
  PASS  Build bundle
  PASS  diff_logits (battery)
  PASS  diff_layers
  PASS  test_runner_parity
----------------------------------------
  4 passed, 0 failed
========================================
```

**Agent decision tree from validation output**:

| Failure Pattern | Diagnosis | Agent Action |
|---|---|---|
| Build fails | Config parsing or weight loading error | Fix `matches()` or `load_weights()` in plugin |
| Plane 2 fails at layer 0 | Embedding or first-layer weight mapping | Check embedding table shape, transposition |
| Plane 2 fails at layer N | Weight mapping error at layer N | Check Q/K/V/O/gate/up/down mapping for that layer |
| Plane 2: TRT_std != HF_std | Rotation or permutation error | Check RoPE mode, head dimension order |
| Plane 2: TRT_std == HF_std but high MaxDiff | Scaling or normalization error | Check norm epsilon, activation variant, Gemma +1.0 |
| Plane 3 fails only after K steps | Cache eviction or position clamping bug | Check `max_cache_length` parameter |
| Plane 3: `code` prompt fails, others pass | Tokenizer or rare-token embedding issue | Check tokenizer extraction, special tokens |
| Plane 4 fails but Planes 1-3 pass | C++ runtime disagreement | Mirror runner bug — check mask/position/cache logic |
| Plane 5 vision features fail | Preprocessor or vision encoder error | Check pixel ordering, patch_size, merge_size |

This decision tree is what makes the framework agentic — the diagnostic output maps directly to corrective actions in the plugin code.

### Innovation 2: Composable Builder Stack Designed for Agent Iteration

The builder stack is structured so that each layer of the framework corresponds to a layer of validation, creating a tight feedback loop:

**Layer 1 — `graph_ops.py` (Atomic TRT Operations)** → validated by **Plane 1** (atomic op oracle). Pure tensor-in/tensor-out functions: `add_rms_norm`, `add_apply_rope`, `add_layer_norm`, `add_gelu_new`, etc. An agent adding a new op writes the function, then runs Plane 1 to verify numerical correctness in isolation.

**Layer 2 — `graph_blocks.py` (Weight-Aware Composable Blocks)** → validated by **Plane 2** (per-layer oracle). `add_attention_block`, `add_swiglu_mlp`, `add_gelu_fc_mlp`. Blocks explicitly do NOT apply residual connections — the caller composes residuals. This means the same attention block works for sequential, parallel, and DeepStack-injected residual patterns. An agent composing blocks runs Plane 2 to see which layer first diverges.

**Layer 3 — `standard_decoder_builder.py` + Plugins** → validated by **Plane 3** (trajectory oracle). Parameterized by `norm_type`, `mlp_type`, `position_type`, `activation`, etc. An agent creating a new plugin selects parameters, runs Plane 3, and gets per-step feedback across 4 prompt types.

**Layer 4 — C++ Runtime** → validated by **Plane 4** (cross-language oracle). The mirror runner (`TrtRunner`) is the specification; the C++ runtime must match it exactly.

**The builder-validation correspondence is the key design insight**: an agent working at Layer N can trust that Layers 1..N-1 are correct (validated by Planes 1..N-1), and focus debugging on Layer N using Plane N's diagnostic output.

### Innovation 3: Zero-Registration Auto-Discovery Plugin Architecture

Adding a new HF model family requires creating a single Python file — the framework validates it automatically. Plugins are discovered via `pkgutil.iter_modules()` scanning for `.py` files with a module-level `plugin` attribute. The `FamilyPlugin` protocol uses structural typing. Four customization levels (pure delegation, weight transformation, custom loading, fully custom graph) let the agent select minimal effort for the model's complexity. Scaffolding via `new_family.py` generates a working plugin from `config.json`.

### Innovation 4: Two-Stage Bundle Architecture with Engine-Introspection

Build-time Python produces `.trtfb` bundles; C++ runtime loads them. The runtime discovers capabilities by probing TRT engine tensor names (`has_io_tensor`), not config fields. An agent adds new Python-side capabilities without C++ changes — the runtime auto-detects new tensor names.

### Innovation 5: Device-Resident KV Cache for Speed-of-Light Inference

`DeviceKvCache`: persistent GPU allocations, zero per-step malloc, ~2 KB PCIe per step (vs ~2 GB in prior art). Sliding-window D2D eviction. Single CUDA stream. Pre-allocated I/O via engine introspection.

### Innovation 6: Branchless Soft-Switch for Vision-Language Models

Single static TRT engine for all inference phases via scalar-multiplication gating: `hidden = (1-flag)*token + flag*embed`. DeepStack multi-level injection: `post_attn += embed * active`. Merge-group pixel preprocessing eliminates in-engine gather/scatter. Build-time baking of GQA expansion, RoPE tables, ALiBi slopes.

---

## 6. What are the specific technical differences (not benefits) between the solution described in the previous question and prior solutions?

### Difference 1: Five-Plane Differential Testing Oracle vs. Single-Level Validation

**Prior art**: Correctness validation uses golden-file comparison (brittle, cannot localize errors), end-to-end integration tests (pass/fail only, no localization), or manual per-layer inspection (O(layers * models), unscalable).

**This invention**: Five independent validation planes, each providing error-class-specific diagnostics:
- Plane 1: Per-operation TRT-vs-PyTorch comparison at `atol=1e-5`, testing each graph_ops function in isolation across multiple parameter combinations
- Plane 2: Per-layer hidden state comparison using TRT identity layers to expose intermediate outputs without affecting graph fusion, with diagnostic `TRT_std`/`HF_std` columns that distinguish scaling from permutation errors
- Plane 3: Multi-step autoregressive logit trajectory across a battery of 4 prompt types (factual/reasoning/code/multi-turn), where different failure patterns map to different error classes (step-0 = weight mapping, late divergence = cache bug, code-only = tokenizer issue)
- Plane 4: Cross-language parity via a behaviorally-identical Python mirror runner using cuda-python, asserting exact string equality of generated text between Python and C++
- Plane 5: Decomposed VL validation (vision features, embed gate, full generation, C++ parity) with GPU memory isolation and cosine-similarity hard gate

All planes are invoked by a single `validate_family.sh` command. Each plane's diagnostic output maps to specific corrective actions in the plugin code, forming an agent decision tree.

### Difference 2: Behaviorally-Identical Dual-Language Mirror Runner vs. No Cross-Language Validation

**Prior art**: No existing TRT inference system maintains a second implementation of the runtime in a different language for parity testing. Cross-language correctness is validated only at the end-to-end level.

**This invention**: `TrtRunner` in Python reimplements every algorithm in the C++ runtime — attention mask construction, position clamping, sliding-window cache eviction, VL scalar gating, DeepStack injection — using cuda-python for GPU memory management. The mirror auto-detects engine capabilities via tensor name probing, identical to the C++ `DeviceResources` constructor. Token-level divergence between Python and C++ is treated as a bug in either implementation. The mirror serves simultaneously as a testing oracle, a living specification, and the engine that powers Planes 2 and 3.

### Difference 3: Builder-Validation Layer Correspondence vs. Unstructured Test Suites

**Prior art**: Tests are organized by test type (unit, integration, e2e) with no structural correspondence to the system's architectural layers.

**This invention**: Each builder layer has a corresponding validation plane: Layer 1 (graph_ops) → Plane 1, Layer 2 (graph_blocks) → Plane 2, Layer 3 (standard_decoder_builder + plugins) → Plane 3, Layer 4 (C++ runtime) → Plane 4. An agent working at Layer N can trust that Layers 1..N-1 are correct and focus debugging using Plane N's output. The `debug_layer_outputs=True` flag in the builder produces a debug engine with intermediate outputs at every layer boundary, enabling Plane 2 without changing the builder's production output.

### Difference 4: Prompt Battery for Error-Class Triangulation vs. Single-Prompt Testing

**Prior art**: Logit comparison tests use a single test prompt.

**This invention**: The battery of 4 prompts (factual, reasoning, code, multi-turn) is designed so that different error classes produce different failure patterns across the battery. A weight mapping error fails all prompts at step 0. A cache bug fails all prompts at the same step. A tokenizer issue fails only the code prompt. An activation variant error causes gradual divergence across all prompts. This triangulation maps failure patterns to specific plugin code locations, guiding the agent's fix.

### Difference 5: Single-File Agentic Extension with Validation Gate vs. Multi-File Manual Integration

**Prior art** (TensorRT-LLM): Adding a new model requires editing 10-20 files with no automated correctness gate.

**This invention**: Single `.py` file, auto-discovered, validated by one command. Scaffolding generates working plugins from `config.json`. The validation gate provides localized diagnostics that guide the agent's iterations.

### Difference 6: Engine-Introspection Feature Detection vs. Configuration-Driven Dispatch

**Prior art**: Runtime reads config fields to determine capabilities.

**This invention**: C++ runtime probes TRT engine tensor names. An agent adds new capabilities (new tensor outputs) without C++ changes — the runtime auto-detects. The mirror runner uses the same probing pattern, keeping parity tests synchronized.

### Difference 7: Fully Device-Resident KV Cache vs. Host-Managed Cache

**Prior art**: KV cache transferred H2D/D2H each step.

**This invention**: KV cache on GPU for entire generation. ~2 KB PCIe per step. D2D sliding-window eviction. Single CUDA stream with single sync point.

### Difference 8: Branchless Scalar-Gate VL vs. Separate Engines

**Prior art**: Separate engines per mode or dynamic shapes.

**This invention**: One static engine. Scalar multiplication gates: `(1-flag)*token + flag*embed`. `cudaMemset` zeros inactive buffers — exact arithmetic zero. Merge-group pixel preprocessing eliminates in-engine gather. Build-time baking of GQA, RoPE, ALiBi.

---

## Additional Notes

### The Agent-Guided Development Loop in Practice

The system has been used to bridge 24+ model families. The agent-framework interaction follows a tight loop:

```
Agent reads config.json → Scaffolds plugin → Runs validate_family.sh
                                                    │
                                    ┌───────────────┴───────────────┐
                                    │                               │
                                All PASS                      Failure(s)
                                    │                               │
                            Model bridged ✓             Read diagnostic output
                                                               │
                                                    Map failure pattern to
                                                    error class (decision tree)
                                                               │
                                                    Fix specific plugin code
                                                               │
                                                    Re-run validate_family.sh
                                                               │
                                                          (loop back)
```

| Model Complexity | Plugin Level | Typical Agent Iterations | Guided By |
|---|---|---|---|
| Standard decoder (Qwen, LLaMA) | Level 1 (~30 lines) | 0-1 | Scaffolding produces working plugin |
| Unusual norms (Gemma) | Level 2 (~50 lines) | 1-2 | Plane 2 shows norm layers diverging |
| Fused weights (Phi-3) | Level 3 (~100 lines) | 2-3 | Plane 2 shows Q/K/V layers diverging |
| Custom architecture (Mamba) | Level 4 (~300 lines) | 3-5 | Plane 1 validates new ops, Plane 3 validates trajectory |

### Scaling Properties

| Dimension | Scaling Behavior |
|---|---|
| New model family | O(1): single file + validation gate |
| New architectural feature | O(1) in C++ via tensor-name auto-detection |
| New TRT operation | O(1): add to Layer 1, validate with Plane 1 |
| New validation to E2E suite | O(1): one JSON manifest file |
| Agent iterations per family | 0-5, guided by diagnostic output |

### Key Implementation Files

| Component | Files |
|---|---|
| **Differential Testing Framework** | |
| Mirror runner (Python ↔ C++ oracle) | `trtf_build/trtf_build/debug_runner.py` |
| Plane 1: Atomic op oracle | `tools/test_graph_ops.py` |
| Plane 2: Per-layer oracle | `tools/diff_layers.py` |
| Plane 3: Logit trajectory oracle | `tools/diff_logits.py` |
| Plane 4: Cross-language parity | `tools/test_runner_parity.py` |
| Plane 5: VL pipeline oracle | `tools/diff_vl.py` |
| Validation gate (all planes) | `scripts/validate_family.sh` |
| E2E test framework | `tests/e2e/test_full_pipeline.py`, `tests/e2e/conftest.py` |
| Per-model manifests | `tests/e2e/models/*.json` |
| **Composable Builder Stack** | |
| Layer 1: atomic ops | `trtf_build/trtf_build/graph_ops.py` |
| Layer 2: composable blocks | `trtf_build/trtf_build/graph_blocks.py` |
| Layer 3: decoder builder | `trtf_build/trtf_build/standard_decoder_builder.py` |
| **Plugin Architecture** | |
| Plugin auto-discovery | `trtf_build/trtf_build/families/__init__.py` |
| Plugin protocol | `trtf_build/trtf_build/families/base.py` |
| Family plugins (24+) | `trtf_build/trtf_build/families/*.py` |
| Plugin scaffolding | `scripts/new_family.py` |
| Engine builder orchestrator | `trtf_build/trtf_build/engine_builder.py` |
| **Runtime** | |
| Device-resident KV cache | `src/runtime/trt/device_kv_cache.{h,cpp}` |
| Engine-introspection dispatch | `src/cabi/trtf_c.cpp`, `src/cabi/bundle_helpers.{h,cpp}` |
| Bundle format | `src/bundle/bundle_format.{h,cpp}`, `trtf_build/trtf_build/bundle_writer.py` |
| VL vision encoder builder | `trtf_build/trtf_build/qwen_vl_vision_builder.py` |
| Image preprocessor (4 strategies) | `src/runtime/trt/image_preprocessor.{h,cpp}` |
