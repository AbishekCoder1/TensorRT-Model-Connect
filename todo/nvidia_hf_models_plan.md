# Plan: Support NVIDIA HuggingFace Models in trtf

## Context

NVIDIA publishes 100+ models on HuggingFace (`nvidia/` org). Many are fine-tunes of architectures we already support (LLaMA, Mistral, Qwen). Others use novel architectures (Hymba hybrid, Nemotron-Nano Mamba-2 hybrid) that need new work. This plan categorizes every relevant NVIDIA text-generation and VL model by implementation effort.

---

## Group A: Already Supported (no changes needed)

These are fine-tunes/derivatives of architectures with existing family plugins. They work today with `trtf-build build <repo> -o out.trtfb`.

| Model | HF Repo | Base model_type | Existing Plugin |
|-------|---------|----------------|-----------------|
| Llama-3.1-Nemotron-70B-Instruct | `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | `llama` | `llama.py` |
| Llama-3.1-Nemotron-Nano-8B | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | `llama` | `llama.py` |
| Llama-3.1-Nemotron-Nano-4B | `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | `llama` | `llama.py` |
| Llama-3.3-Nemotron-Super-49B | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | `llama` | `llama.py` |
| Llama-3.1-Nemotron-Ultra-253B | `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | `llama` | `llama.py` |
| Llama-3.1-Minitron-4B (depth) | `nvidia/Llama-3.1-Minitron-4B-Depth-Base` | `llama` | `llama.py` |
| Llama-3.1-Minitron-4B (width) | `nvidia/Llama-3.1-Minitron-4B-Width-Base` | `llama` | `llama.py` |
| Mistral-NeMo-12B | `nvidia/Mistral-NeMo-12B-Base` | `mistral` | `mistral.py` |
| Mistral-NeMo-Minitron-8B | `nvidia/Mistral-NeMo-Minitron-8B-Base` | `mistral` | `mistral.py` |

**Count: ~9 models, zero work needed.**

---

## Group B: New Plugin — Standard Decoder (Python-only, easy)

### B1. Nemotron-4 family (`nemotron` model_type)

**Models:** Nemotron-4-340B-Base, Nemotron-4-340B-Instruct, Nemotron-Mini-4B, nemotron-3-8b, nemo-megatron-gpt-20B/1.3B

**Architecture:** Standard decoder-only transformer with:
- RoPE position embeddings
- **Squared ReLU activation** (`relu(x)^2`) — NOT currently in `standard_decoder_builder.py`
- Grouped Query Attention
- RMSNorm (or LayerNorm depending on variant)
- SentencePiece tokenizer

**What's needed:**
1. Add `squared_relu` activation to `standard_decoder_builder.py` (1 new TRT op: `relu` → `elementwise_pow(x, 2)`)
2. Add `graph_ops.py:add_squared_relu()` helper
3. New `families/nemotron.py` plugin with weight mapping
4. Verify with `diff_logits.py` / `diff_layers.py`

**Effort: ~2 days. Mostly trivial, the only real work is the new activation function.**

**Files to modify:**
- `trtf_build/trtf_build/graph_ops.py` — add `squared_relu` op
- `trtf_build/trtf_build/standard_decoder_builder.py` — add `squared_relu` activation option
- `trtf_build/trtf_build/families/nemotron.py` — new file

**Note:** 340B requires multi-GPU (8x H100). Validation can use smaller Nemotron-Mini-4B or nemotron-3-8b.

---

## Group C: VL Models — New Vision Encoders (Python + minor C++)

These use text decoders we already support (Qwen) but need new vision encoder builders and possibly new preprocessing strategies.

### C1. Cosmos-Reason1-7B / Cosmos-Reason2-8B

**Base:** Qwen2.5-VL-7B / Qwen3-VL-8B (post-trained)

**What's needed:** Likely works with existing `qwen_vl.py` since it's the same base architecture. Needs testing.

**Effort: ~0.5 days (just validation).**

### C2. NVLM-D-72B

**Base decoder:** Qwen2-72B (`qwen` model_type — already supported)
**Vision encoder:** InternViT-6B-448px (ViT, different from Qwen2.5-VL's vision encoder)
**Connector:** 2-layer MLP (12800 → 29568 → 8192)
**Image handling:** 1D tile-tagging, dynamic high-resolution

**What's needed:**
1. New vision encoder TRT builder for InternViT-6B (similar pattern to `qwen_vl_vision_builder.py` but different ViT architecture)
2. New `families/nvlm.py` plugin
3. MLP connector weights mapped into the vision engine or as a separate projection
4. C++ `image_preprocessor.cpp` — likely reuse `simple_chw` or `center_crop_chw` strategy (already implemented)

**Effort: ~5 days.**

### C3. Eagle2-9B

**Base decoder:** Qwen2.5-7B (`qwen` model_type — already supported)
**Vision encoder:** Dual — SigLIP (448x448 tiles) + ConvNeXt-XXLarge (512x512 tiles)
**Connector:** Channel-wise concatenation + MLP

**What's needed:**
1. Two vision encoder TRT builders (SigLIP ViT + ConvNeXt)
2. Concatenation + MLP connector logic
3. New `families/eagle.py` plugin
4. C++ VL backend would need to handle dual vision engines (currently supports 1 vision engine)

**Effort: ~7 days (most complex VL addition due to dual encoders).**

### C4. Llama-3.1-Nemotron-Nano-VL-8B

**Base decoder:** LLaMA 3.1 architecture
**Vision:** Document intelligence VLM

**What's needed:** New VL plugin with LLaMA decoder (currently VL only supports Qwen decoder). Need to investigate vision encoder type.

**Effort: ~5 days.**

---

## Group D: Hybrid Architectures — New C++ Backends (hard)

These require fundamentally new runtime strategies in C++ because they mix attention (KV cache) with SSM (recurrent state) in the same model.

### D1. Nemotron-Nano-9B-v2 (Mamba-2 + Attention hybrid)

**Architecture:** 62 layers — 6 attention + 28 Mamba-2 + 28 FFN
**model_type:** `nemotron` (or `nemotron_h`)

**What's needed:**
1. **New C++ hybrid backend** (`HybridBackend`) managing both KV cache (for 6 attention layers) and Mamba-2 SSM state (for 28 Mamba-2 layers)
2. **New `IStepState` implementation** (`HybridStepState`) that holds per-layer state of the correct type
3. **New runtime_strategy:** `hybrid_mamba_attention`
4. **Python side:** New TRT graph builder that mixes attention blocks and Mamba-2 blocks
5. **Mamba-2 (SSD) graph ops** — Mamba-2 differs from Mamba-1 (uses structured state-space duality). Need new `graph_ops.py` functions for SSD
6. New `families/nemotron_hybrid.py` plugin
7. New Python debug runner (`HybridTrtRunner`)

**Files to create/modify:**
- `src/runtime/trt/hybrid_backend.{h,cpp}` — new C++ backend
- `src/runtime/trt/hybrid_step_state.{h,cpp}` — new state management
- `src/cabi/fast_path_config.{h,cpp}` — add `hybrid_mamba_attention` strategy
- `src/cabi/trtf_c.cpp` — dispatch to new backend
- `trtf_build/trtf_build/graph_ops.py` — Mamba-2 SSD ops
- `trtf_build/trtf_build/families/nemotron_hybrid.py` — new plugin
- `trtf_build/trtf_build/debug_runner.py` — `HybridTrtRunner`

**Effort: ~2 weeks.**

### D2. Hymba-1.5B (Parallel Attention + SSM Heads)

**Architecture:** Each block has **parallel** attention heads and Mamba SSM heads (not interleaved layers — parallel within each block). Plus:
- 128 learnable meta-tokens prepended to every prompt
- Cross-layer KV cache sharing (10x cache reduction)
- Sliding window attention on 90% of attention layers

**model_type:** `hymba`

**What's needed:**
1. Everything from D1 (hybrid state management), PLUS:
2. **Per-block parallel head routing** in TRT graph (attention + SSM computed in parallel, outputs merged)
3. **Meta-token injection** in C++ runtime (prepend 128 learned embeddings before user prompt)
4. **Cross-layer KV sharing** in C++ state management (multiple layers reading from same KV cache)
5. **Sliding window attention** support in the TRT graph

**Effort: ~3 weeks (most complex NVIDIA model).**

### D3. Nemotron-3-Nano-30B-A3B (Triple Hybrid: Attention + Mamba-2 + MoE)

**Architecture:** 52 layers — 6 attention + 23 Mamba-2 + 23 MoE (128 experts + 1 shared, top-6)
**Special:** Squared ReLU in MoE MLPs, sigmoid gating router

**What's needed:**
1. Everything from D1 (hybrid backend), PLUS:
2. **MoE layer support** in hybrid graph builder (already have MoE patterns from Mixtral/Phi-MoE)
3. **Sigmoid routing** (not softmax — different from existing MoE plugins)
4. **Shared expert** pattern (1 expert always active + top-k from remaining 128)

**Effort: ~2.5 weeks (builds on D1 + existing MoE patterns).**

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
1. **Validate Group A models** — confirm existing plugins work for all Llama/Mistral Nemotron fine-tunes
2. **Validate Cosmos-Reason models** (C1) — test with existing `qwen_vl.py`
3. **Implement Nemotron-4 plugin** (B1) — add `squared_relu` activation + `nemotron.py` family

### Phase 2: VL Expansion (Weeks 2-3)
4. **NVLM-D-72B** (C2) — InternViT vision encoder builder
5. **Llama-Nemotron-VL** (C4) — LLaMA-based VL support
6. **Eagle2-9B** (C3) — dual vision encoder support

### Phase 3: Hybrid Architecture Foundation (Weeks 4-5)
7. **Mamba-2 SSD graph ops** — prerequisite for all hybrid models
8. **Hybrid backend in C++** — `HybridBackend` + `HybridStepState`
9. **Nemotron-Nano-9B-v2** (D1) — first hybrid model validation

### Phase 4: Advanced Hybrid Models (Weeks 6-8)
10. **Nemotron-3-Nano-30B-A3B** (D3) — triple hybrid with MoE
11. **Hymba-1.5B** (D2) — parallel heads, meta-tokens, KV sharing

---

## Summary

| Group | Models | Work Required | Effort |
|-------|--------|--------------|--------|
| **A: Already works** | 9 Llama/Mistral fine-tunes | Validation only | 0 |
| **B: Standard decoder** | Nemotron-4 family (~5 models) | New activation + plugin | ~2 days |
| **C: VL models** | 4-5 VL models | New vision encoder builders | ~2-3 weeks |
| **D: Hybrid architectures** | 3 novel models (Nano-9B, Nano-30B, Hymba) | New C++ backend + Mamba-2 ops | ~5-8 weeks |

**Total NVIDIA models supportable: ~20+ text-gen/VL models**
**Already working today: ~9 (45%)**
**Quick additions (< 1 week): ~14 (adding Nemotron-4 family)**
**Full coverage: ~8 weeks of work**

---

## Verification Plan

For each group:
- **Group A:** `trtf-build build nvidia/Llama-3.1-Nemotron-Nano-8B-v1 -o test.trtfb && ./build/trtf run test.trtfb --prompt "Hello"`
- **Group B:** `python3 tools/diff_logits.py --model nvidia/Nemotron-Mini-4B-Instruct --atol 1e-3 --battery`
- **Group C:** `python3 tools/diff_vl.py --bundle nvlm.trtfb --image test.jpg --model nvidia/NVLM-D-72B`
- **Group D:** New parity tests for hybrid runner vs HF transformers
- **All:** `pytest tests/ -v` (unit tests must pass), E2E in container
