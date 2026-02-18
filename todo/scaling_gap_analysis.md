# Gap Analysis: Scaling trtf to All Major HuggingFace Architectures

## Context

trtf currently supports ~22 model families (19 decoder-only plugins + 2 MoE + 1 SSM + 1 VL). HuggingFace hosts 80+ distinct text-generation/VL architectures. This analysis identifies every missing piece — graph ops, builder capabilities, C++ runtime components — needed for comprehensive coverage, organized by effort and impact.

---

## Current Coverage Snapshot

**Supported runtime strategies (C++):** 4
- `decoder_kv_cache` — standard attention (TrtBackendFastPath + KvCacheStepState)
- `decoder_moe` — same runtime, MoE routing in TRT graph
- `ssm_recurrent` — Mamba SSM (MambaBackendFastPath + MambaStepState)
- `vision_language` — vision encoder + text decoder (VLBackendFastPath)

**Supported graph ops (`graph_ops.py`):** 19 functions
- Norms: RMSNorm, LayerNorm
- Activations: silu, gelu_new, gelu, relu
- Positions: RoPE (standard/interleaved/partial), learned, ALiBi, sinusoidal
- Attention: self-attention (with/without RoPE, windowed)
- VL: patch_embed_3d, spatial_merge

**Standard decoder builder axes:**
- `norm_type`: rmsnorm | layernorm
- `mlp_type`: swiglu | gelu_fc
- `position_type`: rope | learned | alibi
- `activation`: silu | gelu_new | gelu | relu
- Flags: partial_rotary_factor, interleaved_rope, parallel_residual, scale_attn_weights, embed_input

---

## Tier 1: Already Supported (~22 families, 0 work)

Works today with existing plugins or trivially (same model_type match):

| Family | Plugin | model_type | Notes |
|--------|--------|-----------|-------|
| Qwen 1.5/2/2.5/3/QwQ | qwen.py | qwen*, qwq* | |
| LLaMA 2/3/3.1/3.2/3.3 | llama.py | llama* | Includes CodeLlama, TinyLlama, Solar |
| Mistral v0.1-v0.3 | mistral.py | mistral* | Includes Zephyr, OpenHermes |
| Gemma 1 | gemma.py | gemma* | +1.0 gamma, sqrt(hidden) scale |
| Falcon | falcon.py | falcon* | |
| StableLM | stablelm.py | stablelm* | |
| OLMo 1.0 | olmo.py | olmo | |
| Granite | granite.py | granite* | |
| InternLM 1/2 | internlm.py | internlm* | |
| StarCoder2 | starcoder2.py | starcoder2 | |
| Phi-3/3.5 | phi.py | phi* | Fused QKV/gate_up split |
| GPT-2 | gpt2.py | gpt2 | |
| GPT-Neo | gpt_neo.py | gpt_neo | |
| GPT-NeoX / Pythia | gpt_neox.py | gpt_neox | |
| BLOOM / BLOOMZ | bloom.py | bloom | ALiBi |
| CodeGen | codegen.py | codegen | Interleaved RoPE |
| OPT | opt.py | opt | |
| XGLM | xglm.py | xglm | Sinusoidal positions |
| Mixtral | mixtral.py | mixtral | MoE top-k softmax |
| Phi-MoE | phi_moe.py | phimoe | MoE SparseMixer |
| Mamba-1 | mamba.py | mamba | SSM recurrent |
| Qwen2.5-VL | qwen_vl.py | qwen*vl* | Vision-language |

---

## Tier 2: New Plugin Only — Python, No C++ Changes (~12 families)

Standard decoder-only models that differ only in weight key names or minor preprocessing. Each needs only a new `families/<name>.py` file.

| Model Family | HF model_type | What's Different | Effort |
|-------------|--------------|-----------------|--------|
| **Yi (01.AI)** | yi | LLaMA clone, different weight keys | 0.5 day |
| **DeepSeek v1** | deepseek | LLaMA-style, standard layout | 0.5 day |
| **Baichuan 2** | baichuan | RMSNorm + SwiGLU, ALiBi or RoPE | 1 day |
| **StarCoder 1** | gpt_bigcode | LayerNorm + GELU_FC + learned, MQA (num_kv_heads=1) | 1 day |
| **MPT** | mpt | LayerNorm + ALiBi, no position embeds | 1 day |
| **OLMo 2** | olmo2 | Minor variation on olmo.py | 0.5 day |
| **Persimmon** | persimmon | RMSNorm + SwiGLU + RoPE | 0.5 day |
| **GPT-J** | gptj | Parallel residual + interleaved RoPE (verify codegen.py coverage) | 0.5 day |
| **Cohere Command-R** | cohere | Standard decoder, log-scaled attention | 1 day |
| **OpenELM** | openelm | Per-layer varying head counts, custom weight map | 1-2 days |
| **Qwen2-Audio (text)** | qwen2_audio | Same Qwen decoder, different model_type match | 0.5 day |
| **NVIDIA Nemotron LLaMA-based** | llama | Already works via llama.py | 0 |

**Total effort: ~1-2 weeks. Zero risk (no C++ changes).**

---

## Tier 3: New Graph Ops Needed — Python Only, No New C++ Runtime

Models that use standard KV-cache decoding but need new TRT graph operations or builder parameters. The C++ runtime is unchanged because the ops run inside the TRT engine.

### 3A. Trivial Graph Op Additions (1-3 lines each)

| Graph Op | Where | Models Unlocked | Effort |
|----------|-------|----------------|--------|
| `squared_relu`: `relu(x)^2` | `graph_ops.py` + `standard_decoder_builder.py` activation list | Nemotron-4, Nemotron-Mini | 0.5 day |
| `geglu`: `gelu(gate) * up` | `standard_decoder_builder.py` MLP variant | GLU-variant models | 0.5 day |
| `logit_softcap`: `tanh(logits/cap)*cap` | `graph_ops.py` + attention block | Gemma 2, Cohere v2 | 0.5 day |
| `sigmoid_router`: sigmoid instead of softmax for expert selection | MoE block builder | DBRX, some DeepSeek variants | 0.5 day |

**Total: ~2 days, unlocks ~5-7 families**

### 3B. Medium-Complexity Graph Additions

| Feature | Description | Models Unlocked | Effort |
|---------|------------|----------------|--------|
| **Shared-expert MoE** | 1 always-active expert + top-k from remaining N. New `_add_shared_expert_moe_block()` ~100 lines | Qwen-MoE, DeepSeek-V2 (partial), Nemotron-Nano-30B (partial) | 3-5 days |
| **Double-norm pattern** | Pre-attention norm + post-attention norm (Gemma 2 style) | Gemma 2 | 1-2 days |
| **Sliding window attention** (compile-time) | Already have `add_windowed_self_attention_with_rope` — verify usable for alternating global/local layers | Gemma 2, Mistral (long context) | 1 day |

**Total: ~1-2 weeks, unlocks ~4-6 additional families**

### 3C. High-Complexity Graph Additions

| Feature | Description | Models Unlocked | Effort |
|---------|------------|----------------|--------|
| **Multi-head Latent Attention (MLA)** | Compresses KV into low-rank latent before caching. Different attention computation: Q compresses via W_dq, KV uses absorbed W_uk/W_uv. Changes KV cache layout (latent dim != head_dim x num_kv_heads). Needs new `add_mla_attention_block()` ~200 lines + modified cache management. | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 | 2-3 weeks |
| **Relative position bias (T5-style)** | Learned bias table indexed by relative distance with log bucketing. Not RoPE, not ALiBi, not learned absolute. New `add_relative_position_bias()` + `make_relative_position_buckets()`. | T5, Flan-T5, mT5 (prerequisite for encoder-decoder) | 1 week |

**MLA is the highest-impact single item — DeepSeek-R1 alone is extremely popular.**

**Note on MLA and KV cache:** MLA changes the cache shape. Currently `KvCacheStepState` stores `[max_cache_length x attention_size]` per layer. With MLA, the cache stores compressed latents `[max_cache_length x latent_dim]` which is smaller. The C++ `KvCacheStepState` and `DecoderStepEngine` read cache dimensions from the TRT engine bindings, so they should adapt automatically IF the engine's cache tensors have different sizes. Need to verify `kv_cache_step_state.cpp` doesn't hardcode attention_size — it reads from `fp_cfg.attention_size`, so `fast_path_config` would need to support per-model cache dimensions, or MLA engines would set attention_size = latent_dim. **This may actually need minor C++ config changes.**

---

## Tier 4: New C++ Runtime Strategies Needed

These require new `IStepState` implementations, new backend classes, and new dispatch branches in `trtf_c.cpp`.

### 4A. Encoder-Decoder Runtime (`encoder_decoder`)

**Models unlocked:** T5, Flan-T5, mT5, BART, mBART, PEGASUS, MarianMT, M2M-100, LED (~10 families, each with many size variants. T5 is top-5 most-used on HuggingFace.)

**Architecture:** Separate encoder (bidirectional self-attention, run once) + decoder (causal self-attention + cross-attention to encoder output, run autoregressively).

**New C++ components:**

| Component | File | Description |
|-----------|------|-------------|
| `EncoderDecoderBackend` | `src/runtime/trt/encoder_decoder_backend.{h,cpp}` | Two-phase: run encoder once -> cache output -> run decoder autoregressively with cross-attention K/V from encoder |
| `CrossAttentionStepState` | `src/runtime/trt/cross_attention_step_state.{h,cpp}` | Holds decoder self-attention KV cache AND fixed encoder output (cross-attention K/V never changes) |
| Config parsing | `src/cabi/fast_path_config.{h,cpp}` | Add encoder-specific fields: `encoder_num_layers`, `encoder_hidden_size`, `encoder_num_heads`, `encoder_max_length` |
| Dispatch branch | `src/cabi/trtf_c.cpp` | New `if (strategy == "encoder_decoder")` branch, deserialize 2 engines |
| Bundle format | `src/bundle/bundle_format.{h,cpp}` | New section: `encoder_engine_plan` alongside existing `engine_plan` |

**New Python components:**

| Component | File | Description |
|-----------|------|-------------|
| Encoder builder | `trtf_build/trtf_build/encoder_builder.py` | New: builds encoder TRT engine (bidirectional attention, no KV cache, no causal mask) |
| Cross-attention op | `trtf_build/trtf_build/graph_ops.py` | `add_cross_attention()`: Q from decoder hidden, K/V from encoder output |
| Relative pos bias | `trtf_build/trtf_build/graph_ops.py` | `add_relative_position_bias()` for T5-style |
| Encoder-decoder builder | `trtf_build/trtf_build/encoder_decoder_builder.py` | Orchestrates building both engines |
| Family plugins | `trtf_build/trtf_build/families/t5.py`, `bart.py` | Weight mapping + config |
| Debug runner | `trtf_build/trtf_build/debug_runner.py` | `EncoderDecoderTrtRunner` |
| FamilyPlugin protocol | `trtf_build/trtf_build/families/base.py` | Add optional `build_encoder_engine()` method |

**Effort: ~3-4 weeks. Largest single runtime addition.**

### 4B. Hybrid Attention + SSM Runtime (`hybrid_mamba_attention`)

**Models unlocked:** Jamba (AI21), Zamba (Zyphra), Nemotron-Nano-9B-v2, Hymba, Falcon-H1 (~5-6 families)

**Architecture:** Single model with heterogeneous layers — some are attention (KV cache), others are Mamba-2/SSD (conv+SSM state). Layer type specified in model config.

**New C++ components:**

| Component | File | Description |
|-----------|------|-------------|
| `HybridBackend` | `src/runtime/trt/hybrid_backend.{h,cpp}` | Manages mixed state, single TRT engine with both cache types |
| `HybridStepState` | `src/runtime/trt/hybrid_step_state.{h,cpp}` | Per-layer: KvCache OR Mamba state. Holds layer-type map. |
| Config parsing | `src/cabi/fast_path_config.{h,cpp}` | Add `layer_types` array (e.g., `["attention", "mamba", "mamba", "attention", ...]`), Mamba-2 dims |
| Dispatch branch | `src/cabi/trtf_c.cpp` | New `if (strategy == "hybrid_mamba_attention")` branch |

**New Python components:**

| Component | File | Description |
|-----------|------|-------------|
| Mamba-2 SSD ops | `trtf_build/trtf_build/graph_ops.py` | Mamba-2 differs from Mamba-1: matrix-valued state transitions, SSD duality. ~200 lines new ops |
| Hybrid decoder builder | `trtf_build/trtf_build/hybrid_decoder_builder.py` | Builds single engine with mixed attention + Mamba-2 layers |
| Family plugins | `trtf_build/trtf_build/families/jamba.py`, `zamba.py`, `nemotron_hybrid.py` | Weight mapping |
| Debug runner | `trtf_build/trtf_build/debug_runner.py` | `HybridTrtRunner` |

**Dependencies:** Requires understanding Mamba-2/SSD (significantly different from Mamba-1).

**Effort: ~3-4 weeks.**

### 4C. Broader Vision-Language Support

**Models unlocked:** LLaVA 1.5/1.6/Next, InternVL 2/2.5, Phi-3-Vision, Phi-4-Vision, Pixtral, Idefics2/3, CogVLM, MiniCPM-V, Molmo, Llama-3.2-Vision (~8-12 families)

**Current limitation:** VL is hardcoded to Qwen2.5-VL's vision encoder architecture. The C++ VL runtime is actually fairly generic — it just runs a vision engine + text decoder with embed_input. The bottleneck is Python-side: only one vision encoder builder exists (`qwen_vl_vision_builder.py`).

**What's needed (mostly Python):**

| Component | File | Description |
|-----------|------|-------------|
| Generic ViT builder | `trtf_build/trtf_build/generic_vit_builder.py` | Parameterized ViT engine builder: patch embedding (2D, not Qwen's 3D), standard multi-head attention, LayerNorm, configurable pool/projection. Handles SigLIP, CLIP, InternViT. |
| Projection builders | Per-plugin or shared | Linear, MLP, Perceiver resampler. Most can be folded into the vision engine. |
| New graph ops | `graph_ops.py` | `add_patch_embed_2d()` (simpler than existing 3D), `add_cls_pool()`, `add_perceiver_resampler()` |
| Family plugins | `families/llava.py`, `internvl.py`, `phi_vision.py`, `pixtral.py` | Weight mapping + VL config |
| Multi-image support | `src/runtime/trt/vl_backend.cpp` (C++) | Currently single-image. Need to loop over images and concatenate features. Minor C++ change. |
| VL with non-Qwen decoder | `families/*.py` | LLaVA uses LLaMA decoder (already supported), InternVL uses InternLM (already supported). Just need to combine existing text plugins with new vision builders. |

**C++ changes are minimal** — existing VL backend handles generic vision engine -> embed_input flow.

**Effort: ~2-3 weeks.**

### 4D. Audio Models (Whisper, etc.)

**Models:** Whisper, SeamlessM4T (~3-5 families)

**Prerequisite:** Encoder-decoder runtime (4A) must exist first.

**Additional work on top of 4A:**
- Audio preprocessing (log-mel spectrogram) — can be Python-only or add `audio_preprocessor.{h,cpp}` in C++
- `families/whisper.py` plugin
- Encoder input is continuous features (not token embeddings) — minor encoder builder variant

**Effort: ~1-2 weeks (on top of 4A).**

---

## Tier 5: Fundamentally Different — Major Architectural Changes

### 5A. Encoder-Only Models (BERT, RoBERTa, DeBERTa)

**Use case:** Embeddings/classification, NOT text generation. Different API surface entirely.

**What's needed:**
- New `IEmbeddingBackend` interface (not `IGenerationBackend`)
- New `encoder_only_backend.{h,cpp}`: single forward pass, no autoregressive loop
- New C ABI function: `trtf_embed(pipeline, input_ids, output_embeddings)`
- New CLI subcommand: `trtf embed <bundle> --text "..."`

**Effort: ~2 weeks. Different product surface.**

### 5B. DeepSeek MLA — KV Cache Layout Change

**Critical note:** Multi-head Latent Attention compresses K/V before caching. The cache stores `[max_cache_length x latent_dim]` instead of `[max_cache_length x attention_size]`. This means:
- `KvCacheStepState` currently uses `fp_cfg.attention_size` to size cache buffers
- MLA needs `latent_dim` per layer (typically much smaller — e.g., 512 vs 4096)
- C++ would need to read cache dims from engine tensor shapes rather than config

**This may or may not need C++ changes** depending on whether `attention_size` in config can be set to `latent_dim` and everything still works. Needs investigation.

### 5C. RWKV (Linear RNN)

Custom WKV (weighted key-value) computation unlike anything in TRT's standard ops. May need TRT custom plugins or creative graph construction. Low priority unless specifically requested.

### 5D. Speculative Decoding

Not a model architecture but an inference optimization. Requires running two engines simultaneously, multi-token verification. Cross-cutting change to `IGenerationBackend::generate()`. Large effort, orthogonal to model coverage.

---

## Priority Matrix — What to Build First

| Priority | Item | Effort | Models Unlocked | Cumulative Coverage |
|----------|------|--------|----------------|-------------------|
| **P1** | Tier 2: New plugins (Yi, DeepSeek v1, StarCoder1, MPT, etc.) | 1-2 weeks | +12 families | ~34 |
| **P2** | Tier 3A: Trivial graph ops (squared_relu, geglu, softcap, sigmoid router) | 2 days | +5-7 families | ~40 |
| **P3** | Tier 3B: Shared-expert MoE + double-norm | 1-2 weeks | +4-6 families (Qwen-MoE, DBRX, Gemma 2) | ~46 |
| **P4** | Tier 4C: Generic ViT builder for broader VL | 2-3 weeks | +8-12 VL families | ~56 |
| **P5** | Tier 3C: MLA (DeepSeek-V2/V3/R1) | 2-3 weeks | +3 families (extremely popular) | ~59 |
| **P6** | Tier 4A: Encoder-decoder runtime | 3-4 weeks | +10 families (T5, BART, etc.) | ~69 |
| **P7** | Tier 4B: Hybrid attention+SSM | 3-4 weeks | +5-6 families | ~75 |
| **P8** | Tier 4D: Audio (Whisper) | 1-2 weeks | +3-5 families | ~78 |
| **P9** | Tier 5A: Encoder-only (BERT) | 2 weeks | +5 families (different use case) | ~83 |

### Dependency graph:
```
P1, P2, P3  — independent, can start immediately
P4          — independent
P5          — independent (may need minor C++ config tweak for cache dims)
P6          — independent, prerequisite for P8
P7          — independent, needs Mamba-2 SSD understanding
P8          — depends on P6 (encoder-decoder runtime)
P9          — independent (different API surface)
```

### Quick wins (first 4 weeks, P1+P2+P3):
- ~24 new model families
- Zero C++ changes
- Coverage jumps from 22 -> 46 families (~2x)

### Medium-term (weeks 5-10, P4+P5):
- +11-15 more families including DeepSeek-R1 and broad VL
- Minor C++ changes (multi-image in VL, possible cache config tweak for MLA)
- Coverage: ~59 families

### Longer-term (weeks 11-18, P6+P7+P8):
- Encoder-decoder + hybrid = two new C++ runtime strategies
- +18 more families
- Coverage: ~78 families

---

## New Runtime Components Summary

### New C++ backends needed: 2 (encoder-decoder, hybrid)
### New IStepState implementations needed: 2 (cross-attention, hybrid)
### New Python builders needed: 3 (encoder, encoder-decoder, hybrid-decoder)
### New graph ops needed: ~10 (squared_relu, geglu, softcap, sigmoid_router, shared_expert_moe, cross_attention, relative_pos_bias, mla_attention, mamba2_ssd, patch_embed_2d)
### FamilyPlugin protocol changes: 1 (add optional `build_encoder_engine()`)
### IGenerationBackend changes: 0 for model coverage (speculative decoding would need changes but is orthogonal)

---

## Key Files That Need Changes (by priority)

**Always touched (for any new model):**
- `trtf_build/trtf_build/families/<new>.py` — new plugin file
- `trtf_build/trtf_build/graph_ops.py` — if new ops needed

**For new graph ops (P2, P3, P5):**
- `trtf_build/trtf_build/graph_ops.py`
- `trtf_build/trtf_build/standard_decoder_builder.py` (if new activation/norm/position variant)

**For encoder-decoder (P6):**
- New: `src/runtime/trt/encoder_decoder_backend.{h,cpp}`
- New: `src/runtime/trt/cross_attention_step_state.{h,cpp}`
- Modify: `src/cabi/fast_path_config.{h,cpp}`
- Modify: `src/cabi/trtf_c.cpp`
- Modify: `src/bundle/bundle_format.{h,cpp}` (new `encoder_engine_plan` section)
- New: `trtf_build/trtf_build/encoder_builder.py`
- New: `trtf_build/trtf_build/encoder_decoder_builder.py`
- Modify: `trtf_build/trtf_build/families/base.py` (protocol extension)
- Modify: `trtf_build/trtf_build/debug_runner.py`

**For hybrid att+SSM (P7):**
- New: `src/runtime/trt/hybrid_backend.{h,cpp}`
- New: `src/runtime/trt/hybrid_step_state.{h,cpp}`
- Modify: `src/cabi/fast_path_config.{h,cpp}`
- Modify: `src/cabi/trtf_c.cpp`
- New: `trtf_build/trtf_build/hybrid_decoder_builder.py`
- Modify: `trtf_build/trtf_build/graph_ops.py` (Mamba-2 SSD ops)
- Modify: `trtf_build/trtf_build/debug_runner.py`

---

## The Long Tail (unique one-off architectures, deprioritize unless requested)

| Model | Why It's Hard | Priority |
|-------|-------------|----------|
| RWKV v4/v5/v6 | Custom WKV computation, may need TRT custom plugins | Low |
| RetNet | Retention mechanism != attention | Low |
| Hyena | Long convolution, no attention | Low |
| ChatGLM / GLM | Prefix-LM (bidirectional prefix + causal rest), 2D position encoding | Medium-Low |
| Longformer / BigBird | Sparse attention (local + global), hard to express in dense TRT graphs | Low |
| Mixture-of-Depths | Dynamic layer skipping, can't express in static TRT graphs | Very Low |
| Diffusion Transformers | Different paradigm (denoising loop), out of scope for text gen | Out of scope |
