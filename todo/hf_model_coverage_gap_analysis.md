# Research: HuggingFace Model Coverage Gap Analysis

## Context

The trtf framework currently supports **19 model families** across 4 runtime strategies. This document analyzes which additional HF models can be supported, categorized by implementation difficulty.

## Current Coverage (19 families)

| # | Family | model_type | Pattern | Special Handling |
|---|--------|-----------|---------|-----------------|
| 1 | Qwen | `qwen*` (excl VL) | Standard decoder | RMSNorm + SwiGLU + RoPE |
| 2 | LLaMA | `llama*` | Standard decoder | RMSNorm + SwiGLU + RoPE |
| 3 | Mistral | `mistral*` | Standard decoder | RMSNorm + SwiGLU + RoPE |
| 4 | Gemma | `gemma*` | Standard decoder | +1.0 RMSNorm gamma, embed scale |
| 5 | Phi | `phi*` (excl MoE) | Standard decoder | Fused QKV + gate_up splitting |
| 6 | Granite | `granite*` | Standard decoder | 4 multiplier scalars absorbed |
| 7 | Falcon | `falcon*` | Standard decoder | LayerNorm + GELU FC + RoPE |
| 8 | StableLM | `stablelm*` | Standard decoder | LayerNorm + partial RoPE |
| 9 | GPT-2 | `gpt2` | Standard decoder | Learned pos, fused Conv1D QKV |
| 10 | OPT | `opt` | Standard decoder | Learned pos (offset=2), ReLU |
| 11 | GPT-Neo | `gpt_neo` | Standard decoder | Learned pos, no attn scaling |
| 12 | GPT-NeoX | `gpt_neox` | Standard decoder | Parallel residual, partial RoPE |
| 13 | CodeGen | `codegen` | Standard decoder | Parallel residual, interleaved RoPE |
| 14 | OLMo | `olmo` | Standard decoder | Non-parametric LayerNorm |
| 15 | XGLM | `xglm` | Standard decoder | Sinusoidal pos embeddings |
| 16 | StarCoder2 | `starcoder2` | Standard decoder | LayerNorm + GELU FC + RoPE |
| 17 | BLOOM | `bloom` | Standard decoder | ALiBi positions, embed LayerNorm |
| 18 | InternLM | `internlm*` | Standard decoder | Fused group-interleaved QKV |
| 19 | Mixtral | `mixtral` | MoE | Router + N experts, SwiGLU |
| 20 | Phi-MoE | `phimoe` | MoE | SparseMixer routing |
| 21 | Mamba | `mamba` | SSM recurrent | Custom C++ backend |
| 22 | Qwen2.5-VL | `qwen*vl*` | Vision-Language | Dual engines + preprocessing |

## Framework Capabilities (what's already built)

**Standard decoder builder supports:** RMSNorm, LayerNorm, SwiGLU, GELU FC, RoPE (full/partial/interleaved), learned pos, ALiBi, sinusoidal pos, GQA, parallel residual, embedding LayerNorm, activations {silu, gelu_new, gelu, relu}.

**Runtime strategies:** `decoder_kv_cache` (standard), `decoder_moe` (transparent to C++), `ssm_recurrent` (Mamba), `vision_language` (dual-engine VL).

**Key constraint:** MoE is transparent to C++ (routing lives in TRT graph). Adding a new MoE family = Python-only plugin work.

---

## Tier 1: Easy Additions (Python plugin only, ~1-2 days each)

These use standard decoder or existing MoE patterns. Only need a new `families/*.py` file with weight mapping. **No C++ changes.**

### Standard Decoder (reuse `standard_decoder_builder.py` directly)

| Family | model_type | Downloads | Architecture Notes | Effort |
|--------|-----------|-----------|-------------------|--------|
| **Yi** | `yi` | Very high (01.AI) | LLaMA-like, RMSNorm + SwiGLU + RoPE, GQA | Trivial (same as LLaMA) |
| **Baichuan** | `baichuan` | High (Chinese market) | RMSNorm + SwiGLU + RoPE, some use ALiBi | Low |
| **Cohere Command-R** | `cohere` | High (enterprise) | RMSNorm + SwiGLU + RoPE, 128K context | Low |
| **OLMo2** | `olmo2` | Growing (Allen AI) | RMSNorm variant, SwiGLU + RoPE | Low |
| **MPT** | `mpt` | Moderate (MosaicML) | LayerNorm + GELU FC + ALiBi | Low (ALiBi already supported) |
| **Gemma3** | `gemma3` | High (Google) | Gemma2 variant, same +1.0 gamma trick | Low |
| **Phi-3.5** | `phi3_5` | High (Microsoft) | Same as Phi-3, fused QKV | Trivial (extend phi.py match) |
| **ChatGLM/GLM-4** | `chatglm` | High (Chinese) | RMSNorm + SwiGLU + RoPE, custom QKV interleave | Medium (weight mapping) |
| **DeepSeek (v1)** | `deepseek` | Very high | LLaMA-like standard decoder | Trivial |

### MoE (reuse existing MoE graph pattern)

| Family | model_type | Downloads | Architecture Notes | Effort |
|--------|-----------|-----------|-------------------|--------|
| **Qwen2-MoE** | `qwen2_moe` | High | Standard top-k softmax routing + RMSNorm/SwiGLU experts | Low |
| **Qwen3-MoE** | `qwen3_moe` | Very high | Same pattern, 235B/22B active | Low |
| **DBRX** | `dbrx` | Moderate (Databricks) | Fine-grained MoE (16 experts, top-4), RMSNorm + SwiGLU | Medium (fine-grained routing) |
| **Arctic** | `arctic` | Moderate (Snowflake) | Dense + residual MoE hybrid | Medium |

**Tier 1 total: ~13 new families, covering a large fraction of popular HF models.**

---

## Tier 2: Medium Additions (Python + minor C++ or new graph ops, ~3-5 days each)

### Sliding Window Attention

| Family | model_type | Notes | What's Needed |
|--------|-----------|-------|--------------|
| **Mistral v0.2+** | `mistral` | Already matched, but sliding window not implemented | KV cache eviction policy in C++ (or just set `max_cache_length` as workaround) |
| **Gemma2** | `gemma2` | Alternating global + sliding window layers | Per-layer window config in C++ backend |

### Grouped Query Attention Variants

| Family | model_type | Notes | What's Needed |
|--------|-----------|-------|--------------|
| **Phi-3.5-MoE** | `phi3_5_moe` | MoE + GQA | Extend phi_moe.py with GQA |

### VL Models (reuse VL backend with new vision encoder)

| Family | model_type | Notes | What's Needed |
|--------|-----------|-------|--------------|
| **LLaVA** | `llava` | CLIP-ViT + LLaMA decoder | New vision encoder builder + `simple_chw` preprocessor (already in C++) |
| **Phi-3-Vision** | `phi3v` | SigLIP-ViT + Phi decoder | New vision encoder builder |
| **InternVL** | `internvl` | InternViT + InternLM decoder | New vision encoder builder |

---

## Tier 3: Hard Additions (New C++ backend or major graph ops, ~1-2 weeks each)

### Multi-Head Latent Attention (MLA)

| Family | model_type | Notes | What's Needed |
|--------|-----------|-------|--------------|
| **DeepSeek-V2** | `deepseek_v2` | MLA: low-rank KV compression | New TRT graph ops for latent projection/inverse. New KV cache format (compressed latent vectors instead of full K/V). C++ state management changes. |
| **DeepSeek-V3** | `deepseek_v3` | MLA + MoE (671B/37B active) | Same as V2 + MoE routing |

### Hybrid Architectures (Transformer + SSM)

| Family | model_type | Notes | What's Needed |
|--------|-----------|-------|--------------|
| **Jamba** | `jamba` | Alternating Mamba + attention + MoE | New hybrid C++ backend that manages both KV cache (for attention layers) and SSM state (for Mamba layers). Significant C++ work. |
| **Bamba** | `bamba` | Mamba2 variant | Mamba2's SSD layer differs from Mamba1 |

### Linear Attention / SSM Variants

| Family | model_type | Notes | What's Needed |
|--------|-----------|-------|--------------|
| **RWKV-6/7** | `rwkv` | Linear attention variant | New TRT graph ops for linear attention (token shift, WKV computation). New C++ state management (not KV cache, not SSM — its own recurrent state). |
| **Mamba-2** | `mamba2` | Improved SSM with SSD | Updated Mamba TRT graph + potentially different state shape |

---

## Summary

| Tier | Count | C++ Changes? | Effort per family |
|------|-------|-------------|-------------------|
| **Tier 1: Easy** | ~13 | None | 1-2 days |
| **Tier 2: Medium** | ~6 | Minor (VL encoder builders, sliding window) | 3-5 days |
| **Tier 3: Hard** | ~6 | Major (MLA, hybrid backends, linear attention) | 1-2 weeks |
| **Total new** | **~25** | | |

**Current: 22 families. Potential: ~47 families total (more than doubling coverage).**

### Highest-ROI additions (popularity x ease):

1. **Yi** — trivial, massive Chinese market
2. **DeepSeek v1** — trivial, extremely popular
3. **Cohere Command-R** — low effort, enterprise demand
4. **Qwen3-MoE** — low effort, most downloaded MoE
5. **Gemma3** — low effort, Google ecosystem
6. **Phi-3.5** — trivial extension of existing phi.py
7. **ChatGLM/GLM-4** — medium effort, huge Chinese market
8. **Baichuan** — low effort, popular Chinese model
9. **LLaVA** — medium effort, most popular open VLM
10. **DeepSeek-V3** — hard but highest strategic value (MLA + MoE)
