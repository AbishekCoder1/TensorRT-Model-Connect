# Coverage-Based Test Selection (Test Impact Analysis)

**Status**: Planning
**Created**: 2026-02-26
**Goal**: Given a set of changed files, automatically determine the minimal set of E2E model tests to run, reducing full-suite runtime from ~2-3 hours to minutes for typical changes.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Industry Research & Best Practices](#3-industry-research--best-practices)
4. [Repository Dependency Analysis](#4-repository-dependency-analysis)
5. [Complete File-to-Model Mapping](#5-complete-file-to-model-mapping)
6. [Design: 3-Tier File Classification](#6-design-3-tier-file-classification)
7. [Implementation Plan](#7-implementation-plan)
8. [Phase 1: pytest conftest.py Plugin](#8-phase-1-pytest-conftestpy-plugin)
9. [Phase 2: Standalone CLI Tool](#9-phase-2-standalone-cli-tool)
10. [Phase 3: CI/CD Integration](#10-phase-3-cicd-integration)
11. [Phase 4: Function-Level Refinement](#11-phase-4-function-level-refinement)
12. [Safety Net & Guardrails](#12-safety-net--guardrails)
13. [Expected Impact](#13-expected-impact)
14. [Validation Plan](#14-validation-plan)
15. [Appendix A: Complete Model Manifest Database](#appendix-a-complete-model-manifest-database)
16. [Appendix B: Complete C++ File Inventory](#appendix-b-complete-c-file-inventory)
17. [Appendix C: Complete Python Builder File Inventory](#appendix-c-complete-python-builder-file-inventory)
18. [Appendix D: E2E Harness Plugin Inventory](#appendix-d-e2e-harness-plugin-inventory)

---

## 1. Executive Summary

This plan introduces **Test Impact Analysis (TIA)** — a system that maps every source file in the repository to the specific E2E model tests it can affect. When a developer changes a file, the system computes the minimal set of models that need E2E testing, skipping all unaffected models.

**Key design decision**: We use a **static, manifest-driven approach** rather than runtime coverage tracing. This is because:
- Our E2E tests invoke the C++ binary (`./build/trtf`) as a subprocess — Python coverage tools (pytest-testmon, coverage.py) **cannot trace into the C++ binary**
- Our manifest schema already encodes the critical dependency metadata (`family`, `runtime_strategy`)
- Static rules are deterministic, require no baseline run, and cover both Python and C++ code paths
- The dependency structure is clean and well-documented (family plugins → models, runtime_strategy → C++ backends)

**Implementation**: ~300 lines of Python across 2 files (a pytest plugin + a standalone selector script), with zero new dependencies.

---

## 2. Problem Statement

### Current State
- Full E2E suite: **50 models**, ~2-3 hours on GPU
- Every change triggers the same decision: "which models do I need to test?"
- Developers currently rely on tribal knowledge or run everything
- The `--e2e-task-strategy` filter helps but is too coarse (26 models for `text_generation_causal`)

### Desired State
- `git diff` → automatic model selection → `pytest tests/test_e2e.py --affected-only`
- Change `families/qwen.py` → run only `qwen3-0.6b`, `qwen3-4b-instruct-2507` (2 models, ~5 min)
- Change `mamba_backend.cpp` → run only `mamba-130m` (1 model, ~3 min)
- Change `graph_ops.py` → run all 50 models (unavoidable, but correctly identified as global)
- Nightly full suite as safety net regardless

### Constraints
- C++ subprocess execution is opaque to Python tracing → must use static rules for C++ changes
- Family plugin auto-discovery (`pkgutil.iter_modules`) means dependency is implicit → parse manifests instead
- E2E harness uses registry pattern (runners, comparators, references auto-discovered) → map by strategy name

---

## 3. Industry Research & Best Practices

### 3.1 Approaches Evaluated

| Approach | How It Works | Accuracy | Fits Us? | Why / Why Not |
|----------|-------------|----------|----------|---------------|
| **Static manifest-based** | Map files → tests via code structure rules | File-level | **Best fit** | Manifests have `family` + `runtime_strategy`; covers C++ |
| **pytest-testmon** | Runtime line-level coverage per test, stored in SQLite | Line-level | Partial | Cannot trace C++ subprocess; good for Python-only refinement |
| **coverage.py dynamic contexts** | Per-test coverage recording via `sys.settrace()` | Line-level | Partial | Same limitation: only traces Python, not C++ binary |
| **Tach (Gauge)** | Static Python import graph analysis via AST | Module-level | Partial | Cannot handle C++ or auto-discovered plugins |
| **Bazel rdeps** | Build-graph reverse dependency queries | Target-level | No | We don't use Bazel |
| **Pants** | Static dependency inference + test selection | Module-level | No | Requires migration to Pants build system |
| **Meta TestPilot (ML-based)** | Gradient-boosted trees on historical test results | Predictive | Future | Needs substantial historical data |
| **Microsoft TIA (Azure DevOps)** | Call-graph recording during baseline run | Function-level | No | Azure-specific, Windows-centric |

### 3.2 What Large Companies Do

**Google (TAP)**: Uses Bazel's build graph. `rdeps` query finds all test targets transitively depending on changed files. Works because BUILD files explicitly declare all dependencies.

**Meta (Predictive Test Selection)**: Goes beyond dependency analysis. Uses ML model trained on historical {diff → test failure} data. Catches 99.9% of regressions while running only ~33% of tests. Deployed since 2018.

**Pinterest**: Custom Go CLI (`build-collector`) takes commits, identifies changed files, runs `bazel query 'kind(".*_test", rdeps(//..., set(<changed_files>)))'`. Achieved 50%+ CI speedup.

**Pants users**: Automatic Python import inference via AST parsing. No BUILD file declarations needed. Real-world report: 70% reduction in CI test load.

### 3.3 Why Static Manifest-Based is Best for This Repo

1. **C++ opacity**: ~50% of the execution path is inside `./build/trtf` (invoked as subprocess). No Python tracing tool can see inside it. Static rules are the *only* way to handle C++ changes.

2. **Clean manifest schema**: Every model's JSON manifest already declares `family` and `runtime_strategy` — the two fields that determine which code paths execute. No additional metadata needed.

3. **Deterministic**: No baseline run, no stale databases, no cache invalidation. Rules are computed from source code structure on every invocation.

4. **Zero dependencies**: Pure Python, works with existing pytest infrastructure.

5. **Incrementally refinable**: Start with file-level rules, later layer on function-level tracing (pytest-testmon) for shared Python files only.

### 3.4 Key References

- [pytest-testmon](https://github.com/tarpas/pytest-testmon) — Most mature Python TIA tool (71K weekly downloads)
- [pytest-difftest](https://github.com/PaulM5406/pytest-difftest) — Newer, Rust AST parsing + coverage
- [Tach](https://github.com/tach-org/tach) — Static import graph analysis (Rust implementation)
- [Predictive Test Selection (Meta)](https://engineering.fb.com/2018/11/21/developer-tools/predictive-test-selection/)
- [Taming Google-Scale Continuous Testing](https://research.google.com/pubs/archive/45861.pdf)
- [Coverage.py Measurement Contexts](https://coverage.readthedocs.io/en/latest/contexts.html)
- [Pants Dependency Inference](https://www.pantsbuild.org/blog/2022/10/27/why-dependency-inference)
- [Cutting CI test load by 70% using Pants](https://tkbadamdorj.github.io/blog/2025/pants/)
- [Test Impact Analysis: 2x Faster CI (Instawork)](https://engineering.instawork.com/test-impact-analysis-the-secret-to-faster-pytest-runs-e44021306603)

---

## 4. Repository Dependency Analysis

### 4.1 The Natural Dependency Chain

The repo has a clean, layered dependency structure that maps directly to test selection:

```
Changed file
  ↓
Family plugin (1:N models via manifest "family" field)
  ↓
runtime_strategy (selects C++ backend at runtime)
  ↓
task_strategy (selects E2E runner + comparator + reference backend)
  ↓
E2E test parametrization (specific model names in pytest)
```

### 4.2 RUNTIME_TO_TASK_STRATEGY Mapping (from contracts.py:420)

This is the canonical mapping that connects manifest `runtime_strategy` to the E2E harness `task_strategy`, which in turn selects the runner, comparator, and reference backend:

```python
RUNTIME_TO_TASK_STRATEGY = {
    "decoder_kv_cache":        "text_generation_causal",
    "decoder_moe":             "text_generation_causal",
    "ssm_recurrent":           "text_generation_causal",
    "rwkv_recurrent":          "text_generation_causal",
    "hybrid_mamba_attention":   "text_generation_causal",
    "vision_language":          "vision_language_generation",
    "speech_to_text":           "speech_to_text",
    "text_to_audio":            "text_to_audio",
    "speech_to_speech":         "speech_to_speech",
    "segmentation":             "segmentation",
    "prompted_segmentation":    "prompted_segmentation",
    "object_detection":         "object_detection",
    "embedding":                "embedding",
    "reranking":                "reranking",
    "encoder_only":             "encoder_only_nlp",
    "neural_operator":          "neural_operator",
    "diffusion":                "diffusion_media_generation",
    "omni_multimodal":          "omni_multimodal",
}
```

### 4.3 Models Grouped by Family (from manifest JSON files)

| Family | Models | Count |
|--------|--------|-------|
| `bark` | bark-large, bark-small | 2 |
| `bert` | bert-base-uncased | 1 |
| `bloom` | bloom-560m | 1 |
| `codegen` | codegen-350m | 1 |
| `deepseek_v2` | deepseek-v2-lite, deepseek-v2-tiny | 2 |
| `eagle_vlm` | eagle-embed-vl-1b-v2, eagle-rerank-vl-1b-v2 | 2 |
| `falcon` | falcon-rw-1b | 1 |
| `flux` | flux-schnell | 1 |
| `gemma` | gemma-2-2b | 1 |
| `gpt2` | gpt2-125m | 1 |
| `gpt_neo` | gpt-neo-125m | 1 |
| `gpt_neox` | pythia-70m | 1 |
| `granite` | granite-3.1-2b | 1 |
| `internlm` | internlm2-1.8b | 1 |
| `internvl` | internvl3-8b | 1 |
| `llama` | falcon3-1b, minitron-4b-depth, minitron-4b-width, nemotron-nano-4b, tinyllama-1.1b | 5 |
| `mamba` | mamba-130m | 1 |
| `mistral` | mistral-7b, riva-translate-4b | 2 |
| `mixtral` | mixtral-stories-15m | 1 |
| `nemotron` | nemotron-hindi-4b, nemotron-mini-4b | 2 |
| `nemotron_h` | nemotron-h-nano-9b | 1 |
| `olmo` | olmo-1b | 1 |
| `opt` | opt-125m | 1 |
| `personaplex` | personaplex-7b | 1 |
| `phi` | phi3-mini | 1 |
| `phi4_multimodal` | phi4-multimodal | 1 |
| `phi_moe` | phi-moe | 1 |
| `qwen` | qwen3-0.6b, qwen3-4b-instruct-2507 | 2 |
| `qwen_moe` | qwen3-moe-30b-a3b | 1 |
| `qwen_vl` | qwen25vl-3b, qwen3-vl-2b | 2 |
| `rwkv` | rwkv-169m | 1 |
| `sam` | sam-vit-base | 1 |
| `segformer` | segformer-b0-ade | 1 |
| `stablelm` | stablelm2-1.6b | 1 |
| `starcoder2` | starcoder2-3b | 1 |
| `wan_t2v` | wan21-t2v-1.3b | 1 |
| `whisper` | whisper-tiny | 1 |
| `xglm` | xglm-564m | 1 |
| `z_image` | z-image-turbo | 1 |

**Total: 38 families, 50 models**

### 4.4 Models Grouped by Runtime Strategy

| Runtime Strategy | Task Strategy | Models | Count |
|-----------------|---------------|--------|-------|
| `decoder_kv_cache` | `text_generation_causal` | bloom-560m, codegen-350m, deepseek-v2-lite, deepseek-v2-tiny, falcon-rw-1b, falcon3-1b, gemma-2-2b, gpt-neo-125m, gpt2-125m, granite-3.1-2b, internlm2-1.8b, minitron-4b-depth, minitron-4b-width, mistral-7b, nemotron-hindi-4b, nemotron-mini-4b, nemotron-nano-4b, olmo-1b, opt-125m, phi-moe, phi3-mini, pythia-70m, qwen3-0.6b, qwen3-4b-instruct-2507, riva-translate-4b, stablelm2-1.6b, starcoder2-3b, tinyllama-1.1b, xglm-564m | 29 |
| `decoder_moe` | `text_generation_causal` | mixtral-stories-15m, qwen3-moe-30b-a3b | 2 |
| `ssm_recurrent` | `text_generation_causal` | mamba-130m | 1 |
| `rwkv_recurrent` | `text_generation_causal` | rwkv-169m | 1 |
| `hybrid_mamba_attention` | `text_generation_causal` | nemotron-h-nano-9b | 1 |
| `vision_language` | `vision_language_generation` | internvl3-8b, phi4-multimodal, qwen25vl-3b, qwen3-vl-2b | 4 |
| `diffusion` | `diffusion_media_generation` | flux-schnell, wan21-t2v-1.3b, z-image-turbo | 3 |
| `text_to_audio` | `text_to_audio` | bark-large, bark-small | 2 |
| `embedding` | `embedding` | eagle-embed-vl-1b-v2 | 1 |
| `reranking` | `reranking` | eagle-rerank-vl-1b-v2 | 1 |
| `encoder_only` | `encoder_only_nlp` | bert-base-uncased | 1 |
| `speech_to_text` | `speech_to_text` | whisper-tiny | 1 |
| `speech_to_speech` | `speech_to_speech` | personaplex-7b | 1 |
| `segmentation` | `segmentation` | segformer-b0-ade | 1 |
| `prompted_segmentation` | `prompted_segmentation` | sam-vit-base | 1 |

---

## 5. Complete File-to-Model Mapping

This is the core of the TIA system — a complete, exhaustive mapping from every source file to the models it affects.

### 5.1 Python Family Plugins → Models

Each plugin file maps to models via the `family` field in their JSON manifest.

| Plugin File | Family Name | Affected Models |
|------------|-------------|-----------------|
| `trtf_build/trtf_build/families/bark.py` | bark | bark-large, bark-small |
| `trtf_build/trtf_build/families/bert.py` | bert | bert-base-uncased |
| `trtf_build/trtf_build/families/bloom.py` | bloom | bloom-560m |
| `trtf_build/trtf_build/families/codegen.py` | codegen | codegen-350m |
| `trtf_build/trtf_build/families/deepseek_ocr.py` | deepseek_ocr | *(no manifest yet)* |
| `trtf_build/trtf_build/families/deepseek_v2.py` | deepseek_v2 | deepseek-v2-lite, deepseek-v2-tiny |
| `trtf_build/trtf_build/families/eagle_vlm.py` | eagle_vlm | eagle-embed-vl-1b-v2, eagle-rerank-vl-1b-v2 |
| `trtf_build/trtf_build/families/falcon.py` | falcon | falcon-rw-1b |
| `trtf_build/trtf_build/families/flux.py` | flux | flux-schnell |
| `trtf_build/trtf_build/families/gemma.py` | gemma | gemma-2-2b |
| `trtf_build/trtf_build/families/gpt2.py` | gpt2 | gpt2-125m |
| `trtf_build/trtf_build/families/gpt_neo.py` | gpt_neo | gpt-neo-125m |
| `trtf_build/trtf_build/families/gpt_neox.py` | gpt_neox | pythia-70m |
| `trtf_build/trtf_build/families/granite.py` | granite | granite-3.1-2b |
| `trtf_build/trtf_build/families/internlm.py` | internlm | internlm2-1.8b |
| `trtf_build/trtf_build/families/internvl.py` | internvl | internvl3-8b |
| `trtf_build/trtf_build/families/llama.py` | llama | falcon3-1b, minitron-4b-depth, minitron-4b-width, nemotron-nano-4b, tinyllama-1.1b |
| `trtf_build/trtf_build/families/mamba.py` | mamba | mamba-130m |
| `trtf_build/trtf_build/families/mistral.py` | mistral | mistral-7b, riva-translate-4b |
| `trtf_build/trtf_build/families/mixtral.py` | mixtral | mixtral-stories-15m |
| `trtf_build/trtf_build/families/nemotron.py` | nemotron | nemotron-hindi-4b, nemotron-mini-4b |
| `trtf_build/trtf_build/families/nemotron_h.py` | nemotron_h | nemotron-h-nano-9b |
| `trtf_build/trtf_build/families/olmo.py` | olmo | olmo-1b |
| `trtf_build/trtf_build/families/opt.py` | opt | opt-125m |
| `trtf_build/trtf_build/families/personaplex.py` | personaplex | personaplex-7b |
| `trtf_build/trtf_build/families/phi.py` | phi | phi3-mini |
| `trtf_build/trtf_build/families/phi4_multimodal.py` | phi4_multimodal | phi4-multimodal |
| `trtf_build/trtf_build/families/phi_moe.py` | phi_moe | phi-moe |
| `trtf_build/trtf_build/families/qwen.py` | qwen | qwen3-0.6b, qwen3-4b-instruct-2507 |
| `trtf_build/trtf_build/families/qwen3_omni.py` | qwen3_omni | *(no manifest yet)* |
| `trtf_build/trtf_build/families/qwen_moe.py` | qwen_moe | qwen3-moe-30b-a3b |
| `trtf_build/trtf_build/families/qwen_vl.py` | qwen_vl | qwen25vl-3b, qwen3-vl-2b |
| `trtf_build/trtf_build/families/rwkv.py` | rwkv | rwkv-169m |
| `trtf_build/trtf_build/families/sam.py` | sam | sam-vit-base |
| `trtf_build/trtf_build/families/segformer.py` | segformer | segformer-b0-ade |
| `trtf_build/trtf_build/families/stablelm.py` | stablelm | stablelm2-1.6b |
| `trtf_build/trtf_build/families/starcoder2.py` | starcoder2 | starcoder2-3b |
| `trtf_build/trtf_build/families/wan_t2v.py` | wan_t2v | wan21-t2v-1.3b |
| `trtf_build/trtf_build/families/whisper.py` | whisper | whisper-tiny |
| `trtf_build/trtf_build/families/xglm.py` | xglm | xglm-564m |
| `trtf_build/trtf_build/families/z_image.py` | z_image | z-image-turbo |

### 5.2 C++ Runtime Backends → Models (via runtime_strategy)

| C++ Backend File(s) | Runtime Strategy | Affected Models | Count |
|---------------------|-----------------|-----------------|-------|
| `src/runtime/trt/trt_backend_shared.cpp/h` | decoder_kv_cache, decoder_moe, vision_language | *(see §4.4 — all decoder + VL)* | 35 |
| `src/runtime/trt/device_kv_cache.cpp/h` | decoder_kv_cache, decoder_moe, vision_language | *(all decoder + VL)* | 35 |
| `src/runtime/trt/trt_decode_runtime.cpp/h` | decoder_kv_cache, decoder_moe, vision_language, hybrid_mamba_attention | *(all decoder + VL + hybrid)* | 36 |
| `src/runtime/trt/mamba_backend.cpp/h` | ssm_recurrent | mamba-130m | 1 |
| `src/runtime/trt/mamba_decode_runtime.cpp/h` | ssm_recurrent | mamba-130m | 1 |
| `src/runtime/trt/mamba_step_state.cpp/h` | ssm_recurrent | mamba-130m | 1 |
| `src/runtime/trt/rwkv_backend.cpp/h` | rwkv_recurrent | rwkv-169m | 1 |
| `src/runtime/trt/rwkv_decode_runtime.cpp/h` | rwkv_recurrent | rwkv-169m | 1 |
| `src/runtime/trt/rwkv_step_state.cpp/h` | rwkv_recurrent | rwkv-169m | 1 |
| `src/runtime/trt/hybrid_backend.cpp/h` | hybrid_mamba_attention | nemotron-h-nano-9b | 1 |
| `src/runtime/trt/vl_backend.cpp/h` | vision_language | qwen25vl-3b, qwen3-vl-2b, internvl3-8b, phi4-multimodal | 4 |
| `src/runtime/trt/vision_engine.cpp/h` | vision_language | qwen25vl-3b, qwen3-vl-2b, internvl3-8b, phi4-multimodal | 4 |
| `src/runtime/trt/image_preprocessor.cpp/h` | vision_language | qwen25vl-3b, qwen3-vl-2b, internvl3-8b, phi4-multimodal | 4 |
| `src/runtime/trt/whisper_backend.cpp/h` | speech_to_text | whisper-tiny | 1 |
| `src/runtime/trt/bark_backend.cpp/h` | text_to_audio | bark-large, bark-small | 2 |
| `src/runtime/trt/speech_backend.cpp/h` | speech_to_speech | personaplex-7b | 1 |
| `src/runtime/trt/segmentation_backend.cpp/h` | segmentation | segformer-b0-ade | 1 |
| `src/runtime/trt/sam_backend.cpp/h` | prompted_segmentation | sam-vit-base | 1 |
| `src/runtime/trt/encoder_backend.cpp/h` | encoder_only | bert-base-uncased | 1 |
| `src/runtime/trt/embedding_backend.cpp/h` | embedding | eagle-embed-vl-1b-v2 | 1 |
| `src/runtime/trt/reranking_backend.cpp/h` | reranking | eagle-rerank-vl-1b-v2 | 1 |
| `src/runtime/trt/omni_backend.cpp/h` | omni_multimodal | *(no manifest yet)* | 0 |
| `src/runtime/trt/wan_diffusion_backend.cpp/h` | diffusion (wan) | wan21-t2v-1.3b | 1 |
| `src/runtime/trt/flux_diffusion_backend.cpp/h` | diffusion (flux) | flux-schnell | 1 |
| `src/runtime/trt/z_image_diffusion_backend.cpp/h` | diffusion (z_image) | z-image-turbo | 1 |
| `src/runtime/trt/diffusion_backend_base.cpp` | diffusion (all) | flux-schnell, wan21-t2v-1.3b, z-image-turbo | 3 |
| `src/runtime/trt/diffusion_backend.cpp/h` | diffusion (all) | flux-schnell, wan21-t2v-1.3b, z-image-turbo | 3 |

### 5.3 Python Builder Modules → Models

| Builder Module | Usage | Affected Models |
|---------------|-------|-----------------|
| `trtf_build/trtf_build/standard_decoder_builder.py` | Used by all standard decoder families | **ALL decoder models** (~34 models) |
| `trtf_build/trtf_build/graph_blocks.py` | Composable blocks used by standard_decoder_builder + others | **ALL models** |
| `trtf_build/trtf_build/graph_ops.py` | Atomic TRT ops used by everything | **ALL models** |
| `trtf_build/trtf_build/checkpoint_mapper.py` | Weight loading for all models | **ALL models** |
| `trtf_build/trtf_build/bundle_writer.py` | Bundle writing for all models | **ALL models** |
| `trtf_build/trtf_build/engine_builder.py` | Build orchestrator for all models | **ALL models** |
| `trtf_build/trtf_build/config.py` | Config parsing for all models | **ALL models** |
| `trtf_build/trtf_build/debug_runner.py` | Python TRT runner (parity-critical) | **ALL models** |
| `trtf_build/trtf_build/pipeline.py` | Subprocess wrapper | **ALL models** |
| `trtf_build/trtf_build/cli.py` | CLI entry point | **ALL models** |
| `trtf_build/trtf_build/__init__.py` | Public API | **ALL models** |
| `trtf_build/trtf_build/__main__.py` | CLI entry | **ALL models** |
| `trtf_build/trtf_build/qwen_vl_vision_builder.py` | Qwen VL vision encoder builder | qwen25vl-3b, qwen3-vl-2b |
| `trtf_build/trtf_build/qwen3_encoder_builder.py` | Qwen3-VL DeepStack builder | qwen3-vl-2b |
| `trtf_build/trtf_build/internvit_vision_builder.py` | InternVL vision encoder | internvl3-8b |
| `trtf_build/trtf_build/phi4mm_vision_builder.py` | Phi4 multimodal vision | phi4-multimodal |
| `trtf_build/trtf_build/vision_encoder_builder.py` | Generic ViT builder | *(VL models that use it)* |
| `trtf_build/trtf_build/clip_encoder_builder.py` | CLIP encoder for Z-Image | z-image-turbo |
| `trtf_build/trtf_build/flux_dit_builder.py` | FLUX DiT builder | flux-schnell |
| `trtf_build/trtf_build/flux_vae_builder.py` | FLUX VAE decoder | flux-schnell |
| `trtf_build/trtf_build/z_image_dit_builder.py` | Z-Image DiT builder | z-image-turbo |
| `trtf_build/trtf_build/standard_dit_builder.py` | Standard DiT (shared diffusion) | flux-schnell, z-image-turbo, wan21-t2v-1.3b |
| `trtf_build/trtf_build/causal_vae_3d_builder.py` | 3D VAE for Wan T2V | wan21-t2v-1.3b |
| `trtf_build/trtf_build/vae_2d_builder.py` | 2D VAE for Z-Image | z-image-turbo |
| `trtf_build/trtf_build/t5_encoder_builder.py` | T5 text encoder (Bark, Wan) | bark-large, bark-small, wan21-t2v-1.3b |
| `trtf_build/trtf_build/encodec_builder.py` | Audio codec (Bark) | bark-large, bark-small |
| `trtf_build/trtf_build/encoder_builder.py` | SegFormer encoder | segformer-b0-ade |
| `trtf_build/trtf_build/diffusion_runner.py` | Python diffusion runner | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| `trtf_build/trtf_build/schedulers/` | Diffusion schedulers | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| `trtf_build/trtf_build/onnx_vision_builder.py` | *(unused/legacy)* | — |

### 5.4 E2E Harness Files → Models (via task_strategy)

**Strategy Runners** (from `tests/e2e_harness/runners/`):

| Runner File | strategy_name(s) | Affected Models |
|------------|-------------------|-----------------|
| `runners/text_generation.py` | text_generation_causal | ALL text gen models (34) |
| `runners/vision_language.py` | vision_language_generation | qwen25vl-3b, qwen3-vl-2b, internvl3-8b, phi4-multimodal |
| `runners/audio_speech.py` | speech_to_text, text_to_audio, speech_to_speech | whisper-tiny, bark-large, bark-small, personaplex-7b |
| `runners/diffusion.py` | diffusion_media_generation | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| `runners/segmentation.py` | segmentation, prompted_segmentation | segformer-b0-ade, sam-vit-base |
| `runners/embedding.py` | embedding | eagle-embed-vl-1b-v2 |
| `runners/reranking.py` | reranking | eagle-rerank-vl-1b-v2 |
| `runners/encoder_only.py` | encoder_only_nlp | bert-base-uncased |
| `runners/omni.py` | omni_multimodal, composite_pipeline | *(no manifest yet)* |
| `runners/neural_operator.py` | neural_operator | *(no manifest yet)* |
| `runners/object_detection.py` | object_detection | *(no manifest yet)* |

**Comparators** (from `tests/e2e_harness/comparators/`):

| Comparator File | task_strategy | Affected Models |
|----------------|---------------|-----------------|
| `comparators/text.py` | text_generation_causal | ALL text gen models (34) |
| `comparators/vision_language.py` | vision_language_generation | qwen25vl-3b, qwen3-vl-2b, internvl3-8b, phi4-multimodal |
| `comparators/speech_to_text.py` | speech_to_text | whisper-tiny |
| `comparators/text_to_audio.py` | text_to_audio | bark-large, bark-small |
| `comparators/speech_to_speech.py` | speech_to_speech | personaplex-7b |
| `comparators/audio.py` | *(alias for speech_to_text)* | whisper-tiny |
| `comparators/diffusion.py` | diffusion_media_generation | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| `comparators/segmentation.py` | segmentation, prompted_segmentation | segformer-b0-ade, sam-vit-base |
| `comparators/embedding.py` | embedding | eagle-embed-vl-1b-v2 |
| `comparators/reranking.py` | reranking | eagle-rerank-vl-1b-v2 |
| `comparators/encoder_only.py` | encoder_only_nlp | bert-base-uncased |
| `comparators/omni.py` | omni_multimodal | *(no manifest yet)* |
| `comparators/neural_operator.py` | neural_operator | *(no manifest yet)* |

**Reference Backends** (from `tests/e2e_harness/references/`):

| Reference File | Used By Task Strategies | Affected Models |
|---------------|------------------------|-----------------|
| `references/hf_transformers.py` | text_generation_causal, vision_language_generation, speech_to_text, text_to_audio, segmentation, prompted_segmentation, embedding, reranking, encoder_only_nlp | **MOST models** (~45) |
| `references/hf_diffusers.py` | diffusion_media_generation | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| `references/torch_reference.py` | speech_to_speech, omni_multimodal | personaplex-7b |
| `references/golden_snapshot.py` | *(any, regression)* | Depends on usage |
| `references/custom_python.py` | *(any, specialized)* | Depends on usage |
| `references/invariant_only.py` | *(any, property-based)* | Depends on usage |

**Harness Infrastructure** (always global):

| Harness File | Affected Models |
|-------------|-----------------|
| `tests/e2e_harness/contracts.py` | **ALL** |
| `tests/e2e_harness/orchestrator.py` | **ALL** |
| `tests/e2e_harness/registry.py` | **ALL** |
| `tests/e2e_harness/manifest_loader.py` | **ALL** |
| `tests/e2e_harness/artifact_sink.py` | **ALL** |

### 5.5 C++ Shared Infrastructure → ALL Models

| File | Component |
|------|-----------|
| `src/runtime/trt/trt_common.cpp/h` | TRT logger, CUDA helpers (CudaBuffer, CudaStream) |
| `src/runtime/trt/trt_engine_lifecycle.cpp/h` | DecoderStepEngine, tensor validation |
| `src/runtime/trt/step_state.h` | IStepState interface |
| `src/runtime/trt/stb_impl.cpp` | Image loading (stb_image) |
| `src/bundle/bundle_format.cpp/h` | .trtfb file reading |
| `src/cabi/trtf_c.cpp` | C ABI entry point |
| `src/cabi/fast_path_config.cpp/h` | Bundle metadata parsing |
| `src/cabi/bundle_helpers.cpp/h` | Shared plumbing |
| `src/tokenizer/vocab_tokenizer.cpp` | Vocab-based tokenizer |
| `src/tokenizer/hf_python_tokenizer.cpp` | HF tokenizer subprocess bridge |
| `src/tokenizer/hf_python_tokenizer_helpers.h` | Tokenizer helpers |
| `include/trtf/backend.h` | Public API: backend interface |
| `include/trtf/bundle.h` | Public API: bundle loading |
| `include/trtf/generation.h` | Public API: generation |
| `include/trtf/pipeline.h` | Public API: pipeline |
| `include/trtf/tokenizer.h` | Public API: tokenizer |
| `CMakeLists.txt` | Build system |

### 5.6 Manifest Files → Self (1:1)

| File Pattern | Affected Model |
|-------------|---------------|
| `tests/e2e/models/<model-name>.json` | `<model-name>` (only that specific model) |

### 5.7 Test Infrastructure → No E2E Impact (Unit Test Only)

| File Pattern | Impact |
|-------------|--------|
| `tests/builder/*.py` | Unit tests only, no E2E impact |
| `tests/tools/*.py` | Tool self-tests only, no E2E impact |
| `tests/cpp/*.cpp` | C++ unit tests only, no E2E impact |
| `tools/*.py` | Diff framework tools, no E2E impact |
| `scripts/*.py` | Utility scripts, no E2E impact |
| `docs/*` | Documentation, no impact |

---

## 6. Design: 3-Tier File Classification

### Tier 1 — Leaf Files (1-5 models affected)

Changes that affect a small, well-defined set of models. Typical time savings: **95-99%**.

```
families/<name>.py                              → manifests where family=<name>
tests/e2e/models/<model>.json                   → [<model>]
runners/segmentation.py                          → segmentation + prompted_segmentation models
runners/embedding.py                             → embedding models
runners/reranking.py                             → reranking models
runners/encoder_only.py                          → encoder_only_nlp models
comparators/speech_to_text.py                    → whisper-tiny
comparators/text_to_audio.py                     → bark-large, bark-small
comparators/speech_to_speech.py                  → personaplex-7b
comparators/embedding.py                         → eagle-embed-vl-1b-v2
comparators/reranking.py                         → eagle-rerank-vl-1b-v2
comparators/encoder_only.py                      → bert-base-uncased
src/runtime/trt/mamba_*.cpp/h                    → mamba-130m
src/runtime/trt/rwkv_*.cpp/h                     → rwkv-169m
src/runtime/trt/hybrid_backend.cpp/h             → nemotron-h-nano-9b
src/runtime/trt/whisper_backend.cpp/h            → whisper-tiny
src/runtime/trt/bark_backend.cpp/h               → bark-large, bark-small
src/runtime/trt/speech_backend.cpp/h             → personaplex-7b
src/runtime/trt/segmentation_backend.cpp/h       → segformer-b0-ade
src/runtime/trt/sam_backend.cpp/h                → sam-vit-base
src/runtime/trt/encoder_backend.cpp/h            → bert-base-uncased
src/runtime/trt/embedding_backend.cpp/h          → eagle-embed-vl-1b-v2
src/runtime/trt/reranking_backend.cpp/h          → eagle-rerank-vl-1b-v2
src/runtime/trt/wan_diffusion_backend.cpp/h      → wan21-t2v-1.3b
src/runtime/trt/flux_diffusion_backend.cpp/h     → flux-schnell
src/runtime/trt/z_image_diffusion_backend.cpp/h  → z-image-turbo
trtf_build/trtf_build/qwen_vl_vision_builder.py  → qwen25vl-3b, qwen3-vl-2b
trtf_build/trtf_build/qwen3_encoder_builder.py   → qwen3-vl-2b
trtf_build/trtf_build/internvit_vision_builder.py→ internvl3-8b
trtf_build/trtf_build/phi4mm_vision_builder.py   → phi4-multimodal
trtf_build/trtf_build/clip_encoder_builder.py    → z-image-turbo
trtf_build/trtf_build/flux_dit_builder.py        → flux-schnell
trtf_build/trtf_build/flux_vae_builder.py        → flux-schnell
trtf_build/trtf_build/z_image_dit_builder.py     → z-image-turbo
trtf_build/trtf_build/vae_2d_builder.py          → z-image-turbo
trtf_build/trtf_build/causal_vae_3d_builder.py   → wan21-t2v-1.3b
trtf_build/trtf_build/t5_encoder_builder.py      → bark-large, bark-small, wan21-t2v-1.3b
trtf_build/trtf_build/encodec_builder.py         → bark-large, bark-small
trtf_build/trtf_build/encoder_builder.py         → segformer-b0-ade
```

### Tier 2 — Strategy-Scoped Files (4-35 models affected)

Changes that affect an entire runtime strategy group. Time savings: **30-90%**.

```
src/runtime/trt/vl_backend.cpp/h                 → vision_language (4 models)
src/runtime/trt/vision_engine.cpp/h              → vision_language (4 models)
src/runtime/trt/image_preprocessor.cpp/h         → vision_language (4 models)
src/runtime/trt/diffusion_backend_base.cpp       → diffusion (3 models)
src/runtime/trt/diffusion_backend.cpp/h          → diffusion (3 models)
src/runtime/trt/device_kv_cache.cpp/h            → decoder_kv_cache + decoder_moe + VL (35 models)
src/runtime/trt/trt_backend_shared.cpp/h         → decoder_kv_cache + decoder_moe + VL (35 models)
src/runtime/trt/trt_decode_runtime.cpp/h         → all decoder + VL + hybrid (36 models)
runners/text_generation.py                        → text_generation_causal (34 models)
runners/vision_language.py                        → vision_language_generation (4 models)
runners/audio_speech.py                           → speech_to_text + text_to_audio + speech_to_speech (4 models)
runners/diffusion.py                              → diffusion_media_generation (3 models)
comparators/text.py                               → text_generation_causal (34 models)
comparators/vision_language.py                    → vision_language_generation (4 models)
comparators/diffusion.py                          → diffusion_media_generation (3 models)
comparators/segmentation.py                       → segmentation + prompted_segmentation (2 models)
references/hf_diffusers.py                        → diffusion_media_generation (3 models)
references/torch_reference.py                     → speech_to_speech (1 model)
trtf_build/trtf_build/standard_decoder_builder.py→ all standard decoder families (~34 models)
trtf_build/trtf_build/standard_dit_builder.py    → all diffusion models (3 models)
trtf_build/trtf_build/diffusion_runner.py        → all diffusion models (3 models)
trtf_build/trtf_build/vision_encoder_builder.py  → VL models using generic ViT
```

### Tier 3 — Global Files (ALL 50 models)

Changes to these files potentially affect every model. No test reduction possible.

```
# Python core builder infrastructure
trtf_build/trtf_build/graph_ops.py
trtf_build/trtf_build/graph_blocks.py
trtf_build/trtf_build/checkpoint_mapper.py
trtf_build/trtf_build/bundle_writer.py
trtf_build/trtf_build/engine_builder.py
trtf_build/trtf_build/config.py
trtf_build/trtf_build/debug_runner.py
trtf_build/trtf_build/pipeline.py
trtf_build/trtf_build/__init__.py
trtf_build/trtf_build/__main__.py
trtf_build/trtf_build/cli.py
trtf_build/trtf_build/families/__init__.py       # Plugin auto-discovery
trtf_build/trtf_build/families/base.py           # Plugin protocol

# C++ shared infrastructure
src/runtime/trt/trt_common.cpp/h
src/runtime/trt/trt_engine_lifecycle.cpp/h
src/runtime/trt/step_state.h
src/runtime/trt/stb_impl.cpp
src/bundle/bundle_format.cpp/h
src/cabi/trtf_c.cpp
src/cabi/fast_path_config.cpp/h
src/cabi/bundle_helpers.cpp/h
src/tokenizer/vocab_tokenizer.cpp
src/tokenizer/hf_python_tokenizer.cpp
src/tokenizer/hf_python_tokenizer_helpers.h
include/trtf/*.h                                  # All public API headers

# E2E harness infrastructure
tests/e2e_harness/contracts.py
tests/e2e_harness/orchestrator.py
tests/e2e_harness/registry.py
tests/e2e_harness/manifest_loader.py
tests/e2e_harness/artifact_sink.py
tests/test_e2e.py
references/hf_transformers.py                     # Used by ~45 models

# Build system
CMakeLists.txt
pyproject.toml
conftest.py
```

---

## 7. Implementation Plan

### Overview

| Phase | Deliverable | Effort | Dependencies |
|-------|------------|--------|-------------|
| **Phase 1** | pytest `conftest.py` plugin (`--affected-only`) | ~150 LOC | None |
| **Phase 2** | Standalone CLI tool (`scripts/test_selector.py`) | ~200 LOC | Phase 1 |
| **Phase 3** | CI/CD integration (GitLab CI / GitHub Actions) | ~50 LOC | Phase 2 |
| **Phase 4** | Function-level refinement (pytest-testmon) | ~100 LOC config | Phase 1 |

### Timeline

- Phase 1: Implement and validate in 1 session
- Phase 2: Extract into CLI tool, add JSON output
- Phase 3: Integrate into CI pipeline
- Phase 4: (Optional) Layer testmon for graph_ops.py refinement

---

## 8. Phase 1: pytest conftest.py Plugin

### 8.1 Design

Add `--affected-only` and `--changed-files` options to the existing `conftest.py`. When `--affected-only` is passed, the plugin:

1. Runs `git diff --name-only <base>..HEAD` to get changed files (or reads `--changed-files`)
2. Classifies each file using the 3-tier mapping rules
3. Computes the union of affected model names
4. Deselects unaffected tests via `pytest_collection_modifyitems`
5. Prints a summary: "TIA: testing N/50 models due to changes in: [file list]"

### 8.2 New File: `tests/test_impact.py`

This module contains the core mapping logic, separate from conftest.py for testability.

```python
"""Test Impact Analysis — map changed files to affected E2E model tests.

Usage:
    from tests.test_impact import compute_affected_models

    affected = compute_affected_models(
        changed_files=["trtf_build/trtf_build/families/qwen.py"],
        manifest_dir="tests/e2e/models",
    )
    # Returns: {"qwen3-0.6b", "qwen3-4b-instruct-2507"}
"""

import json
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Manifest database (loaded lazily)
# ---------------------------------------------------------------------------

def _load_manifest_db(manifest_dir: str | Path) -> dict:
    """Load all manifests and build lookup indexes.

    Returns dict with:
        - models: {name: {family, runtime_strategy, task_strategy, ...}}
        - by_family: {family_name: [model_names]}
        - by_runtime_strategy: {strategy: [model_names]}
        - by_task_strategy: {strategy: [model_names]}
        - all_model_names: set of all model names
    """
    ...


# ---------------------------------------------------------------------------
# RUNTIME_TO_TASK_STRATEGY (mirrored from contracts.py)
# ---------------------------------------------------------------------------

RUNTIME_TO_TASK_STRATEGY = {
    "decoder_kv_cache": "text_generation_causal",
    "decoder_moe": "text_generation_causal",
    "ssm_recurrent": "text_generation_causal",
    "rwkv_recurrent": "text_generation_causal",
    "hybrid_mamba_attention": "text_generation_causal",
    "vision_language": "vision_language_generation",
    "speech_to_text": "speech_to_text",
    "text_to_audio": "text_to_audio",
    "speech_to_speech": "speech_to_speech",
    "segmentation": "segmentation",
    "prompted_segmentation": "prompted_segmentation",
    "object_detection": "object_detection",
    "embedding": "embedding",
    "reranking": "reranking",
    "encoder_only": "encoder_only_nlp",
    "neural_operator": "neural_operator",
    "diffusion": "diffusion_media_generation",
    "omni_multimodal": "omni_multimodal",
}


# ---------------------------------------------------------------------------
# Mapping rules
# ---------------------------------------------------------------------------

# C++ backend file -> runtime_strategies it serves
CPP_BACKEND_TO_STRATEGIES = {
    # Shared decoder
    "trt_backend_shared": ["decoder_kv_cache", "decoder_moe", "vision_language"],
    "device_kv_cache": ["decoder_kv_cache", "decoder_moe", "vision_language"],
    "trt_decode_runtime": [
        "decoder_kv_cache", "decoder_moe", "vision_language",
        "hybrid_mamba_attention",
    ],
    # SSM
    "mamba_backend": ["ssm_recurrent"],
    "mamba_decode_runtime": ["ssm_recurrent"],
    "mamba_step_state": ["ssm_recurrent"],
    # RWKV
    "rwkv_backend": ["rwkv_recurrent"],
    "rwkv_decode_runtime": ["rwkv_recurrent"],
    "rwkv_step_state": ["rwkv_recurrent"],
    # Hybrid
    "hybrid_backend": ["hybrid_mamba_attention"],
    # Vision-language
    "vl_backend": ["vision_language"],
    "vision_engine": ["vision_language"],
    "image_preprocessor": ["vision_language"],
    # Speech/audio
    "whisper_backend": ["speech_to_text"],
    "bark_backend": ["text_to_audio"],
    "speech_backend": ["speech_to_speech"],
    # Segmentation
    "segmentation_backend": ["segmentation"],
    "sam_backend": ["prompted_segmentation"],
    # Encoder/embedding
    "encoder_backend": ["encoder_only"],
    "embedding_backend": ["embedding"],
    "reranking_backend": ["reranking"],
    # Omni
    "omni_backend": ["omni_multimodal"],
    # Diffusion (specific)
    "wan_diffusion_backend": ["diffusion"],  # further filtered to wan models
    "flux_diffusion_backend": ["diffusion"],  # further filtered to flux models
    "z_image_diffusion_backend": ["diffusion"],  # further filtered to z_image models
    # Diffusion (shared)
    "diffusion_backend_base": ["diffusion"],
    "diffusion_backend": ["diffusion"],
}

# Builder module -> specific family/models (not global)
BUILDER_TO_FAMILIES = {
    "qwen_vl_vision_builder": ["qwen_vl"],
    "qwen3_encoder_builder": ["qwen_vl"],  # Qwen3-VL DeepStack
    "internvit_vision_builder": ["internvl"],
    "phi4mm_vision_builder": ["phi4_multimodal"],
    "clip_encoder_builder": ["z_image"],
    "flux_dit_builder": ["flux"],
    "flux_vae_builder": ["flux"],
    "z_image_dit_builder": ["z_image"],
    "vae_2d_builder": ["z_image"],
    "causal_vae_3d_builder": ["wan_t2v"],
    "t5_encoder_builder": ["bark", "wan_t2v"],
    "encodec_builder": ["bark"],
    "encoder_builder": ["segformer"],
    "diffusion_runner": ["flux", "wan_t2v", "z_image"],
    "standard_dit_builder": ["flux", "wan_t2v", "z_image"],
}

# Runner file -> task_strategies
RUNNER_TO_STRATEGIES = {
    "text_generation": ["text_generation_causal"],
    "vision_language": ["vision_language_generation"],
    "audio_speech": ["speech_to_text", "text_to_audio", "speech_to_speech"],
    "diffusion": ["diffusion_media_generation"],
    "segmentation": ["segmentation", "prompted_segmentation"],
    "embedding": ["embedding"],
    "reranking": ["reranking"],
    "encoder_only": ["encoder_only_nlp"],
    "omni": ["omni_multimodal", "composite_pipeline"],
    "neural_operator": ["neural_operator"],
    "object_detection": ["object_detection"],
}

# Comparator file -> task_strategies
COMPARATOR_TO_STRATEGIES = {
    "text": ["text_generation_causal"],
    "vision_language": ["vision_language_generation"],
    "speech_to_text": ["speech_to_text"],
    "text_to_audio": ["text_to_audio"],
    "speech_to_speech": ["speech_to_speech"],
    "audio": ["speech_to_text"],  # alias
    "diffusion": ["diffusion_media_generation"],
    "segmentation": ["segmentation", "prompted_segmentation"],
    "embedding": ["embedding"],
    "reranking": ["reranking"],
    "encoder_only": ["encoder_only_nlp"],
    "omni": ["omni_multimodal"],
    "neural_operator": ["neural_operator"],
}

# Reference file -> task_strategies
REFERENCE_TO_STRATEGIES = {
    "hf_transformers": None,  # Used by almost all — treat as global
    "hf_diffusers": ["diffusion_media_generation"],
    "torch_reference": ["speech_to_speech", "omni_multimodal"],
    "golden_snapshot": None,  # Generic — treat as global
    "custom_python": None,  # Generic — treat as global
    "invariant_only": None,  # Generic — treat as global
}

# Global files — change triggers ALL models
GLOBAL_PATTERNS = [
    # Python core builder
    r"trtf_build/trtf_build/graph_ops\.py",
    r"trtf_build/trtf_build/graph_blocks\.py",
    r"trtf_build/trtf_build/checkpoint_mapper\.py",
    r"trtf_build/trtf_build/bundle_writer\.py",
    r"trtf_build/trtf_build/engine_builder\.py",
    r"trtf_build/trtf_build/config\.py",
    r"trtf_build/trtf_build/debug_runner\.py",
    r"trtf_build/trtf_build/pipeline\.py",
    r"trtf_build/trtf_build/__init__\.py",
    r"trtf_build/trtf_build/__main__\.py",
    r"trtf_build/trtf_build/cli\.py",
    r"trtf_build/trtf_build/families/__init__\.py",
    r"trtf_build/trtf_build/families/base\.py",
    # C++ shared infrastructure
    r"src/runtime/trt/trt_common\.",
    r"src/runtime/trt/trt_engine_lifecycle\.",
    r"src/runtime/trt/step_state\.h",
    r"src/runtime/trt/stb_impl\.cpp",
    r"src/bundle/",
    r"src/cabi/",
    r"src/tokenizer/",
    r"include/trtf/",
    # E2E harness infrastructure
    r"tests/e2e_harness/contracts\.py",
    r"tests/e2e_harness/orchestrator\.py",
    r"tests/e2e_harness/registry\.py",
    r"tests/e2e_harness/manifest_loader\.py",
    r"tests/e2e_harness/artifact_sink\.py",
    r"tests/test_e2e\.py",
    r"tests/e2e_harness/references/hf_transformers\.py",
    # Build system
    r"CMakeLists\.txt",
    r"pyproject\.toml",
    r"conftest\.py$",
]

# No-impact patterns (unit tests, docs, scripts)
NO_IMPACT_PATTERNS = [
    r"tests/builder/",
    r"tests/tools/",
    r"tests/cpp/",
    r"tools/",
    r"scripts/",
    r"docs/",
    r"todo/",
    r"\.md$",
    r"Dockerfile",
    r"docker-compose",
    r"\.gitignore",
    r"\.claude/",
]


def classify_file(filepath: str, db: dict) -> set[str]:
    """Classify a single changed file and return affected model names.

    Returns:
        Set of model names, or None for "ALL models",
        or empty set for "no E2E impact".
    """
    ...  # Implementation uses the rules above


def compute_affected_models(
    changed_files: list[str],
    manifest_dir: str | Path = "tests/e2e/models",
) -> Optional[set[str]]:
    """Compute the set of affected model names from changed files.

    Returns:
        Set of affected model names, or None if ALL models
        should be tested (global change detected).
    """
    ...
```

### 8.3 conftest.py Additions

```python
# In conftest.py — add options
parser.addoption("--affected-only", action="store_true", default=False,
                 help="Only run E2E tests affected by git changes")
parser.addoption("--changed-files", default=None,
                 help="Comma-separated list of changed files (overrides git)")
parser.addoption("--tia-base", default="origin/master",
                 help="Base ref for git diff (default: origin/master)")

# In conftest.py — deselection hook
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--affected-only"):
        return

    from tests.test_impact import compute_affected_models

    # Get changed files
    changed = config.getoption("--changed-files")
    if changed:
        changed_files = [f.strip() for f in changed.split(",")]
    else:
        import subprocess
        base = config.getoption("--tia-base")
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base}..HEAD"],
            capture_output=True, text=True
        )
        changed_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

    affected = compute_affected_models(changed_files)

    if affected is None:
        # Global change detected — run everything
        print(f"\nTIA: global change detected, running all models")
        return

    if not affected:
        print(f"\nTIA: no E2E-relevant changes detected")
        # Deselect everything
        config.hook.pytest_deselected(items=items[:])
        items.clear()
        return

    # Deselect unaffected tests
    selected = []
    deselected = []
    for item in items:
        if hasattr(item, "callspec") and "model_name" in item.callspec.params:
            model_name = item.callspec.params["model_name"]
            if model_name in affected:
                selected.append(item)
            else:
                deselected.append(item)
        else:
            selected.append(item)  # Non-parametrized tests always run

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected

    print(f"\nTIA: testing {len(affected)}/{len(affected) + len(deselected)} models")
    print(f"  Affected: {sorted(affected)}")
    print(f"  Changed files: {changed_files}")
```

### 8.4 Usage Examples

```bash
# Auto-detect changes vs origin/master:
pytest tests/test_e2e.py --affected-only \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python

# Explicit changed files:
pytest tests/test_e2e.py --affected-only \
  --changed-files "trtf_build/trtf_build/families/qwen.py" \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python
# Output: TIA: testing 2/50 models
#   Affected: ['qwen3-0.6b', 'qwen3-4b-instruct-2507']

# Against a different base:
pytest tests/test_e2e.py --affected-only --tia-base origin/feature-branch \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python

# Combine with existing filters (intersection):
pytest tests/test_e2e.py --affected-only --e2e-task-strategy text_generation_causal \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python
```

---

## 9. Phase 2: Standalone CLI Tool

### 9.1 File: `scripts/test_selector.py`

A standalone CLI tool that can be used outside of pytest, e.g. in CI scripts or pre-commit hooks.

```bash
# From git diff:
python scripts/test_selector.py --base origin/master --head HEAD
# Output (JSON):
{
  "changed_files": [
    "trtf_build/trtf_build/families/qwen.py",
    "tests/e2e/models/qwen3-0.6b.json"
  ],
  "affected_models": ["qwen3-0.6b", "qwen3-4b-instruct-2507"],
  "total_models": 50,
  "affected_count": 2,
  "tier": "leaf",
  "is_global": false,
  "pytest_filter": "-k 'qwen3-0.6b or qwen3-4b-instruct-2507'",
  "pytest_cmd": "pytest tests/test_e2e.py --affected-only --changed-files 'trtf_build/trtf_build/families/qwen.py,tests/e2e/models/qwen3-0.6b.json'",
  "classification": {
    "trtf_build/trtf_build/families/qwen.py": {
      "tier": 1,
      "reason": "family plugin: qwen",
      "models": ["qwen3-0.6b", "qwen3-4b-instruct-2507"]
    },
    "tests/e2e/models/qwen3-0.6b.json": {
      "tier": 1,
      "reason": "manifest: qwen3-0.6b",
      "models": ["qwen3-0.6b"]
    }
  }
}

# From explicit file list:
python scripts/test_selector.py --files "src/runtime/trt/mamba_backend.cpp"
# Output:
{
  "affected_models": ["mamba-130m"],
  "tier": "leaf",
  ...
}

# Human-readable mode:
python scripts/test_selector.py --base origin/master --format human
# Output:
# Test Impact Analysis
# ====================
# Changed files: 2
# Affected models: 2/50 (tier: leaf)
#
# trtf_build/trtf_build/families/qwen.py
#   Tier 1 (leaf) — family plugin: qwen
#   → qwen3-0.6b, qwen3-4b-instruct-2507
#
# tests/e2e/models/qwen3-0.6b.json
#   Tier 1 (leaf) — manifest: qwen3-0.6b
#   → qwen3-0.6b
#
# Run: pytest tests/test_e2e.py -k 'qwen3-0.6b or qwen3-4b-instruct-2507'

# Dry-run: just show what would be tested
python scripts/test_selector.py --base origin/master --dry-run
```

### 9.2 Features

- `--format json|human|pytest` — output format
- `--base` / `--head` — git refs for diff
- `--files` — explicit file list (comma-separated)
- `--include-core` — always include `core: true` models regardless
- `--dry-run` — show classification without running tests
- Exit codes: 0 = some tests selected, 1 = no tests needed, 2 = global (all tests)

---

## 10. Phase 3: CI/CD Integration

### 10.1 GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - analyze
  - test-unit
  - test-e2e-affected
  - test-e2e-full  # nightly only

# Stage 1: Determine affected tests
test-impact-analysis:
  stage: analyze
  script:
    - python scripts/test_selector.py --base origin/master --head HEAD --format json > tia_result.json
    - cat tia_result.json
  artifacts:
    paths:
      - tia_result.json
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# Stage 2: Run only affected E2E tests
e2e-affected:
  stage: test-e2e-affected
  needs: [test-impact-analysis]
  script:
    - |
      AFFECTED=$(python -c "import json; d=json.load(open('tia_result.json')); print(d['pytest_cmd'])")
      if [ "$AFFECTED" = "NONE" ]; then
        echo "No E2E tests needed"
        exit 0
      fi
      eval "$AFFECTED" \
        --engine-dir /mnt/storage/trt-transformers/engines \
        --trtf-binary ./build/trtf --hf-python .venv/bin/python
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# Stage 3: Nightly full suite (safety net)
e2e-full-nightly:
  stage: test-e2e-full
  script:
    - pytest tests/test_e2e.py -v \
        --engine-dir /mnt/storage/trt-transformers/engines \
        --trtf-binary ./build/trtf --hf-python .venv/bin/python \
        --rebuild-engines --e2e-artifacts-dir /tmp/e2e_artifacts
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"  # nightly
```

### 10.2 GitHub Actions Example

```yaml
# .github/workflows/e2e-affected.yml
name: E2E (Affected Only)
on:
  pull_request:
    branches: [master]

jobs:
  analyze:
    runs-on: ubuntu-latest
    outputs:
      affected_models: ${{ steps.tia.outputs.affected_models }}
      pytest_cmd: ${{ steps.tia.outputs.pytest_cmd }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - id: tia
        run: |
          result=$(python scripts/test_selector.py --base origin/master --format json)
          echo "affected_models=$(echo $result | jq -r '.affected_models | join(",")')" >> $GITHUB_OUTPUT
          echo "pytest_cmd=$(echo $result | jq -r '.pytest_cmd')" >> $GITHUB_OUTPUT

  e2e-affected:
    needs: analyze
    if: needs.analyze.outputs.affected_models != ''
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: |
          ${{ needs.analyze.outputs.pytest_cmd }} \
            --engine-dir /mnt/storage/trt-transformers/engines \
            --trtf-binary ./build/trtf --hf-python .venv/bin/python
```

---

## 11. Phase 4: Function-Level Refinement

### 11.1 Problem: Tier 3 Over-Selection

When `graph_ops.py` changes, TIA selects all 50 models. But if only `add_alibi_bias()` changed, we'd ideally only run models that use ALiBi (bloom, falcon, etc.), not all 50.

### 11.2 Solution: pytest-testmon for Python-Only Refinement

Layer `pytest-testmon` on top of the static rules for Tier 3 Python files only:

```bash
# Baseline run (weekly, on main):
pytest tests/test_e2e.py --testmon --testmon-nocollect \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python

# PR CI: testmon refines within Tier 3 Python files
pytest tests/test_e2e.py --testmon --affected-only \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python
```

### 11.3 Limitations

- testmon can only refine **Python** Tier 3 files (graph_ops.py, graph_blocks.py, etc.)
- C++ Tier 3 files remain at full-suite level (testmon can't trace subprocess)
- Requires persisting `.testmondata` between CI runs (cache artifact)
- Adds ~5% overhead to E2E test execution time

### 11.4 Alternative: Manual Op-to-Model Mapping

For higher precision without testmon overhead, extend manifests with explicit op dependencies:

```json
{
  "name": "bloom-560m",
  "family": "bloom",
  "depends_on_ops": ["alibi", "layer_norm", "gelu", "attention_kv_cache"],
  "depends_on_blocks": ["add_attention_block", "add_mlp_block"]
}
```

Then parse `graph_ops.py` changes at the function level (AST diff) and match against declared ops. This is higher effort but maximally precise.

---

## 12. Safety Net & Guardrails

### 12.1 Core Model Always-Run

Models with `"core": true` in their manifest always run regardless of TIA. This provides a smoke-test safety net even for narrowly-scoped changes.

Currently `qwen3-0.6b` has `core: true`. Consider adding 1-2 more across modalities:
- `qwen3-0.6b` — text gen smoke test
- `qwen25vl-3b` or `qwen3-vl-2b` — VL smoke test
- `whisper-tiny` — audio smoke test

### 12.2 Nightly Full Suite

Run ALL 50 models nightly (scheduled CI), regardless of changes. This catches:
- Transitive dependencies missed by static rules
- Environment drift (new TRT version, CUDA update)
- Subtle numerical changes from upstream library updates

### 12.3 Conservative Fallback

**Any unrecognized file** (not matching any Tier 1/2/3 or no-impact pattern) triggers ALL models. Better to over-test than to miss a regression.

### 12.4 Override Flags

```bash
# Force full suite (bypass TIA):
pytest tests/test_e2e.py --all-e2e

# Force specific models regardless of TIA:
pytest tests/test_e2e.py --affected-only --also-run qwen3-0.6b,mamba-130m
```

### 12.5 TIA Validation

Before trusting TIA in CI, validate it against the known "what to run when" table from CLAUDE.md:

| Change Type | Expected TIA Output | Verify |
|-------------|-------------------|--------|
| `families/qwen.py` only | `{qwen3-0.6b, qwen3-4b-instruct-2507}` | Tier 1 |
| `mamba_backend.cpp` only | `{mamba-130m}` | Tier 1 |
| `image_preprocessor.cpp` only | `{qwen25vl-3b, qwen3-vl-2b, internvl3-8b, phi4-multimodal}` | Tier 2 |
| `device_kv_cache.cpp` only | 35 models (decoder_kv + decoder_moe + VL) | Tier 2 |
| `graph_ops.py` only | ALL 50 models | Tier 3 |
| `tests/builder/test_config.py` only | 0 models (unit test, no E2E impact) | No-impact |
| `families/qwen.py` + `graph_ops.py` | ALL 50 models (Tier 3 dominates) | Tier 3 |
| `README.md` only | 0 models | No-impact |

---

## 13. Expected Impact

### 13.1 Test Time Reduction by Change Type

| Change Scenario | Models Today | Models with TIA | Time Saved |
|----------------|-------------|-----------------|------------|
| Single family plugin | 50 | 1-5 | ~95% |
| Single C++ backend | 50 | 1-4 | ~92-98% |
| VL-specific change | 50 | 4 | ~92% |
| Diffusion-specific change | 50 | 3 | ~94% |
| Audio-specific change | 50 | 1-4 | ~92-98% |
| Manifest-only change | 50 | 1 | ~98% |
| `device_kv_cache.cpp` | 50 | 35 | ~30% |
| `graph_ops.py` | 50 | 50 | 0% |
| Unit test / docs only | 50 | 0 | 100% |

### 13.2 Realistic Distribution

Based on typical development patterns, most changes fall into Tier 1 (family plugins, model-specific builders, individual backends). Estimated distribution:

| Tier | % of Changes | Models Run | Typical Time |
|------|-------------|------------|-------------|
| Tier 1 (leaf) | ~60% | 1-5 | 3-15 min |
| Tier 2 (strategy) | ~20% | 4-35 | 15-90 min |
| Tier 3 (global) | ~10% | 50 | 2-3 hours |
| No-impact | ~10% | 0 | 0 min |

**Weighted average time**: ~25 min (vs 2-3 hours today) — **~85% reduction**.

---

## 14. Validation Plan

### 14.1 Unit Tests for test_impact.py

```python
# tests/test_impact_test.py (or tests/builder/test_impact.py)

def test_family_plugin_selects_correct_models():
    affected = compute_affected_models(
        ["trtf_build/trtf_build/families/qwen.py"]
    )
    assert affected == {"qwen3-0.6b", "qwen3-4b-instruct-2507"}

def test_manifest_selects_single_model():
    affected = compute_affected_models(
        ["tests/e2e/models/mamba-130m.json"]
    )
    assert affected == {"mamba-130m"}

def test_cpp_mamba_backend():
    affected = compute_affected_models(
        ["src/runtime/trt/mamba_backend.cpp"]
    )
    assert affected == {"mamba-130m"}

def test_vl_backend_selects_vl_models():
    affected = compute_affected_models(
        ["src/runtime/trt/vl_backend.cpp"]
    )
    assert affected == {"qwen25vl-3b", "qwen3-vl-2b", "internvl3-8b", "phi4-multimodal"}

def test_graph_ops_is_global():
    affected = compute_affected_models(
        ["trtf_build/trtf_build/graph_ops.py"]
    )
    assert affected is None  # None = ALL models

def test_readme_has_no_impact():
    affected = compute_affected_models(["README.md"])
    assert affected == set()

def test_unit_test_has_no_impact():
    affected = compute_affected_models(["tests/builder/test_config.py"])
    assert affected == set()

def test_mixed_changes_union():
    affected = compute_affected_models([
        "trtf_build/trtf_build/families/qwen.py",
        "trtf_build/trtf_build/families/mamba.py",
    ])
    assert affected == {"qwen3-0.6b", "qwen3-4b-instruct-2507", "mamba-130m"}

def test_global_overrides_leaf():
    affected = compute_affected_models([
        "trtf_build/trtf_build/families/qwen.py",
        "trtf_build/trtf_build/graph_ops.py",
    ])
    assert affected is None  # Global change detected

def test_unknown_file_is_global():
    affected = compute_affected_models(["some/unknown/file.py"])
    assert affected is None  # Conservative: unknown = ALL

def test_runner_file_selects_strategy_models():
    affected = compute_affected_models(
        ["tests/e2e_harness/runners/diffusion.py"]
    )
    assert affected == {"flux-schnell", "wan21-t2v-1.3b", "z-image-turbo"}

def test_comparator_file_selects_strategy_models():
    affected = compute_affected_models(
        ["tests/e2e_harness/comparators/speech_to_text.py"]
    )
    assert affected == {"whisper-tiny"}

def test_device_kv_cache_selects_decoder_and_vl():
    affected = compute_affected_models(
        ["src/runtime/trt/device_kv_cache.cpp"]
    )
    # Should include all decoder_kv_cache, decoder_moe, and vision_language models
    assert "qwen3-0.6b" in affected
    assert "mixtral-stories-15m" in affected  # decoder_moe
    assert "qwen25vl-3b" in affected  # vision_language
    assert "mamba-130m" not in affected  # ssm_recurrent — not affected
    assert "rwkv-169m" not in affected  # rwkv_recurrent — not affected
```

### 14.2 Integration Validation

```bash
# Verify against known change scenarios from CLAUDE.md's "what to run when" table:

# Python builder logic → Tiers 1, 2
python scripts/test_selector.py --files "trtf_build/trtf_build/families/qwen.py" --format human
# Expected: 2 models (qwen3-0.6b, qwen3-4b-instruct-2507)

# C++ runtime → Tier 1-3
python scripts/test_selector.py --files "src/runtime/trt/device_kv_cache.cpp" --format human
# Expected: ~35 models (decoder_kv_cache + decoder_moe + vision_language)

# Graph ops → Tier 3
python scripts/test_selector.py --files "trtf_build/trtf_build/graph_ops.py" --format human
# Expected: ALL 50 models

# Unit test only → No impact
python scripts/test_selector.py --files "tests/builder/test_config.py" --format human
# Expected: 0 models
```

---

## Appendix A: Complete Model Manifest Database

All 50 models from `tests/e2e/models/*.json`:

| # | Model Name | Family | Runtime Strategy | Task Strategy | Core? |
|---|-----------|--------|-----------------|---------------|-------|
| 1 | bark-large | bark | text_to_audio | text_to_audio | — |
| 2 | bark-small | bark | text_to_audio | text_to_audio | — |
| 3 | bert-base-uncased | bert | encoder_only | encoder_only_nlp | — |
| 4 | bloom-560m | bloom | decoder_kv_cache | text_generation_causal | — |
| 5 | codegen-350m | codegen | decoder_kv_cache | text_generation_causal | — |
| 6 | deepseek-v2-lite | deepseek_v2 | decoder_kv_cache | text_generation_causal | — |
| 7 | deepseek-v2-tiny | deepseek_v2 | decoder_kv_cache | text_generation_causal | — |
| 8 | eagle-embed-vl-1b-v2 | eagle_vlm | embedding | embedding | — |
| 9 | eagle-rerank-vl-1b-v2 | eagle_vlm | reranking | reranking | — |
| 10 | falcon-rw-1b | falcon | decoder_kv_cache | text_generation_causal | — |
| 11 | falcon3-1b | llama | decoder_kv_cache | text_generation_causal | — |
| 12 | flux-schnell | flux | diffusion | diffusion_media_generation | — |
| 13 | gemma-2-2b | gemma | decoder_kv_cache | text_generation_causal | — |
| 14 | gpt-neo-125m | gpt_neo | decoder_kv_cache | text_generation_causal | — |
| 15 | gpt2-125m | gpt2 | decoder_kv_cache | text_generation_causal | — |
| 16 | granite-3.1-2b | granite | decoder_kv_cache | text_generation_causal | — |
| 17 | internlm2-1.8b | internlm | decoder_kv_cache | text_generation_causal | — |
| 18 | internvl3-8b | internvl | vision_language | vision_language_generation | — |
| 19 | mamba-130m | mamba | ssm_recurrent | text_generation_causal | — |
| 20 | minitron-4b-depth | llama | decoder_kv_cache | text_generation_causal | — |
| 21 | minitron-4b-width | llama | decoder_kv_cache | text_generation_causal | — |
| 22 | mistral-7b | mistral | decoder_kv_cache | text_generation_causal | — |
| 23 | mixtral-stories-15m | mixtral | decoder_moe | text_generation_causal | — |
| 24 | nemotron-h-nano-9b | nemotron_h | hybrid_mamba_attention | text_generation_causal | — |
| 25 | nemotron-hindi-4b | nemotron | decoder_kv_cache | text_generation_causal | — |
| 26 | nemotron-mini-4b | nemotron | decoder_kv_cache | text_generation_causal | — |
| 27 | nemotron-nano-4b | llama | decoder_kv_cache | text_generation_causal | — |
| 28 | olmo-1b | olmo | decoder_kv_cache | text_generation_causal | — |
| 29 | opt-125m | opt | decoder_kv_cache | text_generation_causal | — |
| 30 | personaplex-7b | personaplex | speech_to_speech | speech_to_speech | — |
| 31 | phi-moe | phi_moe | decoder_kv_cache | text_generation_causal | — |
| 32 | phi3-mini | phi | decoder_kv_cache | text_generation_causal | — |
| 33 | phi4-multimodal | phi4_multimodal | vision_language | vision_language_generation | — |
| 34 | pythia-70m | gpt_neox | decoder_kv_cache | text_generation_causal | — |
| 35 | qwen25vl-3b | qwen_vl | vision_language | vision_language_generation | — |
| 36 | qwen3-0.6b | qwen | decoder_kv_cache | text_generation_causal | core |
| 37 | qwen3-4b-instruct-2507 | qwen | decoder_kv_cache | text_generation_causal | — |
| 38 | qwen3-moe-30b-a3b | qwen_moe | decoder_moe | text_generation_causal | — |
| 39 | qwen3-vl-2b | qwen_vl | vision_language | vision_language_generation | — |
| 40 | riva-translate-4b | mistral | decoder_kv_cache | text_generation_causal | — |
| 41 | rwkv-169m | rwkv | rwkv_recurrent | text_generation_causal | — |
| 42 | sam-vit-base | sam | prompted_segmentation | prompted_segmentation | — |
| 43 | segformer-b0-ade | segformer | segmentation | segmentation | — |
| 44 | stablelm2-1.6b | stablelm | decoder_kv_cache | text_generation_causal | — |
| 45 | starcoder2-3b | starcoder2 | decoder_kv_cache | text_generation_causal | — |
| 46 | tinyllama-1.1b | llama | decoder_kv_cache | text_generation_causal | — |
| 47 | wan21-t2v-1.3b | wan_t2v | diffusion | diffusion_media_generation | — |
| 48 | whisper-tiny | whisper | speech_to_text | speech_to_text | — |
| 49 | xglm-564m | xglm | decoder_kv_cache | text_generation_causal | — |
| 50 | z-image-turbo | z_image | diffusion | diffusion_media_generation | — |

---

## Appendix B: Complete C++ File Inventory

### `src/runtime/trt/` (60 files)

**Strategy-specific backends:**
| File | Backend | Runtime Strategy | Affected Models |
|------|---------|-----------------|-----------------|
| bark_backend.cpp/h | Bark TTS | text_to_audio | bark-large, bark-small |
| diffusion_backend.cpp/h | Diffusion shared | diffusion | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| diffusion_backend_base.cpp | Diffusion base | diffusion | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| embedding_backend.cpp/h | Embedding | embedding | eagle-embed-vl-1b-v2 |
| encoder_backend.cpp/h | Encoder-only | encoder_only | bert-base-uncased |
| flux_diffusion_backend.cpp/h | FLUX | diffusion | flux-schnell |
| hybrid_backend.cpp/h | Mamba+Attention | hybrid_mamba_attention | nemotron-h-nano-9b |
| mamba_backend.cpp/h | SSM | ssm_recurrent | mamba-130m |
| mamba_decode_runtime.cpp/h | SSM decode | ssm_recurrent | mamba-130m |
| mamba_step_state.cpp/h | SSM state | ssm_recurrent | mamba-130m |
| omni_backend.cpp/h | Omni multimodal | omni_multimodal | *(no manifest)* |
| reranking_backend.cpp/h | Reranking | reranking | eagle-rerank-vl-1b-v2 |
| rwkv_backend.cpp/h | RWKV | rwkv_recurrent | rwkv-169m |
| rwkv_decode_runtime.cpp/h | RWKV decode | rwkv_recurrent | rwkv-169m |
| rwkv_step_state.cpp/h | RWKV state | rwkv_recurrent | rwkv-169m |
| sam_backend.cpp/h | SAM | prompted_segmentation | sam-vit-base |
| segmentation_backend.cpp/h | SegFormer | segmentation | segformer-b0-ade |
| speech_backend.cpp/h | Speech | speech_to_speech | personaplex-7b |
| wan_diffusion_backend.cpp/h | Wan T2V | diffusion | wan21-t2v-1.3b |
| whisper_backend.cpp/h | Whisper | speech_to_text | whisper-tiny |
| z_image_diffusion_backend.cpp/h | Z-Image | diffusion | z-image-turbo |

**Shared decoder infrastructure (Tier 2/3):**
| File | Used By | Affected Models |
|------|---------|-----------------|
| device_kv_cache.cpp/h | decoder_kv_cache, decoder_moe, vision_language | 35 models |
| trt_backend_shared.cpp/h | decoder_kv_cache, decoder_moe, vision_language | 35 models |
| trt_decode_runtime.cpp/h | all decoder + VL + hybrid | 36 models |
| image_preprocessor.cpp/h | vision_language | 4 models |
| vision_engine.cpp/h | vision_language | 4 models |
| vl_backend.cpp/h | vision_language | 4 models |

**Global shared (Tier 3):**
| File | Component |
|------|-----------|
| trt_common.cpp/h | TRT logger, CUDA helpers |
| trt_engine_lifecycle.cpp/h | Engine management |
| step_state.h | IStepState interface |
| stb_impl.cpp | Image loading |
| trt_graph_builder.cpp | Graph builder stub |

### `src/bundle/` (2 files)
| File | Impact |
|------|--------|
| bundle_format.cpp/h | **Global** — all models read bundles |

### `src/cabi/` (5 files)
| File | Impact |
|------|--------|
| trtf_c.cpp | **Global** — C ABI entry point |
| fast_path_config.cpp/h | **Global** — bundle metadata parsing |
| bundle_helpers.cpp/h | **Global** — shared plumbing |

### `src/tokenizer/` (3 files)
| File | Impact |
|------|--------|
| vocab_tokenizer.cpp | **Global** — fallback tokenizer |
| hf_python_tokenizer.cpp | **Global** — HF tokenizer bridge |
| hf_python_tokenizer_helpers.h | **Global** — tokenizer helpers |

### `include/trtf/` (5 files)
| File | Impact |
|------|--------|
| backend.h | **Global** — public API |
| bundle.h | **Global** — public API |
| generation.h | **Global** — public API |
| pipeline.h | **Global** — public API |
| tokenizer.h | **Global** — public API |

---

## Appendix C: Complete Python Builder File Inventory

### `trtf_build/trtf_build/` (32 files + families/)

**Global shared (Tier 3):**
| File | Component |
|------|-----------|
| `__init__.py` | Public API exports |
| `__main__.py` | CLI entry |
| `cli.py` | CLI: build, inspect, version |
| `config.py` | ModelConfig from config.json |
| `checkpoint_mapper.py` | HF safetensors → weight dict |
| `bundle_writer.py` | .trtfb file writing |
| `engine_builder.py` | Orchestrator: load → build → bundle |
| `graph_ops.py` | Atomic TRT graph ops (RoPE, RMSNorm, attention, etc.) |
| `graph_blocks.py` | Composable blocks (attention, MLP, norm) |
| `standard_decoder_builder.py` | Standard decoder engine builder |
| `debug_runner.py` | Python TRT runner (parity-critical) |
| `pipeline.py` | Subprocess wrapper |

**Family-scoped builders (Tier 1):**
| File | Affected Families | Affected Models |
|------|------------------|-----------------|
| `qwen_vl_vision_builder.py` | qwen_vl | qwen25vl-3b, qwen3-vl-2b |
| `qwen3_encoder_builder.py` | qwen_vl (DeepStack) | qwen3-vl-2b |
| `internvit_vision_builder.py` | internvl | internvl3-8b |
| `phi4mm_vision_builder.py` | phi4_multimodal | phi4-multimodal |
| `clip_encoder_builder.py` | z_image | z-image-turbo |
| `flux_dit_builder.py` | flux | flux-schnell |
| `flux_vae_builder.py` | flux | flux-schnell |
| `z_image_dit_builder.py` | z_image | z-image-turbo |
| `vae_2d_builder.py` | z_image | z-image-turbo |
| `causal_vae_3d_builder.py` | wan_t2v | wan21-t2v-1.3b |
| `t5_encoder_builder.py` | bark, wan_t2v | bark-large, bark-small, wan21-t2v-1.3b |
| `encodec_builder.py` | bark | bark-large, bark-small |
| `encoder_builder.py` | segformer | segformer-b0-ade |

**Strategy-scoped (Tier 2):**
| File | Affected Models |
|------|-----------------|
| `standard_dit_builder.py` | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| `diffusion_runner.py` | flux-schnell, wan21-t2v-1.3b, z-image-turbo |
| `vision_encoder_builder.py` | VL models using generic ViT |
| `schedulers/` | flux-schnell, wan21-t2v-1.3b, z-image-turbo |

**Unused/Legacy:**
| File | Notes |
|------|-------|
| `onnx_vision_builder.py` | Legacy — ONNX not used |

### `trtf_build/trtf_build/families/` (43 plugin files)

See §5.1 for the complete family → model mapping. Each plugin is Tier 1 (leaf).

---

## Appendix D: E2E Harness Plugin Inventory

### Strategy Runners (`tests/e2e_harness/runners/`)

| File | strategy_name(s) | Affected Task Strategies |
|------|------------------|--------------------------|
| `text_generation.py` | text_generation_causal | text_generation_causal |
| `vision_language.py` | vision_language_generation | vision_language_generation |
| `audio_speech.py` | speech_to_text, text_to_audio, speech_to_speech | 3 strategies |
| `diffusion.py` | diffusion_media_generation | diffusion_media_generation |
| `segmentation.py` | segmentation, prompted_segmentation | 2 strategies |
| `embedding.py` | embedding | embedding |
| `reranking.py` | reranking | reranking |
| `encoder_only.py` | encoder_only_nlp | encoder_only_nlp |
| `omni.py` | omni_multimodal, composite_pipeline | 2 strategies |
| `neural_operator.py` | neural_operator | neural_operator |
| `object_detection.py` | object_detection | object_detection |

### Comparators (`tests/e2e_harness/comparators/`)

| File | Primary Plugin Class | Task Strategy |
|------|---------------------|---------------|
| `text.py` | TextComparator | text_generation_causal |
| `vision_language.py` | VisionLanguageComparator | vision_language_generation |
| `speech_to_text.py` | SpeechToTextComparator | speech_to_text |
| `text_to_audio.py` | TextToAudioComparator | text_to_audio |
| `speech_to_speech.py` | SpeechToSpeechComparator | speech_to_speech |
| `audio.py` | SpeechToTextComparator (alias) | speech_to_text |
| `diffusion.py` | DiffusionComparator | diffusion_media_generation |
| `segmentation.py` | SegmentationComparator + PromptedSegmentationComparator + ObjectDetectionComparator | segmentation, prompted_segmentation, object_detection |
| `embedding.py` | EmbeddingComparator | embedding |
| `reranking.py` | RerankingComparator | reranking |
| `encoder_only.py` | EncoderOnlyComparator | encoder_only_nlp |
| `omni.py` | OmniComparator | omni_multimodal |
| `neural_operator.py` | NeuralOperatorComparator | neural_operator |

### Reference Backends (`tests/e2e_harness/references/`)

| File | Used By | Impact |
|------|---------|--------|
| `hf_transformers.py` | ~45 models | **Global** (Tier 3) |
| `hf_diffusers.py` | 3 diffusion models | Tier 2 |
| `torch_reference.py` | personaplex-7b | Tier 1 |
| `golden_snapshot.py` | Generic | **Global** if used |
| `custom_python.py` | Generic | **Global** if used |
| `invariant_only.py` | Generic | **Global** if used |

### Shared Harness Infrastructure (always Tier 3)

| File | Role |
|------|------|
| `contracts.py` | Domain types, RUNTIME_TO_TASK_STRATEGY mapping |
| `orchestrator.py` | Lifecycle coordinator |
| `registry.py` | Plugin auto-discovery |
| `manifest_loader.py` | JSON manifest → E2ECase |
| `artifact_sink.py` | Artifact persistence |

---

## Summary

This plan provides a complete, exhaustive mapping from every source file in the repository to the E2E model tests it can affect. The implementation is:

1. **Simple**: ~300 LOC across 2 Python files, no new dependencies
2. **Correct**: Conservative by design — unknown files trigger all tests
3. **Deterministic**: Computed from source structure, no stale databases
4. **Incrementally refinable**: Start with file-level rules, later add function-level tracing
5. **Safe**: Core model always-run + nightly full suite + override flags

Expected result: **~85% average reduction in E2E test time** for typical development workflows.
