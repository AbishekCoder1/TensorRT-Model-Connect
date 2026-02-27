# Testing Strategy Comparison: HF Transformers, TRT-LLM, vLLM, SGLang vs. trtf

*Generated: 2026-02-26*

## The Core Insight: What Gives These Projects Confidence

The key pattern across ALL four major projects is a **3-layer testing pyramid** that avoids full E2E on every model:

| Layer | What it tests | How | GPU needed? | Run frequency |
|-------|--------------|-----|-------------|---------------|
| **Op/Kernel correctness** | Each atomic operation in isolation | Random tensors -> optimized op -> compare to PyTorch reference | Sometimes | Every PR |
| **Model structure validation** | Model graph builds, shapes match, save/load round-trips | Tiny random-weight models (2 layers, hidden=32) | No | Every PR |
| **Numerical accuracy** | Full model logits match reference | Real (small) models, logprob comparison vs HF | Yes | Subset on PR, full nightly |

**The single most impactful pattern: tiny random-weight model testing.** This is what lets HF Transformers test 427 architectures on every PR without GPUs.

---

## Per-Framework Breakdown

### HF Transformers (the gold standard for model testing)

**Key architecture: Mixin inheritance + ModelTester factories**

```
LlamaModelTest
  +-- ModelTesterMixin        -> 25+ auto-tests (determinism, save/load, attention shapes, SDPA equivalence)
  +-- GenerationTesterMixin   -> 15+ auto-tests (greedy, beam, sampling, static cache, KV format)
  +-- PipelineTesterMixin     -> Pipeline integration tests
```

Each model test file is **~20 lines of actual code**. The `LlamaModelTester` just defines:
- `batch_size=13, seq_length=7, hidden_size=32, num_hidden_layers=2, vocab_size=99`
- `get_config()` -> creates a tiny config
- `prepare_config_and_inputs()` -> random `input_ids` + causal mask

From those 20 lines, **100+ tests run automatically** validating:
- Forward pass shapes
- Determinism (same input -> same output)
- Batching equivalence (single sample = batch of 1)
- Save/load round-trip
- Attention mask correctness
- Gradient checkpointing
- `eager` vs `SDPA` vs `FlashAttention2` equivalence
- KV cache format
- Generation (greedy, beam, sampling)
- Left-padding compatibility

**CI strategy:** Smart test selection via `tests_fetcher.py` (reverse dependency analysis). Only affected tests run on PRs. Full suite nightly with `@slow` tests that use real pretrained models.

### TRT-LLM

**Key architecture: 70 functional op tests + per-model allclose-to-HF**

Strongest at **op-level isolation testing**. Each of their 70 TRT graph ops gets its own test:
```python
# Build single-op TRT network -> run -> compare to torch reference
x = torch.rand(shape, device="cuda")
session = create_session(builder, network_with_one_gelu)
output = run_session(session, {"x": x})
torch.testing.assert_close(output, torch.nn.functional.gelu(x), atol=1e-5, rtol=2e-3)
```

Model-level tests: 29 per-model test files with `test_<model>_allclose_to_hf()` that loads real HF weights into both backends and compares logits.

**Unique pattern: Adaptive model scaling** -- `reduce_llama_config()` dynamically shrinks `num_hidden_layers` based on available GPU memory.

**Unique pattern: Statistical accuracy testing** -- hypothesis testing framework for benchmark accuracy (not arbitrary thresholds), computing p-values to determine pass/fail.

### vLLM

**Key architecture: Model registry enforcement + logprob comparison**

Their strongest innovation is the **test registry with enforcement**:
```python
# test_registry.py -- FAILS if any supported architecture lacks a test entry
def test_hf_registry_coverage():
    assert all(arch in TEST_REGISTRY for arch in ModelRegistry.get_supported_archs())
```

Model correctness = logprob comparison against HF:
```python
# Not exact output match -- top-k logprob overlap
check_logprobs_close(hf_outputs, vllm_outputs, name_0="hf", name_1="vllm")
```

**Unique pattern: `load_format="dummy"` initialization test** -- every architecture gets tested with random weights + mocked KV cache, just verifying the model graph can be constructed. Runs in 45 min for all 100+ architectures.

**Unique pattern: `dummy_hf_overrides()`** -- shrinks any HF config to 1 layer for fast testing while preserving architecture semantics.

### SGLang

**Key architecture: Server-based integration + registration-based CI**

Tests launch actual inference servers, then compare via HTTP API:
```python
with HFRunner(model) as hf, SRTRunner(model) as srt:
    check_close_model_outputs(hf.forward(prompts), srt.forward(prompts))
```

Comparison checks 3 dimensions: ROUGE-L on text, prefill logprobs, decode logprobs.

**Unique pattern: Registration-based test discovery** -- tests self-register into CI suites via AST-parsed decorators:
```python
register_cuda_ci(est_time=228, suite="stage-b-test-large-1-gpu")
```
Automatic load balancing via Longest Processing Time heuristic.

**Unique pattern: Real models only for correctness** -- no random-weight models for accuracy testing. Smallest model tested: GPT-2 (124M). They use `--load-format dummy` only for throughput/infrastructure tests.

---

## Comparative Matrix

| Capability | HF Transformers | TRT-LLM | vLLM | SGLang | **trtf** |
|-----------|----------------|---------|------|--------|----------|
| **Op-level isolation tests** | N/A (PyTorch ops) | **70 ops** | Kernel tests (52 files) | Kernel tests (52 files) | **18 graph ops** |
| **Tiny random-weight model tests** | **427 models** | Adaptive scaling | `load_format="dummy"` | No (real models only) | **None** |
| **Model structure validation (no GPU)** | 25+ auto-tests per model | Sanity tests | Init tests | No | Config + bundle tests |
| **Mixin/inheritance test reuse** | **100+ tests from 20 lines** | Shared patterns | Parametrized fixtures | Shared runners | **No reuse across families** |
| **Logprob/logit comparison vs HF** | `@slow` integration | allclose-to-HF | `check_logprobs_close` | Logprob + ROUGE-L | `diff_logits.py` |
| **Test registry enforcement** | N/A | No | **Enforced** | No | `>=20 plugins` count |
| **Smart CI test selection** | `tests_fetcher.py` | Pre/post-merge YAML | Source-file deps | Stage-based + labels | **None** (manual tiers) |
| **Attention impl. equivalence** | **Parametrized** (eager/SDPA/FA2) | Multi-backend | Multi-backend | Multi-backend | N/A (TRT only) |
| **KV cache unit tests** | KV format validation | C++ cache tests | **14 block manager tests** | Radix cache unit tests | `test_cache_state_machine.py` |
| **Statistical accuracy testing** | No (fixed thresholds) | **Hypothesis testing** | lm-eval thresholds | Benchmark thresholds | MMLU spot-check |
| **Per-layer hidden state comparison** | No | No | No | `compare.py` debug tool | `diff_layers.py` |

---

## Gap Analysis: What trtf Is Missing

### Gap 1: No "ModelTesterMixin" -- The Biggest Gap

**What HF does:** Every model gets 100+ tests from 20 lines of code via mixin inheritance. A `QwenModelTester` defines tiny config params, and `ModelTesterMixin` automatically tests shapes, determinism, save/load, attention masking, etc.

**What we have:** Each family plugin is tested only through:
- `test_families.py` -- plugin discovery + `match()` (pure Python, no graph building)
- E2E `diff_logits.py` -- full model with real weights (requires GPU + time)

**The gap:** There's no middle layer that tests "does this family plugin produce a valid TRT engine graph with correct I/O tensor names and shapes?" using a tiny random-weight config. If someone adds a new family with a typo in weight mapping, the only way to catch it is running the full E2E pipeline (~5-20 min per model with GPU).

**Recommendation:** Create a `FamilyPluginTesterMixin` that, for each family plugin:
1. Creates a tiny config (2 layers, hidden=64, vocab=100)
2. Generates random weights matching the plugin's weight map
3. Builds a TRT engine graph (requires GPU but takes seconds, not minutes)
4. Validates I/O tensor names match expected contract
5. Runs one forward pass with random inputs, checks output shape
6. Optionally: serializes to `.trtfb` bundle and reads back

This would catch **90% of family plugin bugs** in seconds instead of minutes.

### Gap 2: No Random-Weight Graph Construction Tests Per Family

**What TRT-LLM does:** `test_<model>_sanity()` builds a model with random weights, runs a forward pass, validates tensor shapes. No HF reference needed.

**What vLLM does:** `load_format="dummy"` tests every architecture can be initialized.

**What we have:** `test_standard_decoder.py` tests tensor naming and I/O names but only for a generic decoder, not per-family.

**Recommendation:** For each family, create a test that:
```python
def test_qwen_builds_valid_engine():
    tiny_config = make_tiny_config("qwen", num_layers=2, hidden=64, vocab=100)
    plugin = qwen.plugin
    weights = plugin.generate_random_weights(tiny_config)  # new method on FamilyPlugin
    engine = build_engine(tiny_config, weights, plugin)
    assert "input_ids" in engine.io_names
    assert "logits" in engine.io_names
    # Forward pass with random input
    output = run_engine(engine, random_input_ids=[1, 2, 3])
    assert output.shape == (1, 3, tiny_config.vocab_size)
```

### Gap 3: No Automatic Test Inheritance When Adding a New Family

**What HF does:** Add a 20-line file -> get 100+ tests automatically. Zero edits to shared files.

**What vLLM does:** `test_hf_registry_coverage()` fails if you add a model without a test entry.

**What we have:** Family plugins are auto-discovered, but there's no enforcement that a new plugin has corresponding tests. Someone could add `families/newmodel.py` and never write a test.

**Recommendation:**
1. Add a test that asserts every discovered family plugin has a corresponding entry in an E2E model manifest OR a per-family unit test
2. Add a parametrized test that iterates over all family plugins and validates basic properties (match patterns, runtime_strategy, weight_map structure)

### Gap 4: No Tiny-Config Factory Pattern

**What HF does:** `CausalLMModelTester` creates configs with `hidden_size=32, num_hidden_layers=2, vocab_size=99`.

**What TRT-LLM does:** `reduce_llama_config()` shrinks configs to fit available memory.

**What vLLM does:** `dummy_hf_overrides()` reduces any config to 1 layer.

**What we have:** `ModelConfig` parses real `config.json` files. No facility to generate minimal configs for testing.

**Recommendation:** Add a `ModelConfig.create_tiny(model_type, **overrides)` factory method that returns a minimal valid config for any supported family. This is the foundation for Gap 1 and 2.

### Gap 5: No Test for Weight Mapping Correctness Without Full E2E

**What TRT-LLM does:** `test_<model>_allclose_to_hf()` loads HF weights into both backends, compares logits. But they also have sanity tests with random weights that just check shapes.

**What we have:** `checkpoint_mapper.py` tests cover `_transpose_2d` and GQA expansion, but don't test per-family weight mapping (the `load_weights()` method on each plugin).

**The risk:** A family plugin's `load_weights()` could map weight "A" to slot "B", and we'd only catch it during full E2E. This is the #1 source of bugs when adding new families.

**Recommendation:** For each family, create a test that:
1. Loads a real (small) HF model's weight names from `config.json`
2. Calls the plugin's `load_weights()`
3. Asserts every weight slot in the engine builder's expected map is filled
4. Asserts no "unmapped" or "unexpected" weight names remain

### Gap 6: No Source-File-Dependency CI Triggering

**What HF does:** `tests_fetcher.py` analyzes Python imports to map changed files -> affected tests.

**What vLLM does:** YAML-based `source_file_dependencies` per test area.

**What SGLang does:** `dorny/paths-filter` in GitHub Actions.

**What we have:** Manual tier selection ("if you changed graph_ops, run Tier 1+2").

**Recommendation:** If CI (GitHub Actions) is adopted, add path-based triggering:
```yaml
on:
  pull_request:
    paths:
      - 'trtf_build/trtf_build/graph_ops.py'
      - 'trtf_build/trtf_build/graph_blocks.py'
jobs:
  graph-op-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - run: pytest tests/builder/test_graph_ops.py -v -m trt
```

---

## Priority-Ordered Recommendations

| Priority | Gap | Effort | Impact | What to do |
|----------|-----|--------|--------|------------|
| **P0** | Tiny-config factory | Small | Enables P1+P2 | Add `ModelConfig.create_tiny()` |
| **P1** | FamilyPluginTesterMixin | Medium | 15+ tests per family from 5 lines | HF ModelTesterMixin pattern adapted for TRT engine builds |
| **P2** | Registry enforcement | Small | Prevents untested plugins | Assert every plugin has an E2E manifest (vLLM pattern) |
| **P3** | Core model marking | Small | Fast gate on every commit | 8 models marked `"core": true` cover all backends |
| **P4** | waives.txt + XFAIL | Small | Central failure tracking | Replace JSON `"skip"` with platform-aware waives file |
| **P5** | LPT test partitioning | Medium | 4x speedup with parallel agents | SGLang's auto_partition() for N-way parallel test runs |
| **P6** | Staged execution (A→B→C) | Medium | Don't run expensive tests on broken commits | 3-stage pipeline with sequential gates |
| **P7** | GPU isolation fixtures | Small | Prevent cascade failures | torch_empty_cache + cuda_error_early_quit (TRT-LLM pattern) |
| **P8** | Statistical accuracy testing | Medium | Better regression detection for benchmarks | Hypothesis testing for MMLU/GSM8K (keep atol for logits) |
| **P9** | CI source-file triggering | Small | Faster CI feedback | Path-based workflow triggers |

The P0-P1 items close the biggest gap: **catching family plugin bugs without GPU E2E** in seconds instead of minutes. P2-P3 ensure nothing ships untested. P4-P7 enable scaling to 100+ parallel agents.

---

## Deep-Dive Implementation Details (from source code study)

### FamilyPluginTesterMixin Design (P1)

Adapted from HF Transformers' `CausalLMModelTester` + `ModelTesterMixin` pattern.
Each per-family test file is **5 lines**:

```python
class QwenPluginTester(FamilyPluginTester):
    plugin_module = "trtf_build.families.qwen"
    model_type = "qwen3"

class TestQwenEngine(FamilyPluginTestMixin):
    tester_class = QwenPluginTester
```

The mixin auto-provides **15+ test methods** across 3 tiers:
- **Tier 0 (no GPU):** plugin.matches(), protocol compliance, name/strategy
- **Tier 1 (no GPU):** load_weights() returns correct keys, shapes, dtypes, determinism
- **Tier 2 (TRT GPU):** build_engine() succeeds, I/O tensor names match contract, logits shape correct
- **Tier 3 (TRT GPU):** one forward pass produces finite logits, deterministic output

Families with structural differences override `make_hf_tensors()` and `expected_*()`:
- Standard decoders (Qwen, LLaMA, Mistral, etc.): zero overrides
- Gemma: add test for +1.0 norm offset and embedding scaling
- Falcon: override for LayerNorm + GELU FC MLP
- Mamba: override weight structure, I/O names, config dict
- MoE: override for expert routing weights

### Registry Enforcement (P2)

From vLLM's `test_hf_registry_coverage()`:

```python
def test_all_plugins_have_e2e_manifest():
    manifest_families = {json.load(open(f))["family"] for f in models_dir.glob("*.json")}
    plugin_names = {p.name for p in _ALL_PLUGINS}
    uncovered = plugin_names - manifest_families - _EXEMPT_PLUGINS
    assert not uncovered, f"Plugins without manifests: {uncovered}"
```

Currently 2 plugins would fail: `deepseek_ocr`, `qwen3_omni`.

### Core Model Set (P3)

8 models covering all C++ backends:

| Model | Backend | Time |
|-------|---------|------|
| qwen3-0.6b | TrtBackendFastPath | 90s |
| mamba-130m | MambaBackend | 60s |
| mixtral-stories-15m | TrtBackendFastPath (MoE) | 60s |
| bert-base-uncased | EncoderOnlyBackend | 30s |
| whisper-tiny | WhisperBackend | 90s |
| bark-small | BarkBackend | 120s |
| segformer-b0-ade | SegmentationBackend | 60s |
| qwen25vl-3b | VLBackend | 180s |

Total core gate: ~12 min (serial) or ~3 min (4-way parallel).

### waives.txt (P4)

Replace 7 JSON `"skip"` fields with centralized platform-aware tracking:

```
# tests/e2e/waives.txt
falcon-rw-1b             XFAIL (ALiBi rel_l2 1.7x — needs graph op investigation)
gemma-2-2b               SKIP  (gated model, needs HF_TOKEN)
internlm2-1.8b           SKIP  (DynamicCache.from_legacy_cache removed in transformers 5.x)
nemotron-h-nano-9b       SKIP  (requires mamba-ssm package)
phi4-multimodal           SKIP  (SlidingWindowCache removed in transformers 5.x)
eagle-embed-vl-1b-v2     SKIP  (HF repo 404)
eagle-rerank-vl-1b-v2    SKIP  (HF repo 404)
# GB300/whisper-tiny      XFAIL (aarch64 tokenizer issue)
```

Benefits: single-file view of all known issues; XFAIL tracks issues without hiding them; platform scoping.

### LPT Test Partitioning (P5)

From SGLang's `auto_partition()`:

```python
def partition_models(models, num_agents, agent_id):
    sorted_models = sorted(models, key=lambda m: (-m["est_time_seconds"], m["name"]))
    partitions = [[] for _ in range(num_agents)]
    sums = [0.0] * num_agents
    for model in sorted_models:
        lightest = min(range(num_agents), key=sums.__getitem__)
        partitions[lightest].append(model)
        sums[lightest] += model["est_time_seconds"]
    return partitions[agent_id]
```

With 4 agents and cached bundles: ~38 min wall-clock (vs ~200 min serial = 5.3x speedup).

### Staged Execution (P6)

```
Stage A (1 agent, ~3 min) — core models only, blocks everything
    ↓ pass
Stage B (4 agents, LPT-partitioned, ~35 min) — all non-diffusion models
    ↓ pass
Stage C (2 agents, ~35 min) — diffusion + large models
```

### GPU Isolation Fixtures (P7)

From TRT-LLM's `conftest.py`:

```python
@pytest.fixture(autouse=True)
def gpu_memory_cleanup():
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()

@pytest.fixture(autouse=True)
def cuda_error_early_quit():
    try:
        yield
        torch.cuda.synchronize()
    except RuntimeError as e:
        if 'CUDA error:' in str(e):
            os._exit(1)  # kill worker, let master restart
        raise
```

---

## Current trtf Test Coverage (for reference)

### What we already do well

| Area | Coverage |
|------|----------|
| **Graph ops** | 18 functions tested in isolation (TRT GPU tests) |
| **Config parsing** | 40+ model types + edge cases |
| **Weight mapping primitives** | Transpose, GQA expansion, checkpoint loading |
| **Bundle format** | Magic, sections, round-trip read/write |
| **Plugin discovery** | >=20 family plugins, match() logic |
| **C++ runtime** | Bundle reading, config parsing, argmax/top-k, image preprocessing |
| **Per-layer comparison** | `diff_layers.py` -- unique advantage over all 4 frameworks |
| **E2E inference** | 28 models x 5 runtime strategies |
| **Diff framework self-tests** | Tolerance logic, argmax match, top-k overlap |

### Test tiers

| Tier | Time | GPU | What |
|------|------|-----|------|
| 1: Unit tests | ~60s | No | Config, checkpoint, bundle, families, tools self-tests, C++ |
| 2: Graph-op GPU | ~2 min | Yes | 18 TRT graph operations |
| 3: E2E smoke | ~5 min | Yes | 1 model (qwen3-0.6b) |
| 4: Full E2E | ~90 min | Yes | All 28 models |
| 5: Performance | ~10 min/model | Yes | Manual perf regression |

---

## Sources

### HF Transformers
- [HuggingFace Testing Documentation](https://huggingface.co/docs/transformers/en/testing)
- [tests/test_modeling_common.py (ModelTesterMixin)](https://github.com/huggingface/transformers/blob/main/tests/test_modeling_common.py)
- [tests/models/llama/test_modeling_llama.py](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py)
- [tests/causal_lm_tester.py](https://github.com/huggingface/transformers/blob/main/tests/causal_lm_tester.py)
- [tests/generation/test_utils.py (GenerationTesterMixin)](https://github.com/huggingface/transformers/blob/main/tests/generation/test_utils.py)
- [utils/create_dummy_models.py](https://github.com/huggingface/transformers/blob/main/utils/create_dummy_models.py)
- [utils/tests_fetcher.py](https://github.com/huggingface/transformers/blob/main/utils/tests_fetcher.py)

### TRT-LLM
- [TensorRT-LLM Repository](https://github.com/NVIDIA/TensorRT-LLM)
- [Tests README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/README.md)
- [Integration README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/README.md)
- [Accuracy Test README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/defs/accuracy/README.md)

### vLLM
- [vLLM Unit Testing Documentation](https://docs.vllm.ai/en/stable/contributing/model/tests/)
- [vLLM tests/models/ directory](https://github.com/vllm-project/vllm/tree/main/tests/models)
- [vLLM tests/kernels/ directory](https://github.com/vllm-project/vllm/tree/main/tests/kernels)
- [vLLM Model Registry](https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py)

### SGLang
- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [SGLang Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)
- [SGLang CI Improvement Plan (Issue #8845)](https://github.com/sgl-project/sglang/issues/8845)
