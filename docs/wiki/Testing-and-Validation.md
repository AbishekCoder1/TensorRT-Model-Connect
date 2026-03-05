# Testing and Validation

Comprehensive manual for the trt-transformers-cpp test infrastructure. Covers every abstraction layer, source file locations, intentions, pytest markers, and commands for running each suite.

---

## Test Architecture at a Glance

The test suite is organized into **six abstraction layers**, each with a distinct
purpose, dependency profile, and speed:

| Layer | Directory | Files | Tests | GPU? | Time | Purpose |
|-------|-----------|:--:|:--:|:--:|------|---------|
| 1. Builder unit | `tests/builder/` | 50 | ~940 | No | ~10 min | Python build logic in isolation |
| 2. C++ runtime unit | `tests/cpp/` | 19 | 20 | Mix | ~8 s | C++ runtime correctness |
| 3. Tools self-tests | `tests/tools/` | 11 | ~160 | No | ~35 s | Diff framework + comparison utilities |
| 4. Graph-op GPU | `tests/builder/test_graph_*.py` | 3 | ~70 | TRT | ~2 min | TRT graph operations on real GPU |
| 5. Unified E2E | `tests/test_e2e.py` + `tests/e2e_harness/` | 50 manifests | 50 | GPU | 2-3 h | Full pipeline (build + infer + compare) |
| 6. Diff framework | `tools/diff.py` + `tools/diff_framework/` | 6 checks | -- | GPU | varies | Ad-hoc TRT-vs-HF model comparison |

**Philosophy**: Every TRT engine must produce output matching HuggingFace
Transformers (the ground truth). Testing validates this at multiple
granularities: per-layer hidden states, per-step logits, full generation text,
and (for non-text modalities) modality-specific quality metrics.

---

## Layer 1: Python Builder Unit Tests

**Directory**: `tests/builder/`

**Intent**: Verify Python build logic in isolation -- config parsing, weight
loading, checkpoint mapping, bundle writing, engine orchestration, family
plugin dispatch, graph ops, and debug runner infrastructure. No GPU required
for the majority of tests.

**How to run**:

```bash
# All builder tests (no GPU needed for most; TRT tests auto-skip when unavailable)
.venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Only unit-tier tests (Tier 0 + Tier 1, never needs GPU)
.venv/bin/python -m pytest tests/builder/ -v -m unit --ignore=tests/builder/test_cli.py

# Only GPU/TRT tests
.venv/bin/python -m pytest tests/builder/ -v -m gpu --ignore=tests/builder/test_cli.py
```

**Pytest markers used**: `@pytest.mark.unit` (no GPU), `@pytest.mark.trt` (needs TRT), `@pytest.mark.gpu` (needs GPU).

**Skip markers**: All files use `try/except` with `pytest.skip(allow_module_level=True)` so they skip cleanly when TRT or `trtf_build` is not installed. GPU tests also use a `@requires_trt` skipif decorator.

### Sub-categories

#### Configuration & data parsing (no GPU)

| File | What it tests |
|------|---------------|
| `test_config.py` | `ModelConfig` parsing, VL `text_config` merge, edge cases, negative dimensions, type mismatches |
| `test_checkpoint_mapper.py` | Weight loading, GQA expansion (including 8x single-head), tied embeddings, biases |
| `test_bundle_writer.py` | Bundle format round-trip, section integrity, corrupted bundle detection (bad magic, truncated header/file) |
| `test_cache_state_machine.py` | Position, mask, cache append/shift logic, edge cases (max_cache_length=0, max_cache_length=1) |
| `test_manifest_validation.py` | E2E manifest schema validation (required fields, type checks, unknown runtime_strategy warnings) |

#### Family plugins (no GPU)

| File | What it tests |
|------|---------------|
| `test_families.py` | Plugin match/dispatch, runtime_strategy, embed_input, `matches()` returns bool |
| `test_family_plugins.py` | 10 family plugins: `load_weights()` correctness |
| `test_family_bert.py` | BERT-specific plugin tests |
| `test_family_phi4mm.py` | Phi-4 multimodal plugin |
| `test_family_qwen_moe.py` | Qwen MoE plugin |
| `test_family_sam.py` | SAM prompted segmentation plugin |
| `test_family_yolox.py` | YOLOX object detection plugin |

#### Per-family engine tests (mixin-based, 3-tier)

These files inherit from `FamilyPluginTestMixin` in `family_plugin_test_mixin.py` and follow a standardized 3-tier pattern:

- **Tier 0** (`@pytest.mark.unit`): Plugin discovery, matching, required methods
- **Tier 1** (`@pytest.mark.unit`): Weight loading from synthetic safetensors -- correct keys, shapes, dtypes, determinism
- **Tier 2** (`@pytest.mark.trt`, `@pytest.mark.gpu`): Build real TRT engine, validate I/O tensor names and logits output shape

| File | Family | model_type | Tier 2 |
|------|--------|------------|:--:|
| `test_engine_bark.py` | bark | `bark` | skip (custom builder) |
| `test_engine_bloom.py` | bloom | `bloom` | yes |
| `test_engine_codegen.py` | codegen | `codegen` | yes |
| `test_engine_falcon.py` | falcon | `falcon` | yes |
| `test_engine_gemma.py` | gemma | `gemma2` | yes |
| `test_engine_gpt2.py` | gpt2 | `gpt2` | yes |
| `test_engine_gpt_neo.py` | gpt_neo | `gpt_neo` | yes |
| `test_engine_gpt_neox.py` | gpt_neox | `gpt_neox` | yes |
| `test_engine_granite.py` | granite | `granite` | yes |
| `test_engine_internlm.py` | internlm | `internlm2` | yes |
| `test_engine_llama.py` | llama | `llama` | yes |
| `test_engine_mamba.py` | mamba | `mamba` | skip (custom builder) |
| `test_engine_mistral.py` | mistral | `mistral` | yes |
| `test_engine_mixtral.py` | mixtral | `mixtral` | yes |
| `test_engine_nemotron.py` | nemotron | `nemotron` | yes |
| `test_engine_olmo.py` | olmo | `olmo` | yes |
| `test_engine_opt.py` | opt | `opt` | yes |
| `test_engine_phi.py` | phi | `phi3` | yes |
| `test_engine_phi_moe.py` | phi_moe | `phimoe` | skip (custom builder) |
| `test_engine_qwen.py` | qwen | `qwen3` | yes |
| `test_engine_rwkv.py` | rwkv | `rwkv6` | skip (custom builder) |
| `test_engine_segformer.py` | segformer | `segformer` | skip (custom builder) |
| `test_engine_stablelm.py` | stablelm | `stablelm` | yes |
| `test_engine_starcoder2.py` | starcoder2 | `starcoder2` | yes |
| `test_engine_whisper.py` | whisper | `whisper` | skip (custom builder) |
| `test_engine_xglm.py` | xglm | `xglm` | yes |

#### Graph operations (needs TRT/GPU)

| File | What it tests |
|------|---------------|
| `test_graph_ops.py` | 18 atomic graph ops: RoPE, ALiBi, RMSNorm, LayerNorm, attention, etc. |
| `test_graph_ops_extended.py` | YaRN RoPE, T5 relative bias, extended ALiBi, conv/norm/ELU/pad ops |
| `test_graph_blocks.py` | Composable blocks: `apply_norm`, SwiGLU MLP, GELU FC MLP |

Graph-op tests use the `trt_runner` fixture from `conftest.py`: a `build_fn(network, inputs)` closure constructs a small TRT graph, the fixture builds an engine, runs inference, and returns NumPy outputs for comparison against PyTorch/NumPy references via `np.testing.assert_allclose`.

#### Builder orchestration (mock-based, no GPU)

| File | What it tests |
|------|---------------|
| `test_engine_builder.py` | Engine builder mock tests |
| `test_engine_builder_extended.py` | `build_bundle` orchestration, GPU name, TRT version |
| `test_pipeline.py` | Pipeline subprocess wrapper, binary detection |
| `test_debug_runner.py` | Debug runner mock tests |
| `test_debug_runner_extended.py` | Bundle section loading, runner cleanup, generate sequencing |
| `test_cli.py` | CLI inspect/build command dispatch (excluded from default run) |

#### Standard decoder & vision (needs TRT)

| File | What it tests |
|------|---------------|
| `test_standard_decoder.py` | Tensor naming contract, debug outputs |
| `test_vision_compute.py` | Vision encoder tests |
| `test_vision_compute_extended.py` | Vision RoPE, DeepStack config, patch embed, spatial merge |

---

## Layer 2: C++ Runtime Unit Tests

**Directory**: `tests/cpp/`

**Intent**: Verify C++ runtime correctness -- bundle parsing, tokenizers,
CUDA RAII wrappers, KV cache device operations, TRT engine lifecycle, image
preprocessing, CLI argument parsing, and helper utilities.

**How to run**:

```bash
# All C++ tests
ctest --test-dir build --output-on-failure

# Specific test
ctest --test-dir build -R test_bundle_format --output-on-failure
```

**Implementation**: Plain `main()` executables with no test framework. A
`check(condition, name)` helper accumulates `failures`; `main()` returns
non-zero if any failed. TRT-dependent tests guard with `#if TRTF_HAS_TRT`
and skip gracefully (exit 0).

**RAII guards** (`test_helpers.h`):
- `EnvVarGuard` -- saves/restores environment variables (prevents env leakage between tests)
- `TempDirGuard` -- creates temp directory on construction, `remove_all` on destruction

### File inventory

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_bundle_format.cpp` | Bundle magic, section parsing, round-trip | No |
| `test_bundle_e2e.cpp` | Bundle build + load round-trip | TRT |
| `test_bundle_helpers.cpp` | `find_bundle_sections` for all bundle types | TRT |
| `test_c_abi_entry.cpp` | C ABI entry point | TRT |
| `test_cli_args.cpp` | CLI argument parsing | No |
| `test_cuda_buffer.cpp` | RAII alloc, move semantics, data round-trip (with index on mismatch) | GPU |
| `test_cuda_stream.cpp` | RAII stream, move semantics | GPU |
| `test_data_dir.cpp` | Source/scripts dir resolution, env overrides (via `EnvVarGuard`) | No |
| `test_decode_runtime.cpp` | Argmax, mask building | TRT |
| `test_device_kv_cache.cpp` | Cache construction, mask progression, position clamping, reset | GPU |
| `test_fast_path_config.cpp` | Config JSON parsing | TRT |
| `test_hf_python_tokenizer.cpp` | Shell quoting, int parsing, HF output sanitization | No |
| `test_image_preprocessor.cpp` | All 4 strategies, config parsing, prompt formatting (via `TempDirGuard`) | No |
| `test_json_helpers.cpp` | JSON extraction helpers | No |
| `test_pipeline_api.cpp` | C API pipeline creation | TRT |
| `test_text_parsers.cpp` | String/file parsing helpers | No |
| `test_trt_engine_lifecycle.cpp` | `layer_tensor_name`, constants | TRT |
| `test_trt_logger.cpp` | Severity names, error storage, env-var controls | TRT |
| `test_vocab_tokenizer.cpp` | Encode/decode, round-trip, case insensitivity | No |

---

## Layer 3: Tools Self-Tests

**Directory**: `tests/tools/`

**Intent**: Verify diff framework and comparison utilities in isolation --
logit comparison, layer diffing, perf benchmarking, audio/segmentation metrics
-- without needing models or GPU.

**How to run**:

```bash
.venv/bin/python -m pytest tests/tools/ -v
```

**Implementation**: Pure Python tests, no TRT/GPU needed. `conftest.py` adds
`tools/` to import path. Tests use `importlib.import_module` for lazy
importing. Comparison logic tested with synthetic NumPy arrays.

### File inventory

| File | What it tests |
|------|---------------|
| `test_tool_helpers.py` | `cosine_sim`, `compare_arrays` |
| `test_diff_logits.py` | Logit comparison, argmax match, top-k overlap |
| `test_diff_layers.py` | Layer-wise hidden state comparison |
| `test_diff_vl.py` | Vision-language diff utilities |
| `test_diff_audio.py` | Energy computation, WAV I/O round-trip, token stats |
| `test_diff_segmentation.py` | Pixel agreement (non-tautological), logit diff, argument parsing |
| `test_diff_framework.py` | `DiffResult`, registry, runner, CLI parsing |
| `test_diffusion_helpers.py` | silu, gelu_tanh, bundle config/weights, timestep embedding |
| `test_parity.py` | Text/token comparison for runner parity |
| `test_perf_compare.py` | Stats, formatting, JSON output, serial GPU execution |
| `test_perf_parity.py` | Performance parity validation |

---

## Layer 4: Graph-Op GPU Tests

**Directory**: `tests/builder/test_graph_ops.py`, `test_graph_ops_extended.py`, `test_graph_blocks.py`

**Intent**: Validate TRT graph operations (RMSNorm, RoPE, attention, conv, etc.)
and composable graph blocks (SwiGLU MLP, GELU MLP, attention block) on real GPU
with real TRT engine execution.

**How to run**:

```bash
# All graph-op GPU tests
.venv/bin/python -m pytest tests/builder/test_graph_ops.py \
  tests/builder/test_graph_ops_extended.py \
  tests/builder/test_graph_blocks.py -v -m trt

# A single op
.venv/bin/python -m pytest tests/builder/test_graph_ops.py::TestRMSNorm -v
```

**Dependency**: Requires TRT + GPU. Uses `trt_runner` conftest fixture for
engine build + execution.

---

## Layer 5: Unified E2E Tests

**Directory**: `tests/test_e2e.py` + `tests/e2e_harness/`

**Intent**: Validate the full pipeline end-to-end -- build bundle from HF,
run C++ inference, compare output against HuggingFace reference. This is the
gold-standard correctness gate. All modalities use the same harness.

**How to run**:

```bash
# Single model (auto-builds bundle if missing)
.venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python

# Force rebuild bundle from HF
.venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines

# All 50 models with artifact output
.venv/bin/python -m pytest tests/test_e2e.py -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines --e2e-artifacts-dir /tmp/e2e_artifacts

# Filter by modality
.venv/bin/python -m pytest tests/test_e2e.py -v \
  --e2e-task-strategy text_generation_causal \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python
```

**Available `--e2e-task-strategy` values**:

| Strategy | Models | Runner |
|----------|:--:|--------|
| `text_generation_causal` | 26 | `text_generation.py` |
| `vision_language_generation` | 5 | `vision_language.py` |
| `diffusion_media_generation` | 3 | `diffusion.py` |
| `text_to_audio` | 2 | `audio_speech.py` |
| `speech_to_text` | 1 | `audio_speech.py` |
| `speech_to_speech` | 1 | `audio_speech.py` |
| `segmentation` | 1 | `segmentation.py` |
| `prompted_segmentation` | 1 | `segmentation.py` |
| `encoder_only_nlp` | 1 | `encoder_only.py` |
| `embedding` | 1 (skip) | `embedding.py` |
| `reranking` | 1 (skip) | `reranking.py` |

### E2E harness architecture (DIP-first)

```
tests/test_e2e.py                    # Single parametrized pytest entrypoint
tests/e2e/models/*.json              # 50 per-model JSON manifests
tests/e2e_harness/
  __init__.py                        # save_full_stderr() helper
  contracts.py                       # E2ECase, StageOutput, CompareResult, protocols
  orchestrator.py                    # Lifecycle: preflight -> build -> run -> compare
  manifest_loader.py                 # JSON -> E2ECase (with schema validation)
  registry.py                        # Auto-discover runners/references/comparators
  artifact_sink.py                   # Persist artifacts (JSON, logits, audio, images)
  runners/                           # TRT strategy runners (one per task_strategy)
    text_generation.py               # Causal LM (decoder, MoE, SSM, RWKV)
    vision_language.py               # VL models (Qwen VL, InternVL)
    audio_speech.py                  # Whisper, Bark, PersonaPlex
    diffusion.py                     # Wan T2V, FLUX, Z-Image
    segmentation.py                  # SegFormer, SAM
    embedding.py                     # Eagle-embed
    reranking.py                     # Eagle-rerank
    encoder_only.py                  # BERT
    omni.py                          # Qwen3-Omni
    object_detection.py              # YOLOX
    neural_operator.py               # DeepONet / FNO
  references/                        # Gold-standard reference backends
    hf_transformers.py               # HF Transformers (text, VL, audio, seg)
    hf_diffusers.py                  # HF Diffusers (Wan, FLUX, Z-Image)
    torch_reference.py               # PyTorch (speech-to-speech)
    custom_python.py                 # Custom Python scripts
    golden_snapshot.py               # Pre-saved reference data
    invariant_only.py                # No external reference (self-consistency)
  comparators/                       # Metric computation + threshold gating
    text.py                          # 6-metric composite: logit cosine, top1 match, NED
    vision_language.py               # Vision cosine + text NED/agreement
    text_to_audio.py                 # RMS, duration ratio, mel/spectral distance
    speech_to_text.py                # Transcript text similarity
    speech_to_speech.py              # Audio quality metrics
    diffusion.py                     # Pixel stats, temporal consistency, PSNR/SSIM
    segmentation.py                  # mIoU, pixel accuracy, boundary F-score
    encoder_only.py                  # Hidden state / CLS cosine similarity
    embedding.py                     # Embedding cosine distance
    reranking.py                     # Score correlation
    omni.py                          # Multi-modal composite
    neural_operator.py               # Field comparison
    audio.py                         # Re-export umbrella
  thresholds/                        # Default + per-model threshold profiles
    defaults/                        # Per-strategy JSON threshold files
```

### Manifest schema

Each model is defined by a JSON manifest in `tests/e2e/models/<model-name>.json`:

```json
{
  "name": "qwen3-0.6b",
  "hf_id": "Qwen/Qwen3-0.6B",
  "bundle": "qwen3-0.6b.trtfb",
  "family": "qwen",
  "runtime_strategy": "decoder_kv_cache",
  "max_cache_length": 256,
  "prompt": "The capital of France is",
  "max_new_tokens": 20,
  "logit_atol": 1e-3,
  "trust_remote_code": false
}
```

**Required fields**: `name` (always); `hf_id` and `family` (when not skipped).

**Type-checked fields**: `max_new_tokens` and `max_cache_length` must be `int`.

**Schema validation**: `manifest_loader._validate_manifest()` runs automatically on load. Unknown `runtime_strategy` values emit a warning.

**Optional**: `skip` (string reason to skip), `threshold_overrides` (per-metric), `test_image` (VL), diffusion-specific fields.

### Comparator diagnostics

Every `CompareResult` returned by any comparator includes:
- `passed` (bool) -- overall pass/fail
- `metrics` (dict) -- raw metric values
- `per_metric_pass` (dict) -- per-metric bool (which individual metrics passed)
- `gate_details` (list of str) -- human-readable explanation of each gate decision
- `message` (str) -- summary including full traceback on exception

### Error diagnostics

- **Full stderr**: When a subprocess fails, `save_full_stderr()` writes the
  complete stderr to `{artifacts_dir}/{case}_{stage}_stderr.log` and includes
  the path in the error message. The inline message shows only the last 2000
  chars.
- **Full tracebacks**: Exception blocks in the orchestrator capture
  `traceback.format_exc()` and include it in the `CompareResult.message`.

### 50 model manifests by category

| Category | Count | Models |
|----------|:--:|---------|
| Standard decoder | 24 | Qwen3, LLaMA, Mistral, Phi, GPT-2, OPT, Bloom, Nemotron, etc. |
| MoE decoder | 3 | Mixtral, Phi-MoE, DeepSeek-V2 |
| SSM | 2 | Mamba, RWKV |
| Encoder-only | 1 | BERT |
| Speech-to-text | 1 | Whisper |
| Text-to-audio | 2 | Bark-small, Bark-large |
| Speech-to-speech | 1 | PersonaPlex |
| Segmentation | 1 | SegFormer |
| Vision-language | 3+2 skip | Qwen2.5-VL, Qwen3-VL, InternVL3 |
| Diffusion (T2V/T2I) | 3 | Wan2.1-T2V, FLUX.1-schnell, Z-Image-Turbo |
| Embedding/Reranking | 2 skip | Eagle-embed, Eagle-rerank |
| Hybrid | 1 skip | Nemotron-H |

---

## Layer 6: Diff Framework

**Directory**: `tools/diff.py` + `tools/diff_framework/`

**Intent**: Ad-hoc GPU-accelerated TRT-vs-HF comparison for development and
debugging. Auto-detects `runtime_strategy` from HF config or bundle header
and runs applicable checks.

**How to run**:

```bash
# List all registered checks
python tools/diff.py list

# Run all applicable checks for a model
python tools/diff.py run --model Qwen/Qwen3-0.6B

# Specific checks with a bundle
python tools/diff.py run --model Qwen/Qwen3-0.6B \
  --bundle qwen3.trtfb --binary ./build/trtf \
  --test logit_diff --test runner_parity

# VL model with test image
python tools/diff.py run --model Qwen/Qwen2.5-VL-3B-Instruct \
  --bundle qwen25vl.trtfb --image test.jpg --binary ./build/trtf
```

### 6 registered checks

| Check | Strategies | Bundle? | What it validates |
|-------|-----------|:--:|---|
| `logit_diff` | decoder_kv_cache, decoder_moe, ssm_recurrent | No | Per-step logit comparison (4-prompt battery) |
| `layer_diff` | decoder_kv_cache, decoder_moe | No | Per-layer hidden state comparison (debug engine) |
| `runner_parity` | decoder_kv_cache, decoder_moe, ssm_recurrent | Yes | Python TrtRunner vs C++ binary (token-for-token) |
| `vl_pipeline` | vision_language | Yes | 4-stage VL: vision features, embed_input, generation, C++ parity |
| `perf_benchmark` | decoder_kv_cache, decoder_moe, ssm_recurrent | No | TRT vs HF latency/throughput |
| `diffusion_components` | diffusion | Yes | 9-step component comparison |

---

## Regression Tiers

Standard regression gate before merging changes. Run in order; each tier
catches progressively harder issues.

### Tier 1: Unit tests (no GPU, ~10 min)

Fast, deterministic tests for logic correctness. Always run first.

```bash
# Python builder unit tests
.venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Tools self-tests
.venv/bin/python -m pytest tests/tools/ -v

# C++ unit tests
ctest --test-dir build --output-on-failure
```

**What's covered**:
- Python: 50 test modules -- config, checkpoint_mapper, bundle_writer, family plugins, 27 per-family engine tests, manifest validation, debug runner, cache state machine
- Tools: 11 modules -- diff framework, logits, layers, audio, segmentation, diffusion helpers, perf_compare
- C++: 19 test executables -- bundle format, tokenizers, CUDA RAII, KV cache, image preprocessor, CLI args

### Tier 1.5: C++ Cyclomatic Complexity Gate (no GPU, <1 min)

Cyclomatic complexity is measured with `lizard`, which is baked into both
container images (`Dockerfile`, `Dockerfile.gb300`) and verified in
`scripts/setup_container.sh`.

Use the repository checker:

```bash
# Report-only scan for C++ runtime sources
python tools/check_cyclomatic_complexity.py src

# Gate: fail if any function is above CCN 10
python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

CI gate job: `check-cyclomatic-complexity` in `.gitlab-ci.yml`.
Threshold can be tuned via:
- `CCM_MAX_CCN`

Current policy and status:
- CI default is strict: fail on any function with `CCN > 10`.
- As of March 4, 2026, repository scan reports `CCN max: 9` and `CCN >= 10: 0`.

### Tier 2: Graph-op GPU tests (~2 min, needs TRT)

```bash
.venv/bin/python -m pytest tests/builder/test_graph_ops.py \
  tests/builder/test_graph_ops_extended.py \
  tests/builder/test_graph_blocks.py -v -m trt
```

### Tier 3: E2E single-model smoke test (~5 min, needs GPU)

```bash
.venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines
```

### Tier 4: Full E2E suite (~2-3 hours, needs GPU)

All 50 models, force-rebuild every bundle. Gold-standard regression gate.

```bash
.venv/bin/python -m pytest tests/test_e2e.py -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines --e2e-artifacts-dir /tmp/e2e_artifacts
```

### Tier 5: Performance regression (~10 min per model)

```bash
python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /mnt/storage/trt-transformers/engines/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" --max-new-tokens 20 --json results.json
```

### What to Run When

| Change type | Tiers to run |
|-------------|-------------|
| Python builder logic | 1, 2 |
| Family plugin | 1 (includes per-family engine tests), 2, 3 (the specific model) |
| C++ runtime | 1 (ctest), 3 |
| Graph ops / graph blocks | 1, 2 |
| KV cache / mask / position logic | 1 (ctest + cache_state_machine), 3, 4 |
| debug_runner.py | 1 (debug_runner_extended), 3 |
| Image preprocessor | 1 (ctest test_image_preprocessor) |
| Tokenizer (vocab or HF) | 1 (ctest test_vocab_tokenizer / test_hf_python_tokenizer) |
| perf_compare.py | 1 (tools tests), 5 |
| Diff tools (audio/seg/diffusion) | 1 (tools tests) |
| Vision encoder / VL pipeline | 1 (vision_compute_extended), 2, 3 |
| New model family | 1, 2, `validate_family.sh`, then add manifest + tier 4 |
| New model (existing family) | Add JSON manifest, run tier 3 with that model |
| E2E harness (runners/comparators) | Tier 3 or 4 (run affected models) |
| Manifest loader changes | 1 (test_manifest_validation.py), tier 3 |

---

## Pytest Markers Reference

Registered in `pyproject.toml`:

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests -- no GPU, no TRT |
| `gpu` | Requires NVIDIA GPU |
| `trt` | Requires TensorRT |
| `slow` | Slow tests (>30s) |
| `e2e` | End-to-end tests |
| `text` | Text generation models |
| `vision` | Vision/VL models |
| `audio` | Audio models (Whisper, Bark, PersonaPlex) |
| `diffusion` | Diffusion models (Wan, FLUX, Z-Image) |

Usage:

```bash
# Run only unit tests (fast, no GPU)
pytest tests/builder/ -m unit -v

# Exclude GPU tests
pytest tests/builder/ -m "not gpu" -v

# Only TRT graph tests
pytest tests/builder/ -m trt -v
```

---

## Coverage

Coverage is configured in `pyproject.toml`:

```bash
# Run with coverage
.venv/bin/python -m pytest tests/builder/ tests/tools/ -v \
  --ignore=tests/builder/test_cli.py --cov --cov-report=term-missing

# HTML report
.venv/bin/python -m pytest tests/builder/ tests/tools/ -v \
  --ignore=tests/builder/test_cli.py --cov --cov-report=html
```

Configuration:
- **Source**: `trtf_build/trtf_build` (the build package)
- **Omit**: `*/tests/*`, `*/__pycache__/*`
- **Excluded lines**: `pragma: no cover`, `if __name__ == "__main__"`, `raise NotImplementedError`

---

## `validate_family.sh` Workflow

The primary validation gate for new model families:

```
validate_family.sh <hf-repo-or-path> [options]
  |
  +-- Step 1: Build bundle
  |     trtf-build build <model> -o /tmp/<name>.trtfb --max-cache-length 256
  |
  +-- Step 2: diff_logits battery
  |     python tools/diff_logits.py --model <model> --atol 1e-3 --battery
  |
  +-- Step 3: diff_layers
  |     python tools/diff_layers.py --model <model> --atol 0.05
  |
  +-- Step 4: runner_parity (if binary exists)
        python tools/test_runner_parity.py --bundle /tmp/<name>.trtfb \
          --binary ./build/trtf --hf-python .venv/bin/python --max-new-tokens 20
```

All 4 steps must pass. Step 4 is skipped if `./build/trtf` is not found.

---

## Adding a Model to the Test Suite

1. **Run `validate_family.sh`** to confirm the model builds and passes diff checks.

2. **Create a manifest JSON** in `tests/e2e/models/<model-name>.json`.

3. **Run Tier 3 smoke test**:
   ```bash
   .venv/bin/python -m pytest tests/test_e2e.py::test_e2e[my-model] -v \
     --engine-dir /mnt/storage/trt-transformers/engines \
     --trtf-binary ./build/trtf --hf-python .venv/bin/python \
     --rebuild-engines
   ```

4. **Run Tier 4 full suite** to confirm no regressions.

---

## Text Generation Comparator: Composite Gating

The text comparator (`tests/e2e_harness/comparators/text.py`) uses 6 metrics
with composite gating. Understanding the gating logic is important for
diagnosing test failures.

### Metrics

| Metric | Compares | Pass criterion |
|--------|----------|----------------|
| `logit_cosine_p5` | TRT debug runner logits vs HF logits | >= 0.99 (5th percentile) |
| `logit_rel_l2_p95` | TRT debug runner logits vs HF logits | <= 0.05 (95th percentile) |
| `stable_top1_match_rate` | Argmax agreement on confident tokens | >= 0.9 |
| `unstable_topk_hit_rate` | Top-k overlap on ambiguous tokens | >= 0.8 |
| `token_agreement_rate` | Raw argmax token-for-token match | >= 0.8 |
| `normalized_text_edit_distance` | C++ binary text vs HF text | <= 0.2 |

### Composite gating rule

```
passed = logit_quality_ok AND token_level_ok AND text_ok
```

Where:
- `logit_quality_ok` = cosine_p5 passes OR rel_l2_p95 passes
- `token_level_ok` = agreement passes OR (stable_top1 passes AND unstable_topk passes)
- `text_ok` = NED passes (with two adjustments below)

### Prompt-echo stripping

The C++ binary outputs `prompt + generation` while HF returns only
`generation`. Before computing NED, the comparator strips the prompt prefix
from the C++ text (using the known prompt from `trt.data["prompt"]`). This
prevents inflated NED from the echo.

### NED hard-fail threshold

When NED >= 0.65, the test fails **regardless** of logit/token metrics. This
catches genuinely broken C++ text (repetition loops, empty output, chat
template bugs) that would otherwise be masked by good debug runner logits.

For NED < 0.65 with good token metrics, the NED failure is treated as
acceptable divergence (minor sampling differences, max_new_tokens budget
differences, etc.).

### Why logits can match but text differs

The comparator compares **debug runner logits** (Python TRT) against **HF logits**.
Both use the same tokenization (`tokenizer.encode(prompt)` with defaults).
The C++ binary uses a separate tokenizer path (`hf_python_tokenizer.py`).
If the C++ tokenizer uses different `add_special_tokens` settings, the input
tokens differ, causing completely different generated text despite "perfect"
logit match. This is why the `tokenizer_add_special_tokens` bundle field
exists — see [Architecture Overview](Architecture-Overview.md#bundle-config-self-describing-runtime-behavior).

---

## Accuracy Tolerances Reference

| Model | Strategy | `logit_atol` | `layer_atol` | Rationale |
|-------|----------|:--:|:--:|---|
| Most standard decoders | `decoder_kv_cache` | `1e-3` | `0.05` | Baseline FP32 precision |
| mamba-130m | `ssm_recurrent` | `2e-3` | -- | Recurrent state drift |
| mixtral-stories-15m | `decoder_moe` | `2e-3` | `0.05` | Expert routing precision |
| phi-moe | `decoder_kv_cache` | `1e-3` | `0.05` | SparseMixer is deterministic |
| VL models | `vision_language` | `1e-3` | `0.05` | Vision features `atol=0.1` |
| wan21-t2v-1.3b | `diffusion` | -- | -- | Cosine sim (0.95 single step, 0.8 full) |

**Why some models need looser tolerances**:
- **Distilled models** (minitron): Pruning amplifies kernel differences
- **Recurrent models** (mamba): FP32 drift accumulates across recurrence
- **MoE models** (mixtral): Expert routing softmax is precision-sensitive
- **Cross-lingual** (xglm): Multi-language vocabulary increases variance
