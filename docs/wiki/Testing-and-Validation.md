# Testing and Validation

Comprehensive guide to testing TRT-Transformers-CPP. Organized by model category with specific checks, tolerances, and workflows for each `runtime_strategy`.

## Test Architecture Overview

### Philosophy

Every TRT engine must produce output matching HuggingFace Transformers (the ground truth). Testing validates this at multiple granularities: per-layer hidden states, per-step logits, full generation text, and (for diffusion) component-by-component pipeline fidelity.

### Three Pillars

1. **Unit tests** -- Fast, deterministic, no GPU. Validate config parsing, checkpoint mapping, bundle I/O, CLI args, diff framework mechanics, and C++ decode runtime logic.
2. **Diff framework** -- GPU-accelerated TRT-vs-HF comparison. The unified `tools/diff.py` CLI auto-detects model type and runs applicable checks. Six registered checks currently cover five core runtime strategies (`decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `vision_language`, `diffusion`); other strategies are covered primarily by unit and E2E tests.
3. **E2E suite** -- Full pipeline tests: build bundle from HF, run C++ inference, compare against HF reference. 28 model manifests in `tests/e2e/models/` drive parametrized pytest tests.

### Test Applicability Matrix

Which checks apply to which `runtime_strategy` (for strategies currently supported by `tools/diff.py` checks):

| Check | `decoder_kv_cache` | `decoder_moe` | `ssm_recurrent` | `vision_language` | `diffusion` |
|-------|:--:|:--:|:--:|:--:|:--:|
| `logit_diff` | Y | Y | Y | -- | -- |
| `layer_diff` | Y | Y | -- | -- | -- |
| `runner_parity` | Y | Y | Y | -- | -- |
| `vl_pipeline` | -- | -- | -- | Y | -- |
| `diffusion_components` | -- | -- | -- | -- | Y |
| `perf_benchmark` | Y | Y | Y | -- | -- |

Additional runtime strategies in the C++ dispatcher (for example `segmentation`, `object_detection`, `prompted_segmentation`, `speech_to_text`, `text_to_audio`, `speech_to_speech`, `embedding`, `reranking`, `neural_operator`, `omni_multimodal`, `rwkv_recurrent`, `hybrid_mamba_attention`) are validated through dedicated unit/E2E suites rather than the unified diff checks above.

---

## Regression Tiers

Standard regression gate before merging. Run in order; each tier catches progressively harder issues.

### Tier 1: Unit tests (no GPU, ~60s)

Fast, deterministic tests for logic correctness. Always run first.

```bash
# Python builder unit tests (config, checkpoint_mapper, bundle_writer, etc.)
.venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Tools self-tests (diff framework, perf_compare stats/formatting/serial ordering)
.venv/bin/python -m pytest tests/tools/ -v

# C++ unit tests (bundle format, decode runtime, image preprocessor)
ctest --test-dir build --output-on-failure
```

**What's covered**:
- Python: `tests/builder/` (11 modules) -- config parsing, checkpoint mapping, bundle writing, graph ops
- Tools: `tests/tools/` (7 modules) -- diff framework protocol/registry/runner, diff_logits, diff_layers, diff_vl, parity, perf_compare
- C++: 11 test executables -- bundle format, C ABI, CLI args, decode runtime, image preprocessor, JSON helpers, text parsers, fast_path_config

### Tier 2: Graph-op GPU tests (~2 min, needs TRT)

Validates TRT graph operations (RMSNorm, RoPE, attention, etc.) on real GPU.

```bash
.venv/bin/python -m pytest tests/builder/test_graph_ops.py -v -m trt
```

### Tier 3: E2E single-model smoke test (~5 min, needs GPU)

Quick sanity with one small model. Always use `--rebuild-engines` to build the bundle from scratch -- avoids testing against stale cached bundles.

```bash
.venv/bin/python -m pytest tests/e2e/test_full_pipeline.py -v -k qwen3-0.6b \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines
```

### Tier 4: Full E2E suite (~90 min, needs GPU)

All models in `tests/e2e/models/`: force-rebuild every bundle, then infer/compare for each. This is the gold-standard regression gate.

```bash
.venv/bin/python -m pytest tests/e2e/ -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines
```

### Tier 5: Performance regression (~10 min per model, needs GPU + bundle)

Spot-check inference speed for key models. Not in CI; run manually for perf-sensitive changes.

```bash
source .venv/bin/activate
python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /mnt/storage/trt-transformers/engines/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" --max-new-tokens 20 --json results.json
```

### What to Run When

| Change type | Tiers to run |
|-------------|-------------|
| Python builder logic | 1, 2 |
| Family plugin | 1, 2, 3 (the specific model) |
| C++ runtime | 1 (ctest), 3 |
| Graph ops | 1, 2 |
| KV cache / mask / position logic | 1, 3, 4 |
| debug_runner.py | 1, 3 |
| perf_compare.py | 1 (tools tests), 5 |
| New model family | 1, 2, `validate_family.sh`, then add to `tests/e2e/models/` + tier 4 |
| Diffusion pipeline (Python or C++) | 1, `test_diffusion_pipeline.py` with `--rebuild-engines` |
| VL pipeline (Python or C++) | 1, `test_vl_pipeline.py` with `--rebuild-engines` |
| Diff framework changes | 1 (`tests/tools/test_diff_framework.py`) |

---

## Unified Diff Framework

### Architecture

The diff framework (`tools/diff_framework/`) is a plugin-based test orchestration layer with a unified CLI (`tools/diff.py`).

```
tools/diff.py                          # CLI: list | run
tools/diff_framework/
  __init__.py                          # Public API exports, auto-discovers checks
  protocol.py                          # DiffResult, TestContext, DiffTest protocol
  registry.py                          # register() decorator, get_all_tests(), get_tests_for_strategy()
  runner.py                            # detect_runtime_strategy(), list_tests(), run_tests()
  checks/
    __init__.py                        # Auto-discovers all check modules via pkgutil
    logit_diff.py                      # LogitDiffTest
    layer_diff.py                      # LayerDiffTest
    runner_parity.py                   # RunnerParityTest
    vl_pipeline.py                     # VLPipelineTest
    perf_benchmark.py                  # PerfBenchmarkTest
    diffusion_components.py            # DiffusionComponentsTest
```

### Core Contracts

**`DiffResult`** (`protocol.py`): Result container returned by every check.

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `str` | Check name (e.g., `"logit_diff"`) |
| `model` | `str` | HF model ID or path |
| `runtime_strategy` | `str` | Detected or specified strategy |
| `passed` | `bool` | Whether the check passed |
| `status` | `str` | `"PASS"`, `"FAIL"`, `"SKIP"`, or `"ERROR"` |
| `message` | `str` | Human-readable summary |
| `metrics` | `dict` | Check-specific numeric results |
| `duration_s` | `float` | Wall-clock time in seconds |
| `details` | `str` | Extended diagnostic info |

Static constructors: `DiffResult.skip(...)` for missing prerequisites, `DiffResult.error(...)` for exceptions.

**`TestContext`** (`protocol.py`): Shared context passed to all checks.

| Field | Default | Description |
|-------|---------|-------------|
| `model` | required | HF model ID or local path |
| `runtime_strategy` | required | Auto-detected or user-specified |
| `bundle_path` | `None` | Path to `.trtfb` bundle (required by some checks) |
| `binary_path` | `None` | Path to `trtf` C++ binary |
| `hf_python` | `None` | Python interpreter with HF deps |
| `image_path` | `None` | Test image for VL models |
| `max_cache_length` | `256` | Engine cache length |
| `max_new_tokens` | `20` | Generation length for comparison |
| `atol` | `1e-3` | Logit absolute tolerance |
| `layer_atol` | `0.05` | Layer hidden state tolerance |
| `trust_remote_code` | `False` | HF trust_remote_code flag |
| `num_inference_steps` | `30` | Diffusion denoising steps |

**`DiffTest`** (Protocol): Interface for checks.

```python
class DiffTest(Protocol):
    name: str
    description: str
    runtime_strategies: list[str]
    requires_bundle: bool
    requires_gpu: bool
    def run(self, ctx: TestContext) -> DiffResult: ...
```

### The 6 Registered Checks

| Check | Strategies | Requires Bundle | What It Validates |
|-------|-----------|:-:|---|
| **`logit_diff`** | `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent` | No | Per-step logit comparison via `diff_logits.py`. Battery of 4 prompts, absolute tolerance check. |
| **`layer_diff`** | `decoder_kv_cache`, `decoder_moe` | No | Per-layer hidden state comparison via `diff_layers.py`. Builds debug engine with `debug_layer_outputs=True`. |
| **`runner_parity`** | `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent` | Yes | Token-for-token comparison: Python `TrtRunner` vs C++ `trtf` binary. |
| **`vl_pipeline`** | `vision_language` | Yes | 4-stage VL validation via `diff_vl.py`: vision features, embed_input, text generation, C++ binary parity. |
| **`perf_benchmark`** | `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent` | No | TRT vs HF latency/throughput comparison via `perf_compare.py`. |
| **`diffusion_components`** | `diffusion` | Yes | 9-step component comparison via `debug_diffusion_pipeline.py`. See [Diffusion testing](#diffusion-diffusion) below. |

### Auto-Detection

The runner auto-detects `runtime_strategy` from:
1. **HF config** (`detect_runtime_strategy()`): Loads `config.json`, finds the family plugin, reads its `runtime_strategy` attribute.
2. **Bundle header** (`detect_runtime_strategy_from_bundle()`): Parses the bundle's embedded `config.json` for the `runtime_strategy` field.

Falls back to `"decoder_kv_cache"` if detection fails.

### CLI Usage

```bash
# List all registered checks
python tools/diff.py list

# List checks applicable to a specific model (strategy auto-detected)
python tools/diff.py list --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Run all applicable checks for a model
python tools/diff.py run --model Qwen/Qwen3-0.6B

# Run specific checks with a bundle
python tools/diff.py run --model Qwen/Qwen3-0.6B \
  --bundle qwen3.trtfb --binary ./build/trtf \
  --test logit_diff --test runner_parity

# Run with JSON output
python tools/diff.py run --model Qwen/Qwen3-0.6B --json results.json

# Diffusion model (run only diffusion component check)
python tools/diff.py run --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --bundle wan21.trtfb --test diffusion_components

# VL model with test image
python tools/diff.py run --model Qwen/Qwen2.5-VL-3B-Instruct \
  --bundle qwen25vl.trtfb --image test.jpg --binary ./build/trtf
```

### Self-Tests

The framework itself is tested in `tests/tools/test_diff_framework.py` (no GPU required):

- `TestDiffResult`: Serialization, `skip()`/`error()` constructors, default fields
- `TestRegistry`: Check registration, strategy filtering, unknown test lookup
- `TestRunner`: Test listing, bundle-skip logic, unknown name error
- `TestCLI`: Module imports, subcommand existence

---

## Testing by Model Category

### Transformer-Based (`decoder_kv_cache` / `decoder_moe`)

**Coverage**: 25 models (23 standard decoder + 2 MoE) in `tests/e2e/models/`.

**Standard decoder models** (23): qwen3-0.6b, qwen3-4b-instruct-2507, phi3-mini, tinyllama-1.1b, nemotron-hindi-4b, nemotron-mini-4b, nemotron-nano-4b, minitron-4b-depth, minitron-4b-width, gpt2-125m, opt-125m, bloom-560m, gpt-neo-125m, codegen-350m, falcon3-1b, granite-3.1-2b, internlm2-1.8b (skipped), olmo-1b, pythia-70m, riva-translate-4b, stablelm2-1.6b, starcoder2-3b, xglm-564m.

**MoE models** (2): phi-moe (`decoder_kv_cache`), mixtral-stories-15m (`decoder_moe`).

**Applicable checks**:
- **`logit_diff`**: Battery of 4 prompts, per-step logit comparison. Default `atol=1e-3`.
- **`layer_diff`**: Builds a debug engine with per-layer hidden state outputs (`debug_layer_outputs=True`). Compares embedding, all decoder layers, and final logits. Default `layer_atol=0.05`.
- **`runner_parity`**: Token-for-token comparison between Python `TrtRunner` and C++ `trtf` binary. Verifies the C++ runtime reproduces Python TRT inference exactly.
- **`perf_benchmark`**: Latency and throughput comparison. Serial GPU execution to avoid OOM on 24GB GPUs.

**MoE note**: Routing is handled entirely in the TRT graph. The C++ runtime uses the same `TrtBackendFastPath` backend as standard decoders. The `decoder_moe` strategy is identical to `decoder_kv_cache` at runtime -- the distinction exists for test selection and documentation.

**E2E test files**:
- `test_full_pipeline.py` -- Build + C++ inference + diff_logits + perf_compare
- `test_inference.py` -- Basic inference sanity (non-empty output, determinism)
- `test_bundle_inspect.py` -- `trtf inspect` output validation
- `test_logit_parity.py` -- `diff_logits.py --battery` wrapper
- `test_runner_parity.py` -- `test_runner_parity.py` wrapper

### SSM / Mamba (`ssm_recurrent`)

**Coverage**: 1 model -- mamba-130m (`state-spaces/mamba-130m-hf`).

**Applicable checks**:
- **`logit_diff`**: Uses `MambaTrtRunner` for pure-Python TRT inference with device-resident recurrent state. `logit_atol=2e-3` (higher than standard because recurrent state accumulates floating-point drift).
- **`runner_parity`**: Python `MambaTrtRunner` vs C++ `MambaBackend`. Same token-for-token parity check.
- **`perf_benchmark`**: TRT vs HF latency comparison.

**Not applicable**: `layer_diff` (no debug engine support for SSM -- Mamba has no standard decoder layer structure).

**Key differences from transformer testing**:
- No prefill phase in C++ (Mamba processes tokens one at a time)
- Recurrent state (conv_state + ssm_state) is constant-size per layer, not growing like KV cache
- Higher logit tolerance due to cumulative FP32 drift in recurrent computations

### Vision-Language (`vision_language`)

**Coverage**: 2 models -- qwen25vl-3b (`Qwen/Qwen2.5-VL-3B-Instruct`), qwen3-vl-2b (`Qwen/Qwen3-VL-2B-Instruct`).

**Applicable checks**:
- **`vl_pipeline`**: 4-stage validation via `diff_vl.py`:
  1. **Vision features**: Run TRT vision encoder on a test image, compare against HF reference features. `atol=0.1`.
  2. **Embed input**: Verify image features are correctly injected into the text decoder's embedding space.
  3. **Text generation**: Compare VL-conditioned text generation output (token match).
  4. **C++ parity**: Compare C++ `trtf` binary VL output against Python `VLTrtRunner`.

**Qwen3-VL specifics**: Additional DeepStack feature validation -- multi-level vision features injected at early text decoder layers. The vision encoder outputs per-level features that are stored and fed to different decoder layers during image token prefill.

**E2E test files**:
- `test_vl_pipeline.py` -- Vision-only smoke test (`diff_vl.py --vision-only`) + VL generation via C++ binary
- `test_full_pipeline.py::test_full_pipeline_vlm` -- Full VL E2E: build, diff_vl, perf

### Diffusion (`diffusion`)

**Coverage**: 1 model -- wan21-t2v-1.3b (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`).

**Why tested differently**: Diffusion models produce video frames, not text tokens. The output is a multi-component pipeline (T5 text encoder, DiT denoiser, VAE decoder) where noise drift accumulates over 30 denoising steps. Absolute logit comparison is meaningless -- instead, validation is component-by-component with cosine similarity thresholds.

**Applicable checks**:
- **`diffusion_components`**: Delegates to `tools/debug_diffusion_pipeline.py`, which runs 9 component-by-component TRT-vs-HF comparisons:

| Step | What It Validates | Pass Criterion |
|------|-------------------|----------------|
| 1. Bundle config | Config fields match HF pipeline | Exact match |
| 2. Text projection | DiT preprocessor weight activation | Exact match |
| 3. T5 encoding | TRT T5 encoder vs HF UMT5EncoderModel | `atol` threshold |
| 4. Timestep embedding | Sinusoidal + MLP timestep computation | Exact match |
| 5. Patch embedding | Patchify vs Conv3D equivalence | Exact match |
| 6. 3D RoPE | Rotary position embeddings (temporal + spatial) | Exact match |
| 7. Single DiT step | One denoiser forward pass | `cosine_sim > 0.95` |
| 8. Scheduler sigmas | Flow-match Euler sigma schedule | Exact match |
| 9. Full denoising | Complete 30-step denoising loop | `cosine_sim > 0.8` at final step |

**Frame quality checks** (E2E test):
- Pixel mean in `[0.15, 0.85]` -- catches all-black or all-white output
- Pixel std >= `0.05` -- catches washed-out or flat-color output
- Correct frame count (17 for Wan2.1-T2V-1.3B)
- PNG files exist on disk

**C++ generate-video check**:
- Runs `trtf generate-video` command
- Verifies correct number of PNG frames produced
- Sample frames (first/middle/last) saved alongside bundle for visual inspection

**E2E test file**: `test_diffusion_pipeline.py` with 4 test functions:
1. `test_diffusion_build` -- Bundle builds successfully
2. `test_diffusion_debug_pipeline` -- All 9 component checks pass
3. `test_diffusion_cpp_generate` -- C++ binary produces correct frame count
4. `test_diffusion_frame_quality` -- Pixel statistics within bounds

---

## Accuracy Tolerances Reference

Default tolerances by model and check type. Models with non-default tolerances are listed explicitly.

| Model | Strategy | `logit_atol` | `layer_atol` | Rationale |
|-------|----------|:--:|:--:|---|
| Most standard decoders | `decoder_kv_cache` | `1e-3` | `0.05` | Baseline FP32 precision |
| mamba-130m | `ssm_recurrent` | `2e-3` | -- | Recurrent state drift |
| mixtral-stories-15m | `decoder_moe` | `2e-3` | `0.05` | Expert routing precision |
| phi-moe | `decoder_kv_cache` | `1e-3` | `0.05` | Standard (SparseMixer is deterministic) |
| pythia-70m | `decoder_kv_cache` | `0.05` | `0.05` | Older architecture, higher numeric variance |
| xglm-564m | `decoder_kv_cache` | `0.1` | `0.05` | Cross-lingual model, higher variance |
| minitron-4b-depth | `decoder_kv_cache` | `0.4` | `0.05` | Distilled model, known high variance |
| minitron-4b-width | `decoder_kv_cache` | `0.5` | `0.05` | Distilled model, known high variance |
| qwen25vl-3b | `vision_language` | `1e-3` | `0.05` | Vision features `atol=0.1` |
| qwen3-vl-2b | `vision_language` | `1e-3` | `0.05` | Vision features `atol=0.1` |
| wan21-t2v-1.3b | `diffusion` | -- | -- | Cosine sim thresholds (0.95 single step, 0.8 full pipeline) |

**Why some models need looser tolerances**:
- **Distilled models** (minitron): Pruning creates sharper weight distributions where small TRT kernel differences amplify
- **Recurrent models** (mamba): FP32 drift accumulates across the recurrence chain
- **MoE models** (mixtral): Expert routing softmax is sensitive to small logit differences
- **Cross-lingual** (xglm): Multi-language vocabulary increases numerical sensitivity

---

## E2E Model Manifests

Each model is defined by a JSON manifest in `tests/e2e/models/`. The conftest.py loads all manifests and parametrizes tests.

### Schema

**Standard decoder fields**:

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
  "layer_atol": 0.05,
  "trust_remote_code": false
}
```

**Additional VL fields**: `test_image` (path to test image).

**Additional diffusion fields**:

```json
{
  "test_type": "diffusion",
  "test_prompt": "A cat sitting on a beach watching the sunset",
  "video_num_frames": 17,
  "video_height": 480,
  "video_width": 832,
  "num_inference_steps": 30,
  "min_pixel_mean": 0.15,
  "max_pixel_mean": 0.85,
  "min_pixel_std": 0.05,
  "build_args": { "max_cache_length": 256 }
}
```

**Optional fields**: `skip` (string reason to skip), `notes` (documentation).

### Current Model Count

28 manifests across 4 strategies (the `wan21-t2v-1.3b` diffusion manifest is in the main branch):

| Strategy | Count | Models |
|----------|:--:|---|
| `decoder_kv_cache` | 23 | qwen3-0.6b, qwen3-4b-instruct-2507, phi3-mini, phi-moe, tinyllama-1.1b, nemotron-hindi-4b, nemotron-mini-4b, nemotron-nano-4b, minitron-4b-depth, minitron-4b-width, gpt2-125m, opt-125m, bloom-560m, gpt-neo-125m, codegen-350m, falcon3-1b, granite-3.1-2b, internlm2-1.8b, olmo-1b, pythia-70m, riva-translate-4b, stablelm2-1.6b, starcoder2-3b, xglm-564m |
| `decoder_moe` | 1 | mixtral-stories-15m |
| `ssm_recurrent` | 1 | mamba-130m |
| `vision_language` | 2 | qwen25vl-3b, qwen3-vl-2b |
| `diffusion` | 1 | wan21-t2v-1.3b |

---

## `validate_family.sh` Workflow

The primary validation gate for new model families. Runs 4 steps sequentially:

```
validate_family.sh <hf-repo-or-path> [options]
  |
  +-- Step 1: Build bundle
  |     trtf-build build <model> -o /tmp/<name>.trtfb --max-cache-length 256
  |
  +-- Step 2: diff_logits battery
  |     python tools/diff_logits.py --model <model> --atol 1e-3 --battery
  |     (4 prompts, per-step logit comparison)
  |
  +-- Step 3: diff_layers
  |     python tools/diff_layers.py --model <model> --atol 0.05
  |     (per-layer hidden state comparison)
  |
  +-- Step 4: runner_parity (if binary exists)
        python tools/test_runner_parity.py --bundle /tmp/<name>.trtfb \
          --binary ./build/trtf --hf-python .venv/bin/python --max-new-tokens 20
        (Python TrtRunner vs C++ trtf binary)
```

### Usage

```bash
# Validate from HF repo ID
./scripts/validate_family.sh Qwen/Qwen3-0.6B

# With trust-remote-code for models that need it
./scripts/validate_family.sh microsoft/Phi-3-mini-4k-instruct --trust-remote-code

# From a local model directory
./scripts/validate_family.sh /mnt/models/my-model
```

### Pass criteria

All 4 steps must pass. Step 4 is skipped if `./build/trtf` is not found.

---

## Adding a Model to the Test Suite

After a model family is validated, add it to the E2E test suite:

1. **Run `validate_family.sh`** -- Confirms the model builds and passes all diff checks.

2. **Create a manifest JSON** in `tests/e2e/models/<model-name>.json`:
   ```json
   {
     "name": "my-model-1b",
     "hf_id": "org/My-Model-1B",
     "bundle": "my-model-1b.trtfb",
     "family": "my_family",
     "runtime_strategy": "decoder_kv_cache",
     "max_cache_length": 256,
     "prompt": "The capital of France is",
     "max_new_tokens": 20,
     "logit_atol": 1e-3,
     "layer_atol": 0.05,
     "trust_remote_code": false
   }
   ```

3. **Run Tier 3 smoke test** with the specific model:
   ```bash
   .venv/bin/python -m pytest tests/e2e/test_full_pipeline.py -v -k my-model-1b \
     --engine-dir /mnt/storage/trt-transformers/engines \
     --trtf-binary ./build/trtf --hf-python .venv/bin/python \
     --rebuild-engines
   ```

4. **Run Tier 4 full suite** to confirm no regressions:
   ```bash
   .venv/bin/python -m pytest tests/e2e/ -v \
     --engine-dir /mnt/storage/trt-transformers/engines \
     --trtf-binary ./build/trtf --hf-python .venv/bin/python \
     --rebuild-engines
   ```

For VL models, add a `test_image` field pointing to a test image. For diffusion models, set `test_type: "diffusion"` and add the diffusion-specific fields (see [schema](#schema) above).
