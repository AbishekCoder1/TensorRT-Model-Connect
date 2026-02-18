# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

C++ bundle-only runtime for TensorRT inference, paired with a Python build package (`trtf_build/`) that converts HuggingFace models into `.trtfb` bundles. The Python package handles model loading, TRT engine building, and bundle packaging. The C++ runtime loads `.trtfb` bundles, deserializes TRT engines, and runs autoregressive inference. Public C++ API lives in `include/trtf/`. Everything is in the `trtf` namespace. C++17, compiled with `-Wall -Wextra -Wpedantic`.

## Build commands

ALWAYS DO EVERYTHING IN CONTAINER. Run command with docker exec

### C++ runtime

Native host build (works without TRT/CUDA; TRT backend auto-disables if deps not found):
```bash
cmake -S . -B build -G Ninja
cmake --build build -j
```

With explicit TRT/CUDA paths:
```bash
cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=<path>/include/zapped_headers \
  -DTRTF_TRT_LIBRARY=<path>/lib/libnvinfer.so \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

### Python build package

```bash
pip install --no-deps -e trtf_build/

# Build from HF repo ID (auto-downloads) or local directory
trtf-build build Qwen/Qwen3-0.6B -o <output.trtfb> [--max-cache-length N] [--verbose]
trtf-build build <model-dir> -o <output.trtfb> [--max-cache-length N] [--verbose]

# Or use the Python API
python3 -c "import trtf_build; trtf_build.build('Qwen/Qwen3-0.6B', 'qwen3.trtfb')"

trtf-build inspect <bundle.trtfb>
trtf-build version
```

## Running tests

C++ runtime unit tests:
```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R test_pipeline --output-on-failure
```

Python builder unit tests:
```bash
pytest tests/builder/ -v
```

Diff framework self-tests:
```bash
pytest tests/tools/ -v
```

E2E tests (requires GPU + engine bundles):
```bash
pytest tests/e2e/ -v --engine-dir /mnt/storage/trt-transformers/engines
```

Full-pipeline E2E tests (build + infer + compare):
```bash
pytest tests/e2e/test_full_pipeline.py -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python

# Force rebuild all bundles from HF:
pytest tests/e2e/test_full_pipeline.py -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python --rebuild-engines
```

All Python tests at once:
```bash
pytest tests/ -v --ignore=tests/cpp
```

C++ tests are plain executables (no framework) in `tests/cpp/`. They use `main()`, print to stderr on failure, and return 0 on success / non-zero on failure. Test names in CMake match their source files.

## Regression test plan

Standard regression gate before merging changes. Run in order; each tier
catches progressively harder issues.

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

### Tier 2: Graph-op GPU tests (~2 min, needs TRT)

Validates TRT graph operations (RMSNorm, RoPE, attention, etc.) on real GPU.

```bash
.venv/bin/python -m pytest tests/builder/test_graph_ops.py -v -m trt
```

### Tier 3: E2E single-model smoke test (~5 min, needs GPU)

Quick sanity with one small model. Always use `--rebuild-engines` to build
the bundle from scratch — avoids testing against stale cached bundles.

```bash
.venv/bin/python -m pytest tests/e2e/test_full_pipeline.py -v -k qwen3-0.6b \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines
```

### Tier 4: Full E2E suite (~90 min, needs GPU)

All models in engines.json: force-rebuild every bundle, then
infer/compare for each. This is the gold-standard regression gate.

```bash
.venv/bin/python -m pytest tests/e2e/ -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines
```

### Tier 5: Performance regression (~10 min per model, needs GPU + bundle)

Spot-check inference speed for key models. Not in CI; run manually for
perf-sensitive changes.

```bash
source .venv/bin/activate
python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /mnt/storage/trt-transformers/engines/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" --max-new-tokens 20 --json results.json
```

### What to run when

| Change type | Tiers to run |
|-------------|-------------|
| Python builder logic | 1, 2 |
| Family plugin | 1, 2, 3 (the specific model) |
| C++ runtime | 1 (ctest), 3 |
| Graph ops | 1, 2 |
| KV cache / mask / position logic | 1, 3, 4 |
| debug_runner.py | 1, 3 |
| perf_compare.py | 1 (tools tests), 5 |
| New model family | 1, 2, validate_family.sh, then add to engines.json + tier 4 |

## Running executables

```bash
./build/trtf run     <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--hf-python PATH]
./build/trtf run     <bundle.trtfb> --prompt "text" --image <image.jpg> [--max-new-tokens N] [--hf-python PATH]
./build/trtf inspect <bundle.trtfb>
./build/trtf version
```

For vision-language models, pass `--image` with the path to an image file. The VL bundle must have been built from a VL model (e.g. Qwen2.5-VL) and contains both a text decoder and vision encoder engine.

## Key environment variables

- `TRTF_TRT_LOG_STDERR=1` / `TRTF_TRT_LOG_MIN_SEVERITY` - TRT logger controls

## Architecture

The system is split into two stages:

1. **Python build** (`trtf_build/`) — takes an HF model directory (with `config.json` + safetensors), builds a TRT engine via TensorRT's Python API, and packages it into a `.trtfb` bundle. Model family plugins in `trtf_build/trtf_build/families/` handle family-specific weight mapping and graph construction.

2. **C++ runtime** — loads a `.trtfb` bundle, deserializes the TRT engine plan, and runs autoregressive inference. The runtime is bundle-only: it does not load HF model directories directly. The `runtime_strategy` field in the bundle's config.json selects the backend:
   - `decoder_kv_cache` (default): standard attention with device-resident KV cache (`TrtBackendFastPath` + `DeviceKvCache`)
   - `decoder_moe`: MoE decoder (same device-resident KV cache, routing handled in TRT graph)
   - `ssm_recurrent`: Mamba/SSM with conv + SSM recurrent state (`MambaBackend`)
   - `vision_language`: VL pipeline with vision encoder + text decoder + image preprocessing. Qwen3-VL adds DeepStack: multi-level vision features injected at early text decoder layers.

Tokenizer implementations (`ITokenizer`):
- `VocabTokenizer` — vocab.txt-based lookup.
- `HfPythonTokenizer` — bridges to HuggingFace tokenizers via Python subprocess.

## Source layout

```
trtf_build/                          # Python package (engine builder)
  trtf_build/
    __init__.py
    __main__.py
    cli.py                           # CLI: trtf-build build|inspect|version
    config.py                        # ModelConfig from config.json
    graph_ops.py                     # Layer 1: Atomic TRT graph ops (tensor-in/tensor-out)
    graph_blocks.py                  # Layer 2: Composable blocks (attention, SwiGLU, GELU MLP, norm)
    standard_decoder_builder.py      # Layer 3: Standard decoder engine builder (uses graph_blocks)
    checkpoint_mapper.py             # HF safetensors -> weight dict
    bundle_writer.py                 # Write .trtfb files
    engine_builder.py                # Orchestrator: load -> build -> bundle
    debug_runner.py                  # TrtRunner + MambaTrtRunner + VLTrtRunner for pure-Python inference
    qwen_vl_vision_builder.py        # Vision encoder builders: Qwen2.5-VL (3D RoPE) + Qwen3-VL (DeepStack)
    families/
      __init__.py                    # Auto-discover plugins
      base.py                        # FamilyPlugin protocol
      qwen.py llama.py mistral.py gemma.py phi.py phi_moe.py
      granite.py internlm.py starcoder2.py gpt2.py opt.py
      falcon.py stablelm.py mamba.py qwen_vl.py olmo.py nemotron.py
      xglm.py gpt_neox.py gpt_neo.py codegen.py bloom.py mixtral.py
  pyproject.toml
src/                                 # C++ bundle-only runtime
  bundle/
    bundle_format.h/cpp              # Read .trtfb files
  cabi/
    trtf_c.cpp                       # C ABI: trtf_create_pipeline_ex()
    fast_path_config.h/cpp           # Parse config.json for bundle metadata
    bundle_helpers.h/cpp             # Shared plumbing: tokenizer extraction, engine init
  runtime/trt/
    trt_common.h/cpp                 # TRT logger, CUDA helpers (CudaBuffer/CudaStream with move semantics)
    trt_engine_lifecycle.h/cpp       # DecoderStepEngine, tensor validation
    trt_decode_runtime.h/cpp         # select_argmax_token, build_attention_mask
    device_kv_cache.h/cpp            # DeviceKvCache, DeviceResources, run_decoder_step_device
    trt_backend_shared.h/cpp         # TrtBackendFastPath autoregressive loop (device-resident cache)
    step_state.h                     # IStepState interface
    mamba_backend.h/cpp              # MambaBackend: SSM autoregressive loop
    mamba_decode_runtime.h/cpp       # MambaStepEngine, run_mamba_step
    mamba_step_state.h/cpp           # MambaStepState: conv + SSM recurrent state
    image_preprocessor.h/cpp         # VL image preprocessing: 4 strategies + configurable interpolation
  tokenizer/
    vocab_tokenizer.cpp              # Word-to-id lookup from vocabulary list
    hf_python_tokenizer.cpp          # HF tokenizers bridge via Python subprocess
  utils/
    data_dir.h/cpp                   # centralized source-dir resolution
    text_parsers.h/cpp               # shared string/file parsing (starts_with, read_file, etc.)
    json_helpers.h/cpp               # shared JSON extraction (extract_json_string, etc.)
tools/                               # Diff test framework (TRT vs HF comparison)
  diff_logits.py                     # E2E logit comparison (trtf vs HF transformers)
  diff_layers.py                     # Per-layer hidden state comparison
  diff_vl.py                         # VL diff testing (vision features, generation, C++ parity)
  test_runner_parity.py              # Python vs C++ runtime parity
  test_graph_ops.py                  # TRT graph operation testing
scripts/                             # Infrastructure & utility scripts
  setup_container.sh                 # One-shot container setup (venv, deps, build, test)
  new_family.py                      # Scaffold a new family plugin from HF repo
  validate_family.sh                 # One-command validation gate (build + diff + parity)
tests/
  cpp/                               # C++ runtime unit tests
    test_helpers.h                   # Shared helpers: temp dirs, safetensors writing
    test_bundle_format.cpp ...       # 12 test executables
  builder/                           # Python builder unit tests (from trtf_build/tests/)
    conftest.py test_config.py ...   # 11 test modules
  tools/                             # Diff framework self-tests
    test_diff_logits.py ...          # Mocked tests for diff tools
  e2e/                               # E2E tests with JSON manifest
    engines.json                     # Model manifest
    test_inference.py ...            # GPU-required E2E tests
```

## Adding a new model family

Adding a new model family is done in the Python build package. For standard decoder, MoE, and extended-decoder families, no C++ changes are needed. Families with fundamentally different state management (e.g., Mamba/SSM) require a new `IStepState` implementation and backend in C++.

### Quick path (scaffolding script)

```bash
# 1. Generate a plugin from a HuggingFace model
python3 scripts/new_family.py \
  --model-type phi3 \
  --hf-repo microsoft/Phi-3-mini-4k-instruct \
  --family-name phi

# 2. Review and customize the generated plugin
$EDITOR trtf_build/trtf_build/families/phi.py

# 3. Validate end-to-end
./scripts/validate_family.sh microsoft/Phi-3-mini-4k-instruct
```

### Manual path

1. **Create `trtf_build/trtf_build/families/<family>.py`** — implement the `FamilyPlugin` protocol (see `base.py`). This handles:
   - Matching HF `model_type` / `architectures`
   - Mapping HF safetensors weight keys to the engine builder's expected names
   - Any family-specific weight pre-processing (e.g., Gemma's +1.0 to RMSNorm gamma, embedding scaling)
   - Graph construction customization if the family diverges from the standard decoder pattern

2. **The plugin is auto-discovered** — `families/__init__.py` uses `pkgutil.iter_modules()` to scan for any `.py` file with a module-level `plugin` attribute. Zero edits to shared files needed.

3. **Validate** — run `./scripts/validate_family.sh <hf-repo-or-path>` which builds a bundle, runs diff_logits (battery), diff_layers, and runner parity tests.

### Example

See `trtf_build/trtf_build/families/qwen.py` for the Qwen3 plugin (standard decoder).
See `trtf_build/trtf_build/families/phi.py` for Phi-3 (fused QKV/gate_up weight splitting).
See `trtf_build/trtf_build/families/phi_moe.py` for Phi-MoE (MoE with SparseMixer routing, uses `graph_blocks.add_attention_block`).
See `trtf_build/trtf_build/families/qwen_vl.py` for Qwen VL (Qwen2.5-VL standard + Qwen3-VL DeepStack via `graph_blocks` composition).
See `trtf_build/trtf_build/families/mamba.py` for Mamba (SSM, custom graph + C++ backend).
See `trtf_build/trtf_build/families/base.py` for the plugin protocol.

### Diff-test framework

Pure-Python TRT-vs-HF comparison (no C++ binary needed). Requires `torch`.

```bash
# E2E logit comparison (per-step logits, text match)
python3 tools/diff_logits.py \
  --model Qwen/Qwen3-0.6B --atol 1e-3 --battery

# Per-layer hidden state comparison (embedding, all layers, logits)
python3 tools/diff_layers.py \
  --model Qwen/Qwen3-0.6B --atol 0.05

# For models requiring custom tokenizer code (e.g., Phi-3), add --trust-remote-code:
python3 tools/diff_logits.py \
  --model microsoft/Phi-3-mini-4k-instruct --atol 1e-3 --battery --trust-remote-code
```

The diff-test framework uses `trtf_build.debug_runner.TrtRunner` for pure-Python TRT inference with device-resident KV cache, matching the C++ `DeviceKvCache` behavior exactly. For Mamba/SSM models, `MambaTrtRunner` handles device-resident recurrent state. For VL models, `VLTrtRunner` combines vision + text decoders with image preprocessing. `diff_layers.py` builds a debug engine with per-layer hidden state outputs via `debug_layer_outputs=True`.

**VL diff testing** (`diff_vl.py`):
```bash
# Vision encoder feature comparison (TRT vs HF)
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --model Qwen/Qwen2.5-VL-3B-Instruct --atol 0.1

# Vision-only smoke test (no HF model needed)
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg --vision-only

# Debug with preprocessor override
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --vision-only --preprocessor-type simple_chw

# Full VL generation + C++ binary parity
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --binary ./build/trtf --hf-python .venv/bin/python
```

**Runner parity guarantee**: If you change the C++ mask/cache/position logic (`trt_decode_runtime.cpp`, `device_kv_cache.cpp`), you MUST also update `debug_runner.py` and verify with:
```bash
python3 tools/test_runner_parity.py \
  --bundle /tmp/qwen3.trtfb --binary ./build/trtf \
  --hf-python .venv/bin/python --max-new-tokens 20
```

## Container workflow (TRT GPU)

Prerequisites: Docker + NVIDIA Container Toolkit. The container is fully self-contained — no host TRT artifacts needed. TRT headers come from apt (`libnvinfer-headers-dev` baked into the Dockerfile), TRT libs come from pip (`tensorrt_cu12`).

### 1) Build and launch the dev container
```bash
./scripts/docker_build.sh
./scripts/docker_run.sh
```

### 2) One-shot setup (inside container)
```bash
./scripts/setup_container.sh
source .venv/bin/activate
```

This creates `.venv`, installs TRT + Python deps, builds the C++ runtime into `build/`, and runs tests.

### 3) Build bundle + validate Qwen3 TRT E2E (inside container)
```bash
source .venv/bin/activate

# Build a bundle (auto-downloads from HuggingFace)
trtf-build build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256

# Inspect it
trtf-build inspect /tmp/qwen3.trtfb

# Run from bundle using the C++ runtime
TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])")
export LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
./build/trtf run /tmp/qwen3.trtfb --prompt "Hello" --max-new-tokens 5 \
  --hf-python $PWD/.venv/bin/python
```

### 4) Build + run a vision-language model (inside container)
```bash
source .venv/bin/activate

# Build a VL bundle (text decoder + vision encoder)
trtf-build build Qwen/Qwen2.5-VL-3B-Instruct -o /tmp/qwen25vl.trtfb --max-cache-length 384

# Run with an image
TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])")
export LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
./build/trtf run /tmp/qwen25vl.trtfb --prompt "Describe this image." \
  --image /path/to/image.jpg --max-new-tokens 30 \
  --hf-python $PWD/.venv/bin/python
```

### 5) Build + run Qwen3-VL with DeepStack (inside container)
```bash
source .venv/bin/activate

# Build a Qwen3-VL bundle (text decoder with DeepStack + vision encoder with multi-level outputs)
trtf-build build Qwen/Qwen3-VL-2B-Instruct -o /tmp/qwen3vl.trtfb --max-cache-length 256

# Text-only inference (DeepStack inactive during text-only)
TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])")
export LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
./build/trtf run /tmp/qwen3vl.trtfb --prompt "The capital of France is" \
  --max-new-tokens 20 --hf-python $PWD/.venv/bin/python

# VL inference with image (DeepStack active during image token prefill)
./build/trtf run /tmp/qwen3vl.trtfb --prompt "Describe this image." \
  --image /path/to/image.jpg --max-new-tokens 30 \
  --hf-python $PWD/.venv/bin/python
```

### 6) MMLU sanity check (inside container)
```bash
$PWD/.venv/bin/python scripts/eval_mmlu.py \
  --backend trtf --model /tmp/qwen3.trtfb \
  --trtf-binary ./build/trtf \
  --subject all --split test \
  --num-samples 4 --min-accuracy 0.0
```
