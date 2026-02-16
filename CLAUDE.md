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

All tests:
```bash
ctest --test-dir build --output-on-failure
```

Single test:
```bash
ctest --test-dir build -R test_pipeline --output-on-failure
# or directly:
./build/test_pipeline
```

Tests are plain C++ executables (no framework). They use `main()`, print to stderr on failure, and return 0 on success / non-zero on failure. Test names in CMake match their source files in `tests/`.

## Running executables

```bash
./build/trtf run     <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--hf-python PATH]
./build/trtf inspect <bundle.trtfb>
./build/trtf version
```

## Key environment variables

- `TRTF_TRT_LOG_STDERR=1` / `TRTF_TRT_LOG_MIN_SEVERITY` - TRT logger controls

## Architecture

The system is split into two stages:

1. **Python build** (`trtf_build/`) — takes an HF model directory (with `config.json` + safetensors), builds a TRT engine via TensorRT's Python API, and packages it into a `.trtfb` bundle. Model family plugins in `trtf_build/trtf_build/families/` handle family-specific weight mapping and graph construction.

2. **C++ runtime** — loads a `.trtfb` bundle, deserializes the TRT engine plan, and runs autoregressive inference. The runtime is bundle-only: it does not load HF model directories directly. The `runtime_strategy` field in the bundle's config.json selects the backend:
   - `decoder_kv_cache` (default): standard attention with KV cache (`TrtBackendFastPath`)
   - `decoder_moe`: MoE decoder (same KV cache, routing handled in TRT graph)
   - `ssm_recurrent`: Mamba/SSM with conv + SSM recurrent state (`MambaBackend`)

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
    graph_ops.py                     # TRT graph ops (RMSNorm, RoPE, etc.)
    standard_decoder_builder.py      # Standard decoder engine builder
    checkpoint_mapper.py             # HF safetensors -> weight dict
    bundle_writer.py                 # Write .trtfb files
    engine_builder.py                # Orchestrator: load -> build -> bundle
    debug_runner.py                  # TrtRunner + MambaTrtRunner for pure-Python inference
    families/
      __init__.py                    # Auto-discover plugins
      base.py                        # FamilyPlugin protocol
      qwen.py llama.py mistral.py gemma.py phi.py phi_moe.py
      granite.py internlm.py starcoder2.py gpt2.py opt.py
      falcon.py stablelm.py mamba.py qwen_vl.py
  pyproject.toml
src/                                 # C++ bundle-only runtime
  bundle/
    bundle_format.h/cpp              # Read .trtfb files
  cabi/
    trtf_c.cpp                       # C ABI: trtf_create_pipeline_ex()
    fast_path_config.h/cpp           # Parse config.json for bundle metadata
  runtime/trt/
    trt_common.h/cpp                 # TRT logger, CUDA helpers
    trt_engine_lifecycle.h/cpp       # DecoderStepEngine, tensor validation
    trt_decode_runtime.h/cpp         # run_decoder_step, sampling
    trt_backend_shared.h/cpp         # TrtBackendFastPath autoregressive loop
    kv_cache_step_state.h/cpp        # KV cache state management
    step_state.h                     # IStepState interface
    mamba_backend.h/cpp              # MambaBackend: SSM autoregressive loop
    mamba_decode_runtime.h/cpp       # MambaStepEngine, run_mamba_step
    mamba_step_state.h/cpp           # MambaStepState: conv + SSM recurrent state
  tokenizer/
    vocab_tokenizer.cpp              # Word-to-id lookup from vocabulary list
    hf_python_tokenizer.cpp          # HF tokenizers bridge via Python subprocess
  utils/
    data_dir.h/cpp                   # centralized source-dir resolution
    text_parsers.h/cpp               # shared string/file parsing (starts_with, read_file, etc.)
    json_helpers.h/cpp               # shared JSON extraction (extract_json_string, etc.)
scripts/
  setup_container.sh                 # One-shot container setup (venv, deps, build, test)
  diff_logits.py                     # E2E logit comparison (trtf vs HF transformers)
  diff_layers.py                     # Per-layer hidden state comparison
  new_family.py                      # Scaffold a new family plugin from HF repo
  validate_family.sh                 # One-command validation gate (build + diff + parity)
tests/
  test_helpers.h                     # Shared helpers: temp dirs, safetensors writing
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
See `trtf_build/trtf_build/families/phi_moe.py` for Phi-MoE (MoE with SparseMixer routing).
See `trtf_build/trtf_build/families/mamba.py` for Mamba (SSM, custom graph + C++ backend).
See `trtf_build/trtf_build/families/base.py` for the plugin protocol.

### Diff-test framework

Pure-Python TRT-vs-HF comparison (no C++ binary needed). Requires `torch`.

```bash
# E2E logit comparison (per-step logits, text match)
python3 scripts/diff_logits.py \
  --model Qwen/Qwen3-0.6B --atol 1e-3 --battery

# Per-layer hidden state comparison (embedding, all layers, logits)
python3 scripts/diff_layers.py \
  --model Qwen/Qwen3-0.6B --atol 0.05

# For models requiring custom tokenizer code (e.g., Phi-3), add --trust-remote-code:
python3 scripts/diff_logits.py \
  --model microsoft/Phi-3-mini-4k-instruct --atol 1e-3 --battery --trust-remote-code
```

The diff-test framework uses `trtf_build.debug_runner.TrtRunner` for pure-Python TRT inference with KV cache management, matching the C++ runtime behavior exactly. For Mamba/SSM models, `MambaTrtRunner` handles recurrent state instead of KV cache. `diff_layers.py` builds a debug engine with per-layer hidden state outputs via `debug_layer_outputs=True`.

**Runner parity guarantee**: If you change the C++ mask/cache/position logic (`trt_decode_runtime.cpp`, `kv_cache_step_state.cpp`), you MUST also update `debug_runner.py` and verify with:
```bash
python3 scripts/test_runner_parity.py \
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

### 4) MMLU sanity check (inside container)
```bash
$PWD/.venv/bin/python scripts/eval_mmlu.py \
  --backend trtf --model /tmp/qwen3.trtfb \
  --trtf-binary ./build/trtf \
  --subject all --split test \
  --num-samples 4 --min-accuracy 0.0
```
