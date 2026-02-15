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
pip install -e trtf_build/
trtf-build build <model-dir> -o <output.trtfb> [--max-cache-length N] [--verbose]
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

2. **C++ runtime** — loads a `.trtfb` bundle, deserializes the TRT engine plan, and runs autoregressive inference. The runtime is bundle-only: it does not load HF model directories directly.

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
    families/
      __init__.py                    # Auto-discover plugins
      base.py                        # FamilyPlugin protocol
      qwen.py llama.py mistral.py gemma.py
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
  tokenizer/
    vocab_tokenizer.cpp              # Word-to-id lookup from vocabulary list
    hf_python_tokenizer.cpp          # HF tokenizers bridge via Python subprocess
  utils/
    data_dir.h/cpp                   # centralized source-dir resolution
    text_parsers.h/cpp               # shared string/file parsing (starts_with, read_file, etc.)
    json_helpers.h/cpp               # shared JSON extraction (extract_json_string, etc.)
scripts/
  diff_logits.py                     # E2E logit comparison (trtf vs HF transformers)
  diff_layers.py                     # Per-layer hidden state comparison
tests/
  test_helpers.h                     # Shared helpers: temp dirs, safetensors writing
```

## Adding a new model family

Adding a new model family is done in the Python build package. Create a new plugin file in `trtf_build/trtf_build/families/`:

1. **Create `trtf_build/trtf_build/families/<family>.py`** — implement the `FamilyPlugin` protocol (see `base.py`). This handles:
   - Matching HF `model_type` / `architectures`
   - Mapping HF safetensors weight keys to the engine builder's expected names
   - Any family-specific weight pre-processing (e.g., Gemma's +1.0 to RMSNorm gamma, embedding scaling)
   - Graph construction customization if the family diverges from the standard decoder pattern

2. **The plugin is auto-discovered** — `trtf_build/trtf_build/families/__init__.py` auto-discovers all plugin files in the directory.

3. **No C++ changes needed** — the C++ runtime is family-agnostic; it only loads pre-built `.trtfb` bundles.

### Example

See `trtf_build/trtf_build/families/qwen.py` for the Qwen3 plugin.
See `trtf_build/trtf_build/families/base.py` for the plugin protocol.

### Diff-test framework

After implementing a new family, validate with:
```bash
# E2E logit comparison
python3 scripts/diff_logits.py \
  --model-dir <hf-model-dir> --binary ./build/trtf \
  --atol 1e-3 --battery

# Per-layer hidden state comparison
python3 scripts/diff_layers.py --model-dir <hf-model-dir>
```

## Container workflow (TRT GPU)

Prerequisites: Docker + NVIDIA Container Toolkit. The container is fully self-contained — no host TRT artifacts needed.

### 1) Build and launch the dev container
```bash
./scripts/docker_build.sh
./scripts/docker_run.sh
```
Or directly:
```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -w /workspace/trt-transformers-cpp \
  trtf-dev bash
```

### 2) Install TRT + Python deps (inside container)
```bash
# TRT headers (for C++ compilation) + CUDA 12 runtime libs (from pip)
apt-get update && apt-get install -y --no-install-recommends libnvinfer-headers-dev

python3 -m venv .venv-hf
source .venv-hf/bin/activate
pip install -U pip
pip install tensorrt_cu12 && pip install tensorrt --no-deps  # CUDA 12 TRT Python
pip install "transformers>=4.57.0" tokenizers safetensors sentencepiece huggingface_hub datasets
pip install --no-deps -e trtf_build/
# Optional for HF-reference parity scripts:
pip install torch accelerate
```

### 3) Download real Qwen3 weights (inside container)
```bash
source .venv-hf/bin/activate
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-0.6B',
    local_dir='models/hf/Qwen__Qwen3-0.6B',
    local_dir_use_symlinks=False,
    allow_patterns=[
        'config.json', 'generation_config.json', 'model.safetensors*',
        'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
        'merges.txt', 'special_tokens_map.json', '*.model',
        'README.md', 'LICENSE', '.gitattributes',
    ],
)
PY
```

### 4) Configure and build C++ runtime (inside container)

The C++ runtime uses apt TRT headers + pip TRT cu12 libs (fully self-contained):
```bash
# Find pip-installed TRT libs
TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])")
# Create libnvinfer.so symlink if needed
[ ! -f "$TRT_LIB_DIR/libnvinfer.so" ] && ln -sf libnvinfer.so.10 "$TRT_LIB_DIR/libnvinfer.so"

cmake -S . -B build-container -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DTRTF_TRT_LIBRARY=$TRT_LIB_DIR/libnvinfer.so \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build-container -j

# Run tests (set LD_LIBRARY_PATH for pip TRT libs)
export LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
ctest --test-dir build-container --output-on-failure
```
Do not reuse host-generated build dirs inside the container. Always use a container-specific build dir.

### 5) Build bundle + validate Qwen3 TRT E2E (inside container)
```bash
source .venv-hf/bin/activate

# Build a bundle using the Python builder
trtf-build build models/hf/Qwen__Qwen3-0.6B \
  -o /tmp/qwen3.trtfb --max-cache-length 256

# Inspect it
trtf-build inspect /tmp/qwen3.trtfb

# Run from bundle using the C++ runtime
export LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
./build-container/trtf run /tmp/qwen3.trtfb --prompt "Hello" --max-new-tokens 5 \
  --hf-python $PWD/.venv-hf/bin/python
```

### 6) MMLU sanity check (inside container)
```bash
$PWD/.venv-hf/bin/python scripts/eval_mmlu.py \
  --backend trtf --model /tmp/qwen3.trtfb \
  --trtf-binary ./build-container/trtf \
  --subject all --split test \
  --num-samples 4 --min-accuracy 0.0
```
