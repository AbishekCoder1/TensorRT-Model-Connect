# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

C++ library mirroring HuggingFace `transformers.pipeline(...)` with TensorRT-first execution. Public API lives in `include/trtf/`. Everything is in the `trtf` namespace. C++17, compiled with `-Wall -Wextra -Wpedantic`.

## Build commands

ALWAYS DO EVERYTHING IN CONTAINER. Run command with docker exec

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
./build/trtf_text_generation [--force-trt|--cpu-only] [model_id] [prompt]
./build/trtf_load_model [--force-trt|--cpu-only] [model_id] [prompt]
```

## Key environment variables

- `TRTF_MAX_NEW_TOKENS` - override max generation tokens
- `TRTF_MAX_CACHE_LENGTH` - cap/override runtime cache length for model definitions
- `TRTF_HF_PYTHON` - path to Python interpreter for HF tokenizer bridge / hf-transformers backend
- `TRTF_TRT_ENGINE_CACHE_DIR` - on-disk cache for serialized TRT engine plans
- `TRTF_DISABLE_ENGINE_CACHE=1` - disable plan cache (force rebuild each process)
- `TRTF_DEBUG_LOGITS_TOPK=N` - print top-N logit debug info per decode step
- `TRTF_TRT_LOG_STDERR=1` / `TRTF_TRT_LOG_MIN_SEVERITY` - TRT logger controls

## Architecture

The pipeline flow has three stages, each with an extension point:

1. **Model resolution** (`ResolveTextGenerationModel`) — turns a `model_id` string into a `ResolvedModelSpec` (decoder-definition, HF-local, or custom). Extension: `RegisterTextGenerationModelResolver(...)`.

2. **HF family registry** (`ResolveHfModelViaFamilyRegistry`) — when resolution sees an HF directory with `config.json` + safetensors, it checks registered `HfModelFamilyRegistration` entries (matched by `model_type`/`architectures`, ordered by priority). Extension: `RegisterHfModelFamily(...)`. This is the preferred way to add new model families.

3. **Runtime assembly** (`BuildRuntimeForTextGeneration`) — creates tokenizer + backend from the resolved model spec, respecting `BackendSelection` (prefer_trt/force_trt). Extension: `RegisterTextGenerationRuntimeAssembler(...)`.

Three backends implement `IGenerationBackend`:
- **trt** — TensorRT graph builder. Dispatches to registered `ITrtGraphBuilder` per family. Builds network via TensorRT C++ API (not ONNX).
- **cpu-reference** — deterministic CPU fallback using transition tables.
- **hf-transformers** — delegates to HuggingFace Python via subprocess.

Tokenizer implementations (`ITokenizer`):
- `ToyTokenizer` — vocab.txt-based lookup for built-in/decoder-definition models.
- `HfPythonTokenizer` — bridges to HuggingFace tokenizers via Python subprocess.

## Source layout

```
src/
  utils/
    text_parsers.h/cpp           # shared string/file parsing (starts_with, read_file, etc.)
    json_helpers.h/cpp           # shared JSON extraction (extract_json_string, etc.)
    tensor_math.h/cpp            # transpose_2d, expand_kv, repeat_head_norm
    trt/engine_cache.h/cpp       # TRT engine plan on-disk cache
  model/
    model_loader.cpp             # generic DecoderModel loading + Qwen checkpoint mapping
    safetensors_loader.h/cpp     # SafetensorReader + TensorSource (single/sharded)
    model_resolver.cpp           # multi-stage model resolution pipeline
    hf_family_registry.cpp       # family registry + builtin registration dispatch
    trt_model_definition.h/cpp   # DecoderModel -> TrtDecoderDefinition conversion
    qwen3_trt_model_definition.h/cpp  # Qwen3 layer stack -> TRT definition
    qwen3_decoder_model_loader.h/cpp  # Qwen-specific LoadDecoderModel wrapper
  models/
    qwen/
      registration.h/cpp         # RegisterQwenFamily() — HF matcher + TRT builder
  runtime/
    trt_backend.cpp              # CreateTrtBackend — family dispatch (registration-based)
    trt_backend_qwen.cpp         # Qwen graph builders (legacy + multi-layer)
    trt_backend_qwen_impl.h      # CreateTrtQwenBackend declaration
    trt/
      trt_common.h/cpp           # TrtLogger, TrtDeleter, CudaStream, CudaBuffer
      trt_graph_ops.h/cpp        # Reusable TRT ops: add_rms_norm, add_rope, matmul, etc.
      trt_engine_lifecycle.h/cpp # DecoderStepEngine, finalize_decoder_step_engine
      trt_decode_runtime.h/cpp   # run_decoder_step, cache management, sampling
      trt_backend_shared.h/cpp   # Generic TrtBackend autoregressive generate loop
      trt_graph_builder.h/cpp    # ITrtGraphBuilder interface + registry
    cpu_reference_backend.cpp
    hf_python_backend.cpp
    runtime_factory.cpp
  pipeline/pipeline.cpp
  tokenizer/*.cpp
```

## Adding a new model family

Preferred path — create files only in `src/models/<family>/`, add to CMakeLists.txt, zero edits to shared code:

1. Create `src/models/<family>/registration.cpp` that calls `RegisterHfModelFamily(...)` with a `matcher` and `model_definition_loader`.
2. The loader returns a `DecoderModel` from `HfModelMetadata`.
3. Optionally register a TRT graph builder via `RegisterTrtGraphBuilder(family, builder)`.
4. Call your `Register<Family>Family()` from `RegisterBuiltinHfModelFamilies()` in `hf_family_registry.cpp`.
5. See `src/models/qwen/registration.cpp` for the Qwen3 built-in registration and `tests/test_hf_family_registry.cpp` for the pattern.

Shared TRT graph ops (RMSNorm, RoPE, attention, SwiGLU) are in `src/runtime/trt/trt_graph_ops.h` — reusable by any family.

## Container workflow (TRT GPU)

Prerequisites: Docker + NVIDIA Container Toolkit, TensorRT host artifacts at a known path (e.g. `/home/yifeif/repos/trt/build/tensorrt-base-dev/rel-10.15-native-x86_64-ubuntu20.04-cuda11.8-auto/cmake`).

### 1) Build and launch the TRT dev container
```bash
./scripts/docker_build.sh
./scripts/docker_run.sh
```
Or directly:
```bash
docker run --rm -it --gpus all \
  -e TRT_ROOT=/opt/trt \
  -e LD_LIBRARY_PATH=/opt/trt/Debug/lib:/opt/trt/Release/lib:/opt/trt/lib:/opt/trt/myelin-ext/myelin/lib/Debug:/usr/local/cuda/lib64 \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -v <trt-artifacts-path>:/opt/trt:ro \
  -w /workspace/trt-transformers-cpp \
  trtf-dev bash
```

### 2) Prepare Python env (inside container)
```bash
python3 -m venv .venv-hf
source .venv-hf/bin/activate
pip install -U pip
pip install "transformers>=4.57.0" tokenizers safetensors sentencepiece huggingface_hub datasets
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

### 4) Configure and build (inside container)
```bash
cmake -S . -B build-container-phase1 -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=/opt/trt/include/zapped_headers \
  -DTRTF_TRT_LIBRARY=/opt/trt/Debug/lib/libnvinfer.so \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build-container-phase1 -j
ctest --test-dir build-container-phase1 --output-on-failure
```
Do not reuse host-generated build dirs inside the container. Always use a container-specific build dir.

### 5) Validate Qwen3 TRT E2E (inside container)
```bash
TRTF_HF_PYTHON=$PWD/.venv-hf/bin/python \
TRTF_MAX_CACHE_LENGTH=1 \
TRTF_MAX_NEW_TOKENS=1 \
./build-container-phase1/trtf_load_model --force-trt QWEN3 "Hello"
```
Expected: `backend=trt`, output starts with `Hello Answer`.

Recommended all-in-one diagnostic script:
```bash
./scripts/test_qwen3_trt_e2e.sh "Hello"
```
Logs to `/tmp/trtf_qwen3_trt_e2e.log`. Override build dir with `TRTF_BUILD_DIR=<dir>`.

### 6) MMLU sanity check (inside container)
```bash
TRTF_HF_PYTHON=$PWD/.venv-hf/bin/python \
TRTF_MAX_NEW_TOKENS=8 \
$PWD/.venv-hf/bin/python scripts/eval_mmlu.py \
  --backend trtf --model QWEN3 \
  --trtf-binary ./build-container-phase1/trtf_load_model \
  --force-trt --subject all --split test \
  --num-samples 4 --min-accuracy 0.0
```

### 7) Optional HF-transformers parity check (inside container)
```bash
source .venv-hf/bin/activate
python3 scripts/compare_hf_pipeline_vs_transformers.py \
  --model-dir models/hf/hf-internal-testing__tiny-random-gpt2 \
  --binary ./build-container-phase1/trtf_text_generation \
  --prompt "Hello from trtf" --max-new-tokens 20
```

## Built-in model IDs

- `trtf/tiny-cake-v1` — bundled tiny decoder model (always available)
- `QWEN3` — prefers real Qwen3 at `models/hf/Qwen__Qwen3-0.6B`, falls back to bundled demo at `models/hf/qwen3`
