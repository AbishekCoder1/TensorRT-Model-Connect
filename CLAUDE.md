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
./build/trtf build   <model-dir> -o <output.trtfb> [--max-cache-length N] [--hf-python PATH]
./build/trtf run     <model-or-bundle> --prompt "text" [--max-new-tokens N] [--force-trt]
                     [--hf-python PATH] [--engine-cache-dir DIR] [--no-engine-cache]
./build/trtf inspect <bundle.trtfb>
./build/trtf version
```

## Key environment variables

- `TRTF_DATA_DIR` - source directory override (internal/dev only)
- `TRTF_TRT_LOG_STDERR=1` / `TRTF_TRT_LOG_MIN_SEVERITY` - TRT logger controls

All other configuration is done via CLI flags or `TrtfPipelineOptions`:
- `--hf-python PATH` (replaces `TRTF_HF_PYTHON`)
- `--engine-cache-dir DIR` (replaces `TRTF_TRT_ENGINE_CACHE_DIR`)
- `--no-engine-cache` (replaces `TRTF_DISABLE_ENGINE_CACHE`)
- `--max-cache-length N` (replaces `TRTF_MAX_CACHE_LENGTH`)
- `--max-new-tokens N` (replaces `TRTF_MAX_NEW_TOKENS`)

## Architecture

The pipeline flow has three stages:

1. **Model resolution** (`ResolveTextGenerationModel`) — turns a `model_id` string into a `ResolvedModelSpec` (decoder-definition or HF-local).

2. **HF family registry** (`ResolveHfModelViaFamilyRegistry`) — when resolution sees an HF directory with `config.json` + safetensors, it checks registered `HfModelFamilyRegistration` entries (matched by `model_type`/`architectures`, ordered by priority). Extension: `RegisterHfModelFamily(...)`. This is the preferred way to add new model families.

3. **Runtime assembly** (`BuildRuntimeForTextGeneration`) — creates tokenizer + backend from the resolved model spec, respecting `BackendSelection` (force_trt).

Two backends implement `IGenerationBackend`:
- **trt** — TensorRT model runtime. Dispatches to registered `IModelRuntime` per family. Each family owns graph construction, state creation, and per-step execution. Builds network via TensorRT C++ API (not ONNX).
- **hf-transformers** — delegates to HuggingFace Python via subprocess.

Tokenizer implementations (`ITokenizer`):
- `VocabTokenizer` — vocab.txt-based lookup.
- `HfPythonTokenizer` — bridges to HuggingFace tokenizers via Python subprocess.

## Source layout

```
src/
  utils/
    data_dir.h/cpp               # centralized source-dir resolution
    text_parsers.h/cpp           # shared string/file parsing (starts_with, read_file, etc.)
    json_helpers.h/cpp           # shared JSON extraction (extract_json_string, etc.)
    tensor_math.h/cpp            # transpose_2d, expand_kv, repeat_head_norm
    trt/engine_cache.h/cpp       # TRT engine plan on-disk cache
  model/
    model_loader.cpp             # generic DecoderModel loading (delegates to checkpoint mapper registry)
    safetensors_loader.h/cpp     # SafetensorReader + TensorSource (single/sharded)
    model_resolver.cpp           # model resolution pipeline
    hf_family_registry.cpp       # family registry + builtin registration dispatch
    checkpoint_mapper.h/cpp      # ICheckpointMapper interface + registry (Registry 2)
    standard_checkpoint_mapper.h/cpp  # Base class for standard HF tensor naming
    standard_decoder_graph_builder.h/cpp  # Pre-RMSNorm+GQA+RoPE+SwiGLU decoder pattern (shared build-time infrastructure)
    trt_model_definition.h/cpp   # DecoderModel -> TrtDecoderDefinition conversion (inlined, no registry)
  models/
    qwen/
      registration.h/cpp         # RegisterQwenFamily() — registers into all 3 registries
      checkpoint_mapper.h/cpp    # QwenCheckpointMapper — HF Qwen tensor keys -> DecoderCheckpoint
    llama/
      registration.h/cpp         # RegisterLlamaFamily()
      checkpoint_mapper.h/cpp    # LlamaCheckpointMapper
    mistral/
      registration.h/cpp         # RegisterMistralFamily()
      checkpoint_mapper.h/cpp    # MistralCheckpointMapper
    gemma/
      registration.h/cpp         # RegisterGemmaFamily()
      checkpoint_mapper.h/cpp    # GemmaCheckpointMapper (adds +1.0 to RMSNorm gamma, scales embedding)
  runtime/
    trt_backend.cpp              # CreateTrtBackend — dispatches via FindModelRuntime() registry
    trt/
      trt_common.h/cpp           # TrtLogger, TrtDeleter, CudaStream, CudaBuffer
      trt_graph_ops.h/cpp        # Reusable TRT ops: add_rms_norm, add_rope, matmul, etc.
      trt_engine_lifecycle.h/cpp # DecoderStepEngine, finalize_decoder_step_engine
      trt_decode_runtime.h/cpp   # run_decoder_step, cache management, sampling
      trt_backend_shared.h/cpp   # TrtBackend + TrtBackendFastPath autoregressive generate loop
      model_runtime.h/cpp        # IModelRuntime interface + registry + factory helpers
      model_runtime_fwd.h        # Lightweight header for family registrations (no TRT/CUDA includes)
      trt_graph_builder.h        # ITrtGraphBuilder interface (header-only, no registry)
      step_state.h               # IStepState interface for per-step state management
      kv_cache_step_state.h/cpp  # KvCacheStepState for attention-based decoders
    hf_python_backend.cpp
    runtime_factory.cpp
  cabi/
    trtf_c.cpp                   # C ABI factory: trtf_create_pipeline(), trtf_create_pipeline_ex()
    fast_path_config.h/cpp       # FastPathModelConfig for zero-weight fast path
  bundle/
    bundle_format.h/cpp          # .trtfb binary format
    bundle_api.cpp               # BuildBundle(), InspectBundle()
  tokenizer/
    vocab_tokenizer.cpp          # Word-to-id lookup from vocabulary list
    hf_python_tokenizer.cpp      # HF tokenizers bridge via Python subprocess
scripts/
  diff_logits.py                 # E2E logit comparison (trtf vs HF transformers)
  diff_layers.py                 # Per-layer hidden state comparison
  generate_op_gold_tensors.py    # Generate gold .safetensors fixtures for TRT ops
  templates/model_family/        # Skeleton for adding a new model family
cmake/
  family_dispatch.cpp.in         # Template for auto-generated RegisterBuiltinHfModelFamilies()
tests/
  gold/                          # Committed gold tensor files for per-op tests
  test_helpers.h                 # Shared helpers: temp dirs, safetensors writing, write_standard_decoder_checkpoint()
  test_trt_graph_ops_gold.cpp    # Per-op gold tensor tests (GPU-only)
```

## Adding a new model family

The plug-and-play architecture uses 3 registries. Adding a new family requires:
1. Files only in `src/models/<family>/` + `tests/`
2. Re-run cmake (GLOB auto-discovers new files — **zero edits to any shared file**)

### The 3 Registries

| # | Registry | Interface | Purpose |
|---|----------|-----------|---------|
| 1 | HF Family | `RegisterHfModelFamily(...)` | Match HF model_type → load DecoderModel |
| 2 | Checkpoint Mapper | `RegisterCheckpointMapper(...)` | Map HF safetensors keys → DecoderCheckpoint |
| 3 | Model Runtime | `RegisterModelRuntime(...)` | Build engine, create state, run per-step execution |

### Steps

1. **Create `src/models/<family>/registration.h/cpp`** — call `Register<Family>Family()` which registers into all 3 registries.
2. **Create `src/models/<family>/checkpoint_mapper.h/cpp`** — implement `ICheckpointMapper` to map HF tensor keys to `DecoderCheckpoint`. Consider subclassing `StandardCheckpointMapper` (`src/model/standard_checkpoint_mapper.h`) if your model uses the standard HF naming convention.
3. **Choose or create a model runtime**:
   - `CreateStandardDecoderRuntime()` works for LLaMA, Qwen, Mistral-dense, Gemma (Pre-RMSNorm + GQA + RoPE + SwiGLU).
   - `CreateKvCacheRuntime(engine_factory)` for non-standard graph architectures (MoE) that still use KV-cache attention.
   - Implement `IModelRuntime` directly for fundamentally different architectures (Mamba/SSM).
4. **Create `tests/test_<family>_family.cpp`** — auto-discovered by CMake GLOB.
5. **Re-run cmake** — `cmake -S . -B build -G Ninja` picks up new sources and generates dispatch.

### Example: Dense decoder (LLaMA-like)

```
src/models/llama/
  registration.h        # void RegisterLlamaFamily();
  registration.cpp      # Registers into all 3 registries
  checkpoint_mapper.h   # LlamaCheckpointMapper : ICheckpointMapper
  checkpoint_mapper.cpp # HF LLaMA tensor keys → DecoderCheckpoint
```

See `scripts/templates/model_family/registration.cpp` for the minimal skeleton.
See `src/models/qwen/registration.cpp` for the full Qwen3 built-in registration.

### Diff-test framework

After implementing a new family, validate with:
```bash
# E2E logit comparison
python3 scripts/diff_logits.py \
  --model-dir <hf-model-dir> --binary ./build/trtf \
  --backend-flag --force-trt --atol 1e-3 --battery

# Per-layer hidden state comparison
python3 scripts/diff_layers.py --model-dir <hf-model-dir>

# Per-op gold tensor tests (generate + run, GPU only)
python3 scripts/generate_op_gold_tensors.py
ctest --test-dir build -R test_trt_ops_gold --output-on-failure
```

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
./build-container-phase1/trtf run QWEN3 --prompt "Hello" --force-trt \
  --hf-python $PWD/.venv-hf/bin/python --max-cache-length 1 --max-new-tokens 1
```
Expected: `backend=trt`, output starts with `Hello Answer`.

Recommended all-in-one diagnostic script:
```bash
./scripts/test_qwen3_trt_e2e.sh "Hello"
```
Logs to `/tmp/trtf_qwen3_trt_e2e.log`. Override build dir with `TRTF_BUILD_DIR=<dir>`.

### 6) Build + run from bundle (inside container)
```bash
# Build a bundle
./build-container-phase1/trtf build models/hf/Qwen__Qwen3-0.6B \
  -o /tmp/qwen3.trtfb --max-cache-length 256 --hf-python $PWD/.venv-hf/bin/python

# Inspect it
./build-container-phase1/trtf inspect /tmp/qwen3.trtfb

# Run from bundle
./build-container-phase1/trtf run /tmp/qwen3.trtfb --prompt "Hello" --max-new-tokens 5 \
  --hf-python $PWD/.venv-hf/bin/python
```

### 7) MMLU sanity check (inside container)
```bash
$PWD/.venv-hf/bin/python scripts/eval_mmlu.py \
  --backend trtf --model QWEN3 \
  --trtf-binary ./build-container-phase1/trtf \
  --force-trt --subject all --split test \
  --num-samples 4 --min-accuracy 0.0
```

## Built-in model IDs

- `QWEN3` — prefers real Qwen3 at `models/hf/Qwen__Qwen3-0.6B`, falls back to bundled demo at `models/hf/qwen3`
