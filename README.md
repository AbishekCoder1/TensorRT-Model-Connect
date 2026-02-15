# trt-transformers-cpp

A C++ library that mirrors HuggingFace `transformers.pipeline(...)` with TensorRT-first execution. Reads HuggingFace model checkpoints directly, builds optimized TensorRT engines from the C++ API (no ONNX), and runs GPU-accelerated inference.

## Current status

- **Library API**: C ABI entry point (`trtf_create_pipeline()`) returns `IPipeline*` virtual interface. ABI-safe across compilers.
- **CLI**: `trtf build/run/inspect/version` subcommands.
- **Bundle format**: `.trtfb` files package compiled TRT engines + tokenizer into a single artifact.
- **CMake install**: `find_package(trtf)` support, installs library + headers + CLI.
- Two backends: `trt` (TensorRT GPU), `hf-transformers` (Python subprocess fallback).
- 3-registry plug-and-play architecture. 4 built-in model families (Qwen, LLaMA, Mistral, Gemma).
- `StandardDecoderGraphBuilder` handles Pre-RMSNorm + GQA + RoPE + SwiGLU (LLaMA, Qwen, Mistral, Gemma).
- Verified TRT output: Qwen3-0.6B, TinyLlama-1.1B, TinyMistral-248M, DeepSeek-R1-Distill-1.5B.

## Quick start

Prerequisites: Docker + NVIDIA Container Toolkit + TensorRT host artifacts.

### 1. Launch the dev container

```bash
./scripts/docker_build.sh
./scripts/docker_run.sh       # starts interactive container named trtf-dev
```

### 2. Build and test (inside container)

All remaining commands run inside the container.

```bash
# Clean any stale host-side CMake cache (safe to skip on first run)
rm -rf build

# Configure and build
cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=/opt/trt/include/zapped_headers \
  -DTRTF_TRT_LIBRARY=/opt/trt/Debug/lib/libnvinfer.so \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j

# Run all tests
ctest --test-dir build --output-on-failure
```

### 3. Set up Python env + download model weights (one-time)

```bash
python3 -m venv .venv-hf
source .venv-hf/bin/activate
pip install -U pip
pip install "transformers>=4.57.0" tokenizers safetensors sentencepiece huggingface_hub

python3 -c "
from huggingface_hub import snapshot_download

allow = ['config.json', 'generation_config.json', 'model.safetensors', 'model-*.safetensors',
         'model.safetensors.index.json',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
         'merges.txt', 'special_tokens_map.json', '*.model']

for repo, local in [
    ('Qwen/Qwen3-0.6B',                           'models/hf/Qwen__Qwen3-0.6B'),
    ('TinyLlama/TinyLlama-1.1B-Chat-v1.0',        'models/hf/TinyLlama__TinyLlama-1.1B-Chat-v1.0'),
    ('Felladrin/TinyMistral-248M-Chat-v2',         'models/hf/Felladrin__TinyMistral-248M-Chat-v2'),
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'models/hf/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B'),
]:
    print(f'Downloading {repo}...')
    snapshot_download(repo_id=repo, local_dir=local,
                      local_dir_use_symlinks=False, allow_patterns=allow)
"
```

### 4. Use the C API: write, compile, and run

```bash
cat > /tmp/hello_trtf.cpp <<'EOF'
#include <trtf/pipeline.h>
#include <iostream>

int main() {
    auto* p = trtf_create_pipeline("QWEN3", TRTF_FORCE_TRT);
    if (!p) { std::cerr << "Error: " << trtf_last_error() << std::endl; return 1; }
    std::cout << "backend: " << p->backend_name() << std::endl;
    std::cout << p->generate("Hello", 1) << std::endl;
    delete p;
}
EOF

c++ -std=c++17 /tmp/hello_trtf.cpp \
  -I include -Lbuild -ltrtf_core \
  -L/opt/trt/Debug/lib -lnvinfer \
  -L/usr/local/cuda/lib64 -lcudart \
  -lstdc++fs -o /tmp/hello_trtf

TRTF_HF_PYTHON=$PWD/.venv-hf/bin/python \
TRTF_MAX_CACHE_LENGTH=1 \
/tmp/hello_trtf
```

Expected output:
```
backend: trt
Hello Answer
```

## C API reference

The entire public API is a single `extern "C"` factory that returns a C++ virtual interface:

```cpp
// <trtf/pipeline.h>

// Factory -- the only extern "C" symbol users need
trtf::IPipeline* trtf_create_pipeline(const char* model_or_bundle, int flags);
//   flags: TRTF_PREFER_TRT (0), TRTF_FORCE_TRT (1), TRTF_CPU_ONLY (2)

// IPipeline -- stable vtable, safe across shared library boundary
pipeline->generate(prompt, max_new_tokens)  // returns const char* (valid until next call)
pipeline->model_id()                        // returns const char*
pipeline->backend_name()                    // returns const char*
pipeline->save_bundle(output_path)          // returns bool
delete pipeline;                            // cleanup

// Utilities
const char* trtf_last_error();  // thread-local error message
const char* trtf_version();
int trtf_has_trt();             // 1 if compiled with TRT support
```

## CLI

The `trtf` CLI is a thin wrapper around the same C API. All examples below assume you are
inside the container and have run the Python/model setup from step 3.

For convenience, set these once per session:
```bash
export TRTF_HF_PYTHON=$PWD/.venv-hf/bin/python
export TRTF_MAX_CACHE_LENGTH=256
export TRTF_MAX_NEW_TOKENS=50
```

### Running each supported model

```bash
# Qwen3-0.6B (alias)
./build/trtf run QWEN3 \
  --prompt "Tell me something about TensorRT" --force-trt

# Qwen3-0.6B (explicit path — same model, just showing the pattern)
./build/trtf run models/hf/Qwen__Qwen3-0.6B \
  --prompt "Tell me something about TensorRT" --force-trt

# TinyLlama-1.1B
./build/trtf run models/hf/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Tell me something about TensorRT" --force-trt

# TinyMistral-248M
./build/trtf run models/hf/Felladrin__TinyMistral-248M-Chat-v2 \
  --prompt "Tell me something about TensorRT" --force-trt

# DeepSeek-R1-Distill-Qwen-1.5B (chain-of-thought model — outputs <think>...</think> reasoning)
./build/trtf run models/hf/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B \
  --prompt "Tell me something about TensorRT" --force-trt
```

### Engine caching (skip recompilation on subsequent runs)

TRT engine compilation can take 10-30+ seconds. Set `TRTF_TRT_ENGINE_CACHE_DIR` to cache
the compiled engine plan to disk. The second run loads instantly:

```bash
# First run: compiles engine and saves to cache (~20s)
TRTF_TRT_ENGINE_CACHE_DIR=/tmp/trtf_engines \
./build/trtf run QWEN3 --prompt "Tell me something about TensorRT" --force-trt

# Second run: loads cached engine (<1s)
TRTF_TRT_ENGINE_CACHE_DIR=/tmp/trtf_engines \
./build/trtf run QWEN3 --prompt "What is CUDA?" --force-trt

# See what got cached
ls -lh /tmp/trtf_engines/
```

### Other commands

```bash
./build/trtf version                    # Print version and TRT support
./build/trtf inspect model.trtfb        # Print .trtfb bundle metadata
```

### Subcommands

| Subcommand | Description |
|-----------|-------------|
| `trtf run <model-or-bundle> --prompt "text"` | Generate text |
| `trtf build <model-dir> -o <output.trtfb>` | Compile model to `.trtfb` bundle |
| `trtf inspect <bundle.trtfb>` | Print bundle metadata |
| `trtf version` | Print version and TRT support status |

### Flags

- `--force-trt`: require TRT backend, fail if unavailable
- `--max-new-tokens N`: override max generation tokens
- `--max-cache-length N`: cap KV cache length (build only)

### Environment variables

| Variable | Description |
|----------|-------------|
| `TRTF_HF_PYTHON` | Path to Python for HF tokenizer bridge |
| `TRTF_MAX_NEW_TOKENS` | Override max generation tokens |
| `TRTF_MAX_CACHE_LENGTH` | Cap KV cache length (saves GPU memory) |
| `TRTF_TRT_ENGINE_CACHE_DIR` | On-disk cache for compiled TRT engines (set this to avoid recompilation) |
| `TRTF_DATA_DIR` | Override source directory for scripts/models |

## Supported models

Any HF model directory with `config.json` + `model.safetensors` that uses a registered `model_type` works automatically.

| Model | HF Repo | `model_type` | Family | Verified |
|-------|---------|-------------|--------|----------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | `qwen3` | Qwen | Yes |
| TinyLlama-1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `llama` | LLaMA | Yes |
| TinyMistral-248M | `Felladrin/TinyMistral-248M-Chat-v2` | `mistral` | Mistral | Yes |
| DeepSeek-R1-Distill-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `qwen2` | Qwen | Yes |
| Yi-Coder-1.5B | `01-ai/Yi-Coder-1.5B` | `llama` | LLaMA | Yes |
| Gemma-2B | `google/gemma-2b` | `gemma` | Gemma | Yes |

The 4 registered families are: **Qwen**, **LLaMA**, **Mistral**, **Gemma**. Any model whose `model_type` matches a family will load automatically.

## Adding a new model family

For standard dense decoders (LLaMA-like), create 2 files + 2 one-line edits:

1. `src/models/<family>/checkpoint_mapper.h/cpp` — subclass `StandardCheckpointMapper`, override `can_map()`
2. `src/models/<family>/registration.h/cpp` — register into HF Family + Checkpoint Mapper + TRT Graph Builder registries
3. Add `Register<Family>Family()` call in `hf_family_registry.cpp`
4. Add source files to `CMakeLists.txt`

See [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) for the full guide, or `src/models/template/registration.cpp` for the skeleton.

## Documentation

| Page | Description |
|------|-------------|
| [Architecture Overview](docs/wiki/Architecture-Overview.md) | Three-stage pipeline, 3-registry system, backend cascade |
| [Pipeline Deep Dive](docs/wiki/Pipeline-Deep-Dive.md) | Full call chain, data structures, safetensors loading |
| [TRT Internals](docs/wiki/TRT-Internals.md) | Decoder layer anatomy, graph ops, engine lifecycle |
| [HF vs TRT Comparison](docs/wiki/HF-vs-TRT-Comparison.md) | Side-by-side comparison with HuggingFace Transformers |
| [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) | Step-by-step guide with code examples |
| [Source Layout](docs/wiki/Source-Layout.md) | File-by-file guide to the codebase |
| [Development Log](docs/WORKLOG.md) | Chronological development history |
| [CLAUDE.md](CLAUDE.md) | Full build/test/container runbook |
