# trt-transformers-cpp

A C++ library that mirrors HuggingFace `transformers.pipeline(...)` with TensorRT-first execution. Reads HuggingFace model checkpoints directly, builds optimized TensorRT engines from the C++ API (no ONNX), and runs GPU-accelerated inference — all from a single `Pipeline` call.

## Why this exists
- Build a C++ API analogous to `transformers.pipeline(...)`.
- Construct TensorRT graphs directly from C++ APIs (no ONNX intermediate).
- Plug-and-play model families: adding a new family requires ~50 lines of code.

## Current status
- End-to-end `text-generation` pipeline in C++ with three backends: `trt`, `cpu-reference`, `hf-transformers`.
- Plug-and-play 4-registry architecture for model families (HF Family, Checkpoint Mapper, TRT Definition Populator, TRT Graph Builder).
- Built-in support for **Qwen** (Qwen, Qwen2, Qwen3, QWQ) and **LLaMA** (LLaMA, TinyLlama) families.
- `StandardDecoderGraphBuilder` handles the dominant LLM pattern (Pre-RMSNorm + GQA + RoPE + SwiGLU) — works for LLaMA, Qwen, Yi, Mistral-dense, DeepSeek-dense.
- Real upstream Qwen3-0.6B and TinyLlama-1.1B produce correct, coherent TRT output.

## Documentation

See the **[Project Wiki](docs/wiki/Home.md)** for detailed documentation:

| Page | Description |
|------|-------------|
| [Architecture Overview](docs/wiki/Architecture-Overview.md) | Three-stage pipeline, 4-registry system, backend cascade |
| [Pipeline Deep Dive](docs/wiki/Pipeline-Deep-Dive.md) | Full call chain, data structures, safetensors loading |
| [TRT Internals](docs/wiki/TRT-Internals.md) | Decoder layer anatomy, graph ops, engine lifecycle |
| [HF vs TRT Comparison](docs/wiki/HF-vs-TRT-Comparison.md) | Side-by-side comparison with HuggingFace Transformers |
| [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) | Step-by-step guide with code examples |
| [Source Layout](docs/wiki/Source-Layout.md) | File-by-file guide to the codebase |

## Quick start

```cpp
#include "trtf/pipeline.h"

int main()
{
    auto pipeline = trtf::Pipeline::CreateTextGeneration("QWEN3");
    std::string output = pipeline.generate("What is the capital of France?");
    // "What is the capital of France? The capital of France is Paris..."
}
```

## Build

```bash
# Host (TRT auto-disables if deps not found)
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure

# With TRT/CUDA
cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=<path>/include/zapped_headers \
  -DTRTF_TRT_LIBRARY=<path>/lib/libnvinfer.so \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

## Run

```bash
./build/trtf_text_generation trtf/tiny-cake-v1 "the secret to baking a really good cake is"
./build/trtf_load_model --force-trt QWEN3 "Hello"
./build/trtf_load_model --cpu-only QWEN3 "Hello"
```

## Built-in model IDs

| Model ID | Description | Backend |
|----------|-------------|---------|
| `trtf/tiny-cake-v1` | Bundled tiny decoder (always available) | CPU-reference |
| `QWEN3` | Qwen3-0.6B (real weights or bundled demo) | TRT or CPU-reference |
| Any HF LLaMA directory | LLaMA, TinyLlama, etc. | TRT or CPU-reference |
| Any HF Qwen directory | Qwen, Qwen2, Qwen3, QWQ | TRT or CPU-reference |

## Adding a new model family

For standard dense decoders (LLaMA-like), create 2 files + 2 one-line edits:

1. `src/models/<family>/checkpoint_mapper.h/cpp` — subclass `StandardCheckpointMapper`, override `can_map()`
2. `src/models/<family>/registration.h/cpp` — register into HF Family + Checkpoint Mapper + TRT Graph Builder registries
3. Add `Register<Family>Family()` call in `hf_family_registry.cpp`
4. Add source files to `CMakeLists.txt`

`StandardTrtModelDefinitionPopulator` (Registry 3) and `StandardDecoderGraphBuilder` (Registry 4) handle the rest automatically.

See [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) for the full guide with code examples, or `src/models/template/registration.cpp` for the skeleton.

## Container workflow (TRT GPU)

See [CLAUDE.md](CLAUDE.md) for the full container runbook including Docker setup, Python env, model download, and E2E validation commands.

## CLI reference

- `--force-trt`: require TRT backend, fail if unavailable
- `--cpu-only`: bypass TRT, use `cpu-reference`
- `TRTF_HF_PYTHON`: path to Python interpreter for HF tokenizer bridge
- `TRTF_MAX_NEW_TOKENS`: override max generation tokens
- `TRTF_MAX_CACHE_LENGTH`: cap KV cache length
- `TRTF_TRT_ENGINE_CACHE_DIR`: on-disk cache for serialized TRT engine plans
- `TRTF_DEBUG_LOGITS_TOPK=N`: print top-N logit debug info per decode step

## Additional docs
- `docs/wiki/` — comprehensive project wiki with architecture diagrams
- `docs/TRANSFORMERS_COVERAGE_ANALYSIS.md` — model family coverage roadmap
- `docs/WORKLOG.md` — development history and decisions
- `CLAUDE.md` — build/test/container runbook
