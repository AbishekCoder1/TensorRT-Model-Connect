# TRT-Transformers-CPP Wiki

A C++ library that mirrors HuggingFace's `transformers.pipeline()` API with TensorRT-first execution. Read HuggingFace model checkpoints directly, build optimized TensorRT engines from the C++ API (no ONNX), and run GPU-accelerated inference — all from a single `Pipeline` call.

## Quick Navigation

| Page | Description |
|------|-------------|
| **[Architecture Overview](Architecture-Overview.md)** | Three-stage pipeline, 4-registry system, backend cascade |
| **[Pipeline Deep Dive](Pipeline-Deep-Dive.md)** | Detailed walkthrough of model resolution, family registry, runtime assembly |
| **[TRT Internals](TRT-Internals.md)** | How TensorRT graph building works, decoder layer anatomy, engine lifecycle |
| **[HF vs TRT Comparison](HF-vs-TRT-Comparison.md)** | Side-by-side comparison of HuggingFace Transformers and this library |
| **[Adding a Model Family](Adding-a-Model-Family.md)** | Step-by-step guide with code examples for onboarding a new model |
| **[Source Layout](Source-Layout.md)** | File-by-file guide to the codebase |

## Core Design Principles

1. **Mirror the HF API**: `pipeline("text-generation", model=...)` becomes `Pipeline::CreateTextGeneration(model_id)`.
2. **TensorRT-first, always-fallback**: Try TRT GPU inference first, fall back gracefully to CPU reference or HF Python subprocess.
3. **Plug-and-play families**: Adding a new model family requires ~50 lines of code in `src/models/<family>/` and two one-line edits.
4. **Direct C++ TRT graph building**: No ONNX intermediate. Build `INetworkDefinition` directly using reusable op primitives.
5. **Shared infrastructure**: StandardCheckpointMapper, StandardTrtModelDefinitionPopulator, and StandardDecoderGraphBuilder handle 95% of modern decoder-only LLMs out of the box.

## Architecture at a Glance

![Pipeline Flow](diagrams/pipeline_flow.svg)

The library processes a model request in three stages:

1. **Model Resolution** — Turns a `model_id` string (e.g., `"QWEN3"`, a path to an HF directory) into a `ResolvedModelSpec`.
2. **HF Family Registry** — When the model is an HF directory, matches `model_type` from `config.json` to a registered family, loads weights via a checkpoint mapper, and produces a unified `DecoderModel`.
3. **Runtime Assembly** — Creates a tokenizer and selects a backend (TRT > CPU-reference > HF-Python), producing a ready-to-use `Pipeline` object.

## Minimum Viable Example

```cpp
#include "trtf/pipeline.h"

int main() {
    // Mirrors: transformers.pipeline("text-generation", model="Qwen/Qwen3-0.6B")
    auto pipeline = trtf::Pipeline::CreateTextGeneration("QWEN3");
    std::string output = pipeline.generate("What is the capital of France?");
    // output: "What is the capital of France? The capital of France is Paris..."
}
```

## Built-in Model Support

| Model ID | Description | Backend |
|----------|-------------|---------|
| `trtf/tiny-cake-v1` | Bundled tiny decoder (always available) | CPU-reference |
| `QWEN3` | Qwen3-0.6B (real weights or bundled demo) | TRT or CPU-reference |
| Any HF LLaMA directory | LLaMA, TinyLlama, etc. | TRT or CPU-reference |
| Any HF Qwen directory | Qwen, Qwen2, Qwen3, QWQ | TRT or CPU-reference |
