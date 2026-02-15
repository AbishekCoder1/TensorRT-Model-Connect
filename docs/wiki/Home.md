# TRT-Transformers-CPP Wiki

A C++ library that mirrors HuggingFace's `transformers.pipeline()` API with TensorRT-first execution. Read HuggingFace model checkpoints directly, build optimized TensorRT engines from the C++ API (no ONNX), and run GPU-accelerated inference — all from a single `trtf_create_pipeline()` call. Distributable as a library with C ABI entry point, `.trtfb` engine bundles, and a CLI.

## Quick Navigation

| Page | Description |
|------|-------------|
| **[Architecture Overview](Architecture-Overview.md)** | Three-stage pipeline, 4-registry system, backend cascade |
| **[Static Design](Static-Design.md)** | Class-level UML diagrams and logical descriptions for every software unit |
| **[Dynamic Design](Dynamic-Design.md)** | Sequence diagrams (Mermaid) and data flow for pipeline creation, generation, TRT engine build |
| **[Pipeline Deep Dive](Pipeline-Deep-Dive.md)** | Detailed walkthrough of model resolution, family registry, runtime assembly |
| **[TRT Internals](TRT-Internals.md)** | How TensorRT graph building works, decoder layer anatomy, engine lifecycle |
| **[HF vs TRT Comparison](HF-vs-TRT-Comparison.md)** | Side-by-side comparison of HuggingFace Transformers and this library |
| **[Adding a Model Family](Adding-a-Model-Family.md)** | Step-by-step guide with code examples for onboarding a new model |
| **[Extensibility Assessment](Architecture-Extensibility-Assessment.md)** | How hard to add MoE, Mamba/SSM, MLA? Blocking assumptions, refactoring roadmap, subagent parallelization strategy |
| **[Source Layout](Source-Layout.md)** | File-by-file guide to the codebase |

## Core Design Principles

1. **Mirror the HF API**: `pipeline("text-generation", model=...)` becomes `trtf_create_pipeline(model_id, flags)`.
2. **C ABI stability**: Single `extern "C"` factory function returns a C++ virtual interface (`IPipeline`). ABI-safe across compiler/STL versions.
3. **TensorRT-first, always-fallback**: Try TRT GPU inference first, fall back gracefully to HF Python subprocess.
4. **Plug-and-play families**: Adding a new model family requires ~50 lines of code in `src/models/<family>/` and two one-line edits.
5. **Direct C++ TRT graph building**: No ONNX intermediate. Build `INetworkDefinition` directly using reusable op primitives.
6. **Shared infrastructure**: StandardCheckpointMapper and StandardDecoderGraphBuilder handle 95% of modern decoder-only LLMs out of the box.
7. **Distributable bundles**: `.trtfb` files package compiled TRT engines + tokenizer into a single artifact.

## Architecture at a Glance

![Pipeline Flow](diagrams/pipeline_flow.svg)

The library processes a model request in three stages:

1. **Model Resolution** — Turns a `model_id` string (e.g., `"QWEN3"`, a path to an HF directory, or a `.trtfb` bundle) into a `ResolvedModelSpec`.
2. **HF Family Registry** — When the model is an HF directory, matches `model_type` from `config.json` to a registered family, loads weights via a checkpoint mapper, and produces a unified `DecoderModel`.
3. **Runtime Assembly** — Creates a tokenizer and selects a backend (TRT > HF-Python), producing a ready-to-use `IPipeline` object.

## Minimum Viable Example

```cpp
#include <trtf/pipeline.h>
#include <iostream>

int main() {
    auto* pipeline = trtf_create_pipeline("QWEN3", TRTF_FORCE_TRT);
    if (!pipeline) {
        std::cerr << trtf_last_error() << std::endl;
        return 1;
    }
    std::cout << pipeline->generate("What is the capital of France?", 30) << std::endl;
    delete pipeline;
}
```

Or using the CLI:
```bash
trtf run QWEN3 --prompt "What is the capital of France?" --force-trt --max-new-tokens 30
```

## Built-in Model Support

| Model ID | Description | Backend |
|----------|-------------|---------|
| `QWEN3` | Qwen3-0.6B (real weights or bundled demo) | TRT |
| Any HF LLaMA directory | LLaMA, TinyLlama, etc. | TRT |
| Any HF Qwen directory | Qwen, Qwen2, Qwen3, QWQ | TRT |
| Any HF Mistral directory | Mistral, TinyMistral | TRT |
| Any HF Gemma directory | Gemma | TRT |
