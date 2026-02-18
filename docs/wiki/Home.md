# TRT-Transformers-CPP Wiki

A split-language system for TensorRT LLM inference: **Python builds** optimized TRT engines from HuggingFace model checkpoints, **C++ runs** them at maximum speed. The Python `trtf_build/` package reads safetensors, constructs TRT networks via the TensorRT Python API, and produces self-contained `.trtfb` bundles. The C++ runtime loads these bundles and runs GPU-accelerated autoregressive generation via a C ABI entry point.

## Quick Navigation

| Page | Description |
|------|-------------|
| **[Architecture Overview](Architecture-Overview.md)** | Python-builds-C++-runs architecture, bundle-only runtime, two CLIs |
| **[Static Design](Static-Design.md)** | Class-level UML diagrams and logical descriptions for every software unit |
| **[Dynamic Design](Dynamic-Design.md)** | Sequence diagrams (Mermaid) and data flow for bundle creation and generation |
| **[Pipeline Deep Dive](Pipeline-Deep-Dive.md)** | Detailed walkthrough of bundle loading and runtime assembly |
| **[TRT Internals](TRT-Internals.md)** | How TensorRT graph building works (now in Python), decoder layer anatomy, engine lifecycle |
| **[HF vs TRT Comparison](HF-vs-TRT-Comparison.md)** | Side-by-side comparison of HuggingFace Transformers and this library |
| **[Adding a Model Family](Adding-a-Model-Family.md)** | Step-by-step guide for adding a new model family in Python |
| **[Extensibility Assessment](Architecture-Extensibility-Assessment.md)** | MoE, Mamba/SSM support status; MLA roadmap |
| **[Source Layout](Source-Layout.md)** | File-by-file guide to the codebase (Python + C++) |

## Core Design Principles

1. **Mirror the HF API**: `pipeline("text-generation", model=...)` becomes `trtf_create_pipeline(bundle_path, flags)`.
2. **C ABI stability**: Single `extern "C"` factory function returns a C++ virtual interface (`IPipeline`). ABI-safe across compiler/STL versions.
3. **Python builds, C++ runs**: Engine building (safetensors loading, checkpoint mapping, TRT graph construction) lives in Python. Runtime inference (engine deserialization, autoregressive loop, KV-cache, CUDA) lives in C++.
4. **Plug-and-play families**: Adding a new model family is a Python-only task -- create a plugin in `trtf_build/` with a checkpoint mapper and graph builder.
5. **No ONNX**: TRT networks are built directly using the TensorRT Python API. No intermediate representation.
6. **Distributable bundles**: `.trtfb` files package compiled TRT engines + tokenizer into a single artifact for instant C++ loading.

## Architecture at a Glance

The system has two phases:

1. **Build phase (Python)** -- `trtf-build build <hf-model-dir> -o model.trtfb`
   - Reads HF `config.json` + `model.safetensors` + `tokenizer.json`
   - Matches `model_type` to a family plugin (see [families/ directory](../../trtf_build/trtf_build/families/) for full list)
   - Maps HF tensor keys to canonical format, builds TRT network, compiles engine
   - Packages engine plan + tokenizer files into a `.trtfb` bundle

2. **Run phase (C++)** -- `trtf run model.trtfb --prompt "text"`
   - Loads `.trtfb` bundle, deserializes TRT engine
   - Dispatches to the correct backend based on `runtime_strategy` in config.json:
     - `decoder_kv_cache` / `decoder_moe`: `TrtBackendFastPath` with KV-cache management
     - `ssm_recurrent`: `MambaBackend` with conv + SSM recurrent state
     - `vision_language`: VL pipeline with vision encoder + text decoder + image preprocessing
   - Creates tokenizer + autoregressive generation loop
   - Runs GPU-accelerated inference

## Minimum Viable Example

```bash
# Build a bundle (Python)
trtf-build build path/to/Qwen3-0.6B -o qwen3.trtfb

# Run inference (C++)
trtf run qwen3.trtfb --prompt "What is the capital of France?" --max-new-tokens 30
```

Or from the C API:
```cpp
#include <trtf/pipeline.h>
#include <iostream>

int main() {
    auto* pipeline = trtf_create_pipeline("qwen3.trtfb", TRTF_FORCE_TRT);
    if (!pipeline) {
        std::cerr << trtf_last_error() << std::endl;
        return 1;
    }
    std::cout << pipeline->generate("What is the capital of France?", 30) << std::endl;
    delete pipeline;
}
```

## Built-in Model Support

Family plugins covering dense decoders, extended decoders, MoE, SSM, and vision-language (auto-discovered from `trtf_build/trtf_build/families/`).

| Family | Model Types | Python Plugin |
|--------|-------------|---------------|
| Qwen | qwen, qwen2, qwen3, qwq | `families/qwen.py` |
| LLaMA | llama | `families/llama.py` |
| Mistral | mistral | `families/mistral.py` |
| Gemma | gemma, gemma2 | `families/gemma.py` |
| Phi | phi, phi3 (not phimoe) | `families/phi.py` |
| Phi-MoE | phimoe | `families/phi_moe.py` |
| Granite | granite | `families/granite.py` |
| InternLM | internlm, internlm2 | `families/internlm.py` |
| StarCoder2 | starcoder2 | `families/starcoder2.py` |
| GPT-2 | gpt2 | `families/gpt2.py` |
| OPT | opt | `families/opt.py` |
| Falcon | falcon | `families/falcon.py` |
| StableLM | stablelm | `families/stablelm.py` |
| Mamba | mamba | `families/mamba.py` |
| Qwen-VL | qwen2_vl, qwen2_5_vl | `families/qwen_vl.py` (vision encoder + text decoder) |
| OLMo | olmo | `families/olmo.py` |
| XGLM | xglm | `families/xglm.py` |
| GPT-NeoX | gpt_neox | `families/gpt_neox.py` |
| GPT-Neo | gpt_neo | `families/gpt_neo.py` |
| CodeGen | codegen | `families/codegen.py` |
| BLOOM | bloom | `families/bloom.py` |
| Mixtral | mixtral | `families/mixtral.py` |
| Nemotron | nemotron | `families/nemotron.py` |

All plugin paths relative to `trtf_build/trtf_build/`.
