# TRT-Transformers-CPP Wiki

A split-language system for TensorRT inference: **Python builds** optimized TRT engines from HuggingFace model checkpoints, **C++ runs** them at maximum speed. The Python `trtf_build/` package reads safetensors, constructs TRT networks via the TensorRT Python API, and produces self-contained `.trtfb` bundles. The C++ runtime loads these bundles and runs GPU-accelerated, strategy-specific inference (text, vision, audio, segmentation/detection, neural operators, diffusion) via a C ABI entry point.

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
| **[Testing and Validation](Testing-and-Validation.md)** | Regression tiers, unified diff framework, per-category test strategies |
| **[Agent Orchestration](../AGENT_ORCHESTRATION.md)** | Autonomous multi-agent HF onboarding flow, validation, and merge gating |
| **[Extensibility Assessment](Architecture-Extensibility-Assessment.md)** | MoE, Mamba/SSM, diffusion support status; MLA roadmap |
| **[Source Layout](Source-Layout.md)** | File-by-file guide to the codebase (Python + C++) |

## Core Design Principles

1. **Mirror the HF API**: `pipeline("text-generation", model=...)` becomes `trtf_create_pipeline(bundle_path, flags)`.
2. **C ABI stability**: Single `extern "C"` factory function returns a C++ virtual interface (`IPipeline`). ABI-safe across compiler/STL versions.
3. **Python builds, C++ runs**: Engine building (safetensors loading, checkpoint mapping, TRT graph construction) lives in Python. Runtime inference (engine deserialization, strategy dispatch, modality-specific execution loops, CUDA) lives in C++.
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

2. **Run phase (C++)** -- `trtf <command> model.trtfb ...`
   - Loads `.trtfb` bundle, deserializes TRT engine
   - Dispatches to the correct backend based on `runtime_strategy` in config.json:
     - Decoder paths: `decoder_kv_cache`, `decoder_moe` (`trt` backend family)
     - Recurrent/hybrid: `ssm_recurrent` (`trt_mamba`), `rwkv_recurrent` (`trt_rwkv`), `hybrid_mamba_attention` (`trt_hybrid`)
     - Vision: `vision_language` (`trt_vl`), `segmentation` (`trt_segmentation`), `prompted_segmentation` (`trt_sam`), `object_detection` (`trt_detection`)
     - Speech/audio: `speech_to_text` (`trt_whisper`), `text_to_audio` (`trt_bark`), `speech_to_speech` (`trt_speech`)
     - Encoder/ranking: `encoder_only` (`trt_encoder`), `embedding` (`trt_embedding`), `reranking` (`trt_reranking`)
     - Other specialized paths: `neural_operator` (`trt_neural_operator`), `omni_multimodal` (`trt_omni`), `diffusion` (`trt_diffusion`)
   - Creates tokenizer and task-specific runtime loop(s)
   - Runs GPU-accelerated inference

## Minimum Viable Example

```bash
# Build a bundle (Python)
trtf-build build path/to/Qwen3-0.6B -o qwen3.trtfb

# Text generation (C++)
trtf run qwen3.trtfb --prompt "What is the capital of France?" --max-new-tokens 30

# Other runtime commands use the same bundle entry point:
trtf encode bert.trtfb --prompt "hello"
trtf segment segformer.trtfb --image input.png --output mask.png
trtf transcribe whisper.trtfb --audio sample.wav
trtf generate-video wan_t2v.trtfb --prompt "A cat riding a bike" --output frames/
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

Family plugins are auto-discovered from `trtf_build/trtf_build/families/`; there is no static registration list to update. Any HF model whose `model_type` maps to a plugin is buildable.

Canonical source of truth in your checkout:

```bash
ls trtf_build/trtf_build/families/*.py \
  | sed 's|.*/||; s|\.py$||' \
  | rg -v '^(__init__|base)$' \
  | sort
```

As of **February 26, 2026**, the repository contains **44** family modules:

`bark`, `bert`, `bloom`, `codegen`, `deeponet`, `deepseek_ocr`, `deepseek_v2`, `eagle_vlm`, `falcon`, `flux`, `fno`, `gemma`, `gpt2`, `gpt_neo`, `gpt_neox`, `granite`, `internlm`, `internvl`, `llama`, `mamba`, `mistral`, `mixtral`, `nemotron`, `nemotron_h`, `olmo`, `opt`, `personaplex`, `phi`, `phi4_multimodal`, `phi_moe`, `qwen`, `qwen3_omni`, `qwen_moe`, `qwen_vl`, `rwkv`, `sam`, `segformer`, `stablelm`, `starcoder2`, `wan_t2v`, `whisper`, `xglm`, `yolox`, `z_image`.

These modules cover decoder LLMs, MoE/hybrid/recurrent models, VLM/multimodal models, segmentation/detection, ASR/TTS/speech-to-speech, neural operators, embedding/reranking, and diffusion/video pipelines.
