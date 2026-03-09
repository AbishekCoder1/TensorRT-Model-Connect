# HF-Style Runtime Redesign

## Goal

Redesign the C++ runtime to mirror HuggingFace's transformers/diffusers architecture:
- **TrtModule**: `model.forward()` equivalent — feed tensors in, get tensors out. All TRT I/O binding, H2D/D2H, execute, sync hidden inside.
- **DeviceTensor**: GPU-resident tensor — data stays on device between engine calls (no CPU round-trips in hot loops).
- **KvCache / RecurrentState**: Stateful wrappers over DeviceTensor vectors with advance/reset semantics.
- **Pipeline per family**: One pipeline class per model family (TextGenerationPipeline serves 25+ decoder-only models). Pipeline composes TrtModule + state + tokenizer + scheduler.
- **Factory**: `PipelineFactory::from_bundle(path)` — reads config.json, dispatches to the right pipeline class, assembles components. Like HF's `from_pretrained()`.

## Current Architecture (REMOVE)

```
C API → StrategyBuilder → Backend → BackendAdapter → Port → Service → Router
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         7 layers of indirection for one generate() call
```

## Target Architecture

```
C API → PipelineFactory::from_bundle()
            │
            ├─ Read config.json from bundle
            ├─ Dispatch on runtime_strategy
            ├─ Load TRT engines into TrtModule instances
            ├─ Create appropriate Pipeline class
            └─ Return IPipeline

IPipeline::generate()
    │
    └─ Pipeline owns TrtModule + KvCache + ITokenizer directly
       Pipeline.generate() calls module.forward() in a loop
       No adapters, no ports, no service wrappers
```

## Layers to Delete

| Layer | Files | Lines | Why |
|-------|-------|-------|-----|
| ITrtPort / TrtPortAdapter | adapters/trt/ | ~350 | Factory calls TRT directly |
| IBundlePort / BundlePortAdapter | adapters/bundle/ | ~350 | Factory reads bundle directly |
| Backend adapters (11 interfaces) | services/common/runtime_service_ports.h | ~500 | Pipelines own backends directly |
| Port wrappers (11 classes) | services/common/runtime_service_ports.cpp | ~400 | No purpose without adapters |
| Service classes | services/*/ | ~2000 | Pipeline IS the service |

## Reusable Components (KEEP and refine)

| Component | Current location | Role | HF equivalent |
|-----------|-----------------|------|---------------|
| TrtModule (NEW) | — | model.forward() abstraction | `nn.Module.__call__()` |
| DeviceTensor (NEW) | — | GPU-resident tensor | `torch.Tensor` on CUDA |
| KvCache (REFACTOR) | trt/core/device_kv_cache.h | Autoregressive KV state | `DynamicCache` / `past_key_values` |
| RecurrentState (NEW) | trt/recurrent/mamba_step_state.h | SSM/RWKV state | `MambaCache` |
| CudaStream / CudaBuffer | trt/core/trt_common.h | RAII GPU resources | `torch.cuda.Stream` / CUDA allocator |
| ITokenizer | tokenizer/ | Text tokenization | `AutoTokenizer` |
| IScheduler (NEW) | — | Diffusion schedulers | `SchedulerMixin` / `FlowMatchEulerDiscreteScheduler` |
| ImagePreprocessor | trt/multimodal/image_preprocessor.h | Vision input processing | `ImageProcessor` |
| BundleFile | bundle/bundle_format.h | .trtfb reader | `from_pretrained()` file loading |
| PipelineFactory | — | Config-driven assembly | `AutoPipelineForText2Image.from_pretrained()` |
| GenerationConfig | include/trtf/backend.h | Sampling params | `GenerationConfig` |

## Pipeline Classes (one per family)

| Pipeline | Models served | Components | HF equivalent |
|----------|--------------|------------|---------------|
| TextGenerationPipeline | 25+ decoder-only (Qwen, LLaMA, Mistral, GPT-2, OPT, Phi, Gemma, Bloom, Falcon, etc.) | TrtModule + KvCache + ITokenizer | `TextGenerationPipeline` (one class, many models) |
| MambaPipeline | Mamba | TrtModule + RecurrentState + ITokenizer | `TextGenerationPipeline` + `MambaForCausalLM` |
| RwkvPipeline | RWKV | TrtModule + RecurrentState + ITokenizer | `TextGenerationPipeline` + `RwkvForCausalLM` |
| HybridPipeline | Nemotron-H | TrtModule + KvCache + RecurrentState + ITokenizer | (no direct HF equivalent) |
| VLPipeline | Qwen2.5-VL, Qwen3-VL, InternVL3, Phi4 | TrtModule×2 + KvCache + ImagePreprocessor + ITokenizer | `Qwen2VLForConditionalGeneration.generate()` |
| WhisperPipeline | Whisper | TrtModule×2 + KvCache + CrossKv + MelProcessor | `WhisperForConditionalGeneration` |
| BarkPipeline | Bark | TrtModule×4 + KvCache×2 + CodecDecoder | `BarkModel.generate()` |
| MagpiePipeline | MagpieTTS | TrtModule×N + KvCache + CodecDecoder | (custom TTS) |
| SpeechPipeline | PersonaPlex | TrtModule×N + RecurrentState + CodecDecoder | (custom speech-to-speech) |
| OmniPipeline | Qwen3-Omni | TrtModule×N + KvCache + AudioEncoder | `Qwen3OmniModel` |
| FluxPipeline | FLUX.1, FLUX.2 | TrtModule×3 + IScheduler + FluxPreprocessor | `FluxPipeline` / `Flux2Pipeline` |
| WanPipeline | Wan2.1-T2V | TrtModule×3 + IScheduler + WanPreprocessor | `WanPipeline` |
| ZImagePipeline | Z-Image-Turbo | TrtModule×3 + IScheduler + ZImagePreprocessor | `DiffusionPipeline` |
| EncoderPipeline | BERT, Eagle-embed/rerank | TrtModule + ITokenizer (single pass) | `FeatureExtractionPipeline` |
| SegmentPipeline | SegFormer | TrtModule + ImagePreprocessor | `ImageSegmentationPipeline` |
| SamPipeline | SAM | TrtModule×2 + ImagePreprocessor | `SamModel` |

---

## HF Architecture Cross-Reference Checklist

After each phase, verify alignment with HF's design by checking these principles:

### Phase 1 checkpoint: Foundation matches HF primitives?

- [ ] **TrtModule ↔ nn.Module**: Does `module.forward(inputs) → outputs` feel like
      calling a PyTorch model? Can you swap one TrtModule for another without the
      caller knowing? (HF: any model is interchangeable at the `model(input_ids)` level)
- [ ] **DeviceTensor ↔ torch.Tensor**: Can data stay on GPU between module calls
      without manual `cudaMemcpy`? (HF: tensors are `.to("cuda")` and stay there)
- [ ] **KvCache ↔ DynamicCache**: Does KvCache have a clean `advance()` + `reset()`
      API that pipelines use without knowing cache internals? (HF: `past_key_values`
      is opaque to the pipeline — `model.forward()` consumes and returns it)
- [ ] **IScheduler ↔ SchedulerMixin**: Can you swap FlowMatchEuler for DDPM without
      changing the pipeline? (HF: `pipe.scheduler = DDPMScheduler()` just works)
- [ ] **ITokenizer ↔ AutoTokenizer**: Can you swap VocabTokenizer for HfPythonTokenizer
      without changing the pipeline? (HF: `pipe.tokenizer = AutoTokenizer.from_pretrained()`)

### Phase 2 checkpoint: Pipelines compose like HF?

- [ ] **One pipeline serves many models**: Does `TextGenerationPipeline` work for
      Qwen, LLaMA, GPT-2, Bloom etc. without any model-specific code? (HF:
      `pipeline("text-generation")` works for all CausalLM models)
- [ ] **Pipeline owns components directly**: Does the pipeline hold `TrtModule`,
      `KvCache`, `ITokenizer` as direct members (not through adapters/ports)?
      (HF: `pipe.model`, `pipe.tokenizer` are direct attributes)
- [ ] **No hidden singletons or registries**: Is all state owned by the pipeline
      instance? Can you create two pipelines simultaneously? (HF: each Pipeline
      instance is independent)
- [ ] **Factory reads config and assembles**: Does `PipelineFactory::from_bundle()`
      read `config.json` → pick pipeline class → load components → return pipeline?
      (HF: `from_pretrained()` reads `model_index.json` → pick class → load components)

### Phase 3 checkpoint: Complex pipelines follow HF patterns?

- [ ] **Diffusion: one class per family**: Are FluxPipeline/WanPipeline/ZImagePipeline
      separate classes with their own `generate_media()` logic? (HF: `FluxPipeline`,
      `WanPipeline` are separate classes, NOT one `DiffusionPipeline` with if/else)
- [ ] **Diffusion: shared components, different orchestration**: Do all diffusion
      pipelines share TrtModule/IScheduler/DeviceTensor but differ in their
      preprocessing and denoising loop? (HF: all share `SchedulerMixin`, differ in `__call__`)
- [ ] **Audio: each family is a class**: Are Whisper/Bark/Magpie separate classes?
      (HF: `WhisperForConditionalGeneration`, `BarkModel` are separate)
- [ ] **Multi-engine pipelines compose TrtModules**: Does WhisperPipeline hold
      `encoder_module` + `decoder_module` as separate TrtModule instances?
      (HF: WhisperModel holds `encoder` + `decoder` as separate nn.Module instances)

### Phase 4 checkpoint: Clean cutover?

- [ ] **No adapters/ports/services remain**: Are ALL intermediate wrapper layers
      deleted? (HF: no adapter/port/service pattern exists — pipeline calls model directly)
- [ ] **C API is thin**: Is `trtf_c.cpp` just
      `PipelineFactory::from_bundle() → IPipeline → return`?
      (HF: CLI is just `pipeline = Pipeline(...); pipeline(input)`)
- [ ] **Adding a new model family = 1 pipeline class + 1 factory case**: Is the
      onboarding cost proportional to the model's actual uniqueness?
      (HF: new model = new `XxxForCausalLM` class + register in `MODEL_MAPPING`)
- [ ] **Binary size decreased**: Did removing 5500 lines of wrappers actually
      shrink the binary? (Target: < 3.0MB from current 3.9MB)
