# Source Layout

This page describes the current repository layout with emphasis on the live service-composed runtime.

## Top-Level Layout

| Path | Purpose |
|------|---------|
| `include/trtf/` | Public C/C++ API headers |
| `include/trtf/runtime/` | Runtime contracts, builder interfaces, and router API |
| `src/cabi/` | C ABI entry and runtime-facing config/bundle helpers |
| `src/runtime/` | Builder-composed runtime implementation |
| `src/bundle/` | Bundle file format read/write support |
| `src/tokenizer/` | Tokenizer implementations and bridges |
| `src/utils/` | Shared utility helpers |
| `trtf_build/` | Python builder package |
| `tests/` | C++, Python, tools, and E2E tests |
| `docs/wiki/` | Architecture and design documentation |

## Public API

| Path | Purpose |
|------|---------|
| `include/trtf/pipeline.h` | Stable `IPipeline` API plus C ABI entrypoints |
| `include/trtf/bundle.h` | Bundle inspection API |
| `include/trtf/backend.h` | Generation backend interfaces used by lower layers |
| `include/trtf/tokenizer.h` | Tokenizer interfaces |
| `include/trtf/generation.h` | Generation config/result types |

## Runtime Composition

### `src/cabi/`

| Path | Purpose |
|------|---------|
| `src/cabi/api/trtf_c.cpp` | Single runtime creation entrypoint: input validation, bundle load, family dispatch, builder invocation, and top-level error handling |
| `src/cabi/config/fast_path_config.*` | `config.json` parsing into runtime-facing configuration |
| `src/cabi/bundle/bundle_helpers.*` | Shared bundle helper functions used at runtime assembly time |

### `include/trtf/runtime/contracts/`

| Path | Purpose |
|------|---------|
| `build.h` | `BuildContext`, `BuildResult`, status types |
| `services.h` | `PipelineServices`, runtime DTOs, and service interfaces |
| `strategy_builder.h` | Builder interface contract |

### `src/runtime/builders/`

| Path | Purpose |
|------|---------|
| `text/` | Builder logic for decoder, MoE, Mamba, RWKV, and hybrid text strategies |
| `encoder/` | Builder logic for encoder-only, embedding, and reranking strategies |
| `vision/` | Builder logic for vision-language, segmentation, SAM, detection, and neural-operator strategies |
| `audio/` | Builder logic for TTS, ASR, speech-to-speech, and omni-multimodal strategies |
| `diffusion/` | Builder logic for diffusion/video strategies |

### `src/runtime/pipeline/`

| Path | Purpose |
|------|---------|
| `router.cpp` / `include/.../router.h` | `PipelineRouter`, the thin `IPipeline` compatibility shell |

## Runtime Services And Ports

### `src/runtime/services/`

| Path | Purpose |
|------|---------|
| `text/` | Text generation and encoding service implementations |
| `vision/` | Segmentation, detection, and vision-language service implementations |
| `audio/` | Audio generation, speech, and transcription service implementations |
| `common/` | Shared service ports and adapters used by the service layer |

The service layer consumes canonical DTOs and depends on fakeable ports rather than raw TensorRT state.

## Runtime Adapters

### `src/runtime/adapters/`

| Path | Purpose |
|------|---------|
| `bundle/` | Bundle-port adapters used by builders |
| `trt/` | TRT runtime/engine adapters used by builders and low-level execution glue |
| `io/` | Media input loaders and artifact writers for WAV/PNG/JSON/frame outputs |

These adapters define the runtime edge. Filesystem paths and artifact persistence should stop here or in `PipelineRouter`.

## TRT Execution Layer

### `src/runtime/trt/`

| Path | Purpose |
|------|---------|
| `core/` | Common TensorRT runtime support such as engine lifecycle and shared decode helpers |
| `audio/` | Audio-family executors and extracted seams for Bark, Whisper, speech, Magpie, and omni paths |
| `diffusion/` | Diffusion-family executors plus scheduler, conditioning, and generation-plan seams |
| `encoder/` | Encoder/embedding executors |
| `multimodal/` | Vision-language and vision-engine execution code |
| `perception/` | Segmentation, SAM, and perception-specific preprocessing/postprocessing seams |
| `recurrent/` | Recurrent runtime execution and state/tensor-binding helpers |

This layer should converge on a consistent internal pattern:

1. planner / validator
2. tensor mapper
3. executor shell
4. postprocessor

## Python Builder Layout

### `trtf_build/trtf_build/`

| Path | Purpose |
|------|---------|
| `build.py` | Public Python entrypoint for bundle creation |
| `config.py` | `config.json` parsing and canonical build-time config extraction |
| `builder.py` | Bundle build orchestration |
| `graph_ops.py` | Atomic TensorRT graph operations |
| `graph_blocks.py` | Composable graph building blocks |
| `standard_decoder_builder.py` | Shared standard decoder engine builder |
| `checkpoint_mapper.py` | Shared checkpoint-mapping helpers |
| `families/` | Auto-discovered Python family plugins |
| `schedulers/` | Diffusion scheduler implementations |

## Tests

| Path | Purpose |
|------|---------|
| `tests/cpp/` | C++ unit and harness tests for builders, services, router, seams, and low-level runtime helpers |
| `tests/builder/` | Python builder tests |
| `tests/tools/` | Tooling and governance self-tests |
| `tests/e2e/` | GPU-backed end-to-end regression tests |

## Documentation Files Worth Reading First

1. [Architecture Overview](Architecture-Overview.md)
2. [Runtime Architecture Standard](Runtime-Target-Architecture.md)
3. [Pipeline Deep Dive](Pipeline-Deep-Dive.md)
4. [Testing and Validation](Testing-and-Validation.md)
