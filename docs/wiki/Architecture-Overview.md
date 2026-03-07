# Architecture Overview

## Python Builds, C++ Runs

The system is split into two phases across two languages.

| Phase | Language | Tool | Input | Output |
|-------|----------|------|-------|--------|
| **Build** | Python | `trtf-build` / `trtf_build.build()` | HF repo ID or local directory | `.trtfb` bundle |
| **Run** | C++ | `trtf` / C ABI | `.trtfb` bundle | Task-specific outputs |

### Build Phase

The Python build pipeline performs the following work:

1. Resolve the model from HuggingFace or a local directory.
2. Parse `config.json` to determine the family and architecture.
3. Load safetensors weights.
4. Apply family-specific checkpoint mapping.
5. Build TRT networks using the TensorRT Python API.
6. Compile the engines.
7. Package engine plans, tokenizer files, and metadata into a `.trtfb` bundle.

The build side is plugin-driven. A new model family is normally added by implementing a Python family plugin under `trtf_build/trtf_build/families/`.

### Run Phase

The C++ runtime loads a bundle and assembles exactly one runtime graph for the bundle's `runtime_strategy`.

```text
trtf_create_pipeline_ex()
  -> validate user input
  -> ReadBundleFile()
  -> BundlePortAdapter + TrtPortAdapter
  -> BuildContext
  -> resolve strategy family
  -> StrategyBuilder::build()
  -> PipelineServices
  -> PipelineRouter
  -> request-time service calls
```

The runtime does not keep multiple competing assembly paths. The live runtime is the builder-composed runtime.

## Runtime Layers

| Layer | Responsibility | Key files |
|------|----------------|-----------|
| **C ABI edge** | Validate user input, load bundle, build canonical context, choose strategy family, translate top-level errors | `src/cabi/api/trtf_c.cpp` |
| **Bundle and TRT adapters** | Provide reusable access to bundle sections, tokenizer extraction, TRT runtime and engine construction | `src/runtime/adapters/bundle/*`, `src/runtime/adapters/trt/*` |
| **Strategy builders** | Validate bundle/config requirements and assemble the exact service graph for a strategy family | `src/runtime/builders/*` |
| **Runtime contracts** | Define request/result DTOs and the capability service interfaces | `include/trtf/runtime/contracts/*` |
| **Pipeline router** | Expose the stable `IPipeline` API and translate path-based compatibility calls into canonical requests | `include/trtf/runtime/pipeline/router.h`, `src/runtime/pipeline/router.cpp` |
| **IO adapters** | Decode input media and persist output artifacts at the edge | `src/runtime/adapters/io/*` |
| **Service layer** | Orchestrate task behavior against fakeable ports and adapters | `src/runtime/services/*` |
| **TRT executors** | Own engine-bound CUDA/TensorRT execution only | `src/runtime/trt/*` |
| **Pure seams** | Hold policy, planning, tensor-shape derivation, and postprocess logic that should be unit-tested directly | extracted seam headers under `src/runtime/trt/**` and `src/runtime/services/**` |

## Service-Composed Runtime Model

### 1. Strategy Resolution Happens Once

`runtime_strategy` is resolved when the bundle is loaded. The runtime does not redispatch across strategy families on each request. Each bundle load produces one composed `PipelineServices` graph.

### 2. `PipelineServices` Is The Core Runtime API

`PipelineServices` is the internal capability model. It groups service interfaces such as:

- `ITextService`
- `IVideoService`
- `IAudioService`
- `ITranscriptionService`
- `IEmbeddingService`
- `IRerankService`
- `ISegmentationService`
- `IDetectionService`
- `ISolveService`

The public `IPipeline` API remains stable for C ABI compatibility, but it is implemented by `PipelineRouter`, not by a monolithic runtime object.

### 3. IO Lives At The Edge

The core runtime works on in-memory DTOs such as decoded images, decoded audio, segmentation artifacts, detection artifacts, audio artifacts, and video-frame artifacts. File paths and artifact persistence belong to `PipelineRouter` and the IO adapters.

### 4. Builders Compose, Services Orchestrate, Executors Execute

Each layer has a narrow responsibility.

- Builders validate and wire dependencies.
- Services implement task semantics.
- Ports isolate side effects.
- Executors perform real TensorRT/CUDA work.
- Pure seams carry policy and planning logic.

## Module Design Rules For Modularity And Testability

1. **No file paths in core execution logic**: path-based APIs stop at the router and IO adapters.
2. **No strategy-wide god objects**: a strategy is represented by composed services, not one monolithic implementation class.
3. **No backend-type switching in the service layer**: services depend on ports, not concrete backend class trees.
4. **No direct artifact writing in services**: services return typed results; adapters persist them.
5. **No large mixed-mode backends**: executor-heavy files must be split into planner, tensor mapper, executor shell, and postprocessor units as they grow.
6. **Every side effect must be fakeable**: filesystem, media decode, TRT runtime, engine enqueue, CUDA copies, and subprocess calls should all sit behind interfaces or seam points.
7. **Every extracted seam must have direct tests**: if logic is pure enough to extract, it is pure enough to unit test.

## Strategy Governance

Runtime strategy coverage is governed by repository checks.

`tools/check_runtime_strategy_matrix.py` validates that strategy declarations stay consistent across:

- `src/cabi/api/trtf_c.cpp`
- `src/runtime/builders/**/*.cpp`
- `tests/runtime_strategy_matrix.yaml`
- `tests/e2e_harness/contracts.py`

The goal is to keep builder registration, CLI/E2E coverage, and task contracts in sync as new strategies are added.

## Adding A New Runtime Strategy

1. Extend config parsing so the bundle exposes the new `runtime_strategy` and any required fields.
2. Add or extend a strategy-family builder under `src/runtime/builders/*`.
3. Implement capability services that depend on fakeable ports.
4. Add low-level TRT executors only for the parts that must own CUDA/TensorRT state.
5. Extract pure planners, validators, or postprocessors instead of embedding policy in executor loops.
6. Add unit tests for planners, services, router behavior, and builder validation.
7. Add smoke/E2E coverage and update the strategy governance matrix.

## Why This Architecture Scales

This structure is intended to scale with both model count and strategy count.

- New model families stay mostly on the Python build side.
- New runtime strategies add one builder/service/executor stack instead of editing a giant central pipeline object.
- Shared execution concerns stay behind ports and adapters.
- Pure logic is separated from TensorRT/CUDA state, which makes unit testing economical.
- The router remains the only compatibility shell, so the public API can stay stable while internals evolve.
