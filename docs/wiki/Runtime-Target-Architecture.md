# Runtime Architecture Standard

This document defines the authoritative architecture for the live C++ runtime. It describes the runtime as it should be implemented and maintained now, not as a migration target.

## Design Goal

The runtime must be:

- **modular**: new strategies and new backend families land as isolated units
- **scalable**: adding many more models does not force edits to a central god object
- **individually testable**: most runtime logic can be validated without real TRT engines or filesystem state
- **operationally stable**: the public `IPipeline` API remains stable while internals evolve

## Architectural Contract

The live runtime uses one composition path:

```text
trtf_c.cpp
  -> BuildContext
  -> strategy-family builder
  -> PipelineServices
  -> PipelineRouter
  -> ports / adapters / TRT executors
```

There is no co-equal alternate runtime architecture.

## Layer Responsibilities

| Layer | Owns | Must not own |
|------|------|---------------|
| **C ABI / CLI edge** | input validation, bundle loading, top-level error handling, family dispatch | strategy-specific execution logic, artifact IO policy |
| **Strategy builder** | bundle validation, dependency wiring, service composition, runtime defaults | request-time modality branching, direct file IO |
| **Runtime contracts** | DTOs, service interfaces, build result/status contracts | side effects, concrete backend ownership |
| **Pipeline router** | `IPipeline` compatibility surface, path-to-DTO translation, artifact writing | strategy selection, concrete TRT policy |
| **IO adapters** | image/audio decode, JSON/PNG/WAV/frame persistence | strategy semantics, TRT state |
| **Service layer** | task orchestration against ports, validation of request DTOs, typed results | direct filesystem writes, raw TRT runtime ownership |
| **Ports** | abstract side effects: bundle access, TRT runtime access, media decode, artifact writing, subprocesses | task policy |
| **TRT executors** | engine deserialization, tensor binding, CUDA copies, enqueue/sync, backend-local buffers | bundle composition, user-visible artifact policy |
| **Pure seams** | planning, stop rules, scheduling, tensor-shape derivation, postprocess logic | CUDA state, file IO |

## Core Runtime Model

### `PipelineServices`

`PipelineServices` is the internal capability model. A strategy builder fills only the services that the strategy supports. Examples include text generation, embeddings, reranking, segmentation, detection, video generation, audio generation, transcription, and speech.

The service layer consumes canonical DTOs and returns typed results. That keeps the core runtime independent from path-based compatibility APIs.

### `PipelineRouter`

`PipelineRouter` is the stable shell that implements `IPipeline`. It is responsible for:

- adapting path-based `IPipeline` calls to decoded media DTOs
- applying runtime defaults
- writing artifacts through edge adapters
- exposing stable `model_id()` and `backend_name()` information

`PipelineRouter` is not a strategy owner. It routes to services that were already composed during bundle load.

### Strategy Builders

A builder is the composition root for one strategy family. It must:

1. validate that the bundle contains the sections required by the strategy
2. derive runtime defaults and config-dependent behavior
3. create ports/adapters needed by the services
4. construct the relevant `PipelineServices`
5. return a typed build result with status and message on failure

Builders must not perform request-time file IO or hide unrelated strategy logic.

## The Required Backend Shape

Every backend family should converge on the same internal shape.

### 1. Planner / Validator

Pure logic that derives:

- loop bounds
- scheduler choices
- stop criteria
- input layout
- tensor sizes and shape expectations
- prompt or conditioning expansion

These units should be header-only or otherwise direct unit-test targets.

### 2. Tensor Mapper

Pure or near-pure logic that maps canonical DTOs and planner output into the tensor views or host buffers expected by the executor.

### 3. Executor Shell

A thin unit that owns:

- tensor binding
- device allocation and copies
- enqueue/sync
- device-to-host output retrieval

Executor shells should be deliberately small. If they grow policy or branching, more seams must be extracted.

### 4. Postprocessor / Result Assembler

Pure logic that interprets executor outputs and produces typed runtime results.

### 5. Service Adapter

The service layer ties together planners, mappers, ports, and executor shells to implement a user-facing capability.

## Port Taxonomy

The runtime should keep these port types explicit.

1. **Bundle ports**
   - read named bundle sections
   - extract tokenizer or engine blobs
2. **TRT ports**
   - create runtimes or engines
   - expose testable wrappers for engine lifecycle work
3. **Media input ports**
   - decode image/audio inputs into canonical DTOs
4. **Artifact writer ports**
   - persist audio, masks, detections, or frame directories
5. **Service execution ports**
   - abstract backend capabilities used by the service layer
6. **Executor ports**
   - fakeable interfaces for tensor binding, memcpy, enqueue, and stream sync when executor shells need direct coverage

## Repository Rules For A Fully Testable Runtime

1. **Core services operate on DTOs, not file paths**.
2. **Every side effect must cross an interface or seam**.
3. **Executor-heavy code must be split before it becomes hard to test**.
4. **Service tests use fake ports, not real engines**.
5. **Router tests validate edge behavior and artifact IO**.
6. **Executor tests use focused harnesses and failure injection**.
7. **Smoke and E2E tests validate real integrated behavior, not routine branch coverage**.
8. **No new central dispatch objects**: add builders and services, not another god pipeline.

## Recommended Test Pyramid

### Unit tests

Cover:

- planners and validators
- tensor mappers
- postprocessors
- service logic with fake ports
- router edge behavior with fake loaders and writers
- builder validation and composition outcomes

Each test should document:

- **intent**: what behavior is being verified
- **preconditions**: the target state or inputs
- **postconditions**: the externally observable result

### Harness tests

Cover:

- executor shells
- bind/copy/enqueue/sync failure paths
- shape mismatch handling
- missing tensor guards

### Smoke and E2E tests

Cover:

- real engine bring-up
- parity against reference implementations
- model-family regression coverage

## Strategy Growth Model

This architecture supports growth in two axes.

### New model families

Usually a Python-side concern:

- add a plugin
- add checkpoint mapping
- add graph builder logic if needed
- reuse an existing runtime strategy when possible

### New runtime strategies

A C++ runtime concern:

- add builder validation and composition
- add or reuse services
- add planners/mappers/postprocessors
- add executor shells only where TensorRT/CUDA ownership is required
- add strategy governance and test coverage

## What Good Runtime Code Looks Like

A runtime module is in good shape when:

- the builder composes it without special-case glue in `trtf_c.cpp`
- the service can be tested with fake ports
- the router can exercise it without real filesystem dependencies beyond the edge adapters
- policy and shape logic are covered directly by unit tests
- the executor shell is thin enough that harness tests can cover its error paths

That is the standard the runtime should keep converging toward.
