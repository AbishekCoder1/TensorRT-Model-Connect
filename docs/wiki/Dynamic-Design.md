# Dynamic Design

This page documents the live build-time and run-time flows. The runtime path is the builder-composed runtime that returns a `PipelineRouter` over `PipelineServices`.

## 1. Bundle Build Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as trtf-build
    participant Plugin as Family Plugin
    participant Mapper as Checkpoint Mapper
    participant Graph as TRT Graph Builder
    participant Bundle as Bundle Writer

    User->>CLI: build <model> -o model.trtfb
    CLI->>Plugin: resolve family from config.json
    Plugin->>Mapper: map HF weights to canonical weights
    Plugin->>Graph: build TRT network(s)
    Graph-->>CLI: compiled engine plans
    CLI->>Bundle: write engine plans + tokenizer + metadata
    Bundle-->>User: model.trtfb
```

## 2. Runtime Creation Flow

```mermaid
sequenceDiagram
    participant User
    participant CABI as trtf_c.cpp
    participant Bundle as BundlePortAdapter
    participant TRT as TrtPortAdapter
    participant Builder as StrategyBuilder
    participant Services as PipelineServices
    participant Router as PipelineRouter

    User->>CABI: trtf_create_pipeline_ex(bundle_path, options)
    CABI->>Bundle: ReadBundleFile(bundle_path)
    Bundle-->>CABI: BundleFile + sections
    CABI->>CABI: BuildContext(strategy, config, defaults, handles)
    CABI->>Builder: build(BuildContext)
    Builder->>Bundle: read sections / tokenizer blobs
    Builder->>TRT: create runtime-dependent objects
    Builder->>Services: compose capability services
    Services-->>Builder: PipelineServices
    Builder-->>CABI: BuildResult(PipelineServices)
    CABI->>Router: construct with services + defaults + IO adapters
    Router-->>User: IPipeline*
```

## 3. Text Generation Request Flow

```mermaid
sequenceDiagram
    participant User
    participant Router as PipelineRouter
    participant Text as ITextService
    participant Ports as RuntimeServicePorts
    participant Exec as TRT executor

    User->>Router: generate(prompt, max_new_tokens)
    Router->>Router: resolve defaults
    Router->>Text: generate(prompt, max_new_tokens)
    Text->>Ports: prepare request / invoke backend capability
    Ports->>Exec: bind tensors + enqueue + copy outputs
    Exec-->>Ports: backend outputs
    Ports-->>Text: typed generation result
    Text-->>Router: const char*
    Router-->>User: const char*
```

## 4. Media Request Flow

Segmentation, detection, speech, transcription, and vision-language calls follow the same pattern: file paths stop at the router boundary.

```mermaid
sequenceDiagram
    participant User
    participant Router as PipelineRouter
    participant Loader as IO adapters
    participant Service as Capability service
    participant Ports as RuntimeServicePorts
    participant Exec as TRT executor
    participant Writer as Artifact writer

    User->>Router: segment(image_path, output_path)
    Router->>Loader: decode image_path
    Loader-->>Router: DecodedImage
    Router->>Service: segment(SegmentationRequest{DecodedImage})
    Service->>Ports: invoke backend capability
    Ports->>Exec: engine-bound execution
    Exec-->>Ports: raw outputs
    Ports-->>Service: typed artifact
    Service-->>Router: SegmentationArtifact
    Router->>Writer: write artifact(output_path)
    Writer-->>Router: status
    Router-->>User: int32_t status
```

## 5. Why This Flow Is Testable

Each stage can be tested independently.

- Builders: validate bundle/config behavior without running full inference.
- Router: validate defaults, decode/load behavior, and artifact writes with fake loaders and writers.
- Services: validate request handling with fake ports.
- Executors: validate bind/copy/enqueue/sync behavior with focused harnesses.
- Pure seams: validate planner and postprocess logic with direct unit tests.

That separation is what keeps the runtime scalable and coverage-friendly.
