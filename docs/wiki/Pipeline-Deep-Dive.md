# Pipeline Deep Dive

This page walks through the live C++ runtime assembly path from `trtf_create_pipeline_ex()` to a ready `PipelineRouter`.

## 1. Runtime Entry Point

The C ABI entry lives in `src/cabi/api/trtf_c.cpp`.

Its responsibilities are deliberately narrow:

1. validate `bundle_path` and `TrtfPipelineOptions`
2. load the `.trtfb` bundle
3. parse runtime configuration into a canonical `BuildContext`
4. resolve the strategy family
5. invoke the matching strategy builder
6. wrap the resulting `PipelineServices` in `PipelineRouter`

The entrypoint does not own modality-specific execution logic.

## 2. Bundle Load And Context Construction

The bundle load stage gathers the inputs needed for composition:

- `BundleFile` metadata
- raw bundle sections such as `config.json` and tokenizer assets
- parsed `FastPathModelConfig`
- runtime defaults derived from metadata or config
- `hf_python` and any runtime handles supplied by options

Conceptually:

```text
bundle_path
  -> ReadBundleFile()
  -> BundleSections
  -> parse config / defaults
  -> BuildContext
```

The resulting `BuildContext` is strategy-neutral. It contains enough information for a builder to decide whether it can assemble a runtime for the bundle.

## 3. Strategy Family Resolution

`trtf_c.cpp` maps `runtime_strategy` to a strategy family.

Current families are grouped as:

- text
- encoder
- vision
- audio
- diffusion

Family dispatch happens once at bundle load. After a builder is selected, request-time calls do not re-enter strategy-family dispatch.

## 4. Strategy Builder Composition

A strategy builder performs four jobs.

### Validate

Check that the bundle contains the required sections, engine blobs, config fields, tokenizer data, and runtime prerequisites.

### Normalize Defaults

Resolve runtime defaults such as max token count, diffusion steps, thresholds, or speech limits.

### Wire Dependencies

Create the ports and adapters needed by the services, such as:

- bundle readers
- TRT runtime adapters
- media loaders or artifact writers where needed
- backend-adapter ports for service execution

### Compose Services

Populate `PipelineServices` with the exact capability services supported by the strategy.

The builder returns a typed build result rather than throwing composition details into a giant pipeline constructor.

## 5. `PipelineServices`

`PipelineServices` is the internal runtime surface.

A strategy can fill any relevant subset of these capabilities:

- text generation and encoding
- video generation
- audio generation and speech
- transcription
- embeddings and reranking
- segmentation and prompted segmentation
- detection
- scientific solve workloads

The key architectural point is that the service graph is now the runtime object model. The router is only the public facade.

## 6. `PipelineRouter`

`PipelineRouter` is the `IPipeline` implementation returned to the caller.

It owns:

- default resolution for optional arguments
- path-to-DTO conversion through input loaders
- artifact persistence through writer adapters
- compatibility with the stable public API

It does not own strategy composition and should remain thin.

## 7. Request-Time Call Flow

### Text generation

```text
IPipeline::generate(prompt)
  -> PipelineRouter
  -> ITextService
  -> service ports
  -> TRT executor
  -> text result
```

### Media tasks

```text
IPipeline::segment(image_path, output_path)
  -> PipelineRouter loads image_path
  -> ISegmentationService receives DecodedImage
  -> service ports invoke backend
  -> service returns SegmentationArtifact
  -> PipelineRouter writes output_path
```

The same pattern applies to detection, transcription, speech, and video generation.

## 8. Why This Design Is Modular

The architecture keeps responsibilities separated.

- `trtf_c.cpp` owns assembly entry only.
- Builders own composition.
- Services own task semantics.
- Router owns compatibility and IO edges.
- TRT executors own engine-bound work.
- Pure seams own planning and postprocessing logic.

That is the basis for both scalability and unit-testability.

## 9. Unit-Test Boundaries

The runtime should be tested from the inside out.

1. **Pure seams**: planner, validator, scheduler, stop-policy, and postprocess helpers.
2. **Service tests**: fake the ports and validate task semantics.
3. **Router tests**: fake media loaders and artifact writers.
4. **Builder tests**: validate composition and failure paths from bundle/config combinations.
5. **Executor harness tests**: validate bind/copy/enqueue behavior and failure injection.
6. **Smoke/E2E**: validate real-engine parity and integrated behavior.

That layered test model is what makes a 100%-coverage goal technically credible.
