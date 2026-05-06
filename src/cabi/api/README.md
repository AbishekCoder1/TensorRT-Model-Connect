# C API Entrypoint

`trtmc_c.cpp` implements exported C API functions and wires API calls to the runtime pipeline.

Focus areas:
- Handle validation and error mapping
- Bundle open/close lifecycle
- Pipeline creation and command dispatch
- Registry-based runtime composition via `PipelineFactory::from_bundle` (strategy → `IPipelinePlugin` → concrete `IPipeline`)
