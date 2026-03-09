# TASK-11: Wire factory to C API + delete old layers

## Status: blocked (needs TASK-05 through TASK-10)
## Phase: 4 (Cutover)
## Risk: high — this is the switch-over commit. Must pass ALL E2E tests.

## Goal

Wire `PipelineFactory::from_bundle()` into `trtf_c.cpp` as the primary creation
path. Then delete all old abstraction layers that are no longer needed.

## Step 1: Wire factory (coexist with old path)

```cpp
// trtf_c.cpp — try new factory first, fall back to old path
TrtfPipeline trtf_create_pipeline_ex(const char* path, const TrtfPipelineOptions* opts) {
    PipelineOptions po = convert_options(opts);

    // New path
    auto pipeline = PipelineFactory::from_bundle(path, po);
    if (pipeline) {
        return wrap(std::move(pipeline));
    }

    // Old path (fallback during migration)
    return try_create_with_old_runtime(path, opts);
}
```

Run full E2E suite. Every model must produce identical output.

## Step 2: Delete old layers (one commit per layer)

Delete in order (each must still compile + pass tests):

1. **Delete service wrappers** (~2000 lines)
   - `src/runtime/services/` — entire directory
   - `include/trtf/runtime/contracts/services.h`

2. **Delete port/adapter sandwich** (~1600 lines)
   - `src/runtime/services/common/runtime_service_ports.h` + `.cpp`
   - `src/runtime/adapters/bundle/bundle_port_adapter.*`
   - `src/runtime/adapters/trt/trt_port_adapter.*`
   - `include/trtf/runtime/ports/bundle_port.h`
   - `include/trtf/runtime/ports/trt_port.h`

3. **Delete strategy builders** (~3700 lines)
   - `src/runtime/builders/` — entire directory
   - `include/trtf/runtime/builders/` — headers

4. **Delete old backends** (after pipelines fully replace them)
   - `src/runtime/trt/core/trt_backend_shared.*` — replaced by TextGenerationPipeline
   - `src/runtime/trt/recurrent/mamba_backend.*` — replaced by MambaPipeline
   - etc.

5. **Delete old pipeline router** (~550 lines)
   - `src/runtime/pipeline/router.cpp`
   - `include/trtf/runtime/pipeline/router.h`

6. **Delete IO adapters** (~350 lines)
   - `src/runtime/adapters/io/` — inline into pipelines or use direct structs

## Step 3: Clean up CMakeLists.txt

Remove all deleted source files from `add_library(trtf_core ...)`.
Add new pipeline source files.

## Acceptance criteria

- [ ] ALL 50 E2E model tests pass
- [ ] C++ unit tests pass (ctest)
- [ ] Binary size reduced (target: < 3.0MB from current 3.9MB)
- [ ] `wc -l src/**/*.cpp src/**/*.h` shows significant line reduction
- [ ] No references to deleted files remain

## Dependencies

ALL previous tasks (TASK-01 through TASK-10)
