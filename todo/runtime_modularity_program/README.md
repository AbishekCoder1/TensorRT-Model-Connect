# Runtime Modularity Program

## Goal

Finish the remaining C++ runtime refactor so the codebase is modular, individually testable, and structurally capable of reaching `100%` line / function / branch coverage.

This program assumes the high-level architecture is already chosen and live:

```text
trtf_c.cpp
  -> strategy-family builder
  -> PipelineServices
  -> PipelineRouter
  -> ports / adapters
  -> TRT executors
```

The remaining work is lower-layer decomposition and coverage buildout.

## Start State

Current known state before this program:

- one live runtime creation path
- legacy runtime deleted from the active code path
- DTO-based core service contracts are in place
- router owns compatibility and IO edges
- many planning and postprocess seams already exist
- C++ CCN gate passes with max `10`
- latest measured C++ coverage snapshot used for planning:
  - line: `38.2%`
  - function: `52.4%`
  - branch: `22.8%`

Main remaining low-coverage pressure points:

- `src/runtime/trt/audio/speech_backend.cpp`
- `src/runtime/trt/audio/bark_backend.cpp`
- `src/runtime/trt/audio/magpie_tts_backend.cpp`
- `src/runtime/trt/audio/omni_backend.cpp`
- `src/runtime/trt/audio/whisper_backend.cpp`
- `src/runtime/trt/diffusion/flux_diffusion_backend.cpp`
- `src/runtime/trt/diffusion/wan_diffusion_backend.cpp`
- `src/runtime/trt/diffusion/z_image_diffusion_backend.cpp`
- `src/runtime/trt/multimodal/vision_engine.cpp`
- `src/runtime/trt/multimodal/vl_backend.cpp`
- `src/runtime/trt/encoder/*`
- `src/runtime/trt/perception/*`
- `src/runtime/trt/recurrent/*`

## End State

The runtime is in good shape when all of the following are true:

1. Every heavy backend follows the same internal shape:
   - planner / validator
   - tensor mapper
   - executor shell
   - postprocessor / result assembler
2. Core services depend on fakeable ports, not concrete backend types.
3. Router owns file/media translation and artifact writing.
4. Executor shells are thin enough that failure injection tests cover bind/copy/enqueue/sync/error branches directly.
5. Low-level policy, scheduling, stop rules, and shape logic live in direct unit-test seams.
6. Full coverage runs are data-driven and reproducible in CI/container.
7. The repository can credibly enforce strict coverage gates again.

## Program Lanes

### Lane A: Audio Executors

Goal: finish decomposition of speech, Bark, Magpie, Omni, and Whisper runtime code.

### Lane B: Diffusion Executors

Goal: split FLUX, Wan, and Z-Image runtime logic into testable planners, mappers, executor shells, and result assemblers.

### Lane C: Multimodal / Perception / Encoder Executors

Goal: make vision, VL, SAM, segmentation, detection, embedding, encoder, and rerank paths directly testable.

### Lane D: Recurrent Executors

Goal: finish Mamba, RWKV, and hybrid runtime decomposition and harness coverage.

### Lane E: Shared Ports / Failure Injection

Goal: make executor error paths deterministic and unit-testable across the runtime.

### Lane F: Coverage Tooling / Data Sweep

Goal: make coverage reporting stable, then use real coverage data to drive the final cleanup wave.

## Parallel Execution Model

Recommended parallel wave allocation:

- Wave 1: `TASK-01`, `TASK-02`, `TASK-03`, `TASK-04`, `TASK-05`, `TASK-06`, `TASK-07`, `TASK-08`, `TASK-10`
- Wave 2: `TASK-09` after the first executor-shell patterns have landed
- Wave 3: `TASK-11` after a fresh coverage report exists
- Wave 4: `TASK-12` for final parity, coverage gate raise, and closeout

The key rule is simple: one owner per file group. Agents should not overlap on the same runtime subtree.

## Definition Of Done

This program is complete when:

1. The executor-heavy runtime files have been decomposed into testable modules.
2. Each remaining backend family has direct unit or harness coverage for success and failure paths.
3. Coverage tooling produces stable line/function/branch reports in the container and CI.
4. The final coverage report shows that the runtime is close enough to `100%` that the remaining misses are small, explicit, and cheap to close.
5. Regression, smoke, and L0/L1 gates still pass.

## How To Use This Folder

1. Read [AGENT_RULES.md](AGENT_RULES.md).
2. Pick one task file only.
3. Own only the files listed in that task.
4. Update [EXECUTION_BOARD.md](EXECUTION_BOARD.md) when a task is started or completed.
5. Do not expand the scope of a task by editing unrelated subtrees.
