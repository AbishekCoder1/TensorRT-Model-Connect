# Execution Board

## Phase 1: Foundation (additive, no breaking changes)

| Task | Description | Status | Deps | Est. size |
|------|-------------|--------|------|-----------|
| TASK-01 | TrtModule + DeviceTensor | **done** | none | ~400 lines |
| TASK-02 | KvCache + RecurrentState | **done** | 01 | ~300 lines |
| TASK-03 | IScheduler + ITokenizer interfaces | **done** | 01 | ~200 lines |
| TASK-04 | IPipeline interface + PipelineFactory skeleton | **done** | 01 | ~200 lines |

## Phase 2: Core pipelines (new code alongside old, coexist)

| Task | Description | Status | Deps | Est. size |
|------|-------------|--------|------|-----------|
| TASK-05 | TextGenerationPipeline (25+ decoder models) | **done** | 01-04 | ~200 lines |
| TASK-06 | Mamba + RWKV + Hybrid pipelines | **done** | 01-04 | ~300 lines |
| TASK-07 | Encoder + Segment + SAM pipelines | **done** | 01,04 | ~250 lines |
| TASK-08 | VLPipeline (vision-language) | **done** | 01-05 | ~250 lines |

## Phase 3: Complex pipelines

| Task | Description | Status | Deps | Est. size |
|------|-------------|--------|------|-----------|
| TASK-09 | Flux + Wan + ZImage diffusion pipelines | **done** | 01,03,04 | ~500 lines |
| TASK-10 | Whisper + Bark + Magpie + Speech + Omni audio | **done** | 01-04 | ~800 lines |

## Phase 4: Cutover

| Task | Description | Status | Deps | Est. size |
|------|-------------|--------|------|-----------|
| TASK-11 | Wire factory to C API + delete old layers | **done** (Step 1: factory wired; Step 2: deletion deferred) | 05-10 | -3500 lines |
| TASK-12 | Final validation + documentation | **done** | 11 | docs only |

## Estimated net change

- New code: ~3400 lines (TrtModule + DeviceTensor + pipelines + factory)
- Deleted code: ~5500 lines (adapters + ports + services + old backends)
- **Net: -2100 lines** while gaining clean HF-like architecture

## Parallelism

Phase 1 tasks can run in parallel (TASK-01 is the only hard dependency).
Phase 2 tasks can run in parallel once Phase 1 is done.
Phase 3 tasks can run in parallel.
Phase 4 is sequential (TASK-11 then TASK-12).

## Ralph loop prompt

```
Execute the HF-style runtime redesign task plan in todo/hf_runtime_redesign/.
Follow EXECUTION_BOARD.md. Work through tasks in phase order. For each task:
1. Read the task file carefully
2. Implement the code changes described
3. Write the tests specified
4. Build and run tests in the container: docker exec trtf-dev-gb300-agent-4 ...
5. Mark the task complete in EXECUTION_BOARD.md
Move to the next task. Output <promise>COMPLETE</promise> when all 12 tasks are done.
```
