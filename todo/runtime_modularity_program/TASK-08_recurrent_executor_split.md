# TASK-08: Recurrent Executor Split

## Objective

Make the recurrent runtime family directly testable at the executor layer.

## Own These Files

- `src/runtime/trt/recurrent/hybrid_backend.*`
- `src/runtime/trt/recurrent/mamba_backend.*`
- `src/runtime/trt/recurrent/mamba_decode_runtime.*`
- `src/runtime/trt/recurrent/mamba_step_state.*`
- `src/runtime/trt/recurrent/rwkv_backend.*`
- `src/runtime/trt/recurrent/rwkv_decode_runtime.*`
- `src/runtime/trt/recurrent/rwkv_step_state.*`
- `src/runtime/trt/recurrent/recurrent_*.h`
- related tests under `tests/cpp/`

## Do Not Edit

- text builders
- shared service ports

## Deliverables

1. Separate state contracts, tensor bindings, executor sequencing, and token/result interpretation.
2. Add harness tests for recurrent bind/copy/enqueue/state-update failure paths.
3. Keep backend wrappers thin over the recurrent executor helpers.

## Acceptance Criteria

- recurrent backends are thin wrappers over testable runtime helpers
- success and failure paths are directly covered
- no regression in current recurrent smoke/unit tests

## Required Validation

- targeted recurrent `ctest`
- full `ctest --output-on-failure`
- CCM gate
