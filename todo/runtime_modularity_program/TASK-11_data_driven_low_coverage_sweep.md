# TASK-11: Data-Driven Low-Coverage Sweep

## Objective

After the major executor refactors and fresh coverage report, close the remaining cheap gaps from measured data rather than intuition.

## Own These Files

- targeted low-coverage seam files identified by the fresh report
- targeted test files under `tests/cpp/`

## Dependencies

- `TASK-10` complete
- a fresh report from the current branch after Wave 1 and Wave 2 integration

## Deliverables

1. Triage the top remaining low-coverage helpers, adapters, and small runtime files.
2. Add focused unit tests for uncovered defensive branches and result-formatting paths.
3. Avoid broad refactors here; this is a cleanup wave, not another architecture rewrite.

## Acceptance Criteria

- every change is directly tied to a measured coverage gap
- top remaining misses shrink materially after the sweep
- no task expands into unrelated backend redesign

## Required Validation

- fresh coverage rerun showing improvement
- targeted and full `ctest`
- CCM gate
