# TASK-10: Coverage Tooling And Fresh Report

## Objective

Make the C++ coverage flow stable and produce the fresh low-coverage report that drives the final cleanup wave.

## Own These Files

- `tools/coverage/*`
- `docs/coverage/*`
- coverage-related tool tests under `tests/tools/`
- coverage output documentation in docs if needed

## Do Not Edit

- runtime source files except when required to fix instrumentation/build issues

## Deliverables

1. Make `cpp_coverage.sh` reliable in the container and CI build graph.
2. Document the exact invocation for local and CI use.
3. Produce a fresh report ranking the remaining low-coverage runtime files by line/function/branch impact.
4. Provide machine-readable outputs that make `TASK-11` data-driven.

## Acceptance Criteria

- full coverage run completes without manual gcovr workarounds
- report clearly lists top low-coverage files and uncovered branches/functions
- docs explain how to reproduce the report in the dev container

## Required Validation

- run the full coverage script in the matching container
- run coverage-related tool tests
