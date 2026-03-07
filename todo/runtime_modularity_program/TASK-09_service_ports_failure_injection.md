# TASK-09: Shared Service Ports And Failure Injection

## Objective

Standardize the fakeable executor/session interfaces used by the service layer so backend error branches can be covered deterministically.

## Own These Files

- `src/runtime/services/common/runtime_service_ports.cpp`
- `src/runtime/services/common/runtime_service_ports.h`
- `src/runtime/services/text/*`
- `src/runtime/services/audio/*`
- `src/runtime/services/vision/*`
- `tests/cpp/test_runtime_service_ports.cpp`
- service-layer tests that need failure injection

## Dependencies

Start after Wave 1 lands enough executor-shell patterns to normalize.

## Deliverables

1. Introduce or normalize executor-facing interfaces for:
   - bind failure
   - copy failure
   - enqueue failure
   - sync failure
   - missing tensor
   - shape mismatch
2. Update service tests to use deterministic failures instead of relying on real TRT/CUDA errors.
3. Keep service logic free of concrete backend class checks.

## Acceptance Criteria

- service-layer error handling is deterministic in tests
- shared ports support the failure modes needed by runtime backends
- no service uses concrete backend RTTI or ad-hoc checks where a port can be used instead

## Required Validation

- targeted service/port `ctest`
- full `ctest --output-on-failure`
- CCM gate
