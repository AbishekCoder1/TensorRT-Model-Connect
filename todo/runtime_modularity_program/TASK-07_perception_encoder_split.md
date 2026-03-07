# TASK-07: Perception And Encoder Runtime Split

## Objective

Finish decomposition of perception and encoder runtime code so direct backend coverage is realistic.

## Own These Files

- `src/runtime/trt/perception/detection_backend.*`
- `src/runtime/trt/perception/segmentation_backend.*`
- `src/runtime/trt/perception/sam_backend.*`
- `src/runtime/trt/perception/neural_operator_backend.*`
- `src/runtime/trt/perception/*.h` seam files if needed
- `src/runtime/trt/encoder/encoder_backend.*`
- `src/runtime/trt/encoder/embedding_backend.*`
- `src/runtime/trt/encoder/reranking_backend.*`
- related tests under `tests/cpp/`

## Do Not Edit

- `src/runtime/trt/multimodal/*`
- `src/runtime/services/common/runtime_service_ports.*`

## Deliverables

1. Move remaining preprocess/postprocess and result formatting logic out of backend executors where needed.
2. Add harness tests for direct backend success/error paths.
3. Close obvious zero-coverage gaps in encoder/perception runtime code.

## Acceptance Criteria

- perception/encoder backends have direct tests beyond builder/service coverage
- backend files are not doing avoidable file/media or artifact work
- error handling is exercised in unit/harness tests

## Required Validation

- targeted perception/encoder `ctest`
- full `ctest --output-on-failure`
- CCM gate
