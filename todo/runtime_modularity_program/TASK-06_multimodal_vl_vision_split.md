# TASK-06: Multimodal VL And Vision Engine Split

## Objective

Make the VL runtime and vision-engine path directly testable by isolating prefill/decode policy, deepstack planning, and engine execution sequencing.

## Own These Files

- `src/runtime/trt/multimodal/image_preprocessor.cpp`
- `src/runtime/trt/multimodal/image_preprocessor.h`
- `src/runtime/trt/multimodal/vision_engine.cpp`
- `src/runtime/trt/multimodal/vision_engine.h`
- `src/runtime/trt/multimodal/vision_execution_plan.h`
- `src/runtime/trt/multimodal/vl_backend.cpp`
- `src/runtime/trt/multimodal/vl_backend.h`
- `src/runtime/trt/multimodal/vl_decode_policy.h`
- `tests/cpp/test_image_preprocessor.cpp`
- `tests/cpp/test_vision_execution_plan.cpp`
- `tests/cpp/test_vl_decode_policy.cpp`
- any new multimodal harness tests

## Do Not Edit

- perception backends
- encoder backends
- shared service ports

## Deliverables

1. Isolate image-prep and deepstack planning from engine execution.
2. Keep `vision_engine.cpp` focused on bind/copy/enqueue/sync only.
3. Keep `vl_backend.cpp` focused on orchestration over pure decode/prefill policy.
4. Add harness coverage for vision engine error paths.

## Acceptance Criteria

- multimodal planning is direct-unit-test friendly
- vision engine sequencing is testable without full model bring-up
- new tests cover edge and error cases

## Required Validation

- targeted multimodal `ctest`
- full `ctest --output-on-failure`
- CCM gate
