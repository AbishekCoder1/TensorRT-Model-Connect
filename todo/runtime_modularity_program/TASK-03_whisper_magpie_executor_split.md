# TASK-03: Whisper And Magpie Executor Split

## Objective

Complete the remaining decomposition of Whisper and Magpie runtime execution into testable units.

## Own These Files

- `src/runtime/trt/audio/whisper_backend.cpp`
- `src/runtime/trt/audio/whisper_backend.h`
- `src/runtime/trt/audio/whisper_*.h`
- `src/runtime/trt/audio/magpie_tts_backend.cpp`
- `src/runtime/trt/audio/magpie_tts_backend.h`
- `src/runtime/trt/audio/magpie_*.h`
- `tests/cpp/test_whisper_*.cpp`
- `tests/cpp/test_magpie_*.cpp`

## Do Not Edit

- `speech_*`
- `bark_*`
- `omni_*`
- shared service ports

## Deliverables

1. Whisper decode loop and cross-K/V handling decomposed into pure policies plus thin execution shells.
2. Magpie text completion, codec staging, and decode loop decisions extracted out of executor-heavy methods.
3. Add harness tests for missing tensor, bad shape, bind failure, enqueue failure, and output interpretation paths.

## Acceptance Criteria

- backend files become orchestration-heavy rather than policy-heavy
- direct tests exist for decode-policy and executor-shell behavior
- no regression in targeted smoke/unit tests

## Required Validation

- targeted `ctest` for Whisper/Magpie tests
- full `ctest --output-on-failure`
- CCM gate
