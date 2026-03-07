# TASK-02: Bark And Omni Executor Split

## Objective

Finish decomposition of Bark and Omni audio runtime code so stage logic is unit-testable without full backend bring-up.

## Own These Files

- `src/runtime/trt/audio/bark_backend.cpp`
- `src/runtime/trt/audio/bark_backend.h`
- `src/runtime/trt/audio/bark_generation_plan.h`
- `src/runtime/trt/audio/omni_backend.cpp`
- `src/runtime/trt/audio/omni_backend.h`
- `src/runtime/trt/audio/omni_audio_plan.h`
- `tests/cpp/test_bark_generation_plan.cpp`
- `tests/cpp/test_omni_audio_plan.cpp`
- any new Bark/Omni-specific harness tests

## Do Not Edit

- `speech_*`
- `whisper_*`
- `magpie_*`
- shared service ports

## Deliverables

1. Split Bark into semantic/coarse/fine/codec planning and thin executor stages.
2. Split Omni into audio encode/decode planning, talker/thinker stage contracts, and thin executor stages.
3. Add harness tests for stage-level failure paths where pure seams are not enough.

## Acceptance Criteria

- Bark and Omni policy is no longer buried in long executor methods
- stage transitions and output shaping are directly testable
- new tests cover both expected and defensive behavior

## Required Validation

- targeted `ctest` for Bark/Omni tests
- full `ctest --output-on-failure`
- CCM gate
