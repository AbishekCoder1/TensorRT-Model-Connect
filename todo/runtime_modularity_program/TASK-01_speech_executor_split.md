# TASK-01: Speech Executor Split

## Objective

Make `SpeechToSpeechBackend` individually testable by separating planning, tensor mapping, executor shells, and waveform/result assembly.

## Own These Files

- `src/runtime/trt/audio/speech_backend.cpp`
- `src/runtime/trt/audio/speech_backend.h`
- `src/runtime/trt/audio/speech_*.h`
- `tests/cpp/test_speech_*.cpp`
- `tests/cpp/test_speech_generation_helpers.cpp`

## Do Not Edit

- `bark_*`
- `omni_*`
- `whisper_*`
- `magpie_*`
- `runtime_service_ports.*`

## Deliverables

1. Extract any remaining planning/state logic out of `speech_backend.cpp`.
2. Introduce a thin executor shell for each TRT-bound stage that still mixes bind/copy/enqueue with policy.
3. Add direct tests for:
   - normal execution planning
   - stop/limit behavior
   - shape/contract validation
   - executor failure paths via fakes or harnesses
4. Leave `speech_backend.cpp` as orchestration glue only.

## Acceptance Criteria

- executor methods are visibly thinner than the current baseline
- planning and output assembly logic live in dedicated seam modules
- success and failure branches are covered by unit/harness tests
- CCM remains within gate

## Required Validation

- targeted `ctest` for all speech tests
- full `ctest --output-on-failure`
- `python tools/check_cyclomatic_complexity.py src --max-ccn 10`

## Test Intent Rule

Each new test must say what speech behavior it verifies, under what preconditions, and what postcondition proves the behavior.
