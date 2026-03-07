# Execution Board

## Program Status

| Task | Lane | Wave | Status | Depends On | Owner Files |
|------|------|------|--------|------------|-------------|
| `TASK-01` | Audio | 1 | `ready` | — | `src/runtime/trt/audio/speech_*`, `speech_backend.*`, speech tests |
| `TASK-02` | Audio | 1 | `ready` | — | `bark_*`, `omni_*`, related tests |
| `TASK-03` | Audio | 1 | `ready` | — | `whisper_*`, `magpie_*`, related tests |
| `TASK-04` | Diffusion | 1 | `ready` | — | `flux_diffusion_backend.*`, flux seam/tests |
| `TASK-05` | Diffusion | 1 | `ready` | — | `wan_*`, `z_image_*`, related seam/tests |
| `TASK-06` | Multimodal | 1 | `ready` | — | `vision_engine.*`, `vl_backend.*`, `image_preprocessor.*`, related tests |
| `TASK-07` | Perception/Encoder | 1 | `ready` | — | `src/runtime/trt/perception/*`, `src/runtime/trt/encoder/*`, related tests |
| `TASK-08` | Recurrent | 1 | `ready` | — | `src/runtime/trt/recurrent/*`, related tests |
| `TASK-09` | Shared Ports | 2 | `blocked` | `TASK-01`..`TASK-08` patterns | `runtime_service_ports.*`, service harness/tests |
| `TASK-10` | Coverage Infra | 1 | `ready` | — | `tools/coverage/*`, coverage docs/tests |
| `TASK-11` | Coverage Sweep | 3 | `blocked` | `TASK-10` + fresh report | low-coverage seam/tests across runtime |
| `TASK-12` | Final Parity/Gates | 4 | `blocked` | all prior tasks | CI/docs/final reports |

## Suggested Agent Allocation

| Agent Slot | Task |
|------------|------|
| Agent A | `TASK-01` |
| Agent B | `TASK-02` |
| Agent C | `TASK-03` |
| Agent D | `TASK-04` |
| Agent E | `TASK-05` |
| Agent F | `TASK-06` |
| Agent G | `TASK-07` |
| Agent H | `TASK-08` |
| Agent I | `TASK-10` |
| Integrator | `TASK-09`, then `TASK-11`, then `TASK-12` |

## Integration Order

1. Land Wave 1 tasks independently.
2. Normalize shared executor failure-injection and service ports in Wave 2.
3. Re-run full coverage and drive the cleanup from the report in Wave 3.
4. Re-enable strict gates and finalize parity in Wave 4.
