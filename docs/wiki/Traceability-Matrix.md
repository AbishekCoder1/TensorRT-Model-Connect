# Traceability Matrix

Defines repository policy for **bi-directional traceability** across:

`Architecture contracts -> Unit design -> Unit tests + Integration tests`

This complements (does not replace) `tests/runtime_strategy_matrix.yaml`:
- `runtime_strategy_matrix.yaml` is machine-checked strategy parity.
- This page is human-maintained intent/design/test traceability.

---

## Mandatory Test Intent Fields

Every test added or modified in this repository must document all three fields:

| Field | Required content |
|------|------------------|
| Intent | The behavior or contract being validated |
| Preconditions | Assumptions/setup needed for the test to be valid |
| Postconditions | Observable outcomes/invariants guaranteed by passing assertions |

Required placement:
- Python: test function/class docstring.
- C++: comment block above the scenario's `check(...)` assertions.

A test is traceability-incomplete if any of the three fields is missing.

---

## Trace ID Scheme

Use stable IDs so rows remain valid as files move:

| Prefix | Layer | Example |
|--------|-------|---------|
| `ARCH-*` | Architecture contract/capability | `ARCH-RT-001` |
| `UD-*` | Unit design element(s) | `UD-FAC-TEXT-001` |
| `UT-*` | Unit test evidence | `UT-TOOLS-STRAT-001` |
| `IT-*` | Integration/E2E evidence | `IT-E2E-QWEN3-001` |

Guidance:
- One `ARCH-*` may map to multiple `UD-*` entries.
- Each `UD-*` must map to at least one `UT-*` and one `IT-*`.
- Every `UT-*`/`IT-*` must point back to exactly one primary `ARCH-*` (plus optional secondary IDs).

---

## Matrix Format

Use this table format when adding or updating trace rows.

| ARCH ID | Architecture contract | UD ID(s) + design units | UT evidence | IT evidence | Evidence artifact | Owner | Status |
|---------|------------------------|--------------------------|-------------|-------------|------------------|-------|--------|
| `ARCH-...` | Contract statement | `UD-...` + concrete files/classes | `UT-...` + test file(s) | `IT-...` + manifest/runner/comparator | command + artifact path/date | team | draft/active/verified |

Minimum completeness rules for each row:
1. Architecture contract is testable (contains a verifiable outcome).
2. Unit design anchors cite concrete implementation points.
3. Unit test evidence names at least one deterministic unit-level assertion source.
4. Integration evidence names at least one E2E path (manifest + runner/comparator chain).

---

## Repository-Specific Example Rows

These examples are tailored to the current builder-based runtime, runners, and comparators in this repository.

| ARCH ID | Architecture contract | UD ID(s) + design units | UT evidence | IT evidence | Evidence artifact | Owner | Status |
|---------|------------------------|--------------------------|-------------|-------------|------------------|-------|--------|
| `ARCH-RT-001` | Text strategies build through the text strategy builder and route through `PipelineRouter` (`decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention`) | `UD-RT-06-TEXT`: `src/runtime/builders/text/text_strategy_builder.cpp`; `UD-RT-05`: `src/runtime/pipeline/router.cpp`; `UD-RUN-TEXT-001`: `tests/e2e_harness/runners/text_generation.py`; `UD-CMP-TEXT-001`: `tests/e2e_harness/comparators/text.py` | `UT-TOOLS-STRAT-001`: `tests/tools/test_runtime_strategy_matrix_checker.py`; `UT-CPP-CABI-001`: `tests/cpp/test_c_abi_entry.cpp`; `UT-BLD-TEXT-001`: `tests/cpp/test_text_strategy_builder.cpp` | `IT-E2E-QWEN3-001`: `tests/e2e/models/qwen3-0.6b.json` via `tests/test_e2e.py` | `pytest tests/tools/test_runtime_strategy_matrix_checker.py -v`; `pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v ...` | Runtime + E2E | active |
| `ARCH-RT-002` | `vision_language` builds through the vision strategy builder and validates outputs with the VL comparator | `UD-RT-06-VISION`: `src/runtime/builders/vision/vision_strategy_builder.cpp`; `UD-RT-05`: `src/runtime/pipeline/router.cpp`; `UD-RUN-VL-001`: `tests/e2e_harness/runners/vision_language.py`; `UD-CMP-VL-001`: `tests/e2e_harness/comparators/vision_language.py` | `UT-TOOLS-STRAT-002`: `tests/tools/test_runtime_strategy_matrix_checker.py`; `UT-BLD-VISION-001`: `tests/cpp/test_vision_strategy_builder.cpp`; `UT-BUILDER-MANIFEST-001`: `tests/builder/test_manifest_validation.py` | `IT-E2E-QWEN3VL-001`: `tests/e2e/models/qwen3-vl-2b.json`; `IT-E2E-QWEN25VL-001`: `tests/e2e/models/qwen25vl-3b.json` | `pytest tests/test_e2e.py::test_e2e[qwen3-vl-2b] -v ...` | Runtime + Vision | active |
| `ARCH-RT-003` | `diffusion` builds through the diffusion strategy builder and dedicated diffusion services | `UD-CABI-DIFF-001`: `src/cabi/api/trtf_c.cpp`; `UD-RT-06-DIFF`: `src/runtime/builders/diffusion/diffusion_strategy_builder.cpp`; `UD-RUN-DIFF-001`: `tests/e2e_harness/runners/diffusion.py`; `UD-CMP-DIFF-001`: `tests/e2e_harness/comparators/diffusion.py` | `UT-TOOLS-STRAT-003`: `tests/tools/test_runtime_strategy_matrix_checker.py`; `UT-BLD-DIFF-001`: `tests/cpp/test_diffusion_strategy_builder.cpp`; `UT-CPP-FASTPATH-001`: `tests/cpp/test_fast_path_config.cpp` | `IT-E2E-WAN21-001`: `tests/e2e/models/wan21-t2v-1.3b.json`; `IT-E2E-FLUX-001`: `tests/e2e/models/flux-schnell.json` | `pytest tests/test_e2e.py::test_e2e[wan21-t2v-1.3b] -v ...` | Runtime + Diffusion | active |
| `ARCH-RT-004` | Segmentation and prompted segmentation strategies build through the vision builder and IO adapters, then validate outputs with segmentation comparators | `UD-RT-06-VISION`: `src/runtime/builders/vision/vision_strategy_builder.cpp`; `UD-RT-02`: `src/runtime/adapters/io/image_io_adapter.cpp`; `UD-RUN-SEG-001`: `tests/e2e_harness/runners/segmentation.py`; `UD-CMP-SEG-001`: `tests/e2e_harness/comparators/segmentation.py` | `UT-TOOLS-STRAT-004`: `tests/tools/test_runtime_strategy_matrix_checker.py`; `UT-BLD-VISION-001`: `tests/cpp/test_vision_strategy_builder.cpp`; `UT-TOOLS-SEG-001`: `tests/tools/test_diff_segmentation.py` | `IT-E2E-SEGFORMER-001`: `tests/e2e/models/segformer-b0-ade.json`; `IT-E2E-SAM-001`: `tests/e2e/models/sam-vit-base.json` | `pytest tests/test_e2e.py --e2e-task-strategy segmentation -v ...` | Runtime + Vision | active |
| `ARCH-RT-005` | Audio/speech strategies (`speech_to_text`, `text_to_audio`, `speech_to_speech`) stay aligned across runtime strategy mapping, builder dispatch, and comparator thresholds | `UD-RT-06-AUDIO`: `src/runtime/builders/audio/audio_strategy_builder.cpp`; `UD-RT-02`: `src/runtime/adapters/io/audio_io_adapter.cpp`; `UD-RUN-AUDIO-001`: `tests/e2e_harness/runners/audio_speech.py`; `UD-CMP-AUDIO-001`: `tests/e2e_harness/comparators/text_to_audio.py`, `tests/e2e_harness/comparators/speech_to_text.py`, `tests/e2e_harness/comparators/speech_to_speech.py` | `UT-TOOLS-STRAT-005`: `tests/tools/test_runtime_strategy_matrix_checker.py`; `UT-BLD-AUDIO-001`: `tests/cpp/test_audio_strategy_builder.cpp`; `UT-BLD-AUDIO-VAL-001`: `tests/cpp/test_audio_bundle_validation.cpp`; `UT-TOOLS-AUDIO-001`: `tests/tools/test_diff_audio.py` | `IT-E2E-WHISPER-001`: `tests/e2e/models/whisper-tiny.json`; `IT-E2E-BARK-001`: `tests/e2e/models/bark-small.json`; `IT-E2E-PERSONAPLEX-001`: `tests/e2e/models/personaplex-7b.json` | `pytest tests/test_e2e.py --e2e-task-strategy speech_to_text -v ...` | Runtime + Audio | active |

---

## Maintenance Workflow (Team Process)

Apply this process for any architecture/design/test change:

1. Update or add the affected `ARCH-*` and `UD-*` rows in this matrix.
2. Update tests to include Intent + Preconditions + Postconditions + Trace IDs.
3. Run the mapped `UT-*` and `IT-*` checks.
4. Record evidence command(s), artifact location(s), and verification date in the row or PR notes.
5. During review, verify both directions:
   - Top-down coverage: architecture contract has design and test evidence.
   - Bottom-up coverage: changed tests map back to architecture intent.

Definition of done for traceability:
- No orphan architecture contracts (without tests).
- No orphan tests (without architecture/design contract).
- Matrix row status can be marked `verified` only after unit + integration evidence is available.
