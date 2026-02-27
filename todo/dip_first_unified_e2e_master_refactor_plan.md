# DIP-First Unified E2E Testing Framework Master Refactor Plan

## 1. Executive Summary

This document defines a complete refactor plan for E2E testing in this repository so that:

1. `pytest tests/test_e2e.py::test_e2e[<ModelName>]` is the single public entrypoint.
2. Every model runs through a unified orchestrator and produces `result.json` + artifacts.
3. High-level E2E policy depends only on interfaces (Dependency Inversion Principle), while model/runtime/tool specifics are pluggable adapters.
4. TRT-vs-reference validation is tolerance-based and modality-aware (no fragile exact-byte assumptions).
5. New model family onboarding is primarily configuration plus interface implementation, not custom test file creation.

This plan is designed to absorb all current and future runtime strategies (`decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `rwkv_recurrent`, `vision_language`, `segmentation`, `prompted_segmentation`, `object_detection`, `speech_to_text`, `text_to_audio`, `speech_to_speech`, `embedding`, `reranking`, `encoder_only`, `neural_operator`, `omni_multimodal`, `diffusion`, `hybrid_mamba_attention`) with minimal framework churn.

---

## 2. Why This Refactor Is Needed

### 2.1 Current System Strengths (keep these)

1. Family plugin onboarding is already strong and auto-discovered (`trtf_build/trtf_build/families/__init__.py`).
2. Runtime strategy metadata exists and is explicit (`runtime_strategy` in bundle config).
3. A partial abstraction exists in `tools/diff_framework` (`TestContext`, `DiffResult`, registry pattern).
4. Model manifests already centralize many E2E settings (`tests/e2e/models/*.json`).

### 2.2 Current System Gaps (must fix)

1. E2E logic is fragmented across multiple pytest files and command wrappers.
2. Significant behavior is controlled by `skip` branches in fixtures/tests, which obscures confidence.
3. High-level tests are tightly coupled to concrete CLI commands and script argument conventions.
4. Not all modalities use a first-class, uniform comparator contract.
5. Diff tooling interface quality is inconsistent (missing/duplicated adapter entrypoints).
6. Threshold semantics are not unified; some tests are output-heuristic only.
7. There is no single confidence model across different oracle strengths (HF backend vs torch reference vs custom reference).

### 2.3 Business/Engineering Intent

1. If E2E passes for a model, users and agents should trust build + infer flow in production-like paths.
2. New family authors should not create ad-hoc tests; they should implement declared interfaces and configuration.
3. The framework must scale to future composite/omni pipelines without architecture rewrites.

---

## 3. Design Principles and Architectural Intents

### 3.1 Dependency Inversion Principle (primary)

1. High-level E2E policy (`orchestrator`) depends on abstract contracts.
2. Low-level implementations (`trtf CLI`, `diff scripts`, HF/diffusers/torch/custom refs) implement these contracts.
3. Both high-level and low-level depend on stable interface definitions in a contracts layer.

### 3.2 Open/Closed Principle

1. Add new modality by adding strategy/comparator/reference adapters.
2. Avoid editing orchestrator core for each new family.

### 3.3 Single Responsibility Principle

1. Manifest loader validates and normalizes model case definitions.
2. Orchestrator only coordinates lifecycle.
3. Strategy runner executes TRT path for one task type.
4. Reference runner executes reference path.
5. Comparator computes and evaluates metrics.
6. Artifact sink persists outputs and result schema.

### 3.4 Interface Segregation

1. Do not force all models to implement all methods.
2. Strategy-specific interfaces for text, VL, diffusion, speech, segmentation/detection, neural operator, omni.

### 3.5 Liskov Substitution

1. Any reference backend should be substitutable if it fulfills interface and returns canonical output schema.
2. Any comparator should be swappable for same `task_strategy` contract.

### 3.6 Test-First Discipline

1. New harness behavior should be specified by contract tests before implementation.
2. Migration should preserve behavior while reducing coupling.

### 3.7 Confidence Semantics Must Be Explicit

1. Every run records `oracle_level` and `confidence_level` in `result.json`.
2. Passing with `torch_reference` is valid but not mislabeled as `hf_transformers` parity.

---

## 4. Target End-State

### 4.1 Public Developer UX

1. Single model:
   - `pytest tests/test_e2e.py::test_e2e[qwen3-0.6b]`
2. All models:
   - `pytest tests/test_e2e.py`
3. Filter by strategy:
   - `pytest tests/test_e2e.py --e2e-task-strategy vision_language_generation`
4. Output artifacts:
   - `--e2e-artifacts-dir /path/to/artifacts`

### 4.2 Functional Guarantees

1. Every model case executes deterministic lifecycle.
2. No silent behavioral skip path (preflight failures are explicit failures with artifacts).
3. Each run emits structured result and all key logs.
4. Comparator thresholds are profile-driven and per-model overridable.

### 4.3 Scalability Guarantees

1. New family with existing modality should require:
   - family plugin
   - one manifest case
   - optional threshold override
2. No new pytest file should be required.

---

## 5. Proposed Layered Architecture

```text
+--------------------------------------------------------------+
| tests/test_e2e.py (single param test)                        |
+--------------------------+-----------------------------------+
                           |
                           v
+--------------------------------------------------------------+
| E2E Orchestrator (policy only, no concrete tool commands)    |
+--------------------------+-----------------------------------+
                           |
     +---------------------+--------------------+
     |                     |                    |
     v                     v                    v
+-----------+      +---------------+      +------------------+
| Strategy  |      | Reference     |      | Comparator       |
| Runner    |      | Runner        |      | (metrics+gates)  |
+-----------+      +---------------+      +------------------+
     |                     |                    |
     +----------+----------+--------------------+
                |
                v
+--------------------------------------------------------------+
| Artifact Sink + Result Schema + Env Fingerprint              |
+--------------------------------------------------------------+
```

### 5.1 Contracts Layer (new, stable)

Create `tests/e2e_harness/contracts.py` (or shared package) containing:

1. Domain types:
   - `E2ECase`
   - `StageSpec`
   - `PreflightRequirement`
   - `ThresholdProfile`
   - `StageOutput`
   - `CompareResult`
   - `E2EResult`
2. Protocols:
   - `TaskStrategyRunner`
   - `ReferenceBackendRunner`
   - `Comparator`
   - `ArtifactSink`
   - `DeterminismPolicy`

The orchestrator imports only these interfaces.

---

## 6. Interface Contracts (Detailed)

## 6.1 Core Case Contract

```python
@dataclass
class E2ECase:
    name: str
    hf_id: str
    family: str
    runtime_strategy: str
    task_strategy: str
    reference_backend: str
    oracle_level: str
    bundle: str
    inputs: dict
    preflight: list[PreflightRequirement]
    stages: list[StageSpec]
    comparison_profile: str
    threshold_overrides: dict
    determinism: dict
    metadata: dict
```

### 6.2 Strategy Runner Contract

```python
class TaskStrategyRunner(Protocol):
    strategy_name: str
    def run_stage(self, case: E2ECase, stage: StageSpec,
                  ctx: RunContext) -> StageOutput: ...
```

### 6.3 Reference Runner Contract

```python
class ReferenceBackendRunner(Protocol):
    backend_name: str
    def run_stage(self, case: E2ECase, stage: StageSpec,
                  ctx: RunContext) -> StageOutput: ...
```

### 6.4 Comparator Contract

```python
class Comparator(Protocol):
    task_strategy: str
    def compare(self, trt: StageOutput, ref: StageOutput,
                threshold: ThresholdProfile,
                stage: StageSpec) -> CompareResult: ...
```

### 6.5 Artifact Sink Contract

```python
class ArtifactSink(Protocol):
    def log_command(self, command: list[str], rc: int,
                    stdout: str, stderr: str) -> None: ...
    def write_stage_output(self, name: str, output: StageOutput) -> None: ...
    def write_compare(self, name: str, result: CompareResult) -> None: ...
    def finalize(self, result: E2EResult) -> str: ...
```

### 6.6 Optional Family-Level E2E Descriptor Contract

Extend family plugin pattern with optional method:

```python
class SupportsE2EContract(Protocol):
    def create_e2e_contract(self, config: ModelConfig) -> dict: ...
```

Purpose:
1. Family author can provide default `task_strategy`, reference backend, and stage graph hints.
2. Manifest can override any default.

---

## 7. Task Strategy Catalog (Comprehensive)

Each model must map to one primary `task_strategy`.

1. `text_generation_causal`
2. `vision_language_generation`
3. `speech_to_text`
4. `text_to_audio`
5. `speech_to_speech`
6. `segmentation`
7. `prompted_segmentation`
8. `object_detection`
9. `diffusion_media_generation`
10. `embedding`
11. `reranking`
12. `encoder_only_nlp`
13. `neural_operator`
14. `omni_multimodal`
15. `composite_pipeline`

### 7.1 Runtime Strategy to Task Strategy Mapping

1. `decoder_kv_cache` -> `text_generation_causal` (default)
2. `decoder_moe` -> `text_generation_causal`
3. `ssm_recurrent` -> `text_generation_causal` (special runner backend)
4. `rwkv_recurrent` -> `text_generation_causal` (special runner backend)
5. `hybrid_mamba_attention` -> `text_generation_causal`
6. `vision_language` -> `vision_language_generation`
7. `speech_to_text` -> `speech_to_text`
8. `text_to_audio` -> `text_to_audio`
9. `speech_to_speech` -> `speech_to_speech`
10. `segmentation` -> `segmentation`
11. `prompted_segmentation` -> `prompted_segmentation`
12. `object_detection` -> `object_detection`
13. `embedding` -> `embedding`
14. `reranking` -> `reranking`
15. `encoder_only` -> `encoder_only_nlp`
16. `neural_operator` -> `neural_operator`
17. `diffusion` -> `diffusion_media_generation` (or `composite_pipeline`)
18. `omni_multimodal` -> `omni_multimodal` (often `composite_pipeline` internally)

---

## 8. Reference Backend Abstraction (Non-HF Compatible)

Do not assume HF Transformers/Diffusers availability for all families.

Supported `reference_backend`:

1. `hf_transformers`
2. `hf_diffusers`
3. `torch_reference`
4. `custom_python`
5. `golden_snapshot`
6. `invariant_only`

### 8.1 Oracle Level Semantics

Add `oracle_level` field to every case:

1. `L1_external_reference`: HF/diffusers/official external parity
2. `L2_internal_reference`: torch/custom reference parity
3. `L3_snapshot_regression`: trusted golden outputs
4. `L4_invariants`: metamorphic/invariant checks only

Pass report must include this level.

---

## 9. Comparator Design and Metric Contracts (Tolerance-Only)

Absolute identity is not a realistic requirement for TRT vs reference across floating-point kernels. All gating should be tolerance and behavior based.

### 9.1 Generic Comparator Rules

1. Always check shape/schema first.
2. Always check numerical health (`nan/inf`, range sanity).
3. Use composite metrics; never gate on a single metric alone.
4. Evaluate percentile thresholds (`p95`, `p99`) instead of raw max where appropriate.
5. Use stage-level and end-to-end criteria.

### 9.2 Text Generation Metrics

Required metrics:

1. `logit_cosine_p5`
2. `logit_rel_l2_p95`
3. `stable_top1_match_rate`
4. `unstable_topk_hit_rate`
5. `token_agreement_rate`
6. `normalized_text_edit_distance`

Margin-aware token stability:

1. For each step, compute HF margin `m = top1 - top2`.
2. If `m >= stable_margin`, require exact top1 match.
3. Else require TRT top1 in HF top-k.

### 9.3 Vision-Language Metrics

1. Vision embedding cosine/distance.
2. Decoder text metrics from text generation.
3. Optional semantic similarity score for caption parity.

### 9.4 Speech-to-Text Metrics

1. Decoder token agreement metrics.
2. Transcript `WER` / `CER`.
3. Timestamp and segmentation sanity if enabled.

### 9.5 Text-to-Audio Metrics

1. Codec token match (if applicable).
2. Mel-spectrogram distance.
3. Log-spectral distance.
4. Duration and RMS bounds.
5. Optional ASR transcript similarity against expected prompt intent.

### 9.6 Speech-to-Speech Metrics

1. Depth/audio token match rate.
2. Frame exact match rate.
3. Output RMS floor.
4. Optional ASR content consistency.

### 9.7 Segmentation Metrics

1. mIoU
2. Pixel accuracy
3. Boundary F-score

Canonicalization steps:

1. Ensure same class ID mapping.
2. Resolve resize method deterministically.

### 9.8 Prompted Segmentation Metrics

1. IoU per point prompt.
2. Mask rank consistency.
3. Number-of-masks consistency.

### 9.9 Detection Metrics

1. mAP@IoU thresholds
2. Class precision/recall
3. Box IoU distribution

Canonicalization steps:

1. Deterministic NMS.
2. Stable sort by (class, score desc, box coordinates).

### 9.10 Diffusion / Image / Video Generation Metrics

1. Stage-level latent trajectory parity.
2. Final-frame metrics (PSNR/SSIM/LPIPS).
3. Temporal consistency metric for video.
4. Frame-level distribution checks.

### 9.11 Embedding / Reranking Metrics

1. Embedding: cosine similarity, top-k neighborhood overlap.
2. Reranking: pairwise ordering agreement, Kendall/Spearman.

### 9.12 Neural Operator Metrics

1. Relative L2 on output fields.
2. Optional PDE residual constraints.

---

## 10. Threshold Profile System

## 10.1 Profile Files

1. `tests/e2e_harness/thresholds/defaults/<task_strategy>.yaml`
2. `tests/e2e_harness/thresholds/overrides/<model_name>.yaml`

Resolution order:

1. defaults for task
2. profile selected by CLI (e.g., `fp16_default`)
3. per-model override
4. manifest inline override

### 10.2 Calibration Workflow

1. Build fixed evaluation corpus per strategy.
2. Run across supported env matrix (GPU + driver + TRT versions).
3. Capture metric distributions.
4. Set thresholds via robust statistics (MAD/percentile bands).
5. Commit thresholds and provenance metadata.

### 10.3 Threshold Governance

1. Threshold relaxations require explicit PR annotation.
2. CI should fail if threshold is weakened without rationale field.

---

## 11. Determinism and Reproducibility Contract

Required for all cases:

1. Fixed seed in manifest.
2. Deterministic generation settings where supported.
3. Tokenizer and preprocessing version fingerprinting.
4. At least two TRT reruns with intra-run consistency metrics.

Record in artifacts:

1. CUDA/TRT/torch/transformers versions.
2. GPU model, driver, compute capability.
3. Runner config and generation parameters.

---

## 12. Composite Pipeline Support (WAN, Omni, Future)

## 12.1 Composite Stage Graph

Introduce stage graph in manifest:

```yaml
stages:
  - name: t5_encode
    required: true
  - name: dit_step
    required: true
  - name: vae_decode
    required: true
  - name: end_to_end_video
    required: true
  - name: crossover_ref_t5_trt_dit
    required: false
  - name: crossover_trt_t5_ref_dit
    required: false
```

### 12.2 WAN Validation Logic

For `wan_t2v` (`runtime_strategy=diffusion`):

1. Validate config invariants.
2. T5 output parity where available.
3. Text projection parity.
4. Timestep embedding parity.
5. Patch embedding / RoPE parity.
6. Single DiT step parity.
7. Scheduler parity.
8. End-to-end generation quality parity.

These correspond to the structure already present in `tools/debug_diffusion_pipeline.py` and should be formalized under strategy interfaces.

### 12.3 Omni Validation Logic

For `qwen3_omni` (`runtime_strategy=omni_multimodal`):

1. Thinker text decode branch parity.
2. Vision encoder branch parity.
3. Audio encoder branch parity.
4. Talker token generation parity.
5. Code2Wav waveform quality parity.
6. End-to-end multimodal scenario tests.

---

## 13. PersonaPlex and Non-HF Reference Models

PersonaPlex (`speech_to_speech`) should be first-class using `torch_reference` and explicit reference token assets.

Required checks:

1. Reference token sequence comparison.
2. Token match and frame-exact thresholds.
3. Audio RMS and waveform/spectral metrics.
4. Optional semantic transcript checks.

No dependency on HF Transformers is required for pass criteria.

---

## 14. Manifest v2 Schema (Detailed)

## 14.1 Required Fields

1. `name`
2. `hf_id` or `model_id`
3. `bundle`
4. `family`
5. `runtime_strategy`
6. `task_strategy`
7. `reference_backend`
8. `oracle_level`
9. `inputs`
10. `stages`
11. `comparison_profile`
12. `determinism`
13. `preflight_requirements`

### 14.2 Optional Fields

1. `threshold_overrides`
2. `notes`
3. `metadata`
4. `trust_remote_code`

### 14.3 Preflight Requirement Model

Examples:

1. `binary_exists`
2. `gpu_memory_min_gb`
3. `hf_auth_token_present`
4. `asset_exists(test_image)`
5. `python_module_available(diffusers)`

If unmet:

1. mark status as `PRECHECK_FAIL`
2. still produce artifacts + result file
3. fail test by default unless profile explicitly marks requirement as non-gating

### 14.4 Skip Field Policy

Current `skip` usage must be removed from test control flow.

Migration policy:

1. `skip` -> `preflight_requirements + known_limitations`.
2. Unmet requirement is explicit failure with reason, not hidden skip.

---

## 15. Unified Orchestrator Lifecycle

For each model case:

1. Initialize artifact sink and write case snapshot.
2. Run preflight checks.
3. Resolve or build bundle.
4. For each stage:
   - execute TRT strategy runner
   - execute reference backend runner
   - compare outputs
   - persist stage artifacts
5. Execute determinism reruns for designated stages.
6. Aggregate stage outcomes to final status.
7. Write final `result.json`.
8. Assert final status in pytest.

### 15.1 Failure Classification

1. `PRECHECK_FAIL`
2. `BUILD_FAIL`
3. `TRT_RUN_FAIL`
4. `REFERENCE_RUN_FAIL`
5. `COMPARE_FAIL`
6. `DETERMINISM_FAIL`
7. `ARTIFACT_WRITE_FAIL`

---

## 16. Artifact and Result Contract

## 16.1 Directory Layout

`<artifacts_dir>/<model_name>/`

1. `result.json`
2. `case.json`
3. `env_fingerprint.json`
4. `commands.json`
5. `stages/<stage>/trt_output.*`
6. `stages/<stage>/ref_output.*`
7. `stages/<stage>/compare.json`
8. `logs/*.stdout`
9. `logs/*.stderr`

### 16.2 `result.json` Core Fields

1. `model`
2. `runtime_strategy`
3. `task_strategy`
4. `reference_backend`
5. `oracle_level`
6. `preflight`
7. `stages` with per-stage metrics and pass/fail
8. `determinism`
9. `status`
10. `failure_type`
11. `timestamp`
12. `env_fingerprint`

---

## 17. Codebase Refactor Plan (Phased)

## 17.1 Phase 0: Stabilization and Cleanup

Goals:

1. Fix adapter inconsistencies in diff tooling.
2. Establish baseline before structural changes.

Tasks:

1. Add `run_as_diff_test(ctx)` to `tools/diff_logits.py`.
2. Remove duplicate definitions in:
   - `tools/test_runner_parity.py`
   - `tools/perf_compare.py`
3. Add tests ensuring every registered diff check has exactly one callable adapter.

Deliverables:

1. Green tests for `tools/diff_framework` registry integrity.

### 17.2 Phase 1: Contracts and Single Entrypoint Skeleton

Goals:

1. Introduce contracts layer and orchestrator skeleton.
2. Keep legacy tests intact.

Tasks:

1. Add new modules:
   - `tests/e2e_harness/contracts.py`
   - `tests/e2e_harness/orchestrator.py`
   - `tests/e2e_harness/manifest_loader.py`
   - `tests/e2e_harness/artifact_sink.py`
   - `tests/e2e_harness/result_schema.py`
2. Add `tests/test_e2e.py` with parameterized `test_e2e[ModelName]`.
3. Implement minimal no-op strategy to exercise lifecycle and artifact writing.

Deliverables:

1. Contract tests for schema + lifecycle pass.

### 17.3 Phase 2: Text Strategy Migration

Goals:

1. Move decoder/encoder text-style models to new harness.

Tasks:

1. Implement `text_generation_causal` strategy runner.
2. Implement `hf_transformers` reference runner.
3. Implement text comparator with composite metrics.
4. Wire thresholds and determinism policy.

Deliverables:

1. Legacy text tests and new harness produce aligned outcomes.

### 17.4 Phase 3: Vision-Language Migration

Goals:

1. Replace VL-specific pytest modules with strategy-based execution.

Tasks:

1. Implement `vision_language_generation` strategy runner.
2. Reuse/adapter from `diff_vl` logic through interface, not shell parsing.
3. Implement VL comparator and required artifacts.

Deliverables:

1. All VL manifests run under single entrypoint.

### 17.5 Phase 4: Diffusion and Composite Pipeline

Goals:

1. First-class staged pipeline support for WAN and similar models.

Tasks:

1. Implement `diffusion_media_generation` + `composite_pipeline` strategy support.
2. Stage graph execution and stage-level comparators.
3. Integrate `debug_diffusion_pipeline` logic as internal stage adapters.

Deliverables:

1. WAN runs through unified orchestrator with stage and end-to-end reports.

### 17.6 Phase 5: Audio and Speech

Goals:

1. Integrate `text_to_audio`, `speech_to_text`, `speech_to_speech`.

Tasks:

1. Implement reference runners:
   - HF where available
   - torch/custom for PersonaPlex-like models
2. Implement audio/speech comparators and thresholds.
3. Ensure assets and token references are managed through manifest contract.

Deliverables:

1. Bark, Whisper, PersonaPlex all run under same entrypoint.

### 17.7 Phase 6: Segmentation/Prompted Segmentation/Detection

Goals:

1. Standardize image-structured output comparators and canonicalization.

Tasks:

1. Implement per-task canonicalization helpers.
2. Implement mIoU/mAP-based gating.
3. Remove ad-hoc detector/segmenter branching from old files.

Deliverables:

1. SegFormer, SAM, and object detection families integrated.

### 17.8 Phase 7: Embedding/Reranking/Encoder-Only/Neural Operator/Omni

Goals:

1. Complete modality coverage from runtime strategies.

Tasks:

1. Add strategy runners and comparators for:
   - embedding
   - reranking
   - encoder-only
   - neural operator
   - omni multimodal
2. Add initial manifests where missing.

Deliverables:

1. Full runtime strategy coverage in harness.

### 17.9 Phase 8: Manifest v2 Migration

Goals:

1. Move all manifests to v2 schema.

Tasks:

1. Write migrator tool `scripts/migrate_e2e_manifest_v2.py`.
2. Convert `skip` semantics to preflight constraints.
3. Add schema validation test gate.

Deliverables:

1. All cases parse and validate under v2 schema.

### 17.10 Phase 9: Legacy Test Decommission

Goals:

1. Remove redundant pytest modules after burn-in.

Tasks:

1. Keep wrappers for one release cycle.
2. Collect parity evidence.
3. Delete old fragmented E2E files.

Deliverables:

1. Only `tests/test_e2e.py` remains as primary E2E entrypoint.

---

## 18. CI/CD and Governance Plan

## 18.1 CI Jobs

1. PR changed-cases job:
   - run only affected models and strategy dependencies.
2. Nightly full-matrix job:
   - run all manifests.
3. Weekly calibration job:
   - produce threshold update report.

### 18.2 Required Artifacts in CI

1. Publish all model artifact folders.
2. Include compact summary table and failure classification.

### 18.3 Merge Gates

1. New family cannot merge without at least one passing E2E case in unified harness.
2. Threshold relaxations require reviewer-approved rationale.
3. Schema changes require migration note and compatibility tests.

---

## 19. Onboarding Workflow for New Model Family Authors

### 19.1 Author Steps

1. Implement family plugin (existing flow).
2. Provide optional `create_e2e_contract()` defaults.
3. Add one manifest v2 case.
4. Add threshold override only if required.
5. Run:
   - `pytest tests/test_e2e.py::test_e2e[<ModelName>]`

### 19.2 Framework Guarantees to Author

1. No need to create new pytest file.
2. No need to hand-wire subprocess commands.
3. Uniform result format and artifact outputs.

### 19.3 Scaffold Tooling Enhancement

Update `scripts/new_family.py` to optionally generate:

1. manifest v2 template
2. default threshold profile stub
3. optional `create_e2e_contract()` method stub

---

## 20. Risk Register and Mitigations

### 20.1 Risk: Overengineering Before Stabilizing Basics

Mitigation:

1. Phase 0 fixes and contract tests first.
2. Strategy-by-strategy migration with feature flags.

### 20.2 Risk: Threshold Drift and Flaky Gates

Mitigation:

1. Profile/versioned thresholds with calibration process.
2. Percentile-based thresholds + robust stats.

### 20.3 Risk: Unsupported References for Some Models

Mitigation:

1. Multiple `reference_backend` types.
2. Explicit `oracle_level` semantics.

### 20.4 Risk: Composite Pipelines Too Complex

Mitigation:

1. Stage graph abstraction.
2. Cross-over isolation tests for root-cause localization.

### 20.5 Risk: Migration Disruption

Mitigation:

1. Parallel-run legacy and new harness during transition.
2. Keep wrapper compatibility until parity confidence is demonstrated.

---

## 21. Definition of Done

The refactor is complete when all are true:

1. `pytest tests/test_e2e.py` is the canonical E2E entrypoint.
2. Every model case emits `result.json` + complete artifacts.
3. No silent behavioral skip paths remain.
4. All runtime strategies map to supported task strategies.
5. WAN-style composite and PersonaPlex-style non-HF references are first-class.
6. New family onboarding is config + interface implementation, no new E2E test files.
7. CI gates and threshold governance are active.
8. Documentation and scaffolding reflect the new architecture.

---

## 22. Immediate Execution Backlog (Actionable)

## 22.1 Week 1

1. Fix `run_as_diff_test` inconsistencies.
2. Add contracts and orchestrator skeleton.
3. Introduce `tests/test_e2e.py` entrypoint.
4. Implement artifact sink and `result.json` schema.

### 22.2 Week 2

1. Migrate text and VL strategies.
2. Add manifest v2 parser and validation tests.
3. Add threshold profile files and loading logic.

### 22.3 Week 3

1. Migrate diffusion composite strategy (WAN).
2. Migrate speech/audio strategies (Bark/Whisper/PersonaPlex).

### 22.4 Week 4

1. Migrate segmentation/detection.
2. Add embedding/reranking/encoder-only coverage.
3. Enable CI gate for unified test entrypoint.

---

## 23. Expected Confidence Outcome After Completion

After implementing this plan, confidence is materially stronger because:

1. Every model follows one deterministic and auditable E2E lifecycle.
2. Confidence claims are tied to explicit oracle strength, not implied.
3. Modalities are validated with task-appropriate metrics, not generic heuristics.
4. Adding new families does not fragment the test framework.
5. All outputs are standardized for humans and agents to inspect quickly and reliably.

This is the core condition needed to "walk away with confidence" that current and future model families can fit and scale in one testing framework.
