# Runtime Modularity Investigation and Migration Risk Register

This file is a historical migration record. References to legacy runtime
components describe risks that existed during the refactor and are not the live
architecture. Use `docs/wiki/` for the current runtime design.

Date: 2026-03-04  
Scope: C++ runtime modularity/scalability audit, with focus on strategy plugin architecture and migration risk.  
Method: parallel multi-agent repo investigation + direct source validation.

## Executive Summary

The current runtime is functional for many paths, but it does not scale cleanly as strategy count grows. The main blockers are:

1. Centralized orchestration in `src/cabi/api/trtf_c.cpp` (2,915 LOC, broad include and dispatch surface).
2. Runtime strategy coupling across dispatch, config parsing, bundle section parsing, CLI contracts, and docs.
3. High-risk correctness issues (notably null dereference in `PipelineImpl` vision checks).
4. Drift between advertised capabilities and implemented/tested behavior.

The safest path is an ABI-preserving migration:

1. Stabilize correctness and contract gaps first.
2. Introduce a registry-driven plugin layer under existing C ABI.
3. Migrate strategies incrementally behind compatibility gates.
4. Enforce a runtime-strategy test matrix in CI.

Validation has been run for the current implementation wave (see status below), with targeted compile and test coverage for the new registry, matrix checker, and diff-framework hardening.

## Implementation Status (2026-03-04)

### Completed in `runtime-refactor`

1. **Phase 0 stabilization items (partial-to-substantial)**:
   - Null-safety and pipeline option handling landed for C ABI creation paths and vision checks.
   - CLI/runner contract alignment landed for detection aliases and neural-operator / omni harness invocation shape.
2. **Phase 1 registry introduction (substantial)**:
   - Added `BackendRegistry` core and tests.
   - `try_create_from_bundle()` now dispatches through registry first.
   - Registry wrappers now cover all non-diffusion strategies:
     - `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `rwkv_recurrent`, `segmentation`,
       `speech_to_text`, `vision_language`, `encoder_only`, `embedding`, `reranking`,
       `text_to_audio`, `hybrid_mamba_attention`, `object_detection`, `prompted_segmentation`,
       `neural_operator`, `omni_multimodal`, `speech_to_speech`.
3. **Phase 4 governance items (substantial)**:
   - Added `tests/runtime_strategy_matrix.yaml`.
   - Added `tools/check_runtime_strategy_matrix.py` with parity checks against C++ dispatch, contracts mapping, runner/comparator declarations, and diff-framework registrations.
   - Added checker unit tests.
   - Hardened diff-framework strategy detection to use explicit statuses (`ok`/`warning`/`skip`/`error`) and remove silent unknown-strategy fallback behavior.
4. **Docs drift remediation (partial)**:
   - Updated architecture wiki pages to remove stale claims around `FastPathModelConfig`, `trtf_c.cpp` scale, and unconditional “Python-only” plugin guidance.

### Validation Snapshot

1. C++ compile checks:
   - `src/cabi/api/trtf_c.cpp` with `TRTF_HAS_TRT=0`: pass.
   - `src/cabi/registry/backend_registry.cpp`: pass.
   - `examples/trtf_cli.cpp`: pass.
2. C++ tests:
   - `tests/cpp/test_cli_args.cpp`: pass.
   - `tests/cpp/test_backend_registry.cpp`: pass.
3. Python checks/tests (targeted):
   - `python3 tools/check_runtime_strategy_matrix.py`: pass.
   - `pytest tests/tools/test_runtime_strategy_matrix_checker.py`: pass.
   - `pytest tests/tools/test_diff_framework.py`: pass.
   - `pytest tests/tools/test_e2e_runner_cli_alignment.py`: pass.
4. Known environment limitations in this shell:
   - Full `tests/tools` sweep fails on missing optional deps (`tensorrt`, `transformers`) not introduced by this change set.

## Validated Findings

### P0 (must fix before structural refactor)

1. `PipelineImpl` null-deref risk in vision methods.
   - `generate(prompt, image_path, ...)` dereferences `mBackend` before null check: `src/cabi/api/trtf_c.cpp:197`.
   - `supports_vision()` dereferences `mBackend` unconditionally: `src/cabi/api/trtf_c.cpp:259-261`.
   - Multiple pipelines are intentionally created with `mBackend == nullptr`, for example:
     - `speech_to_text`: `src/cabi/api/trtf_c.cpp:1273-1275`
     - `segmentation`: `src/cabi/api/trtf_c.cpp:1375-1377`
     - `object_detection`: `src/cabi/api/trtf_c.cpp:1398-1400`
     - `neural_operator`: `src/cabi/api/trtf_c.cpp:1475-1477`
     - `text_to_audio` variants: `src/cabi/api/trtf_c.cpp:1635-1637`, `1843-1845`
     - `encoder_only`: `src/cabi/api/trtf_c.cpp:1878-1880`
     - `embedding`: `src/cabi/api/trtf_c.cpp:1967-1969`
     - `reranking`: `src/cabi/api/trtf_c.cpp:2004-2006`
     - `omni_multimodal`: `src/cabi/api/trtf_c.cpp:2123-2125`
     - `speech_to_speech`: `src/cabi/api/trtf_c.cpp:2617-2619`
   - CLI path can trigger this through `pipeline->supports_vision()` in `run` command when `--image` is passed: `examples/trtf_cli.cpp:389`.

2. Advertised strategies with stub backends.
   - Detection backend is explicitly stubbed and factory returns `nullptr`:
     - `src/runtime/trt/detection_backend.h:2`, `35-39`.
   - Neural operator backend is explicitly stubbed and factory returns `nullptr`:
     - `src/runtime/trt/neural_operator_backend.h:2`, `36-40`.
   - Dispatcher still routes to these factories:
     - `object_detection`: `src/cabi/api/trtf_c.cpp:2741-2745`
     - `neural_operator`: `src/cabi/api/trtf_c.cpp:2755-2759`
   - Result: commands may be documented but not constructible in practice.

### P1 (high architectural drag / scaling blockers)

1. Monolithic dispatch and orchestration hotspot.
   - One translation unit includes nearly every backend header: `src/cabi/api/trtf_c.cpp:8-33`.
   - Global strategy dispatch in one large function: `try_create_from_bundle` at `src/cabi/api/trtf_c.cpp:2625-2826`.
   - `PipelineImpl` mixes multi-modality execution logic and many backend-specific members (`mSegBackend`, `mWhisperBackend`, `mOmniBackend`, etc.): class starts at `src/cabi/api/trtf_c.cpp:133`.
   - This is the dominant merge conflict surface as strategy count increases.

2. Layering inversion (`runtime` depends on `cabi` config header).
   - Runtime headers include `cabi/config/fast_path_config.h`, for example:
     - `src/runtime/trt/whisper_backend.h:7`
     - `src/runtime/trt/embedding_backend.h:6`
     - `src/runtime/trt/diffusion_backend.h:5`
     - `src/runtime/trt/vl_backend.h:8`
     - `src/runtime/trt/speech_backend.h:8`
   - This inverts intended layering and makes runtime plugins less isolated.

3. Config parsing is text-search based and order-sensitive.
   - JSON helpers rely on raw string search (`text.find` patterns): `src/utils/json_helpers.cpp:11-37`, `91-124`.
   - Builder has explicit ordering workaround because parser picks first textual key occurrence:
     - `trtf_build/trtf_build/engine_builder.py:409-427`.
   - This is brittle for nested configs and future plugin metadata.

4. Strategy-specific config and bundle parsing are centralized.
   - `FastPathModelConfig` accumulates fields for many modalities: `src/cabi/config/fast_path_config.h:9-225`.
   - Parser contains large strategy-conditional blocks: `src/cabi/config/fast_path_config.cpp:88-390`.
   - Bundle section discovery is a large name-switch function:
     - `src/cabi/bundle/bundle_helpers.cpp:14-92`.
   - Adding a new strategy currently requires edits in multiple shared files.

5. ABI options drift / partially unused API surface.
   - `TrtfPipelineOptions` defines `max_new_tokens`, `hf_python`, `image_path`: `include/trtf/pipeline.h:198-202`.
   - `trtf_create_pipeline_ex` ignores `max_new_tokens`: `src/cabi/api/trtf_c.cpp:2853`.
   - `options->image_path` is not consumed in `trtf_create_pipeline_ex` construction path.

6. Detection threshold argument ignored.
   - `detect(..., conf_threshold)` discards threshold value: `src/cabi/api/trtf_c.cpp:890`.

7. Dead local utilities in core runtime TU.
   - `shell_quote` and `run_subprocess` are defined but unused:
     - `src/cabi/api/trtf_c.cpp:72-90`, `97-131`.

### P2 (contract and quality drift)

1. CLI and harness contract mismatches.
   - Object detection runner uses unsupported flags:
     - runner: `--output-json`, `--score-threshold` in `tests/e2e_harness/runners/object_detection.py:97-99`
     - CLI expects `--output`, `--threshold`: `examples/trtf_cli.cpp:56-60`, `173-183`, `209-218`
   - Neural operator runner invokes `run` with unsupported flags:
     - `tests/e2e_harness/runners/neural_operator.py:33`, `38`, `47`
   - Omni runner uses `run --stage ...`; `--stage` is not parsed by CLI:
     - runner: `tests/e2e_harness/runners/omni.py:45`
     - unknown flags rejected in parser: `examples/trtf_cli.cpp:335-339`

2. Test coverage drift.
   - `test_cli_args.cpp` reimplements a reduced parser (only `run`/`inspect`) rather than testing production parser behavior:
     - `tests/cpp/test_cli_args.cpp:15-19`, `86-87`.
   - Neural operator config test is a stub:
     - `tests/cpp/test_neural_operator_config.cpp:1-2`.
   - E2E manifests do not currently include runtime strategies such as `object_detection`, `neural_operator`, or `omni_multimodal` (58 manifests audited).

3. Docs vs code drift (selected).
   - `Architecture-Overview` says `FastPathModelConfig` is "Not needed":
     - `docs/wiki/Architecture-Overview.md:169`.
   - `Source-Layout` says `trtf_c.cpp` is "~200 lines":
     - `docs/wiki/Source-Layout.md:99`.
   - `Adding-a-Model-Family` says "Python-only task, no C++ changes":
     - `docs/wiki/Adding-a-Model-Family.md:3`,
     - but later notes SSM needs C++ state management: `docs/wiki/Adding-a-Model-Family.md:23`.
   - `Architecture-Extensibility-Assessment` still marks MLA/hybrid as "Not yet implemented":
     - `docs/wiki/Architecture-Extensibility-Assessment.md:19-20`,
     - while runtime has hybrid dispatch/backend paths: `src/cabi/api/trtf_c.cpp:2817-2821`, `src/runtime/trt/hybrid_backend.cpp`.
   - README family list is stale relative to actual plugin files:
     - README snapshot list at `README.md:157-159`,
     - current family directory has 46 plugins (`trtf_build/trtf_build/families/`).

## Target Runtime Architecture (ABI-Preserving)

### Design constraints

1. Keep C ABI stable: `include/trtf/pipeline.h` virtual surface and exported C functions remain unchanged.
2. Keep bundle contract stable: `runtime_strategy` in `config.json` remains strategy key.
3. Support incremental migration with mixed old/new strategy paths.
4. Avoid shared-file edit requirement for each new backend plugin.

### Proposed module boundaries

| Module | Responsibility | Notes |
|---|---|---|
| `src/cabi/` | ABI facade and error handling only | `trtf_create_pipeline_ex`, thread-local error, API compatibility |
| `src/runtime/core/` | plugin interface + registry + factory context | no modality logic |
| `src/runtime/plugins/<strategy>/` | one plugin per strategy | owns strategy-specific config validation + pipeline assembly |
| `src/runtime/common/` | shared TRT/cuda primitives | cache, stream, engine helpers |
| `src/bundle/` | format IO + section catalog access | no strategy conditionals |
| `src/config/` | structured config parsing | strategy-specific parse delegated to plugins |

### Minimal plugin interfaces (conceptual)

```cpp
struct StrategyFactoryContext {
  BundleSectionsView sections;
  FastPathModelConfig cfg;
  std::string model_id;
  std::string hf_python;
  TrtRuntimeHandles trt;
};

class IRuntimeStrategyPlugin {
public:
  virtual ~IRuntimeStrategyPlugin() = default;
  virtual std::string_view strategy_key() const = 0;
  virtual CapabilityMask capabilities() const = 0;
  virtual bool requires_engine_plan() const = 0;
  virtual std::unique_ptr<trtf::IPipeline> create(StrategyFactoryContext&&) const = 0;
};
```

Key behavior:

1. `trtf_create_pipeline_ex` parses bundle header and config once.
2. Registry resolves plugin by `runtime_strategy`.
3. Plugin owns modality-specific construction.
4. `PipelineImpl` becomes a thin facade over a strategy-owned runtime object.

## Migration Plan With Gates

### Phase 0: Stabilization (before modularization)

Goals:

1. Fix null-deref in `generate(image)` and `supports_vision`.
2. Make detection/neural-operator status explicit (implemented or hard-disabled with clear errors).
3. Align CLI flags and harness runner invocations.
4. Decide behavior for `TrtfPipelineOptions.max_new_tokens` and `image_path` (implement or deprecate).

Exit criteria:

1. No crash path on `supports_vision` for non-generation pipelines.
2. CLI + harness command contracts are aligned for `detect`, `solve`, and omni paths.
3. CI includes at least one test per currently advertised command path.

### Phase 1: Registry introduction (no behavior change)

Goals:

1. Add `BackendRegistry` and plugin descriptors.
2. Route `try_create_from_bundle` through registry while keeping legacy fallback.
3. Move each existing `create_*_pipeline` into isolated factory units.

Exit criteria:

1. Existing bundles load with identical runtime behavior.
2. Strategy resolution logs plugin key and backend name.
3. ABI tests pass unchanged.

### Phase 2: Plugin extraction per strategy

Goals:

1. Move registry-dispatch implementation out of `src/cabi/api/trtf_c.cpp` and into dedicated dispatch/registry units.
2. Extract at least one low-risk vision factory target from `src/cabi/api/trtf_c.cpp` if that extraction lands in this phase.
3. Preserve C ABI surface and legacy fallback behavior while extraction proceeds.

Exit criteria:

1. Pipeline creation no longer depends on registry-dispatch logic embedded directly in `src/cabi/api/trtf_c.cpp`.
2. If a low-risk vision factory extraction is merged in this phase, that strategy path is created through the extracted factory path without changing user-facing behavior.
3. ABI remains unchanged (`include/trtf/pipeline.h` + exported C entrypoints), and registry miss/error paths retain current fallback behavior.

### Phase 3: Config + bundle hardening

Goals:

1. Replace string-find parsing with structured JSON parsing.
2. Replace monolithic `BundleSections` hardcoded fields with map/view descriptors.
3. Let plugins declare required config keys and section names.

Exit criteria:

1. Parser compatibility tests against existing bundles pass.
2. Strategy plugin can define config/section requirements without touching central parser switch.
3. Builder no longer needs key-order hacks to satisfy runtime parser behavior.

### Phase 4: Coverage and governance for scale

Goals:

1. Introduce runtime strategy coverage matrix (strategy x command x tests).
2. Enforce matrix in CI when strategy list changes.
3. Fix docs to represent true runtime/plugin architecture.

Exit criteria:

1. Any new strategy without test/docs registration fails CI.
2. Diff framework behavior for unsupported strategy is explicit (no silent fallback to decoder).
3. Wiki and README strategy/family tables are generated or auto-verified.

## Risk Register

| ID | Category | Probability | Impact | Risk | Leading Indicators | Mitigation | Rollback Trigger |
|---|---|---|---|---|---|---|---|
| R1 | Correctness | High | Critical | Null deref in vision checks on non-generation pipelines | Crash on `run --image` with non-vision strategy | Add null guards and capability checks before backend deref | Any reproducible segfault in command smoke tests |
| R2 | Product | High | High | Stub detection/neural_operator paths advertised as available | Build succeeds but pipeline creation fails for these strategies | Mark unsupported until backend exists; align docs/tests | >1 user-visible failure for documented strategy commands |
| R3 | Architecture | High | High | Centralized dispatch remains merge hotspot | Frequent conflicts in `trtf_c.cpp` on feature branches | Introduce registry and plugin factories | Registry migration causes strategy load regressions |
| R4 | ABI | Medium | Critical | Refactor accidentally changes `IPipeline` ABI/vtable | `test_pipeline_api` or external consumers break | Freeze ABI; adapt internals behind facade only | Any ABI size/symbol change in release candidate |
| R5 | Layering | Medium | High | Runtime stays coupled to cabi config types | Runtime plugins must edit cabi headers | Move shared config contracts into runtime/core | Plugin work still requires `cabi/*` edits |
| R6 | Config | Medium-High | High | String-based parser misreads nested/duplicate keys | Bundle-specific parsing anomalies, order-sensitive behavior | Structured parser + compatibility tests | Mismatch against existing bundle parse outputs |
| R7 | Bundle Extensibility | Medium | High | Hardcoded section switch slows plugin growth | Every new section requires central helper edits | Map-based section view + plugin descriptors | Section lookup regressions in existing strategies |
| R8 | Contract Drift | High | High | CLI and harness diverge as strategies evolve | Harness passes unsupported flags, CLI rejects | Single source-of-truth command schema | Repeated e2e runner failures due to arg mismatch |
| R9 | Testing | High | High | No mandatory per-strategy test registration | New strategy lands with no targeted tests | Strategy matrix in CI + test ownership | Post-merge break in untested strategy path |
| R10 | Performance | Medium | High | Refactor introduces hidden buffer churn/extra sync | Latency or memory regressions in decode loops | Preserve shared kernels/helpers and benchmark each phase | >10% regression in agreed baseline |
| R11 | Docs/Onboarding | High | Medium | Stale docs cause wrong implementation choices | Contributors follow obsolete guidance | Auto-validate docs lists, update architecture pages | Repeated onboarding issues tied to doc inaccuracies |
| R12 | Delivery | Medium | Medium | Big-bang rewrite destabilizes runtime | Long-lived branch, delayed integration | Phased migration with compatibility layers | Phase misses gate criteria twice |

## Test and CI Gating Recommendations

1. Add a `runtime_strategy_matrix.yaml` with, per strategy:
   - CLI command(s)
   - required C++ unit tests
   - required e2e harness runner/comparator
   - diff framework checks or explicit exemption.
2. CI check: strategy added in runtime dispatch or plugin registry must exist in matrix.
3. CI check: docs strategy list and matrix list must match.
4. Remove silent default strategy fallback where possible; emit explicit unsupported errors.

## Documentation Remediation Backlog (Priority)

1. Correct architecture claims around `FastPathModelConfig` and runtime strategy dispatch.
2. Update source layout facts (file sizes, strategy addition workflow).
3. Reconcile "Python-only" claims with runtime-strategy cases that require C++ work.
4. Refresh supported family list from the actual plugin directory automatically.

## Immediate Next Actions (Recommended)

1. Ship P0 fixes (null-deref + strategy stubs + CLI/harness contract alignment).
2. Land registry skeleton with one migrated strategy (`decoder_kv_cache`) behind compatibility mode.
3. Add matrix gate and minimum smoke coverage for every documented CLI command.
4. Update architecture/docs in the same milestone as Phase 1 to prevent further drift.
