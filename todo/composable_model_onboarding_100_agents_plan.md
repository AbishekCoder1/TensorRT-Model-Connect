## Composable Model-Onboarding Architecture for 100 Parallel Agents

### Summary
Design and implement a fully composable onboarding architecture where each new model family can be added without touching shared core files in normal cases, while validation and orchestration scale to 100 concurrent agents with deterministic, reproducible outcomes.

Primary outcomes:
- Model additions are **plugin-local** (builder/runtime/validation declarations live with the plugin).
- Shared core is **stable and narrow** (registry + contracts only).
- Validation is **centralized and cache-backed** (GPU work is scheduled once, consumed many times).
- Orchestration state is **sharded and atomic**, removing global lock contention.

---

## 1. Target Architecture

### 1.1 Extension Model (Plugin-First, Core-Minimal)
Introduce a strict plugin contract with explicit extension points:
- `matches(config) -> bool`
- `weight_loader() -> WeightLoader`
- `builder() -> FamilyBuilder`
- `runtime_adapter() -> RuntimeAdapterSpec`
- `validation_profile() -> ValidationProfile`

Each family provides these in its own module; core discovers and composes them.
Core no longer requires family-specific branches in shared files for standard onboarding.

### 1.2 Build/Runtime Composition
Split current shared builder/runtime logic into layered interfaces:

- **Layer A: Core Contracts** (stable)
  - `FamilyPlugin`, `FamilyBuilder`, `RuntimeAdapterSpec`, `ValidationProfile`
- **Layer B: Reusable Blocks** (shared primitives)
  - attention/rope/mlp/norm/cache ops
- **Layer C: Family Implementations** (plugin-local composition)
  - family-specific assembly and overrides

### 1.3 Validation as a Service
Create a centralized validation scheduler:
- Agents submit `ValidationRequest(bundle_hash, family, profile, env_requirements)`.
- Scheduler deduplicates by hash/profile and executes once.
- Results stored as immutable `ValidationArtifact` (pass/fail + metrics + logs + environment fingerprint).
- Agents consume artifacts instead of rerunning heavy GPU steps.

### 1.4 Sharded Orchestration State
Replace single shared backlog/state files with:
- `state/tasks/<task_id>.json`
- `state/runs/<run_id>.json`
- append-only event log `state/events/<date>.ndjson`
Use atomic writes + optimistic versioning (`etag`/`version`) per task file.

---

## 2. Required API / Interface Additions

### 2.1 Python (`trtf_build`) Contracts
Add new types in `trtf_build/trtf_build/families/base.py`:

- `class RuntimeAdapterSpec(TypedDict):`
  - `strategy: Literal["decoder_kv_cache","decoder_moe","ssm_recurrent","diffusion","vl"]`
  - `required_tensors: list[str]`
  - `optional_tensors: list[str]`
  - `state_schema: dict[str, Any]`

- `class ValidationProfile(TypedDict):`
  - `tiers: list[Literal["tier1","tier2","tier3","tier4"]]`
  - `gates: list[Literal["diff_logits","diff_layers","runner_parity","e2e"]]`
  - `resource_class: Literal["cpu","single_gpu","high_mem_gpu"]`
  - `timeouts_sec: dict[str, int]`
  - `acceptance_thresholds: dict[str, float]`

- `class FamilyBuilder(Protocol):`
  - `build_engines(model_dir, config, out_dir, build_context) -> BuildManifest`

- `class FamilyPlugin(Protocol):`
  - existing methods + required:
    - `runtime_adapter(self, cfg) -> RuntimeAdapterSpec`
    - `validation_profile(self, cfg) -> ValidationProfile`

### 2.2 Build Manifest
Add immutable manifest emitted per build:
- `bundle_hash`
- `engine_hashes`
- `config_hash`
- `family`
- `runtime_strategy`
- `toolchain_fingerprint` (cuda/trt/python/commit)

### 2.3 Validation Service API
Internal API (CLI first, service-ready):
- `validate submit --bundle <path> --profile <json>`
- `validate status --request-id <id>`
- `validate fetch --request-id <id> --out <dir>`

Result schema:
- `request_id`, `dedupe_key`, `state`
- `artifacts[]` (logs, metrics, json reports)
- `pass_fail_by_gate`
- `environment_fingerprint`

### 2.4 Runtime Dispatch Boundary (C++)
Keep runtime strategy dispatch stable and table-driven:
- map `runtime_strategy` -> backend factory
- avoid family-specific logic in dispatch implementation

---

## 3. Concrete Refactor Plan (Decision-Complete)

### Phase 0: Freeze & Baseline
1. Define and publish “core files allowed to change” list.
2. Measure baseline:
   - merge conflict rate on hotspot files
   - median onboarding lead time
   - duplicate GPU validation runs per bundle

### Phase 1: Contract Extraction
1. Add new contract types (`RuntimeAdapterSpec`, `ValidationProfile`, `FamilyBuilder`).
2. Adapt existing families to contract wrappers without behavior changes.
3. Add contract lint:
   - every plugin must expose adapter + validation profile.
   - fail fast if missing.

### Phase 2: Builder/Runtime Decoupling
1. Move family-specific assembly out of shared builder branches into plugin-local builders.
2. Convert `engine_builder.py` to a coordinator that only:
   - resolves plugin
   - invokes plugin builder
   - writes build manifest
3. Replace hardcoded strategy assumptions with adapter-driven strategy mapping.

### Phase 3: Validation Scheduler + Artifact Cache
1. Implement local queue worker (single process) with dedupe key:
   - `sha256(bundle_hash + validation_profile + toolchain_fingerprint)`.
2. Add artifact store layout:
   - `.validation_store/<dedupe_key>/...`
3. Integrate test entrypoints to query scheduler first; run only if miss.
4. Update `validate_family.sh` and e2e helpers to use per-run isolated temp dirs always.

### Phase 4: Sharded Orchestration State
1. Replace single backlog file with per-task files.
2. Add atomic write helper:
   - write temp + fsync + rename.
3. Add optimistic concurrency control (`version` field).
4. Provide migration script from old backlog to sharded store.

### Phase 5: Data-Driven Docs/Registry
1. Generate supported-model docs from structured manifests.
2. Stop manual edits to shared tables/worklogs for routine onboarding.
3. Add “release notes generator” from validation artifacts + merged manifests.

### Phase 6: Scale Trial
1. Run controlled simulation:
   - 20 concurrent agents (week 1), 50 (week 2), 100 (week 3).
2. Gate scale-up on SLOs (defined below).

---

## 4. Testing and Acceptance Criteria

### 4.1 Contract Tests
- Plugin contract completeness test:
  - all discovered plugins provide required methods and valid schemas.
- Adapter strategy compatibility test:
  - adapter-declared tensors match runtime backend requirements.

### 4.2 Determinism Tests
- Same bundle/profile validated twice yields identical pass/fail and metric envelope.
- Dedupe works:
  - second request with same key must be cache hit (no GPU rerun).

### 4.3 Concurrency Tests
- 100 parallel task state updates with no lost updates/corruption.
- 100 parallel validation submissions with dedupe and queue fairness.

### 4.4 Integration Tests
- Add a synthetic new family using only plugin-local files; ensure no core edits required.
- End-to-end onboarding path:
  - build -> submit validation -> consume artifact -> merge gate pass.

### 4.5 SLOs to declare success
- <5% onboarding PRs touch core hotspot files.
- >80% validation requests served from dedupe cache during burst windows.
- 0 state corruption incidents under 100-agent concurrency test.
- P95 onboarding validation wait time within configured resource-class target.

---

## 5. Rollout / Migration

1. **Compatibility mode (2 releases):**
   - support both legacy and new plugin contracts via adapter shim.
2. **Deprecation:**
   - warn on legacy path usage.
3. **Cutover:**
   - enforce new contract in CI.
4. **Cleanup:**
   - remove legacy branches and old backlog path.

---

## 6. Risks and Mitigations

- Risk: Contract too strict for exotic families.
  - Mitigation: `capabilities` field + optional extension hooks with schema validation.
- Risk: Validation scheduler becomes SPOF.
  - Mitigation: artifact store + resumable workers + idempotent request IDs.
- Risk: Early migration churn.
  - Mitigation: compatibility shim and incremental family migration batches.

---

## 7. Assumptions and Defaults

- Default strategy: plugin-first onboarding must not require touching shared core.
- Default validation execution: centralized queue with dedupe cache.
- Default state backend: sharded JSON files with atomic writes (SQLite acceptable as later swap if needed).
- Default temp/artifact policy: per-run isolated directories; no shared mutable `/tmp` paths.
- Default merge gate: validation artifact presence + pass status is required for model manifest merge.
