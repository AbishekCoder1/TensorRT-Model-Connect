# Autonomous Model Onboarding: Architecture and Developer Guide

This document explains the autonomous multi-agent flow for adding Hugging Face models into this repository with minimal manual work while keeping `master` stable.

## 1. What Developers Actually Do

For normal usage, developers only do three actions:

1. Submit HF links.
2. Start or monitor the orchestration loop.
3. Review failures only when a task is marked `failed`.

Everything else is automated: task decomposition, subagent execution, validation, merge gating.

## 2. One-Time Machine Setup

### 2.1 Start container and choose repo path

Use the active development container and a clean checkout for merge operations.

```bash
docker ps
```

Set convenience variables:

```bash
export CONTAINER=trtf-dev-gb300
export REPO_IN_CONTAINER=/workspace/trt-transformers-cpp
```

Optional starter file:

```bash
source agent/config/env.production.sh
```

### 2.2 Configure subagent execution command

`agent/worker.py` calls `agent/run_subagent.sh`, which requires `AGENT_SUBAGENT_CMD_TEMPLATE`.

Required placeholders:
- `{prompt_file}`: rendered task prompt.
- `{worktree}`: isolated git worktree path for that model branch.
- `{task_json}`: full task JSON payload as inline string.
- `{task_json_file}`: path to a temp file containing task JSON.
- `{worker_id}`: scheduler/worker identifier.

Recommended template:

```bash
export AGENT_SUBAGENT_CMD_TEMPLATE='codex exec -C {worktree} --prompt-file {prompt_file}'
```

### 2.3 Configure strict parity hooks (non-decoder modalities)

Decoder/MoE/SSM parity is built-in. For other modalities, strict parity is enforced via hook templates.

| Env Var | Used For | Required placeholders |
|---|---|---|
| `TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE` | encoder-only parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE` | encoder-decoder parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE` | vision encoder parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE` | audio encoder parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE` | diffusion frame/image parity | `{model}` `{bundle}` `{output_dir}` `{binary}` `{hf_python}` `{prompt}` |
| `TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE` | video encoder parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE` | video diffusion parity | `{model}` `{bundle}` `{output_dir}` `{binary}` `{hf_python}` `{prompt}` |
| `TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE` | time-series parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE` | RL policy parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE` | RL value parity | `{model}` `{bundle}` `{binary}` `{hf_python}` |
| `TRTF_VL_IMAGE` | real image used by VL parity | file path |

If these hooks are missing for their modality, validation intentionally fails to avoid false green merges.

Example encoder hook:

```bash
export TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 <your_encoder_parity_script>.py --model {model} --bundle {bundle}'
```

Example diffusion hook:

```bash
export TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE='python3 <your_diffusion_frame_parity_script>.py --bundle {bundle} --out {output_dir} --prompt "{prompt}"'
```

### 2.4 Configure scheduling policy (optional but recommended)

Default scheduling lives in `agent/config/resource_policy.json`.

Use env overrides for quick tuning:

```bash
export AGENT_CPU_SLOT_CAP=48
export AGENT_RAM_RESERVE_GB=256
export AGENT_GPU_SLOT_THRESHOLDS='220:3,160:2,100:1'
export AGENT_MAX_WORKERS_PER_STRATEGY='diffusion=4,video_diffusion=2'
export AGENT_STRATEGY_MUTEXES='diffusion|video_diffusion'
```

## 3. Architecture Overview

Core runtime modules:

- `agent/submit_links.py`: user ingress for HF links.
- `agent/intake_daemon.py`: drains inbox and triggers DAG creation.
- `agent/orchestrator.py`: converts model metadata into dependency-aware tasks.
- `agent/store.py`: git-native task state store (`backlog.json` + lifecycle dirs).
- `agent/scheduler.py`: resource-aware dispatcher (CPU/RAM/GPU slot control).
- `agent/worker.py`: executes task logic in isolated worktrees.
- `agent/validator.py`: standardized modality-specific parity gate.
- `agent/merge_manager.py`: serialized integration and canary gating before merge.
- `agent/config/capability_matrix.json`: strategy readiness matrix used for foundation-task gating.
- `agent/config/resource_policy.json`: CPU/GPU scheduling policy + strategy mutexes.

Single source of truth:
- `agent/backlog.json`

Lifecycle directories:
- `agent/inbox/`, `agent/claimed/`, `agent/completed/`, `agent/failed/`, `agent/artifacts/`.

## 4. Task Graph and State Model

Per HF link, intake creates a task chain similar to:

1. `intake_model`
2. `foundation_*` singleton tasks (only when strategy is not ready)
3. `merge_ready_check` for foundation branch (strategy-level)
4. `implement_plugin` (if needed)
5. `implement_runtime_dispatch` (if needed)
6. `implement_builder` (if needed)
7. `add_manifest`
8. `parity_validation`
9. `merge_ready_check`

Status progression:

- `new -> ready -> dispatched -> running -> completed`
- `completed -> merge_pending -> merged`
- failure exits: `failed`, `blocked`, `deferred`

## 5. End-to-End Developer Journey with Expected State

### Step A: Submit links

```bash
python3 -m agent.submit_links --link https://huggingface.co/Qwen/Qwen3-0.6B
```

Expected state:
- `agent/inbox/links.jsonl` contains queued link(s).
- No task execution yet.

### Step B: Intake

```bash
python3 -m agent.intake_daemon --once
```

Expected state:
- `agent/backlog.json` includes generated tasks for each model.
- `intake:*` tasks marked `completed`.
- inbox entries moved to `agent/inbox/processed.jsonl`.

### Step C: Dispatch + implementation

```bash
python3 -m agent.scheduler --once
```

Expected state:
- dependency-satisfied tasks become `ready` and may become `running`.
- task claim locks appear in `agent/claimed/`.
- task outputs are written into `agent/artifacts/<task-id>/result.json`.

### Step D: Validation

Validation runs automatically when dependency chain reaches `parity_validation`.

Expected state:
- pass: validation task becomes `completed`.
- fail: validation task becomes `failed` with diagnostics in artifacts.

### Step E: Merge gating

```bash
python3 -m agent.merge_manager --once
```

Expected state:
- `merge_ready_check` task becomes `merge_pending`, then `merged` only if canaries pass.
- if any canary fails, task becomes `failed` and `master` remains unchanged.

## 6. Running Modes

### Full autonomous loop

```bash
./agent/agent_loop.sh agent
```

This runs intake, scheduling, and merge passes continuously.

### Container-run loop (recommended)

```bash
docker exec "$CONTAINER" bash -lc "cd $REPO_IN_CONTAINER && ./agent/agent_loop.sh agent"
```

### Manual status view

```bash
python3 -m agent.report_status --store agent
```

## 7. Validation Design Guarantees

Validation is standardized by modality and centralized in `agent/config/validation_profiles.json`.

- Decoder/MoE/SSM: logits battery + bundle build + C++ run + exact HF output parity.
- Encoder: encoder diff battery + strict C++ parity hook.
- Encoder-decoder: strict C++ parity hook.
- VL: `diff_vl.py` against HF with real image (`TRTF_VL_IMAGE`).
- Diffusion: component checks + C++ generation + frame stats + strict frame parity hook.
- Video encoder: strict C++ parity hook.
- Video diffusion: strict frame/video parity hook.
- Time-series: strict tensor parity hook.
- RL inference: strict policy/value hook.

Why hooks are required:
- Non-decoder outputs vary by interface and representation.
- Hooks let each modality enforce exact policy without weakening gate quality.

## 8. Master Stability Rules

`master` protection is enforced operationally:

1. Workers never merge directly to `master`.
2. Merge manager integrates one merge-pending task at a time.
3. Tier-A canaries always run on every merge task.
4. Tier-B canaries run for strategy-foundation merges (`run_tier_b=true`) to avoid unbounded gate time.
5. Failed merge gate marks task failed; merge is not retained.

## 9. Troubleshooting Playbook

- Task stuck in `new`: dependency task not completed.
- Task stuck in `ready`: scheduler capacity constraints or scheduler not running.
- Repeated `failed` on non-decoder parity: missing hook env var.
- VL parity failure: unset or invalid `TRTF_VL_IMAGE`.
- Merge manager refusal due dirty repo: clean the checkout before merge pass.

## 10. New Developer Quick Start Checklist

1. Read this file once.
2. Source `agent/config/env.production.sh` and fill required hooks.
3. Submit one known model link.
4. Run one `--once` cycle for intake, scheduler, merge manager.
5. Verify status with `agent.report_status`.
6. Run continuous loop for batch onboarding.

## 11. Template Validation (recommended)

Before running real onboarding, validate template formatting and substitution:

```bash
source agent/config/env.smoke.sh
python3 -m agent.validate_templates --execute
```

Then switch back to production values:

```bash
source agent/config/env.production.sh
```

## 12. Single-Agent Example You Can Run Now

Use this to verify the flow with one agent before enabling full autonomous mode.

```bash
bash agent/run_single_agent_smoke.sh
```

What this does:
- loads `agent/config/env.smoke.sh` (safe echo templates),
- submits one HF link,
- runs intake once,
- runs scheduler twice,
- prints status and per-task states.

Note: `running_workers` in `agent.report_status` comes from the last scheduler snapshot and can lag when using one-shot scheduler invocations.

Expected task states after this smoke run:
- `intake:*` -> `completed`
- `impl:*:plugin` -> `completed`
- `impl:*:manifest` -> `completed`
- `validate:*:parity` -> `new` (not started yet in this short smoke run)
- `merge:*:gate` -> `new`

Then switch to real settings:

```bash
source agent/config/env.production.sh
# fill strict parity hook env vars
./agent/agent_loop.sh agent
```

This process is designed so onboarding scales with hardware while preserving strict correctness gates and merge safety.
