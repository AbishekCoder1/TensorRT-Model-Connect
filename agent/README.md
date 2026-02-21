# Autonomous Agent Runtime

Git-native orchestration runtime for scalable HF model onboarding.

## User Flow
1. User provides HF links only.
2. Intake converts each link into dependency-aware tasks and checks strategy readiness.
3. If a runtime strategy is not ready, singleton foundation tasks are created first.
4. Scheduler dispatches tasks to subagents based on CPU/GPU capacity.
5. Validator enforces modality-specific parity rules.
6. Merge manager serializes integration and runs canary gates before marking merged.

Queue links:
- `python3 -m agent.submit_links --link https://huggingface.co/Qwen/Qwen3-0.6B`
- `python3 -m agent.submit_links --links-file /path/to/hf_links.txt`

## Runtime Commands
- `python3 -m agent.intake_daemon --once`
- `python3 -m agent.worker_supervisor --poll-interval 5`
- `python3 -m agent.merge_manager --once`

Continuous loop (recommended):
- `while true; do python3 -m agent.intake_daemon --once; python3 -m agent.scheduler --once; python3 -m agent.merge_manager --once; sleep 5; done`

## Container Usage
All build/runtime validation should run inside the project container:
- `docker exec trtf-dev-gb300 bash -lc 'cd /workspace/trt-transformers-cpp && python3 -m agent.scheduler --once'`
- `docker exec trtf-dev-gb300 bash -lc 'cd /workspace/trt-transformers-cpp && python3 -m agent.merge_manager --once'`

Use `AGENT_SUBAGENT_CMD_TEMPLATE` to define how a worker invokes a coding subagent. Supported placeholders:
- `{prompt_file}`, `{worktree}`, `{task_json}`, `{task_json_file}`, `{worker_id}`

Validation hook env vars for strict non-decoder parity:
- `TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE`
- `TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE`
- `TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE`
- `TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE`
- `TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE`
- `TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE`
- `TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE`
- `TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE`
- `TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE`
- `TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE`

Starter template:
- `source agent/config/env.production.sh`
- `python3 -m agent.validate_templates --execute` (with `env.smoke.sh`)
- `bash agent/run_single_agent_smoke.sh` (single-agent smoke demo)

Full runbook:
- `docs/AGENT_ORCHESTRATION.md`

## State Layout
- `backlog.json`: source of truth task graph.
- `inbox/links.jsonl`: queued user links.
- `claimed/`, `completed/`, `failed/`: task lifecycle snapshots.
- `config/capability_matrix.json`: strategy readiness matrix.
- `config/resource_policy.json`: scheduler auto-scaling and mutex policy.
- `config/canaries.json`: two-tier merge gates for `master`.
- `config/validation_profiles.json`: standardized validation defaults.
