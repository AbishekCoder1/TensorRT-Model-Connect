# AI Agent System

This system turns low-value AI cleanup work into an autonomous integration flow while keeping `master` human-controlled.

## Goal

Human review should happen at one aggregate boundary:

```text
task issue -> AI implementation MR -> sanity CI -> ai-staging -> full CI -> promotion MR -> master
```

`master` is ground truth. `ai-staging` is an integration branch for AI-generated changes. Individual AI MRs are disposable.

## Agents

| Agent | Skill | Owns | Must not do |
|---|---|---|---|
| Task discovery | `ai-task-discovery` | Create atomic, verifiable task issues | Implement code |
| Implementation | `ai-task-implementer` | Turn one issue into one MR targeting `ai-staging` | Babysit CI or merge |
| Sanity babysitter | `ai-sanity-babysitter` | Repair individual AI MRs until sanity CI is green | Push to `ai-staging` |
| Staging autopilot | `gitlab-ai-staging-autopilot` | Merge green AI MRs into `ai-staging` one at a time | Diagnose failed CI |
| Staging babysitter | `ai-staging-babysitter` | Keep `ai-staging` full-CI green and promotion-ready | Push or merge to `master` |
| Operator loop | `ai-staging-operator` | Run one bounded full-system cycle under `/loop` | Run forever inside one invocation |

## Shared State

GitLab issues and merge requests are the queue. Do not use a repository file as the queue.

Standard labels are managed by:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers ensure-labels
```

Core labels:

```text
ai:task
ai:ready
ai:claimed
ai:implementing
ai-generated
ai:staging-mr
ai:sanity-pending
ai:sanity-failed
ai:sanity-green
ai:autopilot
ai:staged
ai:staging-failed
ai:dropped
ai:needs-human
ai:promotion
```

## State Machine

```text
ai:task + ai:ready
  -> implementation agent claims issue
  -> agent-2-* branch and MR targeting ai-staging with ai-generated + ai:staging-mr labels
  -> sanity babysitter gets MR green
  -> autopilot rebases and merges MR into ai-staging
  -> staging babysitter verifies full CI
  -> scheduled promotion MR for human review
```

Failure path:

```text
unclear task -> ai:needs-human
invalid or stale task -> ai:dropped
sanity CI failure -> sanity babysitter fixes or drops
full CI failure on ai-staging -> staging babysitter reverts/drops bad AI change
```

## Invariants

- `master` is never pushed by agents.
- Normal MRs targeting `master` keep existing CI behavior.
- AI implementation MRs target `ai-staging`.
- AI implementation MRs carry `ai-generated` and `ai:staging-mr` labels for filtering.
- AI implementation MRs run only sanity CI.
- Promotion MRs from `ai-staging` to `master` run full CI.
- GitLab native auto-merge is not used for AI MRs unless the project globally requires successful pipelines.

## Operator Commands

One-click operator loop:

```text
/loop 20m /ai-staging-operator
```

Start Claude Code normally. Do not use `--dangerously-skip-permissions`.

Preflight:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging preflight
```

Dashboard:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging dashboard
```

Create a dry-run task:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --dry-run create-task \
  --title "docs: remove stale reference to deleted fixture" \
  --scope "docs/wiki/Static-Design.md only" \
  --change "Remove references to the deleted models/hf/qwen3 legacy fixture." \
  --acceptance "No stale fixture reference remains." \
  --verification "rg 'models/hf/qwen3|legacy fixture' docs/wiki/Static-Design.md" \
  --non-goal "Do not edit runtime code or regenerate unrelated docs."
```

Run one autopilot merge action:

```bash
python3 skills/gitlab-ai-staging-autopilot/scripts/ai_staging_autopilot.py \
  --project yifeif/trt-transformers \
  --target ai-staging \
  --source-prefix agent-2- \
  --once
```

Promotion MR:

```bash
python3 tools/ai_staging.py promote --target-branch master
```

Promotion MRs are generated from the actual `origin/master..origin/ai-staging` tree diff. Their descriptions include branch SHAs, staged commit subjects, net file changes, changed paths, diffstat, and a review checklist. Implementation agents must write complete individual MR descriptions with the task link, scope, concrete changes, verification, risk, rollback, and non-goals so the aggregate promotion remains reviewable.
