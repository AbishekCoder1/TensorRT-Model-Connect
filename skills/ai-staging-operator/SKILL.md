---
name: ai-staging-operator
description: Run one bounded autonomous cycle for the AI staging system. Use when Codex or Claude Code should be invoked under /loop to continuously discover small AI tasks, implement one task, babysit ai-staging sanity CI, merge green AI-generated MRs into ai-staging, and verify that scheduled promotion MRs to master are healthy.
---

# AI Staging Operator

## Purpose

Run exactly one bounded cycle of the AI staging system, then stop. Use `/loop`
outside the skill to repeat forever.

This is the umbrella workflow. It coordinates the narrower skills and tools,
but it must keep each invocation finite, idempotent, and safe to repeat.

## Safety Rules

- Never use `--dangerously-skip-permissions` or any equivalent dangerous bypass.
- Never push to `master`.
- Never merge an `ai-staging -> master` promotion MR.
- Treat `origin/master` as ground truth.
- Prefer dropping or reverting low-value AI changes over asking a human to debug them.
- Process small units only: at most one implementation task, one sanity repair,
  and one autopilot merge action per cycle.
- Stop and report if the worktree is dirty in a way that is unrelated to the
  current cycle.

## One-Click Loop Prompt

Start Claude Code normally, without dangerous permission bypasses, then run:

```text
/loop 20m /ai-staging-operator
```

Use `20m` or `30m` for the local operator loop. The GitLab scheduled pipeline is
responsible for creating or refreshing the human-review promotion MR every 2-4
hours.

## Cycle

### 1. Take the lock

Use a short-lived lock so two `/loop` ticks do not overlap:

```bash
LOCK_FILE=/tmp/ai_staging_operator.lock
if [ -f "$LOCK_FILE" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0) ))
  if [ "$age" -lt 7200 ]; then
    echo "Previous AI staging operator cycle still running; skip this tick."
    exit 0
  fi
fi
touch "$LOCK_FILE"
```

Remove the lock before every normal or error exit.

### 2. Preflight

Run:

```bash
git fetch origin master ai-staging
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging preflight
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging ensure-labels
```

If preflight reports failures, stop the cycle after reporting them. Warnings may
be handled in the same cycle when the fix is obvious and low risk.

### 3. Sync staging

Run:

```bash
python3 tools/ai_staging.py --project yifeif/trt-transformers --branch ai-staging full-cycle --push
```

If syncing `origin/master` into `ai-staging` conflicts, use
`ai-staging-babysitter`. Resolve in favor of `origin/master` unless doing so
would silently delete a clearly valid staged change; otherwise mark the issue
`ai:needs-human`.

### 4. Survey

Run:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging dashboard
```

Classify the cycle into the first applicable lane below.

### 5. Repair One Failed Sanity MR

If an AI-generated MR targeting `ai-staging` has failed sanity CI, use
`ai-sanity-babysitter` and repair exactly one MR. Push only to that MR source
branch. Do not merge it.

### 6. Merge One Green MR

If at least one AI-generated MR targeting `ai-staging` is green and approved,
run one autopilot action:

```bash
python3 skills/gitlab-ai-staging-autopilot/scripts/ai_staging_autopilot.py \
  --project yifeif/trt-transformers \
  --target ai-staging \
  --source-prefix agent-2- \
  --required-label ai:staging-mr \
  --once
```

If the script pushes a conflict resolution and leaves the MR open for a new
pipeline, stop the merge lane for this cycle.

### 7. Implement One Ready Task

If there is a ready task and no urgent sanity repair, use `ai-task-implementer`
to claim and implement exactly one `ai:ready` issue. The MR must target
`ai-staging` and include labels `ai-generated` and `ai:staging-mr`.

The MR description must include task link, scope, concrete changes,
verification, risk, rollback, and non-goals.

### 8. Discover Tasks

If the ready queue is empty or nearly empty, use `ai-task-discovery` to create up
to three small, verifiable task issues. Do not implement during the same lane
unless the task is trivial and the cycle has not already changed code.

Good tasks are cleanup-only, narrow, locally verifiable, and easy to revert.

### 9. Promotion Health

Do not merge promotion MRs. Check that at most one open `ai-staging -> master`
MR exists and that the GitLab schedule is active. If there is a tree diff but no
promotion MR and the schedule is unhealthy, run:

```bash
python3 tools/ai_staging.py --project yifeif/trt-transformers --branch ai-staging promote --target-branch master
```

Otherwise let the schedule own promotion MR creation/refresh.

### 10. Report

End with a compact status:

- tasks created or implemented
- MRs repaired
- MRs merged into `ai-staging`
- promotion MR status
- blockers needing human input

Then remove the lock.
