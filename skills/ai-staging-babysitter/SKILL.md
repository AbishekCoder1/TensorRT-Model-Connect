---
name: ai-staging-babysitter
description: Use when maintaining the aggregate ai-staging branch after AI MRs have merged. Keeps ai-staging synced with master, monitors full CI or promotion MR CI, reverts/drops bad low-value AI changes, and ensures a human-review promotion MR exists.
---

# AI Staging Babysitter

## Purpose

Keep `ai-staging` healthy and ready for human review as an aggregate branch.

This agent owns branch health after individual AI MRs have merged. It does not implement new tasks and does not repair individual source MRs unless reverting or dropping a staged AI change.

## Policy

`master` is ground truth. AI-generated changes are disposable.

If `ai-staging` conflicts with `master`, prefer `master`. If full CI fails because of a low-value AI cleanup and the fix is not obvious, revert that AI change from `ai-staging` rather than asking a human to debug it.

## Workflow

1. Check system state:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging dashboard
```

2. Sync `master` into `ai-staging` using existing staging tooling:

```bash
python3 tools/ai_staging.py sync-branch --push
```

3. Ensure the scheduled promotion MR exists or create/update it:

```bash
python3 tools/ai_staging.py promote --target-branch master
```

The promotion MR description must be generated from the current tree diff, not
hand-written from memory. It must state source and target SHAs, whether
`ai-staging` contains `master`, staged commit subjects, net file changes,
changed paths, diffstat, and a review checklist.

4. Monitor full CI on either:

```text
ai-staging branch pipeline
or ai-staging -> master promotion MR pipeline
```

5. If full CI fails:
   - identify the first bad AI commit or MR
   - revert it on `ai-staging`
   - push `ai-staging`
   - rerun full CI

6. Leave the promotion MR for human review only when full CI is green and the diff is understandable.

## Boundaries

- Never push to `master`.
- Never merge the promotion MR unless explicitly instructed.
- Never require a human to resolve individual AI cleanup conflicts.
- Prefer dropping AI changes over weakening CI or changing production behavior.
