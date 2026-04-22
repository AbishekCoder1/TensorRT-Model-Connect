---
name: ai-staging-babysitter
description: Use when rotating the aggregate ai-staging branch for human review. Snapshots ai-staging to a timestamped promotion branch, resets ai-staging to master for future AI MRs, and opens a human-review promotion MR from the snapshot branch to master.
---

# AI Staging Babysitter

## Purpose

Rotate `ai-staging` into a human-review promotion branch.

This agent owns only the aggregate staging rotation. It does not implement tasks, repair source MRs, or merge promotion MRs.

## Policy

`master` is ground truth. `ai-staging` is a temporary accumulation branch for generated MRs.

On each rotation, snapshot the current `ai-staging` tree into a timestamped branch, then reset `ai-staging` to `master` with `--force-with-lease`. The snapshot branch is what humans review.

## Workflow

1. Rotate `ai-staging`:

```bash
python3 tools/ai_staging.py \
  --project yifeif/trt-transformers \
  --branch ai-staging \
  rotate-promotion \
  --target-branch master
```

This command must:

- create `origin/ai-staging-promotion-<UTC timestamp>` from current `origin/ai-staging` when there is a tree diff from `origin/master`
- reset `origin/ai-staging` to current `origin/master` with `--force-with-lease`
- create a human-review MR from the snapshot branch to `master`
- do nothing except exact reset when `ai-staging` has no tree diff from `master`

2. Report only the snapshot branch, reset result, promotion MR URL/status, and blockers.

## Boundaries

- Never push to `master`.
- Never merge the promotion MR unless explicitly instructed.
- Never hand-edit `ai-staging`; use the rotation tool.
- Never use plain force push; reset `ai-staging` only with `--force-with-lease`.
