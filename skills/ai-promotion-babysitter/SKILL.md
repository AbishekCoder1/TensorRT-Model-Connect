---
name: ai-promotion-babysitter
description: Use when maintaining human-review promotion MRs from ai-staging-promotion-* branches to master. Keeps the promotion MR full CI green by clean-rebasing outdated promotion branches, diagnosing failed full pipelines, making minimal fixes on the promotion source branch, and pushing a new pipeline. Never merges, approves, or pushes master.
---

# AI Promotion Babysitter

## Purpose

Keep open promotion MRs ready for human review.

Promotion MRs are from timestamped `ai-staging-promotion-*` branches to `master`. This agent may modify only the promotion source branch. It exists to make the full MR pipeline green before a human reviews or merges.

## Safety Rules

- Never push to `master`.
- Never approve or merge a promotion MR.
- Never use `ci.skip` on promotion fixes; every fix needs a real full MR pipeline.
- Do not modify `ai-staging`; the staging rotation command owns that branch.
- Do not broaden the promotion diff beyond what is required to pass full CI.
- Stop for non-trivial rebase conflicts, product decisions, or fixes whose proof requires unavailable hardware.

## Cycle

1. Classify open promotion MRs:

```bash
python3 tools/ai_staging.py \
  --project yifeif/trt-transformers \
  --branch ai-staging \
  babysit-promotion \
  --target-branch master \
  --max-rebases 1
```

This command lists open `ai-staging-promotion-* -> master` MRs, clean-rebases at most one outdated promotion branch onto `origin/master`, refreshes the generated MR description after a clean rebase, and prints failed full-CI jobs.

2. If a promotion MR is waiting for CI, stop.

3. If a promotion MR is green and up to date with `master`, report that it is ready for human review.

4. If a promotion MR has failed or canceled full CI and has no newer active pipeline, repair exactly one MR:

```bash
git fetch origin master <promotion-source-branch>
mkdir -p .ai-pipeline/promotion-worktrees
git worktree add --detach .ai-pipeline/promotion-worktrees/<promotion-source-branch> origin/<promotion-source-branch>
```

Inside that worktree:

- inspect the failed jobs and logs with GitLab API or `glab`
- download artifacts when logs point to test-result artifacts
- identify whether the failure is caused by the promotion diff, a master breakage, or infrastructure
- make the smallest code or test fix needed on the promotion source branch
- run the most relevant local verification available, preferably CPU/local
- commit the fix with a concise message
- push back to the same source branch with normal CI enabled:

```bash
git push origin HEAD:refs/heads/<promotion-source-branch>
```

5. Update the MR description or add a note with:

- failed job names
- root cause
- files changed
- verification run
- remaining risk or blocker

## Rebase Policy

Clean rebases are allowed. Conflict resolution during promotion babysitting is intentionally conservative:

- If the conflict is purely mechanical and the correct resolution is obvious, resolve it.
- If the conflict changes behavior, crosses task boundaries, or risks dropping promoted work, stop and report the blocker.
- Do not reset the promotion branch to `master`; that would discard the human-review diff.

## Success Criteria

- Promotion source branch is up to date with `origin/master`.
- Latest full MR pipeline is running or green.
- If a fix was pushed, the MR explains the fix and validation.
- Human remains responsible for final review and merge.
