---
name: gitlab-ai-staging-autopilot
description: Use when autonomously managing GitLab merge requests targeting the AI staging branch. Lists MRs targeting ai-staging, waits for successful minimal CI, rebases onto the current target branch, pushes clean rebases with ci.skip, pushes target-wins conflict resolutions with a new pipeline, and merges only after explicit safety checks instead of GitLab native auto-merge.
---

# GitLab AI Staging Autopilot

## Purpose

Run an autonomous queue loop for low-value AI-generated MRs targeting `ai-staging`.

This skill intentionally avoids GitLab native auto-merge. In this project, auto-merge can merge immediately while CI is pending because `Pipelines must succeed` is disabled at the project level.

## Policy

Treat the target branch as ground truth.

- Process one MR at a time. A fast-forward merge moves `ai-staging`, so the next MR must be checked against the new target.
- Merge only open, non-draft MRs targeting the staging branch.
- Require the dedicated `ai:staging-mr` label so AI staging MRs are easy to filter and accidental unlabeled MRs are not merged.
- Require the latest MR head pipeline to be `success` before any clean-rebase merge.
- Require `approvals_left == 0` when the approvals API is available.
- Rebase the source branch onto the latest target branch before merging.
- If the rebase is clean, push the rebased source branch with `ci.skip` and an explicit `--force-with-lease` against the fetched source SHA, then merge by exact SHA.
- If the rebase conflicts, resolve conflicts in favor of the target branch by default, push with an explicit `--force-with-lease` and without `ci.skip`, and do not merge until the new pipeline succeeds in a later loop.
- If conflict resolution makes the MR empty, close or skip it rather than merging a no-op.
- Preserve source branches unless the user explicitly says deletion is acceptable.

## Quick Start

From a clean checkout of this repository:

```bash
python3 skills/gitlab-ai-staging-autopilot/scripts/ai_staging_autopilot.py \
  --project yifeif/trt-transformers \
  --target ai-staging \
  --source-prefix agent-2- \
  --required-label ai:staging-mr \
  --once
```

Use `--dry-run` first when changing options. Dry runs still fetch and may create/rebase a local work branch, but they do not push source branches, close MRs, or merge MRs.

The script uses `glab api` for GitLab API calls and normal `git` for fetch, rebase, push, and merge preparation. Run it from a checkout authenticated for both `glab` and `git push`.

## Workflow

1. Fetch `origin/<target>` and all candidate source branches.
2. List open MRs targeting `<target>` whose source branch matches the configured prefix.
3. Skip MRs that are draft, missing `ai:staging-mr`, unapproved, not mergeable, missing a successful current head pipeline, or from a fork that cannot be pushed.
4. Pick the oldest eligible MR.
5. Create/reset a local work branch from the source branch.
6. Try `git rebase origin/<target>`.
7. On a clean rebase:
   - Push the rebased source branch with `git push -o ci.skip --force-with-lease=<source-ref>:<fetched-source-sha>`.
   - Merge the MR through the GitLab API using the exact rebased SHA.
8. On a rebase conflict:
   - Resolve conflicted paths in favor of the target branch.
   - Continue or skip empty commits.
   - Push the resolved branch without `ci.skip`.
   - Leave the MR open for CI to run.
9. Refetch and repeat on the next loop.

## Failure Handling

Stop and report instead of guessing when:

- GitLab reports the pipeline SHA does not match the MR SHA.
- The source branch is not in the same project.
- The push is rejected.
- The merge API rejects the exact SHA.
- Conflict resolution cannot continue cleanly.
- The working tree is dirty before starting.

For conflicts, prefer dropping conflicting AI changes over modifying target-branch behavior. The target branch is the integration truth.
