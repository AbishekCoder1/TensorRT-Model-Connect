---
name: ai-sanity-babysitter
description: Use when monitoring and repairing AI-generated MRs targeting ai-staging until their minimal sanity CI is green. Focuses only on individual MR sanity pipelines and does not merge to ai-staging.
---

# AI Sanity Babysitter

## Purpose

Make AI-generated MRs targeting `ai-staging` pass the minimal sanity pipeline.

Do not merge MRs. Do not work on the aggregate `ai-staging -> master` promotion MR. Do not perform broad refactors while fixing CI.

## Workflow

1. Survey the queue:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging dashboard
```

2. Pick one MR with:

```text
labels ai:staging-mr
and
pipeline failed/canceled
or detailed_merge_status conflict/need_rebase
or labels ai:sanity-failed
```

3. Check out the MR branch and rebase onto `origin/ai-staging`.
4. If conflicts are trivial, resolve them while preserving the MR's intended task.
5. If the task has become invalid or too expensive, close the MR and mark the linked issue/MR `ai:dropped`.
6. Diagnose failed jobs from GitLab logs/artifacts.
7. Make the minimal fix needed for sanity CI.
8. Push the source branch and leave the MR open for CI.

## Boundaries

- Owns MR repair only until sanity CI is green.
- Does not call the merge API.
- Does not push to `ai-staging`.
- Does not use `ci.skip`; repaired MRs need a real sanity pipeline.

## Success Criteria

The MR is open, targets `ai-staging`, has approvals satisfied, and latest head pipeline is `success`.

After success, optionally update labels:

```text
remove ai:sanity-failed
add ai:sanity-green
add ai:autopilot
```
