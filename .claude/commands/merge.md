Use gitlab-ai-staging-autopilot. Run one bounded merge cycle for the local AI staging pipeline.

Rules:
- Do not use --dangerously-skip-permissions or any equivalent bypass.
- This is the only worker role that may merge AI MRs into `ai-staging`.
- Never push to `master`.
- Never merge a promotion MR.

Find generated MRs targeting `ai-staging` that have a successful current head
pipeline. For each eligible MR, rebase its source branch onto current
`origin/ai-staging`.

- If the rebase is clean, push the rebased source branch with `ci.skip` and
  merge by exact SHA.
- If the rebase conflicts, do not resolve it here. Mark the MR and linked issue
  `ai:needs-rework` and leave it for `/implement`.
- Skip MRs that are still waiting for CI, failed CI, missing labels, missing
  approval, draft, or otherwise not eligible.

Run:

```bash
python3 skills/gitlab-ai-staging-autopilot/scripts/ai_staging_autopilot.py \
  --project yifeif/tensorrt-model-connect \
  --target ai-staging \
  --source-prefix ai-task- \
  --required-label ai:staging-mr \
  --max-actions 5
```

Report only: MRs merged, MRs sent to rework, skipped reasons, and blockers.
