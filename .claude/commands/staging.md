Use ai-staging-babysitter. Run one bounded staging-rotation cycle for the local AI staging pipeline.

Rules:
- Do not use --dangerously-skip-permissions or any equivalent bypass.
- Never push to `master`.
- Never merge a promotion MR.
- Treat `origin/master` as ground truth.

Workflow:
1. Run `git fetch origin master ai-staging`.
2. Snapshot current `origin/ai-staging` to a timestamped promotion branch.
3. Reset `origin/ai-staging` to current `origin/master` with `--force-with-lease` so future small AI MRs start from master.
4. Open a human-review MR from the timestamped snapshot branch to `master`.

Run:

```bash
python3 tools/ai_staging.py \
  --project yifeif/tensorrt-model-connect \
  --branch ai-staging \
  rotate-promotion \
  --target-branch master
```

If `origin/ai-staging` has no tree diff from `origin/master`, no promotion MR is needed; still let the tool reset `ai-staging` to the exact `origin/master` commit when needed.

Report only: snapshot branch, ai-staging reset result, promotion MR URL/status, and blockers.
