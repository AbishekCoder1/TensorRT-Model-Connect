Use ai-promotion-babysitter. Run one bounded promotion-MR babysitting cycle for the local AI staging pipeline.

Rules:
- Do not use --dangerously-skip-permissions or any equivalent bypass.
- Never push to `master`.
- Never approve or merge a promotion MR.
- Only modify timestamped `ai-staging-promotion-*` source branches.
- Keep fixes minimal and directly tied to getting the full promotion MR pipeline green.

Workflow:
1. Run `git fetch origin master`.
2. Run:

```bash
python3 tools/ai_staging.py \
  --project yifeif/trt-transformers \
  --branch ai-staging \
  babysit-promotion \
  --target-branch master \
  --max-rebases 1
```

3. If all open promotion MRs are green or waiting for CI, report that and stop.
4. If a promotion MR has a failed or canceled full pipeline and no newer active pipeline, repair exactly one promotion source branch:
   - create an isolated worktree from `origin/<promotion-source-branch>`
   - inspect failed jobs, logs, and artifacts
   - make the smallest fix needed for full CI
   - run the most relevant local/CPU validation available
   - commit and push back to the same promotion source branch without `ci.skip`
   - update the MR description or add a note summarizing the fix and validation
5. If the branch has a non-trivial rebase conflict against `origin/master`, stop and report the blocker.

Report only: promotion MRs green, promotion MRs waiting, branch rebased, branch fixed, validation run, and blockers.
