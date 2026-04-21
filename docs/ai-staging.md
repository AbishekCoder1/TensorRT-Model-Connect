# AI Staging Branch

`origin/ai-staging` is the integration branch for AI-generated merge requests.
AI-generated work should target this branch first, not `origin/master`.

The branch has three invariants:

- `origin/ai-staging` exists at all times.
- `origin/master` is always merged into `origin/ai-staging`.
- AI-generated source branches, currently `agent-2-*`, open MRs against `ai-staging`.

Human promotion to `master` stays explicit: after the AI queue is clean, create a
normal MR from `ai-staging` to `master` and let the full pipeline run there.
This promotion MR may be filed by a schedule, but it is still reviewed and
merged by a human.

## CI Policy

Pipelines for MRs targeting `ai-staging`, and direct pushes to `ai-staging`, run
only the low-cost gate:

- `build-all`
- `lint-check`
- `ai-staging-sanity`

The sanity job runs the CLI help check, the full C++ CTest unit suite, and the
CPU Python builder/tool test suite. Expensive impact analysis, coverage, graph,
and E2E jobs are skipped for `ai-staging`. Full CI still runs for normal MRs,
scheduled pipelines, web pipelines, and the final promotion MR to `master`.

New AI agent branches should be created from `origin/ai-staging`. That keeps the
source branch CI configuration aligned with the target branch. Existing AI
branches created from old `master` may need to be rebased or merged onto
`origin/ai-staging` once before their MR pipeline reflects this policy.

GitLab evaluates merge-request pipeline configuration from the source branch
SHA. The AI staging CI rules therefore must also exist on `master`, and existing
`agent-2-*` branches must be rebased or otherwise updated after the CI change
lands. Retargeting an old branch to `ai-staging` is not enough by itself.

## Operating Cycle

Run this from a clean checkout:

```bash
python3 tools/ai_staging.py full-cycle --push --retarget
```

That command:

1. Creates `origin/ai-staging` from `origin/master` if it is missing.
2. Fetches `origin/master` and `origin/ai-staging`.
3. Checks out the local `ai-staging` branch.
4. Merges `origin/master` into `ai-staging`.
5. Pushes the updated branch when `--push` is present.
6. Retargets open `agent-2-*` MRs from `master` to `ai-staging` when
   `--retarget` is present.

For a read-only preview:

```bash
python3 tools/ai_staging.py --dry-run full-cycle --push --retarget
```

To inspect the queue without changing anything:

```bash
python3 tools/ai_staging.py list
```

To retarget only after reviewing the list:

```bash
python3 tools/ai_staging.py retarget
```

## Scheduled Promotion MR

The repository includes a lightweight scheduled CI path for filing a promotion
MR from `ai-staging` to `master`. It does not merge automatically.

Configure a GitLab pipeline schedule on the `master` branch with these
variables:

- `AI_STAGING_PROMOTE=1`
- `AI_STAGING_BOT_TOKEN=<masked project access token with API scope>`

The scheduled pipeline runs only `ai-staging-promotion-mr`; normal nightly
build, coverage, graph, and E2E jobs are skipped for this maintenance schedule.
The scheduled job uses the GitLab REST API directly with `AI_STAGING_BOT_TOKEN`;
it does not require `glab` on the runner.

The job is idempotent:

- If `origin/ai-staging` and `origin/master` have identical trees, it exits
  without filing an MR.
- If an open `ai-staging -> master` MR already exists, it reports that MR and
  exits.
- Otherwise it creates `chore: promote ai-staging to master` for human review.

Equivalent local command:

```bash
python3 tools/ai_staging.py promote
```

Use `--source-prefix` more than once if another AI branch prefix is introduced:

```bash
python3 tools/ai_staging.py --source-prefix agent-2- --source-prefix agent-3- list
```

## Conflict Handling

If syncing `ai-staging` with `master` conflicts, stop the cycle and resolve the
branch manually. Do not retarget additional MRs until `origin/master` is again
merged into `origin/ai-staging` and the minimal pipeline is green.

If the final `ai-staging -> master` promotion MR fails full CI, fix forward on a
normal branch targeting `ai-staging`, then rerun the promotion MR. Avoid direct
fixes on `master`; this preserves the integration branch as the single staging
surface for AI-generated cleanup work.
