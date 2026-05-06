Run one bounded implementation-supervisor cycle for the local AI staging pipeline.

Rules:
- Do not use --dangerously-skip-permissions or any equivalent bypass.
- Do not discover new tasks.
- Do not wait on CI after opening or updating an MR.
- Do not push to master.
- This command is a supervisor. It should claim one issue, create one isolated worktree, delegate implementation to a subagent, wait for that subagent to finish, and report the resulting MR.

Workflow:
1. Run `git fetch origin ai-staging`.
2. Run `python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging dashboard`.
3. If there is an open generated MR targeting `ai-staging` with a failed or canceled current head pipeline and no active fix pipeline, run `python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect mark-rework --mr <mr-iid> --skip-if-active-pipeline --reason "<short failed pipeline reason>"`.
4. Run `python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging next-task --json`.
5. If no ready or rework task is available, stop with a concise idle report.
6. Read the issue and verify it has the required task contract: scope, change, acceptance criteria, verification, non-goals, and risk.
7. If the issue has `ai:needs-rework`, run `python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect related-mrs <issue-iid> --json`.
   - If an open related MR targeting `ai-staging` has a head pipeline in `created`, `waiting_for_resource`, `preparing`, `pending`, or `running`, stop and report that a fix pipeline is already in flight.
   - If an open related MR targeting `ai-staging` exists and has no active head pipeline, set `<branch>` to that MR's source branch and repair that existing MR.
   - If no open related MR exists, treat the issue as a normal ready task and create a new MR.
8. Claim the issue with `python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect claim-task <issue-iid>`.
9. For a new task, create a branch name `ai-task-<issue-iid>-<short-slug>` and verify it does not already exist with `git ls-remote --heads origin <branch>`.
10. Create an isolated implementation worktree under `.ai-pipeline/worktrees/<branch>`.
   - For a new task, create it from `origin/ai-staging`.
   - For rework on an existing MR, create it from `origin/<branch>` and push back to that same source branch.
11. Start a subagent in that implementation worktree. Give it the issue JSON/body, related MR JSON when present, and this exact assignment:
   - implement only the requested task
   - stay inside the issue's declared scope
   - run the issue's verification until the acceptance criteria are met
   - commit the change on `<branch>`
   - push `<branch>` to origin
   - for a new task, open an MR targeting `ai-staging`
   - for rework, update the existing MR instead of opening a duplicate MR
   - label the MR `ai-generated`, `ai:staging-mr`, and `ai:sanity-pending`
   - after pushing a rework fix, remove `ai:needs-rework` from the MR
   - write or update a complete MR description with task link, scope, changes, verification result, risk/rollback, and non-goals
12. Wait for the subagent to finish. If it reports the task is stale, ambiguous, or outside scope, label the issue `ai:needs-human` or `ai:dropped` instead of broadening the work.
13. Verify the MR exists and targets `ai-staging`, then report the MR URL and verification result.

Do not keep implementing in the supervisor after the subagent returns. If the subagent failed, report the blocker and leave the worktree for inspection.
