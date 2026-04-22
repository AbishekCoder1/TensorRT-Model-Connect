---
name: ai-task-implementer
description: Use when implementing one ready or rework AI task issue from the GitLab queue. Claims a single ai:ready or ai:needs-rework issue, makes a narrowly scoped change, runs the requested verification, and opens an AI-generated MR targeting ai-staging.
---

# AI Task Implementer

## Purpose

Consume exactly one `ai:ready` or `ai:needs-rework` task issue and turn it into one small MR targeting `ai-staging`.

Do not discover new tasks. Do not broaden scope. Do not wait on CI after opening or updating the MR.

## Workflow

1. Inspect the AI MR queue:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging dashboard
```

If a generated MR targeting `ai-staging` has a failed or canceled current head pipeline and no active fix pipeline, mark it for rework:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers mark-rework --mr <mr-iid> --skip-if-active-pipeline --reason "<short failed pipeline reason>"
```

2. Pick the oldest ready or rework task:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --target ai-staging next-task
```

3. Read the issue and verify it has the required task contract.
4. If the issue has `ai:needs-rework`, inspect related MRs:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers related-mrs <issue-iid>
```

If a related MR targeting `ai-staging` has an active head pipeline, stop. If a related MR targeting `ai-staging` exists with no active pipeline, repair that MR's source branch instead of opening a duplicate MR.

5. Claim it:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers claim-task <issue-iid>
```

6. Create an isolated worktree under `.ai-pipeline/worktrees/<branch>`, then create or check out the source branch there:

```bash
git fetch origin
git switch -c ai-task-<issue-iid>-<short-slug> origin/ai-staging
```

For rework, use the existing related MR source branch.

7. Implement only the requested change and only within the allowed scope.
8. Run the issue's verification command. Add focused tests only when the task changes executable behavior.
9. Commit and push:

```bash
git push -u origin HEAD
```

10. Open a new MR targeting `ai-staging`, or update the existing MR for rework, with labels:

```text
ai-generated
ai:staging-mr
ai:sanity-pending
```

After pushing a rework fix, remove `ai:needs-rework` from the MR.

Link the issue in the MR description.

Use a complete MR description. The promotion MR later depends on individual AI MRs being understandable without re-running the implementation agent:

```markdown
## Task

Closes #<issue-iid>

## Scope

- Files/directories intentionally touched:

## What Changed

- Concrete behavior or reliability improvement performed:

## Verification

- Commands run:
- Result:

## Risk / Rollback

- Risk level:
- Rollback plan:

## Non-goals

- Explicitly out of scope:
```

Do not open an AI-generated MR with an empty or generic description. If the change is too hard to explain crisply, the task is too broad.

## Guardrails

- If the task is ambiguous, label it `ai:needs-human` instead of guessing.
- If the task is already fixed, close it with a short note.
- If implementation requires touching files outside `Scope`, stop and update the issue rather than expanding silently.
- Keep branch names under GitLab's practical limits and always start with `ai-task-`.
