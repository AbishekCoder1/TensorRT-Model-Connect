---
name: ai-task-implementer
description: Use when implementing one ready AI task issue from the GitLab queue. Claims a single ai:ready issue, makes a narrowly scoped change, runs the requested verification, and opens an AI-generated MR targeting ai-staging.
---

# AI Task Implementer

## Purpose

Consume exactly one `ai:ready` task issue and turn it into one small MR targeting `ai-staging`.

Do not discover new tasks. Do not broaden scope. Do not babysit CI after opening the MR.

## Workflow

1. Pick the oldest ready task:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers next-task
```

2. Read the issue and verify it has the required task contract.
3. Claim it:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers claim-task <issue-iid>
```

4. Create a source branch from `origin/ai-staging`:

```bash
git fetch origin
git switch -c agent-2-<issue-iid>-<short-slug> origin/ai-staging
```

5. Implement only the requested change and only within the allowed scope.
6. Run the issue's verification command. Add focused tests only when the task changes executable behavior.
7. Commit and push:

```bash
git push -u origin HEAD
```

8. Open an MR targeting `ai-staging` with labels:

```text
ai-generated
ai:staging-mr
ai:sanity-pending
```

Link the issue in the MR description.

Use a complete MR description. The promotion MR later depends on individual AI MRs being understandable without re-running the implementation agent:

```markdown
## Task

Closes #<issue-iid>

## Scope

- Files/directories intentionally touched:

## What Changed

- Concrete behavior or cleanup performed:

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
- Keep branch names under GitLab's practical limits and always start with `agent-2-`.
