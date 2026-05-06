---
name: ai-task-discovery
description: Use when continuously discovering atomic, low-risk, AI-implementable reliability tasks for the GitLab ai-staging workflow. Creates tightly scoped GitLab task issues with objective exit criteria and verification commands; does not implement code.
---

# AI Task Discovery

## Purpose

Find small, valuable, verifiable tasks that move the project in a more reliable direction, then publish them as GitLab issues for implementation agents.

Do not edit code. Do not create broad tasks. The output of this agent is a queue of issues labeled `ai:task` and `ai:ready`.

Do not limit discovery to cleanup. Good task themes include correctness, testability, CI signal quality, flaky-test reduction, safer developer workflows, clearer diagnostics, docs that prevent operational error, and small maintainability improvements with direct reliability value.

## Task Contract

Every task must include:

- `Scope`: files or directories the implementation agent may touch.
- `Change`: exact requested change and why it improves reliability.
- `Acceptance Criteria`: objective exit criteria for done.
- `Verification`: commands or checks to run.
- `Non-goals`: explicit things not to change.
- `Risk`: low, medium, or high, with rationale and rollback path.

Reject tasks that require broad architectural judgment, unclear ownership, large refactors, speculative features, product decisions, multi-component migrations, or GPU/E2E/manual validation as the primary proof.

Before creating an issue, check existing issues, MRs, and branch names for duplicate or already-in-flight work.

## Atomicity Checklist

Only create an issue when all of these are true:

- It has one concern and one expected outcome.
- It has a bounded file or directory scope.
- It has no dependency on another unmerged task.
- One implementation agent can complete it in one MR.
- The acceptance criteria are objective pass/fail statements.
- The verification path is exact and preferably cheap CPU/local validation.
- The non-goals prevent scope expansion.
- The rollback path is obvious.

If you cannot write clear exit criteria and a reasonable verification path, do not publish the issue.

## Workflow

1. Sync context from `origin/master` and current `ai-staging`.
2. Inspect existing GitLab issues/MRs so the ready backlog stays useful and duplicate work is avoided.
3. Search for reliability-improving tasks: correctness gaps, weak tests, flaky checks, unclear diagnostics, stale operational docs, fragile scripts, orphaned code paths, duplicated test glue, TODOs with obvious closure, and small consistency gaps.
4. For each candidate, prove it is atomic:
   - one concern
   - small allowed file set
   - clear rollback
   - one or two cheap verification commands
5. Create a GitLab issue with `tools/ai_agent_system.py create-task`.
6. Stop before implementation.

## Commands

Ensure labels exist:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect ensure-labels
```

Dry-run a task before creating it:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --dry-run create-task \
  --title "tests: cover AI task contract validation" \
  --scope "tests/tools/test_ai_agent_system.py and tools/ai_agent_system.py only" \
  --change "Add coverage that validate-task rejects issue bodies missing required sections so implementation issues stay actionable." \
  --acceptance "A task body missing Verification or Acceptance Criteria fails validation and names the missing heading." \
  --verification "python3 -m pytest tests/tools/test_ai_agent_system.py -q" \
  --non-goal "Do not change GitLab API behavior or task labels." \
  --label ai:small
```

Validate an issue body:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect validate-task --issue <iid>
```
