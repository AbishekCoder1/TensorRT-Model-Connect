---
name: ai-task-discovery
description: Use when continuously discovering atomic, low-risk, AI-implementable cleanup tasks for the GitLab ai-staging workflow. Creates tightly scoped GitLab task issues with acceptance criteria and verification commands; does not implement code.
---

# AI Task Discovery

## Purpose

Find small, valuable, verifiable cleanup tasks and publish them as GitLab issues for implementation agents.

Do not edit code. Do not create broad tasks. The output of this agent is a queue of issues labeled `ai:task` and `ai:ready`.

## Task Contract

Every task must include:

- `Scope`: files or directories the implementation agent may touch.
- `Change`: exact requested change.
- `Acceptance Criteria`: observable conditions for done.
- `Verification`: commands or checks to run.
- `Non-goals`: explicit things not to change.
- `Risk`: low, medium, or high, with rationale.

Reject tasks that require broad architectural judgment, unclear ownership, large refactors, product decisions, or GPU/E2E validation as the primary proof.

## Workflow

1. Sync context from `origin/master` and current `ai-staging`.
2. Search for stale docs, orphan files, duplicated test glue, TODOs with obvious closure, and small consistency gaps.
3. For each candidate, prove it is atomic:
   - one concern
   - small allowed file set
   - clear rollback
   - one or two cheap verification commands
4. Create a GitLab issue with `tools/ai_agent_system.py create-task`.
5. Stop before implementation.

## Commands

Ensure labels exist:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers ensure-labels
```

Dry-run a task before creating it:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers --dry-run create-task \
  --title "docs: remove stale reference to deleted fixture" \
  --scope "docs/wiki/Static-Design.md only" \
  --change "Remove references to the deleted models/hf/qwen3 legacy fixture." \
  --acceptance "No reference to the deleted fixture remains in docs/wiki/Static-Design.md." \
  --verification "rg 'models/hf/qwen3|legacy fixture' docs/wiki/Static-Design.md" \
  --non-goal "Do not edit runtime code or regenerate unrelated docs." \
  --label ai:small
```

Validate an issue body:

```bash
python3 tools/ai_agent_system.py --project yifeif/trt-transformers validate-task --issue <iid>
```
