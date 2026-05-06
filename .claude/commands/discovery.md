Use ai-task-discovery as a bounded discovery prompt for the local AI staging pipeline.

Rules:
- Do not use --dangerously-skip-permissions or any equivalent bypass.
- Do not edit repository files.
- Do not implement tasks.

Before publishing new issues, inspect the current GitLab issue/MR state. Use
`python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging dashboard`
or equivalent GitLab queries.

If the existing `ai:ready`, `ai:claimed`, and `ai:implementing` backlog is
enough to keep the implementation worker busy, do not create more work.

Otherwise inspect the repository for work that moves the project in a more
reliable direction. This is not limited to cleanup. Good task themes include
correctness, testability, CI signal quality, flaky-test reduction, safer
developer workflows, clearer diagnostics, docs that prevent operational error,
and small maintainability improvements with direct reliability value.

Only publish a task when it is atomic and has clear exit criteria:
- one concern and one expected outcome
- bounded file or directory scope
- no dependency on another unmerged task
- implementable by one agent in one MR
- objective pass/fail acceptance criteria
- exact verification command or check, preferably cheap CPU/local validation
- explicit non-goals so the implementer does not broaden the work
- low or clearly bounded risk with an obvious rollback path

Do not create tasks for broad refactors, speculative features, product
decisions, multi-component migrations, or work whose primary proof requires
GPU/E2E/manual validation. If you cannot write objective exit criteria and a
reasonable verification path, do not publish the issue.

Publish up to three issues per cycle.

Report only: issues created, skipped reason, and blockers.
