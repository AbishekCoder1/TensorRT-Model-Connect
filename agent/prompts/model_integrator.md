# Model Integrator Task

You are implementing a single model onboarding task in an isolated git worktree.

## Task Context
- `task_id`: `{{TASK_ID}}`
- `task_type`: `{{TASK_TYPE}}`
- `model_id`: `{{MODEL_ID}}`
- `runtime_strategy`: `{{RUNTIME_STRATEGY}}`
- `modality`: `{{MODALITY}}`

## Required Outcome
1. Implement only what is needed for this task.
2. Keep iterating until the task acceptance criteria are satisfied.
3. Run focused tests relevant to changed files.
4. Commit your changes on the task branch.
5. Print a short completion report with:
- changed files
- commands run
- remaining risks

## Constraints
- Work only in this worktree/branch.
- Do not ask for user input.
- Do not leave TODO placeholders.
- If blocked, fail with exact diagnostics and command outputs.

## Useful Commands
- `python3 -m pytest tests/builder -v`
- `python3 -m pytest tests/tools -v`
- `ctest --test-dir build --output-on-failure`
- `python3 scripts/validate_family.sh <hf_model_or_dir>`

## Full Task JSON
```json
{{TASK_JSON}}
```
