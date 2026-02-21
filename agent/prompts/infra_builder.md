# Infra Builder Task

You are implementing infrastructure for a new runtime strategy or builder path.

## Task Context
- `task_id`: `{{TASK_ID}}`
- `task_type`: `{{TASK_TYPE}}`
- `model_id`: `{{MODEL_ID}}`
- `runtime_strategy`: `{{RUNTIME_STRATEGY}}`
- `modality`: `{{MODALITY}}`

## Required Outcome
1. Add missing runtime dispatch and/or builder support.
2. Add targeted tests for the new code path.
3. Validate in container when GPU/runtime behavior is touched.
4. Commit cleanly with minimal scope.

## Container Execution
- Prefer `docker exec trtf-dev bash -lc '<command>'`.
- Keep TRT validation commands inside the container.
- Ensure C++ and Python paths are valid inside container.

## Validation Checklist
- Build: `cmake -S . -B build -G Ninja && cmake --build build -j`
- Python tests: `python3 -m pytest tests/builder -v`
- Runtime tests: `ctest --test-dir build --output-on-failure`
- If applicable: `python3 tools/diff_logits.py --model {{MODEL_ID}} --battery`

## Full Task JSON
```json
{{TASK_JSON}}
```
