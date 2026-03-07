# Agent Rules

## Scope Discipline

1. Each task owns a specific file group. Do not edit files outside that group unless the task explicitly lists them.
2. Do not edit `src/cabi/api/trtf_c.cpp` unless a task explicitly allows it.
3. Do not edit `docs/wiki/*` from these tasks unless the task explicitly includes docs.
4. Do not mix executor decomposition with unrelated coverage-only cleanup.

## Testing Rules

1. Every new unit test must state:
   - intent
   - preconditions
   - postconditions
2. Prefer direct seam tests for pure helpers.
3. Prefer fake ports and harnesses for executor shells.
4. Use smoke/E2E only to confirm integrated parity, not to cover routine branches.

## Implementation Rules

1. Extract policy and planning before adding broad mocks around monolithic code.
2. Keep executor shells thin; if a function grows branchy, extract another seam.
3. Preserve behavior first; architecture changes must not silently change outputs.
4. Keep CCN at or below the repository gate.

## Handoff Rules

Each task handoff must include:

1. files changed
2. tests added
3. tests run
4. remaining gaps inside the owned file group

## Container Rule

Run build/test/coverage work in the matching per-agent dev container.
