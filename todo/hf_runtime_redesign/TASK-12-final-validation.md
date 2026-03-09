# TASK-12: Final validation + documentation

## Status: blocked (needs TASK-11)
## Phase: 4 (Cutover)
## Risk: low — validation and cleanup only

## Goal

Validate the complete redesign, update documentation, measure improvements.

## Validation checklist

- [ ] Full E2E suite: `pytest tests/test_e2e.py -v` — all 50 models pass
- [ ] C++ unit tests: `ctest --test-dir build --output-on-failure`
- [ ] Python builder tests: `pytest tests/builder/ -v`
- [ ] Cyclomatic complexity: `python tools/check_cyclomatic_complexity.py src --max-ccn 10`
- [ ] Binary size measurement
- [ ] Line count comparison (before vs after)

## Metrics to capture

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Total source lines (src/) | ~33K | ? | < 25K |
| Binary size (trtf) | 3.9MB | ? | < 3.0MB |
| Abstraction layers | 7 | 3 | 3 |
| Pipeline classes | 0 (monolithic backends) | ~16 | ~16 |
| Interface count | 11 ports + 11 adapters | 1 (IPipeline) | 1 |
| Max CCN | ? | ? | ≤ 10 |

## Documentation updates

- Update CLAUDE.md source layout section
- Update docs/wiki/Architecture-Overview.md
- Update docs/wiki/Static-Design.md
- Write migration guide for adding new model families

## Dependencies

TASK-11 (cutover complete)
