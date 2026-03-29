# Coverage-Based Unit Test Selection

**Date:** 2026-03-28
**Branch:** coverage-based-unit-test
**Status:** Design

## Problem

Unit tests are gated at the tier level: any `src/` change runs all 72 C++ tests, any `trtf_build/` change runs all 71 Python builder tests. Python builder tests are the bottleneck because several build real TRT engines (~40m CI timeout). As the test suite grows, this becomes increasingly wasteful -- most changes only affect a handful of tests.

The existing `test_impact.py` handles E2E model selection well (rule-based, file-path heuristics), but unit tests lack fine-grained selection.

## Solution

Build a unified coverage map system that uses actual test execution data to determine which unit tests exercise which source files. On each MR, only run the tests that cover modified code.

### Approach chosen: Unified Coverage Map (DIY)

Over pytest-testmon (CI story is weak, Python-only) and static heuristics (miss indirect dependencies, degrade over time). The unified map gives one system for both languages, integrates with existing `test_impact.py`, and is self-maintaining -- new tests and source files are automatically incorporated on the next nightly run.

## Coverage Map Format

A single JSON file mapping source files to the tests that exercise them:

```json
{
  "meta": {
    "commit": "abc123",
    "generated_at": "2026-03-28T02:00:00Z",
    "python_tests": 284,
    "cpp_tests": 72
  },
  "source_to_tests": {
    "trtf_build/trtf_build/config.py": [
      "tests/builder/test_config.py::TestModelConfig::test_parse_qwen3",
      "tests/builder/test_config.py::TestModelConfig::test_vl_merge",
      "tests/builder/test_engine_qwen.py::TestQwenEngine::test_config_parsing"
    ],
    "src/tokenizer/vocab_tokenizer.cpp": [
      "test_vocab_tokenizer"
    ],
    "src/bundle/bundle_format.cpp": [
      "test_bundle_format",
      "test_bundle_e2e"
    ]
  }
}
```

- **Python tests** identified by pytest node ID (`file::class::method`)
- **C++ tests** identified by ctest name (each is a separate binary)
- **File-level** granularity for the mapping key (source file). This is the most stable -- line insertions don't invalidate the map, and the map only goes stale when files are added, removed, or renamed.

## Coverage Map Generation

### Python collection

1. Run `pytest tests/builder/ tests/tools/ --cov=trtf_build --cov-context=test --cov-report=`
2. Open the `.coverage` SQLite DB
3. Query for per-test file coverage: `SELECT DISTINCT f.path, c.context FROM line_bits lb JOIN file f ON lb.file_id = f.id JOIN context c ON lb.context_id = c.id`
4. Build reverse map: `{source_path: [test_node_ids]}`

### C++ collection

1. Build with `--coverage -O0 -g` (same flags as existing coverage infrastructure)
2. For each ctest target (parallelizable):
   - `find build -name "*.gcda" -delete`
   - Run the test binary
   - `gcovr --json -o /tmp/cov_<test_name>.json --filter src/ --filter include/`
3. Parse each JSON: extract files with >0 covered lines
4. Build reverse map: `{source_path: [test_names]}`

### Merge

Combine both maps into a single `coverage_map.json` with the meta block.

### CLI

```bash
# Full generation (nightly)
python tools/coverage_map/generate.py --output coverage_map.json

# Python only
python tools/coverage_map/generate.py --python-only --output coverage_map.json

# C++ only
python tools/coverage_map/generate.py --cpp-only --output coverage_map.json

# Validation (check all referenced files/tests still exist)
python tools/coverage_map/generate.py --validate coverage_map.json
```

## Test Selection Logic

### Integration with test_impact.py

The coverage map adds a second layer inside the existing tier-based selection:

```
git diff -> test_impact.py -> unit tiers (builder/cpp/tools) -> coverage_map lookup -> run ONLY affected tests
```

`test_impact.py` still decides which tiers to activate. Within each activated tier, the coverage map narrows to individual tests.

### Selection algorithm

Given a set of changed files from git diff:

1. Look up each changed file in `source_to_tests` -- union all matched tests
2. Add safety tests:
   - New test files (not in the map) -- always run
   - Previously failed tests (from last CI run) -- always re-run
3. Handle unknown source files (not in the map):
   - If the file is in `src/` or `include/` -- fall back to all C++ tests in the tier
   - If the file is in `trtf_build/` -- fall back to all Python builder tests in the tier
   - This preserves the zero-false-negative guarantee
4. Output a filtered test list for pytest `-k` / ctest `-R` selection

### Example scenarios

| Change | Today | With coverage map |
|--------|-------|-------------------|
| Edit `vocab_tokenizer.cpp` | All 72 C++ tests | `test_vocab_tokenizer` + 2-3 others |
| Edit `config.py` | All 71 Python builder tests | ~8 config-related tests |
| Edit `graph_ops.py` | All Python builder tests | `test_graph_ops`, `test_graph_blocks`, + family tests using those ops |
| Add new file `src/foo.cpp` | All C++ tests | All C++ tests (safety fallback) |
| Edit `families/qwen.py` | All Python builder tests | `test_engine_qwen` + tests covering qwen plugin |

### CLI

```bash
# Select tests for current diff
python tools/coverage_map/select_tests.py --coverage-map coverage_map.json --diff <(git diff master)

# Explicit file list
python tools/coverage_map/select_tests.py --coverage-map coverage_map.json --files src/tokenizer/vocab_tokenizer.cpp

# Integrated into test_impact.py
python tools/test_impact.py --coverage-map coverage_map.json --json > impact.json
```

## CI Pipeline Integration

### New nightly job

Runs on master after existing nightly jobs:

```yaml
generate-coverage-map:
  stage: nightly-coverage
  script:
    - python tools/coverage_map/generate.py --output coverage_map.json
  artifacts:
    paths: [coverage_map.json]
    expire_in: 30 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

Estimated nightly cost: ~55m additional (Python +10-15% overhead from coverage context tracking, C++ per-test gcov ~5-10m for 72 fast tests).

### MR pipeline changes

Impact analysis gains coverage map awareness:

```yaml
impact-analysis:
  stage: analyze
  script:
    - python tools/coverage_map/fetch_latest.py --output coverage_map.json
    - python tools/test_impact.py --coverage-map coverage_map.json --json > impact.json
  artifacts:
    paths: [impact.json]
```

Unit test jobs use filtered test lists:

```yaml
test-python-builder:
  script:
    - FILTER=$(python -c "import json; d=json.load(open('impact.json')); print(' or '.join(d['builder_tests']))")
    - if [ -n "$FILTER" ]; then pytest tests/builder/ -k "$FILTER"; fi

test-cpp-unit:
  script:
    - TESTS=$(python -c "import json; d=json.load(open('impact.json')); print('|'.join(d['cpp_tests']))")
    - if [ -n "$TESTS" ]; then ctest --test-dir build -R "$TESTS" --output-on-failure; fi
```

### Artifact fetch strategy

`fetch_latest.py` uses the GitLab Jobs API to download the latest `coverage_map.json` artifact from the `generate-coverage-map` job on the default branch. If the GitLab API is unavailable or the artifact has expired, it falls back to a configurable local path (e.g., a shared NFS mount). If neither source is available, it exits with code 1, signaling the MR pipeline to skip coverage-based selection and run all tests.

### Fallback behavior

- No coverage map available (first run, artifact expired, fetch fails) -- run all tests
- Stale map (>50 commits behind HEAD) -- log warning, still use it
- Map fetch failure -- run all tests in tier (graceful degradation)

## Safety Invariants

1. **Unknown source files** (not in map) -- full tier fallback for that language
2. **New test files** (not in map as values) -- always run
3. **Previously failed tests** -- always re-run
4. **Stale map** -- safe by construction. Staleness causes over-selection (running extra tests), not under-selection. A test that used to cover a file but no longer does will still be selected -- that's a false positive, not a false negative.
5. **Map fetch failure** -- run all tests in tier
6. **Validation** -- `generate.py --validate` checks that every test/source in the map still exists. Run during generation to prune stale entries.

## File Layout

```
tools/coverage_map/
  __init__.py
  generate.py          # Orchestrator: runs Python + C++ collection, merges
  python_collector.py   # Runs pytest with --cov-context, parses .coverage DB
  cpp_collector.py      # Runs per-test gcov isolation, parses gcovr JSON
  select_tests.py       # Given diff + map -> test list (standalone + library)
  fetch_latest.py       # Download coverage map artifact from CI
```

Changes to existing files:
- `tools/test_impact.py` -- add `--coverage-map` flag, integrate `select_tests` into classification pipeline
- `.gitlab-ci.yml` -- add `generate-coverage-map` nightly job, modify unit test jobs to use filtered test lists

## Scope

### In scope

- Coverage map generation for Python builder + C++ unit tests
- Selective test execution in CI MR pipelines
- Local developer usage via `select_tests.py`
- Integration with existing `test_impact.py`

### Out of scope

- E2E model selection (stays rule-based via test_impact.py)
- Graph-op GPU test selection (tier 2 -- small set, always runs if tier triggered)
- Tools test selection (fast, ~18 test files, not worth the complexity)
- ML-based predictive selection
- Line-level or function-level granularity (file-level is sufficient and stable)
- Incremental map updates (optimize later if nightly cost grows)
