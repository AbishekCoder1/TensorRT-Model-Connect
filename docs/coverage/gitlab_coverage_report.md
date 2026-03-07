# GitLab Coverage Report Setup

This repository uses two coverage jobs in `.gitlab-ci.yml`:

- `coverage-python`
- `coverage-cpp`

Both jobs publish GitLab Cobertura reports and act as hard gates.

## Tooling

Container images (`Dockerfile`, `Dockerfile.gb300`) include:
- `coverage` / `pytest-cov` (Python)
- `gcovr` (C++)
- `lcov` (auxiliary tooling)

CI jobs also install pinned runtime tools:
- Python: `coverage[toml]==7.6.10`
- C++: `gcovr==8.2`

## CI Wiring

`coverage-python`:
- Runs `tools/coverage_ci/run_python_coverage.sh`
- Produces:
  - `coverage/python-cobertura.xml`
  - `coverage/python-coverage.txt`
- Enforces:
  - line coverage = 100%
  - branch coverage = 100%
- Emits `PYTHON_COVERAGE_LINE=...%` for GitLab `coverage:` regex parsing.

`coverage-cpp`:
- Runs `tools/coverage_ci/run_cpp_coverage.sh`
- Produces:
  - `coverage/cpp-cobertura.xml`
  - `coverage/cpp-coverage-summary.txt`
- Enforces:
  - line coverage = 100%
  - function coverage = 100%
  - branch coverage = 100%
- Emits `CPP_COVERAGE_LINE=...%` for GitLab `coverage:` regex parsing.

Both jobs publish:

```yaml
artifacts:
  reports:
    coverage_report:
      coverage_format: cobertura
      path: <xml-path>
```

This enables MR diff coverage annotations in GitLab.

## Local Reproduction

From repo root:

```bash
# Python only
tools/coverage/python_coverage.sh -v --ignore=tests/builder/test_cli.py

# C++ only
tools/coverage/cpp_coverage.sh

# Both
tools/coverage/run_coverage_all.sh
```

CI-compatible wrappers:

```bash
bash tools/coverage_ci/run_python_coverage.sh
bash tools/coverage_ci/run_cpp_coverage.sh
```

## Metric Notes

- Python:
  - `coverage.py` provides line + branch metrics.
  - Function coverage is not a native `coverage.py` metric.
- C++:
  - `gcovr` provides line + function + branch metrics directly.

## Operational Notes

- Coverage XML is generated before final gate failure handling, so artifacts remain available for diagnosis when thresholds fail.
- Gates are intentionally strict and will fail until source and tests reach full required coverage.
