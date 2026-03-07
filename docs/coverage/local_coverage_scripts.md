# Local Coverage Scripts (Python + C++)

This note documents local/CI-ready coverage scripts added under `tools/coverage/`.
CI wrapper scripts live under `tools/coverage_ci/` and normalize output file names for GitLab artifacts.
For full GitLab pipeline wiring and report behavior, see `docs/coverage/gitlab_coverage_report.md`.

## Scripts

- `tools/coverage/python_coverage.sh`
  - Runs `pytest` under `coverage.py`.
  - Emits Cobertura XML, HTML, and text summary.
  - Enforces strict Python **line coverage = 100%** and **branch coverage = 100%** gates.

- `tools/coverage/cpp_coverage.sh`
  - Configures/builds/runs `ctest` with coverage flags.
  - Uses `gcovr` to emit Cobertura XML, HTML, and text summary.
  - Enforces strict C++ **line/function/branch = 100%** gates.

- `tools/coverage/run_coverage_all.sh`
  - Runs Python coverage first, then C++ coverage.
  - Intended as a minimal single-entrypoint for CI or local checks.

- `tools/coverage_ci/run_python_coverage.sh`
  - CI wrapper around `tools/coverage/python_coverage.sh`.
  - Writes `coverage/python-cobertura.xml` and prints `PYTHON_COVERAGE_LINE=...%` for GitLab parsing.

- `tools/coverage_ci/run_cpp_coverage.sh`
  - CI wrapper around `tools/coverage/cpp_coverage.sh`.
  - Writes `coverage/cpp-cobertura.xml` and prints `CPP_COVERAGE_LINE/FUNCTION/BRANCH`.

## Suggested Invocation Sequence

1. `tools/coverage/python_coverage.sh`
2. `tools/coverage/cpp_coverage.sh`

Or use:

1. `tools/coverage/run_coverage_all.sh`

## Usage Examples

```bash
# Python coverage (default tests: tests/builder tests/tools)
tools/coverage/python_coverage.sh

# Python coverage with pytest filters
tools/coverage/python_coverage.sh -k tokenizer -q

# C++ coverage, all ctest tests
tools/coverage/cpp_coverage.sh

# C++ coverage, filtered ctest target
tools/coverage/cpp_coverage.sh -R test_bundle_format

# Combined sequence
tools/coverage/run_coverage_all.sh

# Combined sequence with optional forwarded args
PYTHON_ARGS="-k tokenizer -q" CPP_CTEST_ARGS="-R test_bundle_format" \
  tools/coverage/run_coverage_all.sh
```

## Default Output Locations

- Python:
  - `artifacts/coverage/python/cobertura-python.xml`
  - `artifacts/coverage/python/summary.txt`
  - `artifacts/coverage/python/html/index.html`

- C++:
  - `artifacts/coverage/cpp/cobertura-cpp.xml`
  - `artifacts/coverage/cpp/summary.txt`
  - `artifacts/coverage/cpp/index.html`
  - `artifacts/coverage/cpp/gate.log`

## Assumptions / Prerequisites

- Python coverage script:
  - `python3` (or `PYTHON_BIN`), `pytest`, and `coverage` are installed.
  - `pytest` targets default to `tests/builder tests/tools`, but can be overridden via `PYTHON_TEST_TARGETS`.
  - Note: `coverage.py` does not provide a native function-coverage metric; Python gates use line+branch.

- C++ coverage script:
  - `cmake`, `ctest`, C++ compiler toolchain, and `gcovr` are installed.
  - Toolchain supports `--coverage` instrumentation flags.
  - `gcovr` supports `--fail-under-line`, `--fail-under-function`, and `--fail-under-branch`.
  - Coverage filters default to `<repo>/src` and `<repo>/include`, and can be overridden with `GCOVR_FILTERS`.
