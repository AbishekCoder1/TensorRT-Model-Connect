# Coverage-Based Unit Test Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a coverage map system that uses actual test execution data to selectively run only the unit tests (Python builder + C++) that exercise modified source files, integrated with the existing `test_impact.py`.

**Architecture:** A nightly CI job runs all tests with coverage instrumentation and produces a `coverage_map.json` mapping source files to test IDs. MR pipelines fetch this map and pass it to `test_impact.py` via a new `--coverage-map` flag. The selection script looks up changed files in the map and outputs per-tier filtered test lists consumed by CI jobs.

**Tech Stack:** Python 3.12, coverage.py (with `--cov-context=test`), gcovr 8.2, pytest-cov, GitLab CI artifacts API

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `tools/coverage_map/__init__.py` | Package marker |
| `tools/coverage_map/python_collector.py` | Run Python tests with `--cov-context=test`, parse `.coverage` SQLite DB, output `{source: [test_ids]}` |
| `tools/coverage_map/cpp_collector.py` | Run each ctest binary with gcov reset, parse gcovr JSON, output `{source: [test_names]}` |
| `tools/coverage_map/generate.py` | CLI orchestrator: invoke collectors, merge, write `coverage_map.json` |
| `tools/coverage_map/select_tests.py` | Given git diff + coverage map, output filtered test lists per tier |
| `tools/coverage_map/fetch_latest.py` | Download latest `coverage_map.json` artifact from GitLab CI |
| `tests/tools/test_coverage_map.py` | Unit tests for all coverage_map modules |

### Modified files

| File | Change |
|------|--------|
| `tools/test_impact.py` | Add `--coverage-map` flag; extend `ImpactResult` and `format_json` with `builder_tests`, `cpp_tests`, `tools_tests` fields |
| `.gitlab-ci.yml` | Add `generate-coverage-map` nightly job; modify `test-python-builder` and `test-cpp-unit` to use filtered test lists |
| `tests/tools/test_test_impact.py` | Add tests for `--coverage-map` integration |

---

## Task 1: Python Collector

**Files:**
- Create: `tools/coverage_map/__init__.py`
- Create: `tools/coverage_map/python_collector.py`
- Test: `tests/tools/test_coverage_map.py`

- [ ] **Step 1: Write the test for parsing a coverage.py SQLite database**

Create `tests/tools/test_coverage_map.py`:

```python
"""Tests for tools/coverage_map/ -- coverage-based test selection."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from coverage_map.python_collector import parse_coverage_db


def _create_fake_coverage_db(db_path: Path, data: dict) -> None:
    """Create a minimal coverage.py SQLite database.

    data: {source_path: {context_name: [line_numbers]}}
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS coverage_schema (version INTEGER)")
    conn.execute("INSERT INTO coverage_schema VALUES (7)")
    conn.execute("CREATE TABLE IF NOT EXISTS file (id INTEGER PRIMARY KEY, path TEXT UNIQUE)")
    conn.execute("CREATE TABLE IF NOT EXISTS context (id INTEGER PRIMARY KEY, context TEXT UNIQUE)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS line_bits "
        "(file_id INTEGER, context_id INTEGER, numbits BLOB)"
    )
    file_id = 0
    ctx_id = 0
    for src_path, contexts in data.items():
        file_id += 1
        conn.execute("INSERT INTO file VALUES (?, ?)", (file_id, src_path))
        for ctx_name, _lines in contexts.items():
            ctx_id += 1
            conn.execute("INSERT INTO context VALUES (?, ?)", (ctx_id, ctx_name))
            # numbits content doesn't matter for our query -- we only need the row to exist
            conn.execute(
                "INSERT INTO line_bits VALUES (?, ?, ?)",
                (file_id, ctx_id, b"\x01"),
            )
    conn.commit()
    conn.close()


class TestPythonCollector:
    def test_parse_coverage_db_basic(self, tmp_path):
        """Parses a fake .coverage DB and returns source -> test mapping."""
        db_path = tmp_path / ".coverage"
        _create_fake_coverage_db(db_path, {
            "trtf_build/trtf_build/config.py": {
                "tests/builder/test_config.py::TestModelConfig::test_parse|run": [1, 2, 3],
                "tests/builder/test_config.py::TestModelConfig::test_vl|run": [1, 5],
            },
            "trtf_build/trtf_build/graph_ops.py": {
                "tests/builder/test_graph_ops.py::TestRoPE::test_basic|run": [10, 20],
            },
        })
        result = parse_coverage_db(db_path)
        assert "trtf_build/trtf_build/config.py" in result
        assert sorted(result["trtf_build/trtf_build/config.py"]) == [
            "tests/builder/test_config.py::TestModelConfig::test_parse",
            "tests/builder/test_config.py::TestModelConfig::test_vl",
        ]
        assert result["trtf_build/trtf_build/graph_ops.py"] == [
            "tests/builder/test_graph_ops.py::TestRoPE::test_basic",
        ]

    def test_parse_coverage_db_strips_phase(self, tmp_path):
        """Strips |run, |setup, |teardown from context names."""
        db_path = tmp_path / ".coverage"
        _create_fake_coverage_db(db_path, {
            "trtf_build/trtf_build/config.py": {
                "tests/builder/test_config.py::test_a|setup": [1],
                "tests/builder/test_config.py::test_a|run": [2],
                "tests/builder/test_config.py::test_a|teardown": [3],
            },
        })
        result = parse_coverage_db(db_path)
        # Should deduplicate after stripping phase
        assert result["trtf_build/trtf_build/config.py"] == [
            "tests/builder/test_config.py::test_a",
        ]

    def test_parse_coverage_db_empty(self, tmp_path):
        """Empty DB returns empty mapping."""
        db_path = tmp_path / ".coverage"
        _create_fake_coverage_db(db_path, {})
        result = parse_coverage_db(db_path)
        assert result == {}

    def test_parse_coverage_db_missing_file(self, tmp_path):
        """Missing DB file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_coverage_db(tmp_path / "nonexistent.db")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py -v"`
Expected: FAIL with `ModuleNotFoundError: No module named 'coverage_map.python_collector'`

- [ ] **Step 3: Create package and implement python_collector.py**

Create `tools/coverage_map/__init__.py`:

```python
```

Create `tools/coverage_map/python_collector.py`:

```python
"""Collect per-test coverage data from Python tests using coverage.py.

Runs pytest with --cov-context=test, then parses the resulting .coverage
SQLite database to build a {source_file: [test_node_ids]} mapping.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_coverage_db(db_path: Path) -> Dict[str, List[str]]:
    """Parse a coverage.py SQLite database with per-test contexts.

    Returns:
        {source_file_path: [test_node_id, ...]}
        Test node IDs have the |run/|setup/|teardown phase suffix stripped.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Coverage database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT f.path, c.context "
            "FROM line_bits lb "
            "JOIN file f ON lb.file_id = f.id "
            "JOIN context c ON lb.context_id = c.id"
        ).fetchall()
    finally:
        conn.close()

    source_to_tests: Dict[str, set] = {}
    for src_path, context in rows:
        # Strip phase suffix: "test_foo.py::test_bar|run" -> "test_foo.py::test_bar"
        test_id = context.split("|")[0] if "|" in context else context
        if not test_id:
            continue
        source_to_tests.setdefault(src_path, set()).add(test_id)

    return {src: sorted(tests) for src, tests in source_to_tests.items()}


def collect_python_coverage(
    repo_root: Path,
    test_targets: Optional[List[str]] = None,
    cov_source: str = "trtf_build",
    python_bin: str = "python",
) -> Dict[str, List[str]]:
    """Run Python tests with per-test coverage and return source->tests map.

    Args:
        repo_root: Repository root directory.
        test_targets: pytest target directories (default: tests/builder tests/tools).
        cov_source: Package to measure coverage for.
        python_bin: Python executable to use.

    Returns:
        {source_file_path: [test_node_ids]}
    """
    if test_targets is None:
        test_targets = ["tests/builder", "tests/tools"]

    coverage_file = repo_root / ".coverage-map-gen"

    # Clean up any previous coverage data
    coverage_file.unlink(missing_ok=True)
    for wal in [coverage_file.with_suffix(".db-wal"), coverage_file.with_suffix(".db-shm")]:
        wal.unlink(missing_ok=True)

    cmd = [
        python_bin, "-m", "pytest",
        *test_targets,
        f"--cov={cov_source}",
        "--cov-context=test",
        "--cov-report=",  # suppress report output
        "-q",
        "--ignore=tests/builder/test_cli.py",
    ]

    env = {
        "COVERAGE_FILE": str(coverage_file),
        "PYTHONPATH": str(repo_root),
    }

    # Inherit current environment, override specific vars
    import os
    full_env = {**os.environ, **env}

    subprocess.run(
        cmd, cwd=repo_root, env=full_env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    result = parse_coverage_db(coverage_file)

    # Clean up
    coverage_file.unlink(missing_ok=True)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestPythonCollector -v"`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/coverage_map/__init__.py tools/coverage_map/python_collector.py tests/tools/test_coverage_map.py
git commit -m "feat(coverage-map): add Python coverage collector with per-test context parsing"
```

---

## Task 2: C++ Collector

**Files:**
- Create: `tools/coverage_map/cpp_collector.py`
- Modify: `tests/tools/test_coverage_map.py`

- [ ] **Step 1: Write the test for parsing gcovr JSON output**

Append to `tests/tools/test_coverage_map.py`:

```python
from coverage_map.cpp_collector import parse_gcovr_json, build_cpp_map_from_jsons


class TestCppCollector:
    def test_parse_gcovr_json_basic(self, tmp_path):
        """Extracts covered source files from a gcovr JSON report."""
        gcovr_data = {
            "files": [
                {
                    "filename": "/workspace/trt-transformers-cpp/src/tokenizer/vocab_tokenizer.cpp",
                    "line_covered": 20,
                    "line_total": 50,
                },
                {
                    "filename": "/workspace/trt-transformers-cpp/src/bundle/bundle_format.cpp",
                    "line_covered": 0,
                    "line_total": 30,
                },
                {
                    "filename": "/workspace/trt-transformers-cpp/include/trtf/runtime/pipeline_plugin.h",
                    "line_covered": 5,
                    "line_total": 10,
                },
            ]
        }
        json_path = tmp_path / "cov.json"
        json_path.write_text(json.dumps(gcovr_data))
        result = parse_gcovr_json(json_path, repo_root=Path("/workspace/trt-transformers-cpp"))
        # Only files with line_covered > 0
        assert "src/tokenizer/vocab_tokenizer.cpp" in result
        assert "include/trtf/runtime/pipeline_plugin.h" in result
        assert "src/bundle/bundle_format.cpp" not in result

    def test_parse_gcovr_json_empty(self, tmp_path):
        """Empty gcovr report returns empty set."""
        json_path = tmp_path / "cov.json"
        json_path.write_text(json.dumps({"files": []}))
        result = parse_gcovr_json(json_path, repo_root=Path("/workspace/repo"))
        assert result == set()

    def test_build_cpp_map(self, tmp_path):
        """Builds source->tests mapping from per-test gcovr JSONs."""
        # test_vocab_tokenizer covers vocab_tokenizer.cpp + text_parsers.cpp
        (tmp_path / "test_vocab_tokenizer.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/tokenizer/vocab_tokenizer.cpp", "line_covered": 20, "line_total": 50},
            {"filename": "/repo/src/utils/text_parsers.cpp", "line_covered": 5, "line_total": 30},
        ]}))
        # test_bundle_format covers bundle_format.cpp
        (tmp_path / "test_bundle_format.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/bundle/bundle_format.cpp", "line_covered": 10, "line_total": 20},
        ]}))
        result = build_cpp_map_from_jsons(tmp_path, repo_root=Path("/repo"))
        assert result["src/tokenizer/vocab_tokenizer.cpp"] == ["test_vocab_tokenizer"]
        assert result["src/utils/text_parsers.cpp"] == ["test_vocab_tokenizer"]
        assert result["src/bundle/bundle_format.cpp"] == ["test_bundle_format"]

    def test_build_cpp_map_shared_source(self, tmp_path):
        """Two tests covering the same source file are both listed."""
        (tmp_path / "test_a.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/common.cpp", "line_covered": 5, "line_total": 10},
        ]}))
        (tmp_path / "test_b.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/common.cpp", "line_covered": 3, "line_total": 10},
        ]}))
        result = build_cpp_map_from_jsons(tmp_path, repo_root=Path("/repo"))
        assert sorted(result["src/common.cpp"]) == ["test_a", "test_b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestCppCollector -v"`
Expected: FAIL with `ImportError: cannot import name 'parse_gcovr_json'`

- [ ] **Step 3: Implement cpp_collector.py**

Create `tools/coverage_map/cpp_collector.py`:

```python
"""Collect per-test coverage data from C++ tests using gcov/gcovr.

Runs each ctest binary individually with gcda reset between runs,
captures gcovr JSON output, and builds a {source_file: [test_names]} mapping.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set


def parse_gcovr_json(json_path: Path, repo_root: Path) -> Set[str]:
    """Parse a gcovr JSON report and return set of covered source files.

    Only includes files with at least one covered line.
    Paths are returned relative to repo_root.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    covered = set()
    repo_prefix = str(repo_root).rstrip("/") + "/"
    for file_entry in data.get("files", []):
        if file_entry.get("line_covered", 0) > 0:
            path = file_entry["filename"]
            if path.startswith(repo_prefix):
                path = path[len(repo_prefix):]
            covered.add(path)
    return covered


def build_cpp_map_from_jsons(
    json_dir: Path,
    repo_root: Path,
) -> Dict[str, List[str]]:
    """Build source->test mapping from per-test gcovr JSON files.

    Expects files named <test_name>.json in json_dir.
    """
    source_to_tests: Dict[str, set] = {}
    for json_path in sorted(json_dir.glob("*.json")):
        test_name = json_path.stem
        covered_files = parse_gcovr_json(json_path, repo_root)
        for src in covered_files:
            source_to_tests.setdefault(src, set()).add(test_name)
    return {src: sorted(tests) for src, tests in source_to_tests.items()}


def _list_ctest_names(build_dir: Path) -> List[str]:
    """List all registered ctest names."""
    result = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N", "--quiet"],
        capture_output=True, text=True, check=True,
    )
    names = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # ctest -N output: "  Test #1: test_bundle_format"
        if line.startswith("Test #"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                names.append(parts[1].strip())
    return names


def collect_cpp_coverage(
    repo_root: Path,
    build_dir: Path,
    output_dir: Optional[Path] = None,
    gcovr_filters: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Run each ctest individually with gcov and build source->test mapping.

    Args:
        repo_root: Repository root directory.
        build_dir: CMake build directory (must be built with --coverage).
        output_dir: Where to write per-test gcovr JSONs (default: temp dir).
        gcovr_filters: gcovr --filter values (default: src/ and include/).
    """
    import tempfile

    if gcovr_filters is None:
        gcovr_filters = [
            str(repo_root / "src"),
            str(repo_root / "include"),
        ]

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="cpp_covmap_"))

    output_dir.mkdir(parents=True, exist_ok=True)
    test_names = _list_ctest_names(build_dir)

    for test_name in test_names:
        # Reset gcda counters
        for gcda in build_dir.rglob("*.gcda"):
            gcda.unlink()

        # Run the single test
        test_result = subprocess.run(
            ["ctest", "--test-dir", str(build_dir), "-R", f"^{test_name}$",
             "--output-on-failure"],
            capture_output=True, text=True,
        )
        if test_result.returncode != 0:
            # Test failed or was skipped -- still collect what coverage we have
            pass

        # Capture gcovr JSON
        gcovr_cmd = [
            "gcovr",
            "--root", str(repo_root),
            "--object-directory", str(build_dir),
            "--json", "-o", str(output_dir / f"{test_name}.json"),
            "--exclude", str(repo_root / "tests"),
        ]
        for f in gcovr_filters:
            gcovr_cmd.extend(["--filter", f])

        subprocess.run(gcovr_cmd, capture_output=True, text=True)

    return build_cpp_map_from_jsons(output_dir, repo_root)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestCppCollector -v"`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/coverage_map/cpp_collector.py tests/tools/test_coverage_map.py
git commit -m "feat(coverage-map): add C++ coverage collector with per-test gcov isolation"
```

---

## Task 3: Generate Orchestrator

**Files:**
- Create: `tools/coverage_map/generate.py`
- Modify: `tests/tools/test_coverage_map.py`

- [ ] **Step 1: Write the test for map merging and validation**

Append to `tests/tools/test_coverage_map.py`:

```python
from coverage_map.generate import merge_maps, validate_map, load_coverage_map


class TestGenerate:
    def test_merge_maps_disjoint(self):
        """Merging two disjoint maps produces their union."""
        py_map = {"trtf_build/config.py": ["test_config.py::test_a"]}
        cpp_map = {"src/vocab.cpp": ["test_vocab"]}
        merged = merge_maps(py_map, cpp_map)
        assert merged["trtf_build/config.py"] == ["test_config.py::test_a"]
        assert merged["src/vocab.cpp"] == ["test_vocab"]

    def test_merge_maps_overlapping(self):
        """Overlapping keys produce union of test lists, sorted and deduplicated."""
        map_a = {"shared.py": ["test_a", "test_b"]}
        map_b = {"shared.py": ["test_b", "test_c"]}
        merged = merge_maps(map_a, map_b)
        assert merged["shared.py"] == ["test_a", "test_b", "test_c"]

    def test_validate_map_clean(self, tmp_path):
        """No warnings when all files and tests exist."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.cpp").write_text("")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text("")
        coverage_map = {"src/foo.cpp": ["tests/test_foo.py::test_a"]}
        warnings = validate_map(coverage_map, tmp_path)
        assert warnings == []

    def test_validate_map_missing_source(self, tmp_path):
        """Warning when a source file in the map no longer exists."""
        coverage_map = {"src/deleted.cpp": ["test_x"]}
        warnings = validate_map(coverage_map, tmp_path)
        assert any("src/deleted.cpp" in w for w in warnings)

    def test_load_coverage_map(self, tmp_path):
        """Loads and returns the source_to_tests portion of a coverage_map.json."""
        data = {
            "meta": {"commit": "abc", "generated_at": "2026-01-01T00:00:00Z",
                     "python_tests": 10, "cpp_tests": 5},
            "source_to_tests": {"src/a.cpp": ["test_a"]},
        }
        path = tmp_path / "coverage_map.json"
        path.write_text(json.dumps(data))
        result = load_coverage_map(path)
        assert result == {"src/a.cpp": ["test_a"]}

    def test_load_coverage_map_missing_file(self, tmp_path):
        """Returns None when the map file doesn't exist."""
        result = load_coverage_map(tmp_path / "missing.json")
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestGenerate -v"`
Expected: FAIL with `ImportError: cannot import name 'merge_maps'`

- [ ] **Step 3: Implement generate.py**

Create `tools/coverage_map/generate.py`:

```python
#!/usr/bin/env python3
"""Generate a unified coverage map from Python and C++ test coverage data.

Usage:
    python tools/coverage_map/generate.py --output coverage_map.json
    python tools/coverage_map/generate.py --python-only --output coverage_map.json
    python tools/coverage_map/generate.py --cpp-only --output coverage_map.json
    python tools/coverage_map/generate.py --validate coverage_map.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def merge_maps(*maps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge multiple source->tests mappings. Union test lists, deduplicate, sort."""
    merged: Dict[str, set] = {}
    for m in maps:
        for src, tests in m.items():
            merged.setdefault(src, set()).update(tests)
    return {src: sorted(tests) for src, tests in sorted(merged.items())}


def validate_map(
    source_to_tests: Dict[str, List[str]],
    repo_root: Path,
) -> List[str]:
    """Check that source files in the map still exist on disk.

    Returns list of warning strings. Empty list means all clean.
    """
    warnings = []
    for src_path in source_to_tests:
        full_path = repo_root / src_path
        if not full_path.exists():
            warnings.append(f"Source file no longer exists: {src_path}")
    return warnings


def load_coverage_map(path: Path) -> Optional[Dict[str, List[str]]]:
    """Load a coverage_map.json and return the source_to_tests dict.

    Returns None if the file doesn't exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("source_to_tests", {})


def _get_head_commit(repo_root: Path) -> str:
    """Get the current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=repo_root,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate(
    repo_root: Path,
    output_path: Path,
    python_only: bool = False,
    cpp_only: bool = False,
    build_dir: Optional[Path] = None,
    python_bin: str = "python",
) -> Dict[str, List[str]]:
    """Generate the unified coverage map.

    Args:
        repo_root: Repository root.
        output_path: Where to write coverage_map.json.
        python_only: Only collect Python coverage.
        cpp_only: Only collect C++ coverage.
        build_dir: CMake build directory (for C++ collection).
        python_bin: Python executable for pytest.

    Returns:
        The merged source_to_tests mapping.
    """
    from .python_collector import collect_python_coverage
    from .cpp_collector import collect_cpp_coverage

    py_map: Dict[str, List[str]] = {}
    cpp_map: Dict[str, List[str]] = {}
    py_count = 0
    cpp_count = 0

    if not cpp_only:
        print("[coverage-map] Collecting Python coverage...", file=sys.stderr)
        py_map = collect_python_coverage(repo_root, python_bin=python_bin)
        py_count = len({t for tests in py_map.values() for t in tests})
        print(f"[coverage-map] Python: {len(py_map)} source files, {py_count} tests",
              file=sys.stderr)

    if not python_only:
        if build_dir is None:
            build_dir = repo_root / "build"
        print("[coverage-map] Collecting C++ coverage...", file=sys.stderr)
        cpp_map = collect_cpp_coverage(repo_root, build_dir)
        cpp_count = len({t for tests in cpp_map.values() for t in tests})
        print(f"[coverage-map] C++: {len(cpp_map)} source files, {cpp_count} tests",
              file=sys.stderr)

    merged = merge_maps(py_map, cpp_map)

    output = {
        "meta": {
            "commit": _get_head_commit(repo_root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python_tests": py_count,
            "cpp_tests": cpp_count,
        },
        "source_to_tests": merged,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[coverage-map] Written to {output_path} "
          f"({len(merged)} source files)", file=sys.stderr)

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate coverage map for test selection.")
    parser.add_argument("--output", "-o", required=True, help="Output coverage_map.json path")
    parser.add_argument("--python-only", action="store_true", help="Only collect Python coverage")
    parser.add_argument("--cpp-only", action="store_true", help="Only collect C++ coverage")
    parser.add_argument("--build-dir", default=None, help="CMake build directory")
    parser.add_argument("--python-bin", default="python", help="Python executable")
    parser.add_argument("--repo-root", default=None, help="Repository root (default: auto)")
    parser.add_argument("--validate", metavar="MAP_PATH",
                        help="Validate an existing coverage map")
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, check=True,
            )
            repo_root = Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            repo_root = Path.cwd()

    if args.validate:
        source_to_tests = load_coverage_map(Path(args.validate))
        if source_to_tests is None:
            print(f"ERROR: Cannot load {args.validate}", file=sys.stderr)
            return 1
        warnings = validate_map(source_to_tests, repo_root)
        if warnings:
            for w in warnings:
                print(f"  WARN: {w}", file=sys.stderr)
            print(f"Validation: {len(warnings)} warnings.", file=sys.stderr)
        else:
            print("Validation passed: all source files exist.", file=sys.stderr)
        return 0

    build_dir = Path(args.build_dir) if args.build_dir else None
    generate(
        repo_root=repo_root,
        output_path=Path(args.output),
        python_only=args.python_only,
        cpp_only=args.cpp_only,
        build_dir=build_dir,
        python_bin=args.python_bin,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestGenerate -v"`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/coverage_map/generate.py tests/tools/test_coverage_map.py
git commit -m "feat(coverage-map): add generate orchestrator with merge and validation"
```

---

## Task 4: Test Selection Logic

**Files:**
- Create: `tools/coverage_map/select_tests.py`
- Modify: `tests/tools/test_coverage_map.py`

- [ ] **Step 1: Write tests for the selection algorithm**

Append to `tests/tools/test_coverage_map.py`:

```python
from coverage_map.select_tests import select_tests, SelectionResult


class TestSelectTests:
    @pytest.fixture
    def sample_map(self):
        return {
            "src/tokenizer/vocab_tokenizer.cpp": ["test_vocab_tokenizer"],
            "src/bundle/bundle_format.cpp": ["test_bundle_format", "test_bundle_e2e"],
            "src/utils/text_parsers.cpp": ["test_text_parsers", "test_vocab_tokenizer"],
            "trtf_build/trtf_build/config.py": [
                "tests/builder/test_config.py::TestModelConfig::test_parse",
                "tests/builder/test_config.py::TestModelConfig::test_vl",
            ],
            "trtf_build/trtf_build/graph_ops.py": [
                "tests/builder/test_graph_ops.py::TestRoPE::test_basic",
            ],
        }

    def test_known_cpp_file(self, sample_map):
        """Known C++ source file returns its mapped tests."""
        result = select_tests(["src/tokenizer/vocab_tokenizer.cpp"], sample_map)
        assert sorted(result.cpp_tests) == ["test_vocab_tokenizer"]
        assert result.builder_tests == []
        assert result.fallback_tiers == []

    def test_known_python_file(self, sample_map):
        """Known Python source file returns its mapped tests."""
        result = select_tests(["trtf_build/trtf_build/config.py"], sample_map)
        assert sorted(result.builder_tests) == [
            "tests/builder/test_config.py::TestModelConfig::test_parse",
            "tests/builder/test_config.py::TestModelConfig::test_vl",
        ]
        assert result.cpp_tests == []

    def test_unknown_cpp_file_fallback(self, sample_map):
        """Unknown src/ file triggers cpp tier fallback."""
        result = select_tests(["src/new_module.cpp"], sample_map)
        assert "cpp" in result.fallback_tiers
        assert result.cpp_tests == []

    def test_unknown_python_file_fallback(self, sample_map):
        """Unknown trtf_build/ file triggers builder tier fallback."""
        result = select_tests(["trtf_build/trtf_build/new_module.py"], sample_map)
        assert "builder" in result.fallback_tiers
        assert result.builder_tests == []

    def test_multiple_files_union(self, sample_map):
        """Multiple changed files produce union of tests."""
        result = select_tests([
            "src/tokenizer/vocab_tokenizer.cpp",
            "src/bundle/bundle_format.cpp",
        ], sample_map)
        assert sorted(result.cpp_tests) == [
            "test_bundle_e2e", "test_bundle_format", "test_vocab_tokenizer",
        ]

    def test_non_code_file_no_impact(self, sample_map):
        """docs/ file has no test impact."""
        result = select_tests(["docs/README.md"], sample_map)
        assert result.cpp_tests == []
        assert result.builder_tests == []
        assert result.fallback_tiers == []

    def test_include_header_fallback(self, sample_map):
        """Unknown include/ header triggers cpp fallback."""
        result = select_tests(["include/trtf/new_header.h"], sample_map)
        assert "cpp" in result.fallback_tiers

    def test_mixed_cpp_and_python(self, sample_map):
        """Changes in both languages select tests from both."""
        result = select_tests([
            "src/tokenizer/vocab_tokenizer.cpp",
            "trtf_build/trtf_build/config.py",
        ], sample_map)
        assert "test_vocab_tokenizer" in result.cpp_tests
        assert "tests/builder/test_config.py::TestModelConfig::test_parse" in result.builder_tests

    def test_empty_changed_files(self, sample_map):
        """No changed files -> no tests."""
        result = select_tests([], sample_map)
        assert result.cpp_tests == []
        assert result.builder_tests == []
        assert result.fallback_tiers == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestSelectTests -v"`
Expected: FAIL with `ImportError: cannot import name 'select_tests'`

- [ ] **Step 3: Implement select_tests.py**

Create `tools/coverage_map/select_tests.py`:

```python
#!/usr/bin/env python3
"""Select tests to run based on changed files and a coverage map.

Given a list of changed files and a coverage_map.json, determines which
specific unit tests need to run. Falls back to full-tier execution for
files not present in the coverage map (zero false negatives).

Usage:
    python tools/coverage_map/select_tests.py --coverage-map coverage_map.json --files file1,file2
    python tools/coverage_map/select_tests.py --coverage-map coverage_map.json --diff <(git diff master)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Patterns for files that never affect unit tests
_NO_IMPACT_PATTERNS = [
    r"^docs/",
    r"^\.gitignore$",
    r"^\.clang-format$",
    r"^\.editorconfig$",
    r"^\.github/",
    r"^\.gitlab-ci\.yml$",
    r"^\.gitlab/",
    r"^\.claude/",
    r"^LICENSE",
    r"^CLAUDE\.md$",
    r"^recovery-",
    r"^scripts/",
]


@dataclass
class SelectionResult:
    """Per-tier test selection result."""
    cpp_tests: List[str] = field(default_factory=list)
    builder_tests: List[str] = field(default_factory=list)
    tools_tests: List[str] = field(default_factory=list)
    fallback_tiers: List[str] = field(default_factory=list)


def _is_no_impact(path: str) -> bool:
    """Check if a file path has no impact on unit tests."""
    if path.endswith(".md"):
        return True
    for pattern in _NO_IMPACT_PATTERNS:
        if re.match(pattern, path):
            return True
    # tools/ and tests/e2e* don't affect unit tests
    if path.startswith("tools/") and not path.startswith("tools/coverage_map/"):
        return True
    if path.startswith("tests/e2e"):
        return True
    return False


def _classify_tier(path: str) -> Optional[str]:
    """Determine which unit test tier a source file belongs to."""
    if path.startswith("src/") or path.startswith("include/"):
        return "cpp"
    if path == "CMakeLists.txt" or path.startswith("cmake/"):
        return "cpp"
    if path.startswith("trtf_build/"):
        return "builder"
    if path.startswith("tests/builder/"):
        return "builder"
    if path.startswith("tests/cpp/"):
        return "cpp"
    if path.startswith("tests/tools/"):
        return "tools"
    return None


def select_tests(
    changed_files: List[str],
    source_to_tests: Dict[str, List[str]],
) -> SelectionResult:
    """Select tests based on changed files and coverage map.

    For each changed file:
    - If it's in the coverage map: select the specific tests that cover it.
    - If it's a source file NOT in the map: fall back to running all tests
      in that tier (zero false negatives).
    - If it's a no-impact file (docs, scripts, etc.): skip.

    Returns a SelectionResult with per-tier test lists and any fallback tiers.
    """
    cpp_tests: set = set()
    builder_tests: set = set()
    tools_tests: set = set()
    fallback_tiers: set = set()

    for path in changed_files:
        path = path.replace("\\", "/").strip("/")

        if _is_no_impact(path):
            continue

        tier = _classify_tier(path)
        if tier is None:
            continue

        if path in source_to_tests:
            tests = source_to_tests[path]
            for test_id in tests:
                # C++ tests are plain names, Python tests contain "::"
                if "::" in test_id or test_id.startswith("tests/"):
                    builder_tests.add(test_id)
                else:
                    cpp_tests.add(test_id)
        else:
            # Source file not in map -- fall back to full tier
            fallback_tiers.add(tier)

    return SelectionResult(
        cpp_tests=sorted(cpp_tests),
        builder_tests=sorted(builder_tests),
        tools_tests=sorted(tools_tests),
        fallback_tiers=sorted(fallback_tiers),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select unit tests based on coverage map and changed files.",
    )
    parser.add_argument("--coverage-map", required=True, help="Path to coverage_map.json")
    parser.add_argument("--files", help="Comma-separated list of changed files")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    args = parser.parse_args()

    from .generate import load_coverage_map

    source_to_tests = load_coverage_map(Path(args.coverage_map))
    if source_to_tests is None:
        print("ERROR: Coverage map not found. Run all tests.", file=sys.stderr)
        return 1

    changed = [f.strip() for f in args.files.split(",") if f.strip()] if args.files else []
    result = select_tests(changed, source_to_tests)

    if args.json_output:
        print(json.dumps({
            "cpp_tests": result.cpp_tests,
            "builder_tests": result.builder_tests,
            "tools_tests": result.tools_tests,
            "fallback_tiers": result.fallback_tiers,
        }, indent=2))
    else:
        if result.cpp_tests:
            print(f"C++ tests ({len(result.cpp_tests)}):")
            for t in result.cpp_tests:
                print(f"  {t}")
        if result.builder_tests:
            print(f"Builder tests ({len(result.builder_tests)}):")
            for t in result.builder_tests:
                print(f"  {t}")
        if result.fallback_tiers:
            print(f"Fallback tiers (run all): {', '.join(result.fallback_tiers)}")
        if not result.cpp_tests and not result.builder_tests and not result.fallback_tiers:
            print("No unit tests affected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestSelectTests -v"`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/coverage_map/select_tests.py tests/tools/test_coverage_map.py
git commit -m "feat(coverage-map): add test selection logic with fallback safety"
```

---

## Task 5: Integrate with test_impact.py

**Files:**
- Modify: `tools/test_impact.py` (lines ~196-207, ~519-553, ~668-675, ~682-775)
- Modify: `tests/tools/test_test_impact.py`

- [ ] **Step 1: Write test for coverage-map integration in test_impact.py**

Append to `tests/tools/test_test_impact.py`:

```python
class TestCoverageMapIntegration:
    def test_impact_result_has_test_lists(self, imap):
        """ImpactResult with coverage map includes per-tier test lists."""
        coverage_map = {
            "trtf_build/trtf_build/families/qwen.py": [
                "tests/builder/test_engine_qwen.py::TestQwen::test_plugin",
            ],
        }
        result = test_impact.analyze_impact(
            ["trtf_build/trtf_build/families/qwen.py"], imap,
            coverage_map=coverage_map,
        )
        assert "tests/builder/test_engine_qwen.py::TestQwen::test_plugin" in result.builder_tests
        assert "builder" not in result.fallback_tiers

    def test_unknown_file_triggers_fallback(self, imap):
        """File not in coverage map triggers tier fallback."""
        coverage_map = {"trtf_build/trtf_build/config.py": ["tests/builder/test_config.py::test_a"]}
        result = test_impact.analyze_impact(
            ["trtf_build/trtf_build/families/qwen.py"], imap,
            coverage_map=coverage_map,
        )
        # qwen.py not in coverage_map -> builder fallback
        assert "builder" in result.fallback_tiers

    def test_no_coverage_map_no_test_lists(self, imap):
        """Without coverage map, test lists are empty and fallback_tiers empty."""
        result = test_impact.analyze_impact(
            ["trtf_build/trtf_build/families/qwen.py"], imap,
        )
        assert result.builder_tests == []
        assert result.cpp_tests == []
        assert result.fallback_tiers == []

    def test_json_output_includes_test_lists(self, imap):
        """JSON output includes builder_tests, cpp_tests, fallback_tiers."""
        result = test_impact.ImpactResult(
            e2e_models=["qwen3-0.6b"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[],
            builder_tests=["tests/builder/test_config.py::test_a"],
            cpp_tests=[],
            tools_tests=[],
            fallback_tiers=[],
        )
        output = json.loads(test_impact.format_json(result))
        assert output["builder_tests"] == ["tests/builder/test_config.py::test_a"]
        assert output["cpp_tests"] == []
        assert output["fallback_tiers"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_test_impact.py::TestCoverageMapIntegration -v"`
Expected: FAIL with `TypeError` (analyze_impact doesn't accept coverage_map parameter)

- [ ] **Step 3: Modify test_impact.py to integrate coverage map**

Add new fields to `ImpactResult` dataclass (after line 206):

```python
@dataclass
class ImpactResult:
    e2e_models: List[str]
    unit_tiers: List[str]
    rebuild_cpp: bool
    cap_applied: bool
    matched_rules: List[Dict]
    builder_tests: List[str] = field(default_factory=list)
    cpp_tests: List[str] = field(default_factory=list)
    tools_tests: List[str] = field(default_factory=list)
    fallback_tiers: List[str] = field(default_factory=list)
```

Modify `analyze_impact` to accept optional coverage_map (around line 519):

```python
def analyze_impact(
    changed_files: List[str],
    imap: ImpactMap,
    cap: Optional[int] = None,
    coverage_map: Optional[Dict[str, List[str]]] = None,
) -> ImpactResult:
    """Analyze impact of all changed files and return aggregated result."""
    all_models: Set[str] = set()
    all_tiers: Set[str] = set()
    rebuild_cpp = False
    matched_rules: List[Dict] = []

    for fpath in changed_files:
        match = classify_file(fpath, imap)
        all_models.update(match.models)
        all_tiers.update(match.unit_tiers)
        rebuild_cpp = rebuild_cpp or match.rebuild_cpp
        matched_rules.append({
            "file": fpath,
            "rule": match.rule,
            "models": match.models,
        })

    e2e_models = sorted(all_models)
    cap_applied = False
    if cap is not None and len(e2e_models) > cap:
        e2e_models = sorted(imap.core_models)
        cap_applied = True

    # Coverage-map-based unit test selection
    builder_tests: List[str] = []
    cpp_tests: List[str] = []
    tools_tests: List[str] = []
    fallback_tiers: List[str] = []

    if coverage_map is not None:
        from coverage_map.select_tests import select_tests
        sel = select_tests(changed_files, coverage_map)
        builder_tests = sel.builder_tests
        cpp_tests = sel.cpp_tests
        tools_tests = sel.tools_tests
        fallback_tiers = sel.fallback_tiers

    return ImpactResult(
        e2e_models=e2e_models,
        unit_tiers=sorted(all_tiers),
        rebuild_cpp=rebuild_cpp,
        cap_applied=cap_applied,
        matched_rules=matched_rules,
        builder_tests=builder_tests,
        cpp_tests=cpp_tests,
        tools_tests=tools_tests,
        fallback_tiers=fallback_tiers,
    )
```

Update `format_json` (around line 668):

```python
def format_json(result: ImpactResult) -> str:
    return json.dumps({
        "e2e_models": result.e2e_models,
        "unit_tiers": result.unit_tiers,
        "rebuild_cpp": result.rebuild_cpp,
        "cap_applied": result.cap_applied,
        "matched_rules": result.matched_rules,
        "builder_tests": result.builder_tests,
        "cpp_tests": result.cpp_tests,
        "tools_tests": result.tools_tests,
        "fallback_tiers": result.fallback_tiers,
    }, indent=2)
```

Add `--coverage-map` CLI argument (in `main()`, after the `--repo-root` argument):

```python
    parser.add_argument("--coverage-map", default=None,
                        help="Path to coverage_map.json for per-test selection")
```

And load it before calling `analyze_impact` (after the changed-files resolution block):

```python
    # Load coverage map if provided
    coverage_map = None
    if args.coverage_map:
        sys.path.insert(0, str(repo_root / "tools"))
        from coverage_map.generate import load_coverage_map
        coverage_map = load_coverage_map(Path(args.coverage_map))
        if coverage_map is None:
            print(f"WARNING: Coverage map not found at {args.coverage_map}. "
                  "Falling back to tier-level selection.", file=sys.stderr)
```

Then pass `coverage_map=coverage_map` to the `analyze_impact(...)` call.

- [ ] **Step 4: Run all test_impact tests to verify nothing broke**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_test_impact.py -v"`
Expected: All existing tests PASS + 4 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/test_impact.py tests/tools/test_test_impact.py
git commit -m "feat(coverage-map): integrate coverage map into test_impact.py"
```

---

## Task 6: Fetch Latest Artifact

**Files:**
- Create: `tools/coverage_map/fetch_latest.py`
- Modify: `tests/tools/test_coverage_map.py`

- [ ] **Step 1: Write test for fetch fallback logic**

Append to `tests/tools/test_coverage_map.py`:

```python
from coverage_map.fetch_latest import resolve_coverage_map


class TestFetchLatest:
    def test_local_path_exists(self, tmp_path):
        """Returns local path when file exists."""
        path = tmp_path / "coverage_map.json"
        path.write_text('{"meta": {}, "source_to_tests": {}}')
        result = resolve_coverage_map(
            output_path=tmp_path / "output.json",
            local_fallback=str(path),
            gitlab_api_url=None,
        )
        assert result is True
        assert (tmp_path / "output.json").exists()

    def test_local_path_missing(self, tmp_path):
        """Returns False when local fallback doesn't exist and no API."""
        result = resolve_coverage_map(
            output_path=tmp_path / "output.json",
            local_fallback=str(tmp_path / "missing.json"),
            gitlab_api_url=None,
        )
        assert result is False

    def test_copies_to_output(self, tmp_path):
        """Copies the source file to output_path."""
        source = tmp_path / "source.json"
        source.write_text('{"meta": {}, "source_to_tests": {"a.py": ["test_a"]}}')
        output = tmp_path / "subdir" / "output.json"
        result = resolve_coverage_map(
            output_path=output,
            local_fallback=str(source),
            gitlab_api_url=None,
        )
        assert result is True
        data = json.loads(output.read_text())
        assert "a.py" in data["source_to_tests"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestFetchLatest -v"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement fetch_latest.py**

Create `tools/coverage_map/fetch_latest.py`:

```python
#!/usr/bin/env python3
"""Fetch the latest coverage_map.json artifact for CI use.

Tries GitLab Jobs API first, then falls back to a local path.
Exits with code 1 if no map is available (signals full-tier fallback).

Usage:
    python tools/coverage_map/fetch_latest.py --output coverage_map.json
    python tools/coverage_map/fetch_latest.py --output coverage_map.json --local-fallback /shared/coverage_map.json
"""

import argparse
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path


def _try_gitlab_download(api_url: str, output_path: Path) -> bool:
    """Try to download coverage_map.json from GitLab artifacts API.

    Uses CI_JOB_TOKEN if available (CI environment) or PRIVATE_TOKEN.
    """
    import os
    token = os.environ.get("CI_JOB_TOKEN") or os.environ.get("PRIVATE_TOKEN")
    if not token or not api_url:
        return False

    headers = {"PRIVATE-TOKEN": token} if os.environ.get("PRIVATE_TOKEN") else {"JOB-TOKEN": token}

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.read())
            return True
    except (urllib.error.URLError, OSError) as e:
        print(f"WARNING: GitLab API fetch failed: {e}", file=sys.stderr)
        return False


def resolve_coverage_map(
    output_path: Path,
    local_fallback: str = "",
    gitlab_api_url: str = None,
) -> bool:
    """Try to resolve a coverage map, writing it to output_path.

    Tries in order:
    1. GitLab artifacts API (if gitlab_api_url is set)
    2. Local fallback path (if provided and exists)

    Returns True if a map was written to output_path, False otherwise.
    """
    # Try GitLab API
    if gitlab_api_url:
        if _try_gitlab_download(gitlab_api_url, output_path):
            print(f"[fetch] Downloaded coverage map from GitLab API", file=sys.stderr)
            return True

    # Try local fallback
    if local_fallback:
        local_path = Path(local_fallback)
        if local_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, output_path)
            print(f"[fetch] Using local fallback: {local_path}", file=sys.stderr)
            return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch latest coverage_map.json for CI.",
    )
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--local-fallback", default="",
                        help="Local path to fall back to if API fails")
    parser.add_argument("--gitlab-api-url", default=None,
                        help="GitLab artifacts API URL")
    args = parser.parse_args()

    found = resolve_coverage_map(
        output_path=Path(args.output),
        local_fallback=args.local_fallback,
        gitlab_api_url=args.gitlab_api_url,
    )

    if not found:
        print("WARNING: No coverage map available. Run all tests.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py::TestFetchLatest -v"`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/coverage_map/fetch_latest.py tests/tools/test_coverage_map.py
git commit -m "feat(coverage-map): add artifact fetcher with GitLab API and local fallback"
```

---

## Task 7: CI Pipeline Integration

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Add nightly coverage map generation stage**

Add `nightly-coverage` to stages list (after `nightly-e2e`):

```yaml
stages:
  - analyze
  - build
  - tier1-unit
  - tier2-graph-ops
  - tier3-e2e
  - nightly-e2e
  - nightly-coverage
```

Add the `generate-coverage-map` job at the end of the file:

```yaml
# ── Nightly: Coverage Map Generation ──────────────────────────────────────────
# Generates per-test coverage mapping for selective unit test execution.
# MR pipelines fetch this artifact to skip irrelevant unit tests.

generate-coverage-map:
  stage: nightly-coverage
  needs: [build-all]
  script:
    - python -m pip install --disable-pip-version-check --quiet "coverage[toml]==7.6.10" "pytest-cov>=6.0" "gcovr==8.2"
    - python tools/coverage_map/generate.py --output coverage_map.json --python-bin python --build-dir build
    - python tools/coverage_map/generate.py --validate coverage_map.json
    - echo "Coverage map generated successfully"
    - python -c "import json; d=json.load(open('coverage_map.json')); m=d['meta']; print(f'Python tests: {m[\"python_tests\"]}, C++ tests: {m[\"cpp_tests\"]}, Source files: {len(d[\"source_to_tests\"])}')"
  artifacts:
    paths:
      - coverage_map.json
    expire_in: 30 days
  timeout: 90m
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

- [ ] **Step 2: Modify impact-analysis job to fetch and use coverage map**

Update the `impact-analysis` job script:

```yaml
impact-analysis:
  stage: analyze
  before_script: []   # No deps needed — pure Python + git
  script:
    - python3 tools/test_impact.py --validate
    # Try to fetch coverage map (non-fatal if unavailable)
    - |
      python3 tools/coverage_map/fetch_latest.py \
        --output coverage_map.json \
        --local-fallback "${COVERAGE_MAP_PATH:-}" \
        || echo "No coverage map available -- using tier-level selection"
    # Run impact analysis with coverage map if available
    - |
      if [ -f coverage_map.json ]; then
        python3 tools/test_impact.py --base ${CI_MERGE_REQUEST_DIFF_BASE_SHA:-HEAD~1} --coverage-map coverage_map.json --json > impact.json
      else
        python3 tools/test_impact.py --base ${CI_MERGE_REQUEST_DIFF_BASE_SHA:-HEAD~1} --json > impact.json
      fi
    - echo "--- Impact Analysis ---"
    - cat impact.json
    - python3 tools/test_impact.py --base ${CI_MERGE_REQUEST_DIFF_BASE_SHA:-HEAD~1} --verbose
  artifacts:
    paths:
      - impact.json
      - coverage_map.json
    expire_in: 1 day
  timeout: 5m
```

- [ ] **Step 3: Modify test-python-builder to use filtered test list**

```yaml
test-python-builder:
  stage: tier1-unit
  needs: [build-all, impact-analysis]
  script:
    - |
      if ! python3 -c "import json; d=json.load(open('impact.json')); exit(0 if 'builder' in d['unit_tiers'] else 1)"; then
        echo "Skipping: builder tier not affected by this change"
        exit 0
      fi
    - |
      # Check for coverage-map-based selective tests
      BUILDER_TESTS=$(python3 -c "
      import json, sys
      d = json.load(open('impact.json'))
      tests = d.get('builder_tests', [])
      fallback = d.get('fallback_tiers', [])
      if 'builder' in fallback or not tests:
          # Fallback: run all builder tests
          sys.exit(0)
      # Selective: output pytest -k filter
      print(' or '.join(tests))
      " 2>/dev/null) || true
      if [ -n "$BUILDER_TESTS" ]; then
        echo "Selective builder tests: $BUILDER_TESTS"
        python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py -k "$BUILDER_TESTS" -n auto
      else
        echo "Running all builder tests (fallback or no coverage map)"
        python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py -n auto
      fi
  timeout: 40m
```

- [ ] **Step 4: Modify test-cpp-unit to use filtered test list**

```yaml
test-cpp-unit:
  stage: tier1-unit
  needs: [build-all, impact-analysis]
  script:
    - |
      if ! python3 -c "import json; d=json.load(open('impact.json')); exit(0 if 'cpp' in d['unit_tiers'] else 1)"; then
        echo "Skipping: cpp tier not affected by this change"
        exit 0
      fi
    - |
      # Check for coverage-map-based selective tests
      CPP_TESTS=$(python3 -c "
      import json, sys
      d = json.load(open('impact.json'))
      tests = d.get('cpp_tests', [])
      fallback = d.get('fallback_tiers', [])
      if 'cpp' in fallback or not tests:
          sys.exit(0)
      print('|'.join(tests))
      " 2>/dev/null) || true
      if [ -n "$CPP_TESTS" ]; then
        echo "Selective C++ tests: $CPP_TESTS"
        ctest --test-dir build -R "$CPP_TESTS" --output-on-failure
      else
        echo "Running all C++ tests (fallback or no coverage map)"
        ctest --test-dir build --output-on-failure
      fi
  timeout: 20m
```

- [ ] **Step 5: Commit**

```bash
git add .gitlab-ci.yml
git commit -m "ci: integrate coverage-map-based selective unit test execution"
```

---

## Task 8: Run Full Test Suite and Verify

**Files:** None (verification only)

- [ ] **Step 1: Run all coverage_map tests**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_coverage_map.py -v"`
Expected: All tests PASS (approximately 23 tests)

- [ ] **Step 2: Run all test_impact tests (including new integration tests)**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/test_test_impact.py -v"`
Expected: All existing + new tests PASS

- [ ] **Step 3: Run the full tools test suite**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m pytest tests/tools/ -v"`
Expected: All tools tests PASS

- [ ] **Step 4: Validate test_impact.py --validate still passes**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python tools/test_impact.py --validate"`
Expected: Validation passed

- [ ] **Step 5: Dry-run select_tests.py with a sample file**

Run: `docker exec trtf-dev-gb300-agent-2 bash -c "cd /workspace/trt-transformers-cpp && python -m tools.coverage_map.select_tests --coverage-map /dev/null --files src/tokenizer/vocab_tokenizer.cpp 2>&1 || true"`
Expected: Error about missing map (correct behavior -- no map yet)

- [ ] **Step 6: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix: address test suite verification findings"
```
