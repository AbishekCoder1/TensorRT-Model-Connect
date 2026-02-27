"""Unified E2E test entrypoint — single parametrized test for all models.

Usage:
    # Single model:
    pytest tests/test_e2e.py::test_e2e[qwen3-0.6b]

    # All models:
    pytest tests/test_e2e.py

    # Filter by strategy:
    pytest tests/test_e2e.py --e2e-task-strategy text_generation_causal

    # With artifacts:
    pytest tests/test_e2e.py --e2e-artifacts-dir /tmp/e2e_artifacts

    # With legacy options (compat with tests/e2e/conftest.py):
    pytest tests/test_e2e.py --engine-dir /mnt/storage/engines --trtf-binary ./build/trtf
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e_harness.contracts import E2EStatus, RunContext
from tests.e2e_harness.manifest_loader import get_case_by_name, load_all_manifests
from tests.e2e_harness.orchestrator import E2EOrchestrator

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _resolve_binary(config) -> str:
    """Resolve the trtf binary path."""
    cli_val = config.getoption("--trtf-binary", default=None)
    if cli_val:
        # Use absolute() not resolve() to preserve venv symlinks
        return str(Path(cli_val).absolute())
    default = PROJECT_DIR / "build" / "trtf"
    return str(default) if default.is_file() else ""


def _resolve_hf_python(config) -> str:
    """Resolve the Python interpreter with HF tokenizers."""
    cli_val = config.getoption("--hf-python", default=None)
    if cli_val:
        # Use absolute() not resolve() — resolve() follows symlinks,
        # which turns .venv/bin/python into /usr/bin/python3 (no numpy)
        return str(Path(cli_val).absolute())
    venv = PROJECT_DIR / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _resolve_engine_dir(config) -> str:
    """Resolve the engine/bundle directory."""
    cli_val = config.getoption("--engine-dir", default=None)
    if cli_val:
        d = Path(cli_val)
    else:
        d = Path("/mnt/storage/trt-transformers/engines")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _resolve_ld_library_path() -> str:
    """Build LD_LIBRARY_PATH with TRT libs."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); "
             "print(s.submodule_search_locations[0])"],
            capture_output=True, text=True, timeout=10)
        trt_lib_dir = result.stdout.strip()
    except Exception:
        trt_lib_dir = ""
    base = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in [trt_lib_dir, "/usr/local/cuda/lib64", base] if p]
    return ":".join(parts)


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Register CLI options for the unified E2E harness."""
    # Try adding; skip if already registered (e.g. conftest.py also adds these)
    try:
        parser.addoption(
            "--engine-dir", default=None,
            help="Directory containing .trtfb bundles")
    except ValueError:
        pass
    try:
        parser.addoption(
            "--trtf-binary", default=None,
            help="Path to the C++ trtf binary")
    except ValueError:
        pass
    try:
        parser.addoption(
            "--hf-python", default=None,
            help="Python interpreter with HuggingFace tokenizers")
    except ValueError:
        pass
    try:
        parser.addoption(
            "--rebuild-engines", action="store_true", default=False,
            help="Force rebuild of all engine bundles")
    except ValueError:
        pass

    # New harness-specific options
    try:
        parser.addoption(
            "--e2e-task-strategy", default=None,
            help="Filter models by task_strategy (e.g. text_generation_causal)")
    except ValueError:
        pass
    try:
        parser.addoption(
            "--e2e-artifacts-dir", default=None,
            help="Directory for E2E artifacts output")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def _get_case_names(config=None) -> list[str]:
    """Load all case names for parametrization.

    Respects --e2e-task-strategy filter if set.
    """
    strategy_filter = None
    if config is not None:
        strategy_filter = config.getoption("--e2e-task-strategy", default=None)

    cases = load_all_manifests(task_strategy_filter=strategy_filter)
    if not cases:
        return ["__no_models__"]
    return [c.name for c in cases]


# Collect case names at module level for parametrize
_CASE_NAMES = _get_case_names()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_e2e(case_name: str, request) -> None:
    """Unified E2E test — run full lifecycle for one model.

    Each model case goes through:
    1. Preflight checks
    2. Bundle resolution/build
    3. TRT inference
    4. Reference inference
    5. Comparison with tolerance-based gating
    6. Artifact persistence

    The test passes if all required stages pass comparison thresholds.
    """
    if case_name == "__no_models__":
        pytest.skip("No model manifests found")

    # Load the case
    case = get_case_by_name(case_name)
    if case is None:
        pytest.fail(f"Case not found: {case_name}")

    # Handle skip cases (v1 compat — models with skip field)
    if case.metadata.get("skip_reason"):
        pytest.skip(case.metadata["skip_reason"])

    # Build run context
    config = request.config
    artifacts_dir = config.getoption("--e2e-artifacts-dir", default=None) or "/tmp/e2e_artifacts"

    ctx = RunContext(
        case=case,
        artifacts_dir=artifacts_dir,
        binary_path=_resolve_binary(config),
        hf_python=_resolve_hf_python(config),
        ld_library_path=_resolve_ld_library_path(),
        engine_dir=_resolve_engine_dir(config),
        rebuild=config.getoption("--rebuild-engines", default=False),
        verbose=config.getoption("verbose", default=0) > 0,
    )

    # Run orchestrator
    orchestrator = E2EOrchestrator()
    result = orchestrator.run(case, ctx)

    # Assert
    if result.status == E2EStatus.SKIP.value:
        pytest.skip(f"Case {case_name} skipped")
    elif result.status == E2EStatus.PASS.value:
        pass  # Test passes
    else:
        # Collect failure details for the assertion message
        failed_stages = [
            f"  {name}: {cr.message}"
            for name, cr in result.stages.items()
            if not cr.passed
        ]
        failure_msg = (
            f"E2E failed for {case_name} "
            f"(failure_type={result.failure_type}, "
            f"oracle_level={result.oracle_level}):\n"
        )
        if failed_stages:
            failure_msg += "\n".join(failed_stages)
        else:
            failure_msg += f"  status={result.status}"

        pytest.fail(failure_msg)
