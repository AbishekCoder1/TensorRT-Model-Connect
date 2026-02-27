"""Unified E2E testing harness for TRT inference validation.

This package provides a DIP-first framework where high-level orchestration
depends only on abstract contracts, and concrete implementations (strategy
runners, reference backends, comparators) are pluggable adapters.

Public API re-exports the core types from contracts.py.
"""

__version__ = "0.1.0"

import os


def save_full_stderr(stderr: str, artifacts_dir: str, stage_name: str, case_name: str = "") -> tuple:
    """Write full stderr to file, return (truncated_msg, file_path) or (truncated_msg, None) if no artifacts_dir."""
    truncated = stderr[-2000:] if len(stderr) > 2000 else stderr
    if not artifacts_dir:
        return truncated, None
    os.makedirs(artifacts_dir, exist_ok=True)
    prefix = f"{case_name}_" if case_name else ""
    path = os.path.join(artifacts_dir, f"{prefix}{stage_name}_stderr.log")
    with open(path, "w") as f:
        f.write(stderr)
    return truncated, path

from .contracts import (
    RUNTIME_TO_TASK_STRATEGY,
    ArtifactSink,
    Comparator,
    CompareResult,
    DeterminismPolicy,
    E2ECase,
    E2EResult,
    E2EStatus,
    FailureType,
    OracleLevel,
    PreflightRequirement,
    RunContext,
    ReferenceBackendRunner,
    StageOutput,
    StageSpec,
    TaskStrategyRunner,
    ThresholdProfile,
)

__all__ = [
    "__version__",
    # Enums
    "FailureType",
    "OracleLevel",
    "E2EStatus",
    # Dataclasses
    "PreflightRequirement",
    "StageSpec",
    "ThresholdProfile",
    "E2ECase",
    "StageOutput",
    "CompareResult",
    "E2EResult",
    "RunContext",
    # Protocols
    "TaskStrategyRunner",
    "ReferenceBackendRunner",
    "Comparator",
    "ArtifactSink",
    "DeterminismPolicy",
    # Constants
    "RUNTIME_TO_TASK_STRATEGY",
    # Helpers
    "save_full_stderr",
]
