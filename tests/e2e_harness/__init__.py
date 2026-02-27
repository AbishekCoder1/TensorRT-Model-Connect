"""Unified E2E testing harness for TRT inference validation.

This package provides a DIP-first framework where high-level orchestration
depends only on abstract contracts, and concrete implementations (strategy
runners, reference backends, comparators) are pluggable adapters.

Public API re-exports the core types from contracts.py.
"""

__version__ = "0.1.0"

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
]
