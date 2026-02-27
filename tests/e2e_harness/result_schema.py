"""Serialization and deserialization for E2EResult and its nested types.

Converts between the dataclass domain model and plain JSON-serializable
dicts, enabling result.json persistence and cross-process communication.

All serialization is explicit: no pickle, no custom JSON encoders.
"""

from __future__ import annotations

from typing import Any, Dict

from .contracts import CompareResult, E2EResult


def _serialize_compare_result(cr: CompareResult) -> Dict[str, Any]:
    """Serialize a CompareResult to a JSON-safe dict."""
    return {
        "stage_name": cr.stage_name,
        "passed": cr.passed,
        "metrics": dict(cr.metrics),
        "per_metric_pass": dict(cr.per_metric_pass),
        "gate_details": list(cr.gate_details),
        "message": cr.message,
    }


def _deserialize_compare_result(data: Dict[str, Any]) -> CompareResult:
    """Reconstruct a CompareResult from a dict."""
    return CompareResult(
        stage_name=data.get("stage_name", ""),
        passed=data.get("passed", False),
        metrics=data.get("metrics", {}),
        per_metric_pass=data.get("per_metric_pass", {}),
        gate_details=data.get("gate_details", []),
        message=data.get("message", ""),
    )


def serialize_result(result: E2EResult) -> Dict[str, Any]:
    """Serialize an E2EResult to a JSON-serializable dict.

    The returned dict can be written directly with json.dump().
    Nested CompareResult objects are converted to plain dicts.
    """
    stages_ser: Dict[str, Any] = {}
    for stage_name, cr in result.stages.items():
        stages_ser[stage_name] = _serialize_compare_result(cr)

    return {
        "case_name": result.case_name,
        "status": result.status,
        "failure_type": result.failure_type,
        "oracle_level": result.oracle_level,
        "stages": stages_ser,
        "determinism": dict(result.determinism),
        "timing": dict(result.timing),
        "env_fingerprint": dict(result.env_fingerprint),
        "timestamp": result.timestamp,
        "repro_commands": dict(result.repro_commands),
    }


def deserialize_result(data: Dict[str, Any]) -> E2EResult:
    """Reconstruct an E2EResult from a dict (e.g. loaded from result.json).

    Missing fields are filled with defaults. Unknown fields are ignored.
    """
    stages_raw = data.get("stages", {})
    stages: Dict[str, CompareResult] = {}
    for stage_name, cr_data in stages_raw.items():
        stages[stage_name] = _deserialize_compare_result(cr_data)

    return E2EResult(
        case_name=data.get("case_name", ""),
        status=data.get("status", "error"),
        failure_type=data.get("failure_type"),
        oracle_level=data.get("oracle_level", ""),
        stages=stages,
        determinism=data.get("determinism", {}),
        timing=data.get("timing", {}),
        env_fingerprint=data.get("env_fingerprint", {}),
        timestamp=data.get("timestamp", ""),
        repro_commands=data.get("repro_commands", {}),
    )
