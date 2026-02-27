"""Neural operator comparator — compare TRT vs reference field outputs.

Metrics: relative L2 on output fields, max pointwise error, optional
PDE residual constraints.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)


def _try_import_numpy():
    """Lazily import numpy; return None if unavailable."""
    try:
        import numpy as np
        return np
    except ImportError:
        return None


def _relative_l2(trt_field: Any, ref_field: Any) -> float:
    """Compute relative L2 error: ||trt - ref||_2 / ||ref||_2."""
    np = _try_import_numpy()
    if np is not None:
        trt_arr = np.asarray(trt_field, dtype=np.float64)
        ref_arr = np.asarray(ref_field, dtype=np.float64)
        ref_norm = np.linalg.norm(ref_arr)
        if ref_norm < 1e-12:
            return float(np.linalg.norm(trt_arr - ref_arr))
        return float(np.linalg.norm(trt_arr - ref_arr) / ref_norm)

    # Pure Python fallback for flat lists
    if isinstance(trt_field, list) and isinstance(ref_field, list):
        diff_sq = sum((a - b) ** 2 for a, b in zip(trt_field, ref_field))
        ref_sq = sum(b ** 2 for b in ref_field)
        if ref_sq < 1e-24:
            return math.sqrt(diff_sq)
        return math.sqrt(diff_sq / ref_sq)

    return float("inf")


def _max_pointwise_error(trt_field: Any, ref_field: Any) -> float:
    """Compute max pointwise absolute error."""
    np = _try_import_numpy()
    if np is not None:
        trt_arr = np.asarray(trt_field, dtype=np.float64)
        ref_arr = np.asarray(ref_field, dtype=np.float64)
        return float(np.max(np.abs(trt_arr - ref_arr)))

    if isinstance(trt_field, list) and isinstance(ref_field, list):
        return max(abs(a - b) for a, b in zip(trt_field, ref_field))

    return float("inf")


def _load_field(data: Dict[str, Any], key: str = "output_field_path") -> Any:
    """Load field data from StageOutput data dict.

    Supports numpy files via path or inline arrays.
    """
    np = _try_import_numpy()

    # Try loading from file path
    field_path = data.get(key)
    if field_path and np is not None:
        try:
            return np.load(field_path)
        except Exception:
            pass

    # Try inline arrays
    if "output_field" in data:
        return data["output_field"]
    if "field" in data:
        return data["field"]

    return None


class NeuralOperatorComparator:
    """Compare TRT vs reference neural operator field outputs."""

    @property
    def task_strategy(self) -> str:
        return "neural_operator"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        trt_field = _load_field(trt.data)
        ref_field = _load_field(ref.data)

        if trt_field is None or ref_field is None:
            missing = []
            if trt_field is None:
                missing.append("TRT")
            if ref_field is None:
                missing.append("ref")
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics={},
                per_metric_pass={},
                gate_details=[f"early return: missing field data from {', '.join(missing)}"],
                message="Missing field data in TRT or reference output",
            )

        rel_l2 = _relative_l2(trt_field, ref_field)
        max_error = _max_pointwise_error(trt_field, ref_field)

        metrics: Dict[str, float] = {
            "relative_l2": rel_l2,
            "max_pointwise_error": max_error,
        }

        th = threshold.metrics
        per_metric_pass = {}
        gate_details = []

        rel_l2_thresh = th.get("relative_l2", 0.01)
        per_metric_pass["relative_l2"] = rel_l2 <= rel_l2_thresh
        gate_details.append(
            f"relative_l2: {rel_l2:.6e} <= {rel_l2_thresh} -> "
            f"{'PASS' if per_metric_pass['relative_l2'] else 'FAIL'}"
        )

        max_err_thresh = th.get("max_pointwise_error", 0.1)
        per_metric_pass["max_pointwise_error"] = max_error <= max_err_thresh
        gate_details.append(
            f"max_pointwise_error: {max_error:.6e} <= {max_err_thresh} -> "
            f"{'PASS' if per_metric_pass['max_pointwise_error'] else 'FAIL'}"
        )

        # Optional PDE residual constraint (if provided in threshold)
        pde_residual_thresh = th.get("pde_residual")
        if pde_residual_thresh is not None:
            pde_residual = trt.data.get("pde_residual")
            if pde_residual is not None:
                metrics["pde_residual"] = float(pde_residual)
                per_metric_pass["pde_residual"] = float(pde_residual) <= pde_residual_thresh
                gate_details.append(
                    f"pde_residual: {pde_residual:.6e} <= {pde_residual_thresh} -> "
                    f"{'PASS' if per_metric_pass['pde_residual'] else 'FAIL'}"
                )

        passed = all(per_metric_pass.values())

        return CompareResult(
            stage_name=stage.name,
            passed=passed,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=(
                f"Neural operator comparison: relative_l2={rel_l2:.6e}, "
                f"max_pointwise_error={max_error:.6e}"
            ),
        )


plugin = NeuralOperatorComparator()
