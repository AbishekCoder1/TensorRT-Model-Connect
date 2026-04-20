"""Contract plugin for time-series classification models."""

from __future__ import annotations

import numpy as np

from ..contracts import MetricResult
from .base import make_fail, make_pass
from ._time_series_helpers import (
    check_all_finite,
    declared_shape,
    finite_metric,
    max_pointwise_error,
    relative_l2,
    reshape_trt_like_reference,
)


class TimeSeriesClassificationPlugin:
    reference_families = ["time_series_classification"]
    user_contract = "time_series_classification"

    def configure_reference(self, case):
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        trt_arr, ref_arr, shape_error = reshape_trt_like_reference(
            trt_output.data, ref_output.data)
        if shape_error is not None:
            return make_fail("full_inference", {}, message=shape_error)

        ref_name = str(ref_output.data.get("reference_output_name", ""))
        ref_shape = declared_shape(ref_output.data)
        rank_ok = ref_shape is not None and len(ref_shape) in (1, 2)
        output_name_ok = ref_name in {"prediction_logits", "classification_outputs", "logits"}
        finite_ok = check_all_finite(trt_arr) and check_all_finite(ref_arr)
        top1_ok = int(np.argmax(trt_arr.reshape(-1))) == int(np.argmax(ref_arr.reshape(-1)))

        rel_l2 = relative_l2(trt_arr, ref_arr)
        max_err = max_pointwise_error(trt_arr, ref_arr)
        rel_l2_thresh = threshold.metrics.get("relative_l2", 0.01)
        max_err_thresh = threshold.metrics.get("max_pointwise_error", 0.1)

        metrics = {
            "output_name_supported": finite_metric(
                "output_name_supported",
                output_name_ok,
                note=f"reference_output_name={ref_name}",
            ),
            "classification_rank_supported": finite_metric(
                "classification_rank_supported",
                rank_ok,
                note=f"reference_shape={list(ref_shape) if ref_shape else None}",
            ),
            "finite_outputs": finite_metric("finite_outputs", finite_ok),
            "top1_match": MetricResult(
                value=1.0 if top1_ok else 0.0,
                threshold=1.0,
                operator="==",
                passed=top1_ok,
            ),
            "relative_l2": MetricResult(
                value=rel_l2,
                threshold=rel_l2_thresh,
                operator="<=",
                passed=rel_l2 <= rel_l2_thresh,
            ),
            "max_pointwise_error": MetricResult(
                value=max_err,
                threshold=max_err_thresh,
                operator="<=",
                passed=max_err <= max_err_thresh,
            ),
        }

        passed = all(metric.passed for metric in metrics.values())
        if passed:
            return make_pass(
                "full_inference",
                metrics,
                "supported classification output semantics + top1 match + numeric parity",
            )
        return make_fail(
            "full_inference",
            metrics,
            "supported classification output semantics + top1 match + numeric parity",
            f"Classification contract failed for {case.name}",
        )


plugin = TimeSeriesClassificationPlugin()
