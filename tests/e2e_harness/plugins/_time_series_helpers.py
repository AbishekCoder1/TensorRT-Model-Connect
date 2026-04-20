"""Shared helpers for time-series contract plugins."""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any

import numpy as np

from ..contracts import MetricResult


def load_output_field(data: dict[str, Any]) -> np.ndarray | None:
    if "output_field" not in data:
        return None
    return np.asarray(data["output_field"], dtype=np.float64)


def declared_shape(data: dict[str, Any]) -> tuple[int, ...] | None:
    raw = data.get("output_shape")
    if not isinstance(raw, list) or not raw:
        return None
    try:
        shape = tuple(int(dim) for dim in raw)
    except (TypeError, ValueError):
        return None
    if any(dim <= 0 for dim in shape):
        return None
    return shape


def numel(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    return reduce(mul, shape, 1)


def reshape_trt_like_reference(
    trt_data: dict[str, Any], ref_data: dict[str, Any]
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    trt_arr = load_output_field(trt_data)
    ref_arr = load_output_field(ref_data)
    if trt_arr is None or ref_arr is None:
        return trt_arr, ref_arr, "Missing output_field"

    ref_shape = declared_shape(ref_data)
    trt_shape = declared_shape(trt_data)

    if ref_shape is not None:
        expected_numel = numel(ref_shape)
        if expected_numel is not None and trt_arr.size != expected_numel:
            return trt_arr, ref_arr, (
                f"TRT output element count {trt_arr.size} does not match "
                f"reference shape {list(ref_shape)} ({expected_numel} elements)"
            )
        if trt_arr.shape != ref_shape and expected_numel == trt_arr.size:
            trt_arr = trt_arr.reshape(ref_shape)
        if ref_arr.shape != ref_shape and expected_numel == ref_arr.size:
            ref_arr = ref_arr.reshape(ref_shape)

    if trt_shape is not None and ref_shape is not None and trt_shape != ref_shape:
        return trt_arr, ref_arr, (
            f"Declared output shapes differ: TRT {list(trt_shape)} vs ref {list(ref_shape)}"
        )

    if trt_arr.shape != ref_arr.shape:
        return trt_arr, ref_arr, (
            f"Output tensor shapes differ after normalization: TRT {list(trt_arr.shape)} "
            f"vs ref {list(ref_arr.shape)}"
        )

    return trt_arr, ref_arr, None


def relative_l2(trt_arr: np.ndarray, ref_arr: np.ndarray) -> float:
    ref_norm = np.linalg.norm(ref_arr)
    if ref_norm < 1e-12:
        return float(np.linalg.norm(trt_arr - ref_arr))
    return float(np.linalg.norm(trt_arr - ref_arr) / ref_norm)


def max_pointwise_error(trt_arr: np.ndarray, ref_arr: np.ndarray) -> float:
    return float(np.max(np.abs(trt_arr - ref_arr)))


def finite_metric(name: str, ok: bool, note: str = "") -> MetricResult:
    return MetricResult(
        value=1.0 if ok else 0.0,
        threshold=1.0,
        operator="==",
        passed=ok,
        note=note,
    )


def check_all_finite(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr)))
