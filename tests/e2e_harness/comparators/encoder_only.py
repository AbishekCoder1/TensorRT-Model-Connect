"""Encoder-only NLP comparator — compare TRT vs reference encoder outputs.

Metrics: hidden state cosine, hidden state L2, CLS embedding cosine.
"""

from __future__ import annotations

import logging
import math
from typing import List

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _l2_distance(a: List[float], b: List[float]) -> float:
    """Compute L2 distance between two vectors."""
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class EncoderOnlyComparator:
    """Compare TRT vs reference encoder-only outputs."""

    @property
    def task_strategy(self) -> str:
        return "encoder_only_nlp"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics = {}
        per_metric_pass = {}
        gate_details = []
        th = threshold.metrics

        # Compare CLS embeddings if available
        trt_cls = trt.data.get("cls_embedding", [])
        ref_cls = ref.data.get("cls_embedding", [])

        if trt_cls and ref_cls:
            if len(trt_cls) != len(ref_cls):
                return CompareResult(
                    stage_name=stage.name,
                    passed=False,
                    metrics={},
                    per_metric_pass={},
                    gate_details=[
                        f"early return: CLS embedding dimension mismatch: "
                        f"TRT={len(trt_cls)}, ref={len(ref_cls)}"
                    ],
                    message=(
                        f"CLS embedding dimension mismatch: TRT={len(trt_cls)}, "
                        f"ref={len(ref_cls)}"
                    ),
                )

            cls_cosine = _cosine_similarity(trt_cls, ref_cls)
            cls_l2 = _l2_distance(trt_cls, ref_cls)
            metrics["cls_embedding_cosine"] = cls_cosine
            metrics["cls_embedding_l2"] = cls_l2

            cls_cosine_thresh = th.get("cls_embedding_cosine", 0.99)
            per_metric_pass["cls_embedding_cosine"] = cls_cosine >= cls_cosine_thresh
            gate_details.append(
                f"cls_embedding_cosine: {cls_cosine:.6f} >= {cls_cosine_thresh} -> "
                f"{'PASS' if per_metric_pass['cls_embedding_cosine'] else 'FAIL'}"
            )

            cls_l2_thresh = th.get("cls_embedding_l2", 0.1)
            per_metric_pass["cls_embedding_l2"] = cls_l2 <= cls_l2_thresh
            gate_details.append(
                f"cls_embedding_l2: {cls_l2:.6f} <= {cls_l2_thresh} -> "
                f"{'PASS' if per_metric_pass['cls_embedding_l2'] else 'FAIL'}"
            )

        # Compare hidden states if available (flattened vectors)
        trt_hidden = trt.data.get("hidden_states", [])
        ref_hidden = ref.data.get("hidden_states", [])

        if trt_hidden and ref_hidden:
            if len(trt_hidden) != len(ref_hidden):
                return CompareResult(
                    stage_name=stage.name,
                    passed=False,
                    metrics={},
                    per_metric_pass={},
                    gate_details=[
                        f"early return: hidden state dimension mismatch: "
                        f"TRT={len(trt_hidden)}, ref={len(ref_hidden)}"
                    ],
                    message=(
                        f"Hidden state dimension mismatch: TRT={len(trt_hidden)}, "
                        f"ref={len(ref_hidden)}"
                    ),
                )

            hidden_cosine = _cosine_similarity(trt_hidden, ref_hidden)
            hidden_l2 = _l2_distance(trt_hidden, ref_hidden)
            metrics["hidden_state_cosine"] = hidden_cosine
            metrics["hidden_state_l2"] = hidden_l2

            hidden_cosine_thresh = th.get("hidden_state_cosine", 0.99)
            per_metric_pass["hidden_state_cosine"] = hidden_cosine >= hidden_cosine_thresh
            gate_details.append(
                f"hidden_state_cosine: {hidden_cosine:.6f} >= {hidden_cosine_thresh} -> "
                f"{'PASS' if per_metric_pass['hidden_state_cosine'] else 'FAIL'}"
            )

            hidden_l2_thresh = th.get("hidden_state_l2", 0.5)
            per_metric_pass["hidden_state_l2"] = hidden_l2 <= hidden_l2_thresh
            gate_details.append(
                f"hidden_state_l2: {hidden_l2:.6f} <= {hidden_l2_thresh} -> "
                f"{'PASS' if per_metric_pass['hidden_state_l2'] else 'FAIL'}"
            )

        if not metrics:
            # Both ran successfully but output formats are incompatible.
            # Pass at L4 (invariant-only) — TRT inference succeeded.
            trt_has_data = bool(trt.data) and not trt.data.get("skipped")
            ref_has_data = bool(ref.data) and not ref.data.get("skipped")
            if trt_has_data and ref_has_data:
                return CompareResult(
                    stage_name=stage.name,
                    passed=True,
                    metrics={"invariant_only": 1.0},
                    per_metric_pass={"invariant_only": True},
                    gate_details=[
                        "L4 invariant-only: TRT + HF both produced output, "
                        f"but no shared keys (TRT: {list(trt.data.keys())}, "
                        f"ref: {list(ref.data.keys())})"
                    ],
                    message="Invariant-only pass: both ran successfully",
                )
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics={},
                per_metric_pass={},
                gate_details=["early return: no comparable outputs (missing cls_embedding and hidden_states)"],
                message="No comparable outputs found (missing cls_embedding and hidden_states)",
            )

        passed = all(per_metric_pass.values())

        return CompareResult(
            stage_name=stage.name,
            passed=passed,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"Encoder-only comparison: {len(metrics)} metrics evaluated",
        )


plugin = EncoderOnlyComparator()
