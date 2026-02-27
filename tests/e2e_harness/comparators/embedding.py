"""Embedding comparator — compare TRT vs reference embedding outputs.

Metrics: cosine similarity, top-k neighborhood overlap, L2 distance.
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


def _topk_overlap(
    trt_emb: List[float], ref_emb: List[float], k: int
) -> float:
    """Compute overlap of top-k dimensions by absolute magnitude.

    Returns fraction of top-k dimensions that appear in both sets.
    """
    if len(trt_emb) != len(ref_emb) or len(trt_emb) == 0:
        return 0.0
    k = min(k, len(trt_emb))
    trt_topk = set(
        sorted(range(len(trt_emb)), key=lambda i: abs(trt_emb[i]), reverse=True)[:k]
    )
    ref_topk = set(
        sorted(range(len(ref_emb)), key=lambda i: abs(ref_emb[i]), reverse=True)[:k]
    )
    return len(trt_topk & ref_topk) / k


class EmbeddingComparator:
    """Compare TRT vs reference embedding outputs."""

    @property
    def task_strategy(self) -> str:
        return "embedding"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        trt_emb = trt.data.get("embedding", [])
        ref_emb = ref.data.get("embedding", [])

        if not trt_emb or not ref_emb:
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                message="Missing embedding data in TRT or reference output",
            )

        if len(trt_emb) != len(ref_emb):
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                message=(
                    f"Embedding dimension mismatch: TRT={len(trt_emb)}, "
                    f"ref={len(ref_emb)}"
                ),
            )

        cosine = _cosine_similarity(trt_emb, ref_emb)
        l2 = _l2_distance(trt_emb, ref_emb)
        topk_10 = _topk_overlap(trt_emb, ref_emb, k=10)
        topk_100 = _topk_overlap(trt_emb, ref_emb, k=100)

        metrics = {
            "cosine_similarity": cosine,
            "l2_distance": l2,
            "topk_neighborhood_overlap_10": topk_10,
            "topk_neighborhood_overlap_100": topk_100,
        }

        # Gate on thresholds
        th = threshold.metrics
        per_metric_pass = {}
        gate_details = []

        cosine_thresh = th.get("cosine_similarity", 0.99)
        per_metric_pass["cosine_similarity"] = cosine >= cosine_thresh
        gate_details.append(
            f"cosine_similarity: {cosine:.6f} >= {cosine_thresh} -> "
            f"{'PASS' if per_metric_pass['cosine_similarity'] else 'FAIL'}"
        )

        l2_thresh = th.get("l2_distance", 0.1)
        per_metric_pass["l2_distance"] = l2 <= l2_thresh
        gate_details.append(
            f"l2_distance: {l2:.6f} <= {l2_thresh} -> "
            f"{'PASS' if per_metric_pass['l2_distance'] else 'FAIL'}"
        )

        topk10_thresh = th.get("topk_neighborhood_overlap_10", 0.8)
        per_metric_pass["topk_neighborhood_overlap_10"] = topk_10 >= topk10_thresh
        gate_details.append(
            f"topk_neighborhood_overlap_10: {topk_10:.4f} >= {topk10_thresh} -> "
            f"{'PASS' if per_metric_pass['topk_neighborhood_overlap_10'] else 'FAIL'}"
        )

        topk100_thresh = th.get("topk_neighborhood_overlap_100", 0.7)
        per_metric_pass["topk_neighborhood_overlap_100"] = topk_100 >= topk100_thresh
        gate_details.append(
            f"topk_neighborhood_overlap_100: {topk_100:.4f} >= {topk100_thresh} -> "
            f"{'PASS' if per_metric_pass['topk_neighborhood_overlap_100'] else 'FAIL'}"
        )

        passed = all(per_metric_pass.values())

        return CompareResult(
            stage_name=stage.name,
            passed=passed,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"Embedding comparison: cosine={cosine:.6f}, L2={l2:.6f}",
        )


plugin = EmbeddingComparator()
