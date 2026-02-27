"""Reranking comparator — compare TRT vs reference reranking outputs.

Metrics: pairwise ordering agreement, Kendall tau, Spearman rho,
score correlation.
"""

from __future__ import annotations

import logging
import math
from typing import List

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)


def _pairwise_ordering_agreement(a: List[float], b: List[float]) -> float:
    """Fraction of pairs where both rankings agree on relative order."""
    n = len(a)
    if n < 2:
        return 1.0
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            # Same relative order (or tied in both)
            sign_a = (a[i] > a[j]) - (a[i] < a[j])
            sign_b = (b[i] > b[j]) - (b[i] < b[j])
            if sign_a == sign_b:
                concordant += 1
    return concordant / total if total > 0 else 1.0


def _kendall_tau(a: List[float], b: List[float]) -> float:
    """Kendall's tau-b rank correlation coefficient."""
    n = len(a)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = (a[i] > a[j]) - (a[i] < a[j])
            sign_b = (b[i] > b[j]) - (b[i] < b[j])
            if sign_a == 0 and sign_b == 0:
                ties_a += 1
                ties_b += 1
            elif sign_a == 0:
                ties_a += 1
            elif sign_b == 0:
                ties_b += 1
            elif sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    denom = math.sqrt((total_pairs - ties_a) * (total_pairs - ties_b))
    if denom < 1e-12:
        return 1.0 if concordant >= discordant else -1.0
    return (concordant - discordant) / denom


def _spearman_rho(a: List[float], b: List[float]) -> float:
    """Spearman's rank correlation coefficient."""
    n = len(a)
    if n < 2:
        return 1.0

    def _rank(values: List[float]) -> List[float]:
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rank_a = _rank(a)
    rank_b = _rank(b)
    d_sq_sum = sum((ra - rb) ** 2 for ra, rb in zip(rank_a, rank_b))
    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1)) if n > 1 else 1.0


def _score_correlation(a: List[float], b: List[float]) -> float:
    """Pearson correlation of raw scores."""
    n = len(a)
    if n < 2:
        return 1.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = math.sqrt(var_a * var_b)
    if denom < 1e-12:
        return 1.0
    return cov / denom


class RerankingComparator:
    """Compare TRT vs reference reranking outputs."""

    @property
    def task_strategy(self) -> str:
        return "reranking"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        trt_scores = trt.data.get("scores", [])
        ref_scores = ref.data.get("scores", [])

        if not trt_scores or not ref_scores:
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                message="Missing scores in TRT or reference output",
            )

        if len(trt_scores) != len(ref_scores):
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                message=(
                    f"Score count mismatch: TRT={len(trt_scores)}, "
                    f"ref={len(ref_scores)}"
                ),
            )

        pairwise = _pairwise_ordering_agreement(trt_scores, ref_scores)
        tau = _kendall_tau(trt_scores, ref_scores)
        rho = _spearman_rho(trt_scores, ref_scores)
        corr = _score_correlation(trt_scores, ref_scores)

        metrics = {
            "pairwise_ordering_agreement": pairwise,
            "kendall_tau": tau,
            "spearman_rho": rho,
            "score_correlation": corr,
        }

        th = threshold.metrics
        per_metric_pass = {}
        gate_details = []

        pairwise_thresh = th.get("pairwise_ordering_agreement", 0.9)
        per_metric_pass["pairwise_ordering_agreement"] = pairwise >= pairwise_thresh
        gate_details.append(
            f"pairwise_ordering_agreement: {pairwise:.4f} >= {pairwise_thresh} -> "
            f"{'PASS' if per_metric_pass['pairwise_ordering_agreement'] else 'FAIL'}"
        )

        tau_thresh = th.get("kendall_tau", 0.8)
        per_metric_pass["kendall_tau"] = tau >= tau_thresh
        gate_details.append(
            f"kendall_tau: {tau:.4f} >= {tau_thresh} -> "
            f"{'PASS' if per_metric_pass['kendall_tau'] else 'FAIL'}"
        )

        rho_thresh = th.get("spearman_rho", 0.8)
        per_metric_pass["spearman_rho"] = rho >= rho_thresh
        gate_details.append(
            f"spearman_rho: {rho:.4f} >= {rho_thresh} -> "
            f"{'PASS' if per_metric_pass['spearman_rho'] else 'FAIL'}"
        )

        corr_thresh = th.get("score_correlation", 0.9)
        per_metric_pass["score_correlation"] = corr >= corr_thresh
        gate_details.append(
            f"score_correlation: {corr:.4f} >= {corr_thresh} -> "
            f"{'PASS' if per_metric_pass['score_correlation'] else 'FAIL'}"
        )

        passed = all(per_metric_pass.values())

        return CompareResult(
            stage_name=stage.name,
            passed=passed,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=(
                f"Reranking comparison: pairwise={pairwise:.4f}, "
                f"tau={tau:.4f}, rho={rho:.4f}, corr={corr:.4f}"
            ),
        )


plugin = RerankingComparator()
