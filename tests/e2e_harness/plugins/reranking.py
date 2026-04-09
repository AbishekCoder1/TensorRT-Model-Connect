"""Contract test plugin for reranking models."""
from __future__ import annotations
from ..contracts import (CompareResult, E2ECase, MetricResult, StageOutput, ThresholdProfile)
from .base import make_pass, make_fail, make_error


class RerankingPlugin:
    reference_families = ["vl_rerank"]
    user_contract = "ranking_order"

    def configure_reference(self, case):
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        trt_scores = trt_output.data.get("scores", [])
        ref_scores = ref_output.data.get("scores", [])

        if not trt_scores or not ref_scores:
            return make_error("full_inference", "Missing scores in output data")

        if len(trt_scores) != len(ref_scores):
            return make_fail("full_inference", {}, message=f"Score count mismatch: TRT={len(trt_scores)} ref={len(ref_scores)}")

        # Compare ranking order
        trt_order = sorted(range(len(trt_scores)), key=lambda i: -trt_scores[i])
        ref_order = sorted(range(len(ref_scores)), key=lambda i: -ref_scores[i])

        # Pairwise ordering agreement
        n = len(trt_scores)
        concordant = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                trt_cmp = (trt_scores[i] > trt_scores[j]) - (trt_scores[i] < trt_scores[j])
                ref_cmp = (ref_scores[i] > ref_scores[j]) - (ref_scores[i] < ref_scores[j])
                if trt_cmp == ref_cmp:
                    concordant += 1
                total += 1

        agreement = concordant / max(total, 1)
        top1_match = (trt_order[0] == ref_order[0]) if trt_order and ref_order else False

        agreement_threshold = threshold.metrics.get("contract_ranking_agreement", 0.9)

        metrics = {
            "pairwise_agreement": MetricResult(
                value=agreement, threshold=agreement_threshold, operator=">=",
                passed=agreement >= agreement_threshold),
            "top1_match": MetricResult(
                value=1.0 if top1_match else 0.0, threshold=1.0, operator="==",
                passed=top1_match),
        }

        passed = agreement >= agreement_threshold
        if passed:
            return make_pass("full_inference", metrics, "pairwise_agreement >= threshold")
        return make_fail("full_inference", metrics, "pairwise_agreement >= threshold",
                        f"Ranking diverged: agreement={agreement:.3f}")


plugin = RerankingPlugin()
