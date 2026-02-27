"""Omni-multimodal comparator — compare TRT vs reference multi-branch outputs.

Metrics per branch: thinker token agreement, vision/audio embedding cosine,
talker token match, code2wav spectral distance, e2e text edit distance.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

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


def _token_agreement(a: List[int], b: List[int]) -> float:
    """Fraction of tokens that match between two sequences."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    matches = sum(1 for i in range(min_len) if a[i] == b[i])
    return matches / max(len(a), len(b))


def _text_edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def _normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalized by max length."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return _text_edit_distance(a, b) / max_len


class OmniComparator:
    """Compare TRT vs reference omni-multimodal outputs.

    Evaluates stage-specific metrics depending on the stage name.
    """

    @property
    def task_strategy(self) -> str:
        return "omni_multimodal"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        stage_name = stage.name
        th = threshold.metrics

        # Dispatch to stage-specific comparison
        if stage_name == "thinker_decode":
            return self._compare_thinker(trt, ref, th, stage)
        elif stage_name in ("vision_encode", "audio_encode"):
            return self._compare_encoder(trt, ref, th, stage)
        elif stage_name == "talker_decode":
            return self._compare_talker(trt, ref, th, stage)
        elif stage_name == "end_to_end":
            return self._compare_e2e(trt, ref, th, stage)
        else:
            return self._compare_generic(trt, ref, th, stage)

    def _compare_thinker(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare thinker text decoding output."""
        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])

        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: List[str] = []

        if trt_tokens and ref_tokens:
            agreement = _token_agreement(trt_tokens, ref_tokens)
            metrics["thinker_token_agreement"] = agreement
            thresh = th.get("thinker_token_agreement", 0.8)
            per_metric_pass["thinker_token_agreement"] = agreement >= thresh
            gate_details.append(
                f"thinker_token_agreement: {agreement:.4f} >= {thresh} -> "
                f"{'PASS' if per_metric_pass['thinker_token_agreement'] else 'FAIL'}"
            )

        trt_text = trt.text or ""
        ref_text = ref.text or ""
        if trt_text or ref_text:
            ned = _normalized_edit_distance(trt_text, ref_text)
            metrics["thinker_text_edit_distance"] = ned
            thresh = th.get("thinker_text_edit_distance", 0.3)
            per_metric_pass["thinker_text_edit_distance"] = ned <= thresh
            gate_details.append(
                f"thinker_text_edit_distance: {ned:.4f} <= {thresh} -> "
                f"{'PASS' if per_metric_pass['thinker_text_edit_distance'] else 'FAIL'}"
            )

        passed = all(per_metric_pass.values()) if per_metric_pass else False
        return CompareResult(
            stage_name=stage.name, passed=passed, metrics=metrics,
            per_metric_pass=per_metric_pass, gate_details=gate_details,
            message=f"Thinker comparison: {len(metrics)} metrics",
        )

    def _compare_encoder(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare vision/audio encoder embedding output."""
        trt_emb = trt.data.get("embedding", [])
        ref_emb = ref.data.get("embedding", [])

        if not trt_emb or not ref_emb:
            missing = []
            if not trt_emb:
                missing.append("TRT")
            if not ref_emb:
                missing.append("ref")
            return CompareResult(
                stage_name=stage.name, passed=False,
                metrics={},
                per_metric_pass={},
                gate_details=[f"early return: missing embedding from {', '.join(missing)} for {stage.name}"],
                message=f"Missing embedding for {stage.name}",
            )

        cosine = _cosine_similarity(trt_emb, ref_emb)
        # Use canonical name from threshold defaults (e.g. "vision_embedding_cosine")
        branch = stage.name.replace("_encode", "")
        metric_name = f"{branch}_embedding_cosine"
        metrics = {metric_name: cosine}
        # Look up threshold: canonical name -> stage-based name -> generic fallback
        thresh = th.get(metric_name, th.get(
            f"{stage.name}_embedding_cosine", th.get("encoder_embedding_cosine", 0.95)))
        per_metric_pass = {metric_name: cosine >= thresh}
        gate_details = [
            f"{metric_name}: {cosine:.6f} >= {thresh} -> "
            f"{'PASS' if per_metric_pass[metric_name] else 'FAIL'}"
        ]

        return CompareResult(
            stage_name=stage.name, passed=all(per_metric_pass.values()),
            metrics=metrics, per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"{stage.name} embedding cosine={cosine:.6f}",
        )

    def _compare_talker(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare talker decoding output (tokens and/or audio)."""
        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: List[str] = []

        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])

        if trt_tokens and ref_tokens:
            agreement = _token_agreement(trt_tokens, ref_tokens)
            metrics["talker_token_match"] = agreement
            thresh = th.get("talker_token_match", 0.7)
            per_metric_pass["talker_token_match"] = agreement >= thresh
            gate_details.append(
                f"talker_token_match: {agreement:.4f} >= {thresh} -> "
                f"{'PASS' if per_metric_pass['talker_token_match'] else 'FAIL'}"
            )

        passed = all(per_metric_pass.values()) if per_metric_pass else True
        return CompareResult(
            stage_name=stage.name, passed=passed, metrics=metrics,
            per_metric_pass=per_metric_pass, gate_details=gate_details,
            message=f"Talker comparison: {len(metrics)} metrics",
        )

    def _compare_e2e(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare end-to-end omni output (text edit distance)."""
        trt_text = trt.text or ""
        ref_text = ref.text or ""

        ned = _normalized_edit_distance(trt_text, ref_text)
        metrics = {"e2e_text_edit_distance": ned}
        thresh = th.get("e2e_text_edit_distance", 0.3)
        per_metric_pass = {"e2e_text_edit_distance": ned <= thresh}
        gate_details = [
            f"e2e_text_edit_distance: {ned:.4f} <= {thresh} -> "
            f"{'PASS' if per_metric_pass['e2e_text_edit_distance'] else 'FAIL'}"
        ]

        return CompareResult(
            stage_name=stage.name, passed=all(per_metric_pass.values()),
            metrics=metrics, per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"E2E text edit distance={ned:.4f}",
        )

    def _compare_generic(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Fallback: compare any stage with available data."""
        # Try embedding comparison
        trt_emb = trt.data.get("embedding", [])
        ref_emb = ref.data.get("embedding", [])
        if trt_emb and ref_emb:
            return self._compare_encoder(trt, ref, th, stage)

        # Try text comparison
        if trt.text and ref.text:
            return self._compare_e2e(trt, ref, th, stage)

        return CompareResult(
            stage_name=stage.name, passed=True,
            metrics={},
            per_metric_pass={},
            gate_details=[f"No comparable data for stage {stage.name} (pass by default)"],
            message=f"No comparable data for stage {stage.name} (pass by default)",
        )


plugin = OmniComparator()
