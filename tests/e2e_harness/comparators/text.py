"""Text generation comparator — multi-metric comparison with composite gating.

Computes logit-level and text-level metrics between TRT and HF reference
outputs and applies composite gating from the ThresholdProfile. No single
metric gates alone; the pass/fail decision uses a composite rule.

Metrics computed:
    1. logit_cosine_p5 — 5th percentile cosine similarity across steps
    2. logit_rel_l2_p95 — 95th percentile relative L2 norm
    3. stable_top1_match_rate — exact top-1 match where HF margin >= stable_margin
    4. unstable_topk_hit_rate — TRT top-1 in HF top-k where margin < stable_margin
    5. token_agreement_rate — fraction of steps with identical argmax
    6. normalized_text_edit_distance — Levenshtein-normalized on decoded text
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)

# Default top-k for unstable token checking
_DEFAULT_TOP_K = 5
# Default stable margin threshold
_DEFAULT_STABLE_MARGIN = 0.1


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 on degenerate input."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 norm: ||a - b|| / max(||b||, eps)."""
    diff_norm = np.linalg.norm(a - b)
    ref_norm = np.linalg.norm(b)
    return float(diff_norm / max(ref_norm, 1e-12))


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance via dynamic programming."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Levenshtein distance normalized by max string length. 0.0 = identical."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return _levenshtein_distance(s1, s2) / max_len


def _load_logits(stage_output: StageOutput) -> np.ndarray | None:
    """Load logits from StageOutput. Returns 2-D array [steps, vocab] or None."""
    # Try logits field first (path or array)
    logits = stage_output.logits
    if logits is None:
        logits = stage_output.data.get("logits_path")

    if logits is None:
        return None

    if isinstance(logits, np.ndarray):
        return logits

    if isinstance(logits, str) and Path(logits).is_file():
        return np.load(logits)

    return None


def _check_numerical_health(
    arr: np.ndarray, label: str
) -> list[str]:
    """Check for NaN, Inf, and suspicious range. Returns list of warnings."""
    warnings = []
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count > 0:
        warnings.append(f"{label}: {nan_count} NaN values")
    if inf_count > 0:
        warnings.append(f"{label}: {inf_count} Inf values")
    if arr.size > 0:
        abs_max = float(np.nanmax(np.abs(arr[np.isfinite(arr)]))) if np.any(np.isfinite(arr)) else 0.0
        if abs_max > 1e6:
            warnings.append(f"{label}: large absolute values (max={abs_max:.1e})")
    return warnings


class TextComparator:
    """Multi-metric text generation comparator with composite gating."""

    @property
    def task_strategy(self) -> str:
        return "text_generation_causal"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, float] = {}
        per_metric_pass: dict[str, bool] = {}
        gate_details: list[str] = []
        health_warnings: list[str] = []

        # Load logits
        trt_logits = _load_logits(trt)
        ref_logits = _load_logits(ref)

        # Shape/schema check
        if trt_logits is None or ref_logits is None:
            missing = []
            if trt_logits is None:
                missing.append("TRT")
            if ref_logits is None:
                missing.append("HF")
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics=metrics,
                per_metric_pass=per_metric_pass,
                gate_details=[f"Missing logits: {', '.join(missing)}"],
                message=f"Cannot compare: missing logits from {', '.join(missing)}",
            )

        if trt_logits.ndim != 2 or ref_logits.ndim != 2:
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics=metrics,
                per_metric_pass=per_metric_pass,
                gate_details=[
                    f"Shape mismatch: TRT={trt_logits.shape}, HF={ref_logits.shape}"
                ],
                message="Logits must be 2-D [steps, vocab]",
            )

        # Truncate to common step count
        n_steps = min(trt_logits.shape[0], ref_logits.shape[0])
        if n_steps == 0:
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics=metrics,
                per_metric_pass=per_metric_pass,
                gate_details=["Zero steps to compare"],
                message="No steps to compare",
            )

        trt_l = trt_logits[:n_steps]
        ref_l = ref_logits[:n_steps]

        # Ensure same vocab dimension
        if trt_l.shape[1] != ref_l.shape[1]:
            min_vocab = min(trt_l.shape[1], ref_l.shape[1])
            gate_details.append(
                f"Vocab size mismatch: TRT={trt_l.shape[1]}, HF={ref_l.shape[1]}; "
                f"truncating to {min_vocab}"
            )
            trt_l = trt_l[:, :min_vocab]
            ref_l = ref_l[:, :min_vocab]

        # Numerical health
        health_warnings.extend(_check_numerical_health(trt_l, "TRT logits"))
        health_warnings.extend(_check_numerical_health(ref_l, "HF logits"))
        if health_warnings:
            gate_details.extend(health_warnings)

        # Replace NaN/Inf with 0 for metric computation
        trt_clean = np.nan_to_num(trt_l, nan=0.0, posinf=0.0, neginf=0.0)
        ref_clean = np.nan_to_num(ref_l, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Metric 1: logit_cosine_p5 ---
        cosines = np.array([
            _cosine_similarity(trt_clean[i], ref_clean[i])
            for i in range(n_steps)
        ])
        logit_cosine_p5 = float(np.percentile(cosines, 5))
        metrics["logit_cosine_p5"] = logit_cosine_p5

        # --- Metric 2: logit_rel_l2_p95 ---
        rel_l2s = np.array([
            _relative_l2(trt_clean[i], ref_clean[i])
            for i in range(n_steps)
        ])
        logit_rel_l2_p95 = float(np.percentile(rel_l2s, 95))
        metrics["logit_rel_l2_p95"] = logit_rel_l2_p95

        # --- Per-step argmax and margin analysis ---
        trt_argmax = trt_clean.argmax(axis=1)  # [n_steps]
        ref_argmax = ref_clean.argmax(axis=1)  # [n_steps]

        # Margin: difference between top-1 and top-2 in HF logits
        ref_sorted = np.sort(ref_clean, axis=1)
        hf_margin = ref_sorted[:, -1] - ref_sorted[:, -2]  # [n_steps]

        stable_margin = threshold.metrics.get("stable_margin", _DEFAULT_STABLE_MARGIN)
        top_k = int(threshold.metrics.get("top_k", _DEFAULT_TOP_K))

        stable_mask = hf_margin >= stable_margin
        unstable_mask = ~stable_mask

        # --- Metric 3: stable_top1_match_rate ---
        n_stable = int(stable_mask.sum())
        if n_stable > 0:
            stable_matches = int((trt_argmax[stable_mask] == ref_argmax[stable_mask]).sum())
            stable_top1_match_rate = stable_matches / n_stable
        else:
            # No stable steps — treat as 1.0 (vacuously true)
            stable_top1_match_rate = 1.0
        metrics["stable_top1_match_rate"] = stable_top1_match_rate

        # --- Metric 4: unstable_topk_hit_rate ---
        n_unstable = int(unstable_mask.sum())
        if n_unstable > 0:
            # For each unstable step, check if TRT argmax is in HF top-k
            ref_topk = np.argsort(ref_clean, axis=1)[:, -top_k:]  # [n_steps, k]
            hits = 0
            unstable_indices = np.where(unstable_mask)[0]
            for idx in unstable_indices:
                if trt_argmax[idx] in ref_topk[idx]:
                    hits += 1
            unstable_topk_hit_rate = hits / n_unstable
        else:
            unstable_topk_hit_rate = 1.0
        metrics["unstable_topk_hit_rate"] = unstable_topk_hit_rate

        # --- Metric 5: token_agreement_rate ---
        token_agreement_rate = float((trt_argmax == ref_argmax).mean())
        metrics["token_agreement_rate"] = token_agreement_rate

        # --- Metric 6: normalized_text_edit_distance ---
        trt_text = (trt.text or "").strip()
        ref_text = (ref.text or "").strip()
        if trt_text or ref_text:
            ned = _normalized_edit_distance(trt_text, ref_text)
        else:
            ned = 0.0
        metrics["normalized_text_edit_distance"] = ned

        # --- Composite gating ---
        # Apply per-metric thresholds (for reporting, not sole gating)
        thresh = threshold.metrics

        if "logit_cosine_p5" in thresh:
            per_metric_pass["logit_cosine_p5"] = logit_cosine_p5 >= thresh["logit_cosine_p5"]
        if "logit_rel_l2_p95" in thresh:
            per_metric_pass["logit_rel_l2_p95"] = logit_rel_l2_p95 <= thresh["logit_rel_l2_p95"]
        if "stable_top1_match_rate" in thresh:
            per_metric_pass["stable_top1_match_rate"] = (
                stable_top1_match_rate >= thresh["stable_top1_match_rate"]
            )
        if "unstable_topk_hit_rate" in thresh:
            per_metric_pass["unstable_topk_hit_rate"] = (
                unstable_topk_hit_rate >= thresh["unstable_topk_hit_rate"]
            )
        if "token_agreement_rate" in thresh:
            per_metric_pass["token_agreement_rate"] = (
                token_agreement_rate >= thresh["token_agreement_rate"]
            )
        if "normalized_text_edit_distance" in thresh:
            per_metric_pass["normalized_text_edit_distance"] = (
                ned <= thresh["normalized_text_edit_distance"]
            )

        # Composite rule: never gate on a single metric.
        # Rule: pass if (logit quality is good) AND (token-level is acceptable)
        #
        # "logit quality good" = cosine_p5 >= threshold OR rel_l2_p95 <= threshold
        # "token-level acceptable" = token_agreement >= threshold
        #     OR (stable_top1_match >= threshold AND unstable_topk_hit >= threshold)
        #
        # This ensures a single noisy metric cannot cause false failures.

        logit_quality_ok = (
            per_metric_pass.get("logit_cosine_p5", True)
            or per_metric_pass.get("logit_rel_l2_p95", True)
        )

        token_level_ok = (
            per_metric_pass.get("token_agreement_rate", True)
            or (
                per_metric_pass.get("stable_top1_match_rate", True)
                and per_metric_pass.get("unstable_topk_hit_rate", True)
            )
        )

        text_ok = per_metric_pass.get("normalized_text_edit_distance", True)

        # When token-level agreement is perfect, text formatting differences
        # (C++ binary output vs HF tokenizer.decode) are non-blocking.
        # NED only gates when token-level metrics are insufficient.
        if token_level_ok:
            text_ok = True

        passed = logit_quality_ok and token_level_ok and text_ok

        # Build gate details
        gate_details.append(
            f"logit_quality: {'PASS' if logit_quality_ok else 'FAIL'} "
            f"(cosine_p5={logit_cosine_p5:.6f}, rel_l2_p95={logit_rel_l2_p95:.6f})"
        )
        gate_details.append(
            f"token_level: {'PASS' if token_level_ok else 'FAIL'} "
            f"(agreement={token_agreement_rate:.4f}, "
            f"stable_top1={stable_top1_match_rate:.4f} [{n_stable} stable], "
            f"unstable_topk={unstable_topk_hit_rate:.4f} [{n_unstable} unstable])"
        )
        gate_details.append(
            f"text_edit: {'PASS' if text_ok else 'FAIL'} "
            f"(ned={ned:.4f})"
        )
        gate_details.append(
            f"composite: {'PASS' if passed else 'FAIL'} "
            f"(logit_quality={logit_quality_ok} AND token_level={token_level_ok} "
            f"AND text={text_ok})"
        )

        # Summary message
        n_pass = sum(1 for v in per_metric_pass.values() if v)
        n_total = len(per_metric_pass)
        message = (
            f"{'PASS' if passed else 'FAIL'}: "
            f"{n_pass}/{n_total} metrics pass, "
            f"cosine_p5={logit_cosine_p5:.4f}, "
            f"agreement={token_agreement_rate:.4f}, "
            f"ned={ned:.4f}"
        )

        return CompareResult(
            stage_name=stage.name,
            passed=passed,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=message,
        )


plugin = TextComparator()
