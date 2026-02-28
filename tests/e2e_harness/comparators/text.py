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

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)

logger = logging.getLogger(__name__)

# Default top-k for unstable token checking
_DEFAULT_TOP_K = 5
# Default stable margin threshold
_DEFAULT_STABLE_MARGIN = 0.1

_COMPOSITE_RULE = (
    "(cosine_p5 >= T OR rel_l2_p95 <= T) "
    "AND (agreement >= T OR (stable_top1 >= T AND unstable_topk >= T)) "
    "AND ned <= hard_fail"
)


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
        metrics: dict[str, MetricResult] = {}

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
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"Cannot compare: missing logits from {', '.join(missing)}",
            )

        if trt_logits.ndim != 2 or ref_logits.ndim != 2:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"Logits must be 2-D [steps, vocab]: TRT={trt_logits.shape}, HF={ref_logits.shape}",
            )

        # Truncate to common step count
        n_steps = min(trt_logits.shape[0], ref_logits.shape[0])
        if n_steps == 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message="No steps to compare",
            )

        trt_l = trt_logits[:n_steps]
        ref_l = ref_logits[:n_steps]

        # Ensure same vocab dimension
        notes: list[str] = []
        if trt_l.shape[1] != ref_l.shape[1]:
            min_vocab = min(trt_l.shape[1], ref_l.shape[1])
            notes.append(
                f"Vocab size mismatch: TRT={trt_l.shape[1]}, HF={ref_l.shape[1]}; "
                f"truncating to {min_vocab}"
            )
            trt_l = trt_l[:, :min_vocab]
            ref_l = ref_l[:, :min_vocab]

        # Numerical health
        health_warnings = []
        health_warnings.extend(_check_numerical_health(trt_l, "TRT logits"))
        health_warnings.extend(_check_numerical_health(ref_l, "HF logits"))
        notes.extend(health_warnings)

        # Replace NaN/Inf with 0 for metric computation
        trt_clean = np.nan_to_num(trt_l, nan=0.0, posinf=0.0, neginf=0.0)
        ref_clean = np.nan_to_num(ref_l, nan=0.0, posinf=0.0, neginf=0.0)

        thresh = threshold.metrics

        # --- Metric 1: logit_cosine_p5 ---
        cosines = np.array([
            _cosine_similarity(trt_clean[i], ref_clean[i])
            for i in range(n_steps)
        ])
        logit_cosine_p5 = float(np.percentile(cosines, 5))
        cosine_thresh = thresh.get("logit_cosine_p5", 0.99)
        metrics["logit_cosine_p5"] = MetricResult(
            value=logit_cosine_p5, threshold=cosine_thresh,
            operator=">=", passed=logit_cosine_p5 >= cosine_thresh,
        )

        # --- Metric 2: logit_rel_l2_p95 ---
        rel_l2s = np.array([
            _relative_l2(trt_clean[i], ref_clean[i])
            for i in range(n_steps)
        ])
        logit_rel_l2_p95 = float(np.percentile(rel_l2s, 95))
        rel_l2_thresh = thresh.get("logit_rel_l2_p95", 0.05)
        metrics["logit_rel_l2_p95"] = MetricResult(
            value=logit_rel_l2_p95, threshold=rel_l2_thresh,
            operator="<=", passed=logit_rel_l2_p95 <= rel_l2_thresh,
        )

        # --- Per-step argmax and margin analysis ---
        trt_argmax = trt_clean.argmax(axis=1)
        ref_argmax = ref_clean.argmax(axis=1)

        ref_sorted = np.sort(ref_clean, axis=1)
        hf_margin = ref_sorted[:, -1] - ref_sorted[:, -2]

        stable_margin = thresh.get("stable_margin", _DEFAULT_STABLE_MARGIN)
        top_k = int(thresh.get("top_k", _DEFAULT_TOP_K))

        stable_mask = hf_margin >= stable_margin
        unstable_mask = ~stable_mask
        n_stable = int(stable_mask.sum())
        n_unstable = int(unstable_mask.sum())

        # --- Metric 3: stable_top1_match_rate ---
        if n_stable > 0:
            stable_matches = int((trt_argmax[stable_mask] == ref_argmax[stable_mask]).sum())
            stable_top1_match_rate = stable_matches / n_stable
        else:
            stable_top1_match_rate = 1.0
        stable_thresh = thresh.get("stable_top1_match_rate", 0.9)
        metrics["stable_top1_match_rate"] = MetricResult(
            value=stable_top1_match_rate, threshold=stable_thresh,
            operator=">=", passed=stable_top1_match_rate >= stable_thresh,
            note=f"{n_stable} stable steps",
        )

        # --- Metric 4: unstable_topk_hit_rate ---
        if n_unstable > 0:
            ref_topk = np.argsort(ref_clean, axis=1)[:, -top_k:]
            hits = 0
            unstable_indices = np.where(unstable_mask)[0]
            for idx in unstable_indices:
                if trt_argmax[idx] in ref_topk[idx]:
                    hits += 1
            unstable_topk_hit_rate = hits / n_unstable
        else:
            unstable_topk_hit_rate = 1.0
        topk_thresh = thresh.get("unstable_topk_hit_rate", 0.8)
        metrics["unstable_topk_hit_rate"] = MetricResult(
            value=unstable_topk_hit_rate, threshold=topk_thresh,
            operator=">=", passed=unstable_topk_hit_rate >= topk_thresh,
            note=f"{n_unstable} unstable steps",
        )

        # --- Metric 5: token_agreement_rate ---
        token_agreement_rate = float((trt_argmax == ref_argmax).mean())
        ta_thresh = thresh.get("token_agreement_rate", 0.8)
        metrics["token_agreement_rate"] = MetricResult(
            value=token_agreement_rate, threshold=ta_thresh,
            operator=">=", passed=token_agreement_rate >= ta_thresh,
        )

        # --- Metric 6: normalized_text_edit_distance ---
        trt_text = (trt.text or "").strip()
        ref_text = (ref.text or "").strip()

        prompt = (trt.data or {}).get("prompt", "")
        trt_text_for_ned = trt_text
        if prompt and trt_text.startswith(prompt):
            trt_text_for_ned = trt_text[len(prompt):].lstrip()

        if trt_text_for_ned or ref_text:
            ned = _normalized_edit_distance(trt_text_for_ned, ref_text)
        else:
            ned = 0.0
        ned_thresh = thresh.get("normalized_text_edit_distance", 0.2)
        metrics["normalized_text_edit_distance"] = MetricResult(
            value=ned, threshold=ned_thresh,
            operator="<=", passed=ned <= ned_thresh,
        )

        # --- Composite gating ---
        logit_quality_ok = (
            metrics["logit_cosine_p5"].passed
            or metrics["logit_rel_l2_p95"].passed
        )

        token_level_ok = (
            metrics["token_agreement_rate"].passed
            or (
                metrics["stable_top1_match_rate"].passed
                and metrics["unstable_topk_hit_rate"].passed
            )
        )

        text_ok = metrics["normalized_text_edit_distance"].passed
        ned_hard_fail_threshold = 0.65
        if token_level_ok and ned < ned_hard_fail_threshold:
            text_ok = True

        passed = logit_quality_ok and token_level_ok and text_ok

        message = (
            f"{'PASS' if passed else 'FAIL'}: "
            f"cosine_p5={logit_cosine_p5:.4f}, "
            f"agreement={token_agreement_rate:.4f}, "
            f"ned={ned:.4f}"
        )

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=_COMPOSITE_RULE,
            message=message,
        )


plugin = TextComparator()
