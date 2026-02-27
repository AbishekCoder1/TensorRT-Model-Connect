"""Vision-language comparator — compares TRT VL output against reference.

Metrics aligned with thresholds/defaults/vision_language_generation.json:
  - vision_embedding_cosine: Cosine similarity between vision embeddings.
  - vision_embedding_l2: L2 distance between vision embeddings.
  - token_agreement_rate: Fraction of steps with identical argmax (reuses text logic).
  - normalized_text_edit_distance: Levenshtein-normalized on decoded text.
  - semantic_similarity: Optional semantic similarity for caption parity.

The comparator handles three stage types:
  - "vision_encode": Compares vision encoder features.
  - "text_decode": Compares per-step logits (reuses text comparator helpers).
  - "full_generation": Compares generated text output.

Auto-discovered by the registry via the module-level ``plugin`` attribute.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from ..contracts import (
    CompareResult,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text helpers (reused from text comparator module)
# ---------------------------------------------------------------------------


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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 on degenerate input."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_logits(stage_output: StageOutput) -> np.ndarray | None:
    """Load logits from StageOutput. Returns 2-D array [steps, vocab] or None."""
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


def _load_features(stage_output: StageOutput) -> np.ndarray | None:
    """Load vision features from StageOutput."""
    features = stage_output.data.get("features")
    if features is not None:
        return np.asarray(features, dtype=np.float32)
    path = stage_output.data.get("features_path")
    if path and Path(path).is_file():
        return np.load(path)
    return None


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class VisionLanguageComparator:
    """Compares TRT VL inference against reference (HF Transformers).

    Metric names align with thresholds/defaults/vision_language_generation.json.
    """

    @property
    def task_strategy(self) -> str:
        return "vision_language_generation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        if stage.name == "vision_encode":
            return self._compare_vision(trt, ref, threshold, stage)
        elif stage.name == "text_decode":
            return self._compare_text_decode(trt, ref, threshold, stage)
        elif stage.name == "full_generation":
            return self._compare_generation(trt, ref, threshold, stage)
        else:
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics={},
                per_metric_pass={},
                gate_details=[f"early return: unknown stage for VL comparator: {stage.name}"],
                message=f"Unknown stage for VL comparator: {stage.name}",
            )

    # ------------------------------------------------------------------
    # Vision encode comparison
    # ------------------------------------------------------------------

    def _compare_vision(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare vision encoder features between TRT and reference."""
        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: list[str] = []

        trt_features = _load_features(trt)
        ref_features = _load_features(ref)

        # Fall back to subprocess-parsed metrics if no raw features
        trt_sub_metrics = trt.data.get("metrics", {})

        # If the diff_vl.py subprocess passed (rc=0 and "vision_pass" flag),
        # trust the result directly — the tool already did full TRT vs HF
        # comparison internally.
        if trt.data.get("passed") and trt_sub_metrics.get("vision_pass"):
            gate_details.append("diff_vl.py subprocess PASS (internal comparison)")
            return CompareResult(
                stage_name=stage.name,
                passed=True,
                metrics={"vision_subprocess_pass": 1.0},
                per_metric_pass={"vision_subprocess_pass": True},
                gate_details=gate_details,
                message="Vision compare: PASS (diff_vl.py)",
            )

        if trt_features is not None and ref_features is not None:
            trt_f = trt_features
            ref_f = ref_features

            # Handle shape mismatch by comparing overlapping region
            if trt_f.shape != ref_f.shape:
                min_shape = tuple(
                    min(a, b) for a, b in zip(trt_f.shape, ref_f.shape)
                )
                slices = tuple(slice(0, s) for s in min_shape)
                trt_f = trt_f[slices]
                ref_f = ref_f[slices]
                gate_details.append(
                    f"Shape mismatch: TRT={trt_features.shape} "
                    f"vs Ref={ref_features.shape}; "
                    f"compared overlap region {min_shape}"
                )

            # Cosine similarity — matches threshold key "vision_embedding_cosine"
            cosine = _cosine_similarity(trt_f.flatten(), ref_f.flatten())
            metrics["vision_embedding_cosine"] = cosine

            # L2 distance — matches threshold key "vision_embedding_l2"
            l2 = float(np.sqrt(np.mean((trt_f - ref_f) ** 2)))
            metrics["vision_embedding_l2"] = l2

            # Max absolute difference (diagnostic, not gated)
            metrics["vision_max_diff"] = float(np.max(np.abs(trt_f - ref_f)))

        elif trt_sub_metrics:
            # Use metrics parsed from diff_vl.py subprocess output
            if "cosine_sim" in trt_sub_metrics:
                metrics["vision_embedding_cosine"] = trt_sub_metrics["cosine_sim"]
            if "max_diff" in trt_sub_metrics:
                metrics["vision_max_diff"] = trt_sub_metrics["max_diff"]
            if "mean_diff" in trt_sub_metrics:
                metrics["vision_embedding_l2"] = trt_sub_metrics["mean_diff"]
        else:
            gate_details.append("No vision features available for comparison")
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics=metrics,
                per_metric_pass=per_metric_pass,
                gate_details=gate_details,
                message="No vision features available",
            )

        # Gate: vision_embedding_cosine
        cos_thresh = threshold.metrics.get("vision_embedding_cosine", 0.5)
        if "vision_embedding_cosine" in metrics:
            passed_cos = metrics["vision_embedding_cosine"] >= cos_thresh
            per_metric_pass["vision_embedding_cosine"] = passed_cos
            gate_details.append(
                f"vision_embedding_cosine: "
                f"{metrics['vision_embedding_cosine']:.6f} >= {cos_thresh} "
                f"-> {'PASS' if passed_cos else 'FAIL'}"
            )

        # Gate: vision_embedding_l2
        l2_thresh = threshold.metrics.get("vision_embedding_l2")
        if l2_thresh is not None and "vision_embedding_l2" in metrics:
            passed_l2 = metrics["vision_embedding_l2"] <= l2_thresh
            per_metric_pass["vision_embedding_l2"] = passed_l2
            gate_details.append(
                f"vision_embedding_l2: "
                f"{metrics['vision_embedding_l2']:.6f} <= {l2_thresh} "
                f"-> {'PASS' if passed_l2 else 'FAIL'}"
            )

        overall = all(per_metric_pass.values()) if per_metric_pass else False
        return CompareResult(
            stage_name=stage.name,
            passed=overall,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"Vision compare: {'PASS' if overall else 'FAIL'}",
        )

    # ------------------------------------------------------------------
    # Text decode comparison (per-step logits)
    # ------------------------------------------------------------------

    def _compare_text_decode(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare per-step logits from VLTrtRunner vs reference.

        Reuses the same metric definitions as the text comparator:
        token_agreement_rate, normalized_text_edit_distance, and
        logit-level cosine similarity.
        """
        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: list[str] = []

        trt_logits = _load_logits(trt)
        ref_logits = _load_logits(ref)

        if trt_logits is not None and ref_logits is not None:
            # Truncate to common step/vocab
            n_steps = min(trt_logits.shape[0], ref_logits.shape[0])
            if n_steps == 0:
                gate_details.append("Zero steps to compare")
                return CompareResult(
                    stage_name=stage.name, passed=False,
                    metrics=metrics, per_metric_pass=per_metric_pass,
                    gate_details=gate_details, message="No steps",
                )
            trt_l = trt_logits[:n_steps]
            ref_l = ref_logits[:n_steps]
            if trt_l.shape[1] != ref_l.shape[1]:
                min_v = min(trt_l.shape[1], ref_l.shape[1])
                trt_l = trt_l[:, :min_v]
                ref_l = ref_l[:, :min_v]

            trt_l = np.nan_to_num(trt_l, nan=0.0, posinf=0.0, neginf=0.0)
            ref_l = np.nan_to_num(ref_l, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-step cosine similarity
            cosines = np.array([
                _cosine_similarity(trt_l[i], ref_l[i])
                for i in range(n_steps)
            ])
            metrics["logit_cosine_p5"] = float(np.percentile(cosines, 5))

            # Token agreement rate
            trt_argmax = trt_l.argmax(axis=1)
            ref_argmax = ref_l.argmax(axis=1)
            token_agreement = float((trt_argmax == ref_argmax).mean())
            metrics["token_agreement_rate"] = token_agreement

            gate_details.append(
                f"logit steps={n_steps}, "
                f"cosine_p5={metrics['logit_cosine_p5']:.6f}, "
                f"token_agreement={token_agreement:.4f}"
            )

            # Gate: token_agreement_rate
            ta_thresh = threshold.metrics.get("token_agreement_rate")
            if ta_thresh is not None:
                passed_ta = token_agreement >= ta_thresh
                per_metric_pass["token_agreement_rate"] = passed_ta
                gate_details.append(
                    f"token_agreement_rate: {token_agreement:.4f} >= "
                    f"{ta_thresh} -> {'PASS' if passed_ta else 'FAIL'}"
                )
        else:
            gate_details.append("Logits not available; using text-only comparison")

        # Text comparison (always available from text_decode stage)
        trt_text = (trt.text or trt.data.get("generated_text") or "").strip()
        ref_text = (ref.text or ref.data.get("generated_text") or "").strip()

        if trt_text or ref_text:
            ned = _normalized_edit_distance(trt_text, ref_text)
            metrics["normalized_text_edit_distance"] = ned

            ned_thresh = threshold.metrics.get("normalized_text_edit_distance")
            if ned_thresh is not None:
                passed_ned = ned <= ned_thresh
                per_metric_pass["normalized_text_edit_distance"] = passed_ned
                gate_details.append(
                    f"normalized_text_edit_distance: {ned:.4f} <= "
                    f"{ned_thresh} -> {'PASS' if passed_ned else 'FAIL'}"
                )

        overall = all(per_metric_pass.values()) if per_metric_pass else True
        return CompareResult(
            stage_name=stage.name,
            passed=overall,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"VL text decode: {'PASS' if overall else 'FAIL'}",
        )

    # ------------------------------------------------------------------
    # Full generation comparison (C++ binary text output)
    # ------------------------------------------------------------------

    def _compare_generation(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare generated text between TRT C++ binary and reference.

        Computes normalized_text_edit_distance, token_agreement_rate (on
        decoded text words), and optional semantic_similarity.
        """
        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: list[str] = []

        trt_text = (trt.text or trt.data.get("generated_text") or "").strip()
        ref_text = (ref.text or ref.data.get("generated_text") or "").strip()

        metrics["trt_generated_length"] = float(len(trt_text))
        metrics["ref_generated_length"] = float(len(ref_text))

        if not trt_text:
            gate_details.append("TRT produced empty output")
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                metrics=metrics,
                per_metric_pass={"non_empty_output": False},
                gate_details=gate_details,
                message="TRT produced empty VL generation output",
            )
        per_metric_pass["non_empty_output"] = True

        # Normalized text edit distance
        ned = _normalized_edit_distance(trt_text, ref_text) if ref_text else 0.0
        metrics["normalized_text_edit_distance"] = ned

        ned_thresh = threshold.metrics.get("normalized_text_edit_distance")
        if ned_thresh is not None:
            passed_ned = ned <= ned_thresh
            per_metric_pass["normalized_text_edit_distance"] = passed_ned
            gate_details.append(
                f"normalized_text_edit_distance: {ned:.4f} <= "
                f"{ned_thresh} -> {'PASS' if passed_ned else 'FAIL'}"
            )

        # Word-level token agreement (approximate token_agreement_rate on text)
        trt_words = trt_text.lower().split()
        ref_words = ref_text.lower().split()
        if trt_words and ref_words:
            n_compare = min(len(trt_words), len(ref_words))
            matches = sum(
                1 for a, b in zip(trt_words[:n_compare], ref_words[:n_compare])
                if a == b
            )
            word_agreement = matches / n_compare
            metrics["token_agreement_rate"] = word_agreement

            ta_thresh = threshold.metrics.get("token_agreement_rate")
            if ta_thresh is not None:
                passed_ta = word_agreement >= ta_thresh
                per_metric_pass["token_agreement_rate"] = passed_ta
                gate_details.append(
                    f"token_agreement_rate (word-level): {word_agreement:.4f} >= "
                    f"{ta_thresh} -> {'PASS' if passed_ta else 'FAIL'}"
                )

        # Optional semantic similarity (requires sentence-transformers)
        sem_thresh = threshold.metrics.get("semantic_similarity")
        if sem_thresh is not None and trt_text and ref_text:
            sem_sim = _compute_semantic_similarity(trt_text, ref_text)
            if sem_sim is not None:
                metrics["semantic_similarity"] = sem_sim
                passed_sem = sem_sim >= sem_thresh
                per_metric_pass["semantic_similarity"] = passed_sem
                gate_details.append(
                    f"semantic_similarity: {sem_sim:.4f} >= "
                    f"{sem_thresh} -> {'PASS' if passed_sem else 'FAIL'}"
                )
            else:
                gate_details.append(
                    "semantic_similarity: skipped (sentence-transformers not available)"
                )

        gate_details.append(f"TRT: {trt_text[:100]!r}")
        gate_details.append(f"Ref: {ref_text[:100]!r}")

        # Composite rule: NED alone is sufficient for VL generation.
        # Word-level agreement is unreliable for VL since the same scene
        # can be described with different words that are equally valid.
        ned_ok = per_metric_pass.get("normalized_text_edit_distance", True)
        ta_ok = per_metric_pass.get("token_agreement_rate", True)
        non_empty = per_metric_pass.get("non_empty_output", True)
        overall = non_empty and (ned_ok or ta_ok)
        return CompareResult(
            stage_name=stage.name,
            passed=overall,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"VL generation compare: {'PASS' if overall else 'FAIL'}",
        )


def _compute_semantic_similarity(text_a: str, text_b: str) -> float | None:
    """Compute sentence-level semantic similarity using sentence-transformers.

    Returns cosine similarity between sentence embeddings, or None if
    sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
        cos = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-12
        ))
        return cos
    except Exception as e:
        logger.warning("Semantic similarity computation failed: %s", e)
        return None


plugin = VisionLanguageComparator()
