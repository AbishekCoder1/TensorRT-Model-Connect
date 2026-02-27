"""Speech-to-text comparator.

Compares TRT Whisper-style transcription output against reference with metrics:
- Decoder token agreement rate
- Word Error Rate (WER)
- Character Error Rate (CER)
- Timestamp sanity (if timestamps are available)
"""

from __future__ import annotations

import logging

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)


def _levenshtein_distance(s1: list, s2: list) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
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


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = _levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def _compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    distance = _levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def _token_agreement_rate(trt_tokens: list, ref_tokens: list) -> float:
    """Compute token-level agreement rate between two token sequences."""
    if not ref_tokens:
        return 1.0 if not trt_tokens else 0.0
    n = min(len(trt_tokens), len(ref_tokens))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if trt_tokens[i] == ref_tokens[i])
    return matches / max(len(trt_tokens), len(ref_tokens))


class SpeechToTextComparator:
    """Compares TRT transcription against reference transcription."""

    @property
    def task_strategy(self) -> str:
        return "speech_to_text"

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
        thresholds = threshold.metrics
        all_pass = True

        # Check TRT returncode
        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                message=f"TRT transcription failed (rc={trt.data.get('returncode')})",
            )

        trt_transcript = trt.data.get("transcript", trt.text or "")
        ref_transcript = ref.data.get("transcript", ref.text or "")

        # Token agreement
        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])
        if trt_tokens and ref_tokens:
            agreement = _token_agreement_rate(trt_tokens, ref_tokens)
            metrics["token_agreement_rate"] = agreement
            thresh = thresholds.get("token_agreement_rate", 0.8)
            ok = agreement >= thresh
            per_metric_pass["token_agreement_rate"] = ok
            gate_details.append(
                f"{'PASS' if ok else 'FAIL'} token_agreement={agreement:.4f} (>= {thresh})")
            if not ok:
                all_pass = False

        # Word Error Rate
        if trt_transcript and ref_transcript:
            wer = _compute_wer(ref_transcript, trt_transcript)
            metrics["wer"] = wer
            wer_thresh = thresholds.get("wer", 0.1)
            wer_ok = wer <= wer_thresh
            per_metric_pass["wer"] = wer_ok
            gate_details.append(
                f"{'PASS' if wer_ok else 'FAIL'} wer={wer:.4f} (<= {wer_thresh})")
            if not wer_ok:
                all_pass = False

            # Character Error Rate
            cer = _compute_cer(ref_transcript, trt_transcript)
            metrics["cer"] = cer
            cer_thresh = thresholds.get("cer", 0.05)
            cer_ok = cer <= cer_thresh
            per_metric_pass["cer"] = cer_ok
            gate_details.append(
                f"{'PASS' if cer_ok else 'FAIL'} cer={cer:.4f} (<= {cer_thresh})")
            if not cer_ok:
                all_pass = False

        # Timestamp sanity (if available)
        trt_timestamps = trt.data.get("timestamps", [])
        if trt_timestamps:
            ts_sane = self._check_timestamp_sanity(
                trt_timestamps,
                thresholds.get("timestamp_tolerance_s", 0.5),
            )
            metrics["timestamp_sanity"] = 1.0 if ts_sane else 0.0
            per_metric_pass["timestamp_sanity"] = ts_sane
            gate_details.append(
                f"{'PASS' if ts_sane else 'FAIL'} timestamp_sanity")
            if not ts_sane:
                all_pass = False

        return CompareResult(
            stage_name=stage.name,
            passed=all_pass,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"{'PASS' if all_pass else 'FAIL'}: "
                    f"trt='{trt_transcript[:80]}' ref='{ref_transcript[:80]}'",
        )

    @staticmethod
    def _check_timestamp_sanity(timestamps: list, tolerance_s: float) -> bool:
        """Check that timestamps are monotonically increasing within tolerance."""
        if len(timestamps) < 2:
            return True
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1] - tolerance_s:
                return False
        return True


plugin = SpeechToTextComparator()
