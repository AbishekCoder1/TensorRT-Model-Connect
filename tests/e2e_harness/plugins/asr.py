"""Contract test plugin for ASR models (Whisper, Canary)."""
from __future__ import annotations
from ..contracts import (CompareResult, E2ECase, MetricResult, StageOutput, ThresholdProfile)
from .base import (normalize_text, levenshtein_ned, make_pass, make_fail, make_error)

def _word_error_rate(ref_words, hyp_words):
    """WER via Levenshtein on word sequences."""
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            ins = dp[j] + 1
            dele = dp[j - 1] + 1
            sub = prev + (0 if ref_words[i - 1] == hyp_words[j - 1] else 1)
            prev = dp[j]
            dp[j] = min(ins, dele, sub)
    return dp[m] / n

class ASRPlugin:
    reference_families = ["asr_whisper", "asr_canary"]
    user_contract = "exact_transcript"

    def configure_reference(self, case):
        if case.reference_family == "asr_canary":
            return {"auto_class": "AutoModelForSpeechSeq2Seq", "canary": True}
        return {"auto_class": "AutoModelForSpeechSeq2Seq"}

    def verify(self, trt_output, ref_output, case, threshold):
        trt_text = normalize_text(trt_output.data.get("transcript", trt_output.text or ""))
        ref_text = normalize_text(ref_output.data.get("transcript", ref_output.text or ""))

        if not ref_text:
            return make_error("full_generation", "Reference produced empty transcript")

        ned = levenshtein_ned(trt_text, ref_text)

        trt_words = trt_text.split()
        ref_words = ref_text.split()
        wer = _word_error_rate(ref_words, trt_words)

        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.1)
        wer_threshold = threshold.metrics.get("contract_wer_threshold", 0.1)

        metrics = {
            "ned": MetricResult(value=ned, threshold=ned_threshold, operator="<=", passed=ned <= ned_threshold),
            "wer": MetricResult(value=wer, threshold=wer_threshold, operator="<=", passed=wer <= wer_threshold),
        }

        passed = ned <= ned_threshold and wer <= wer_threshold
        rule = "ned <= threshold AND wer <= threshold"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail("full_generation", metrics, rule, f"Transcript diverged: WER={wer:.3f} NED={ned:.3f}")

plugin = ASRPlugin()
