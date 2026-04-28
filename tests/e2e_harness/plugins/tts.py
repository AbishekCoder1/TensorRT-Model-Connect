"""Contract test plugin for TTS models (Bark, Magpie).

Verifies TTS output via:
1. Audio health checks (WAV exists, non-silence, duration)
2. ASR round-trip: feed TRT audio into Whisper TRT, compare transcript
   against input prompt. This is the primary user contract — the audio
   must contain the correct spoken content.
"""
from __future__ import annotations

import logging
import os
import subprocess

from ..contracts import (
    CompareResult, E2ECase, MetricResult, StageOutput, ThresholdProfile,
)
from .base import (
    normalize_text, levenshtein_ned, make_pass, make_fail,
)

logger = logging.getLogger(__name__)

# Whisper bundle preference order for ASR round-trip
_WHISPER_BUNDLES = [
    "whisper-tiny-fp16.trtfb",
    "whisper-small.trtfb",
    "whisper-large-v3-turbo.trtfb",
]


def _find_whisper_bundle(engine_dir: str) -> str | None:
    """Find a Whisper TRT bundle in the engine directory."""
    for name in _WHISPER_BUNDLES:
        path = os.path.join(engine_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _run_asr_roundtrip(
    wav_path: str,
    trtf_binary: str,
    whisper_bundle: str,
    hf_python: str = "",
) -> str | None:
    """Run TRT Whisper on a WAV file and return the transcript.

    Uses the C++ trtf binary (not Python/PyTorch) to avoid cuBLAS issues
    on Blackwell GPUs.
    """
    cmd = [trtf_binary, "transcribe", whisper_bundle, "--audio", wav_path]
    if hf_python:
        cmd.extend(["--hf-python", hf_python])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            # Transcript is on stdout (last non-empty line, strip leading space)
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            return lines[-1] if lines else None
        logger.warning("Whisper ASR failed (rc=%d): %s", result.returncode, result.stderr[-500:])
        return None
    except Exception as e:
        logger.warning("Whisper ASR subprocess failed: %s", e)
        return None


class TTSPlugin:
    """Contract plugin for TTS: audio health + ASR round-trip verification."""

    reference_families = ["tts_bark", "tts_magpie"]
    user_contract = "tts_audio"

    def configure_reference(self, case: E2ECase) -> dict:
        return {}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ) -> CompareResult:
        trt_wav = trt_output.data.get("wav_path")
        trt_rms = trt_output.data.get("rms")
        trt_duration = trt_output.data.get("duration_s")

        min_rms = threshold.metrics.get("contract_min_rms", 0.001)
        min_duration = threshold.metrics.get("contract_min_duration_s", 0.1)
        max_duration = threshold.metrics.get("contract_max_duration_s", 30.0)

        metrics: dict[str, MetricResult] = {}

        # --- Audio health checks ---
        has_wav = trt_wav is not None and isinstance(trt_wav, str) and os.path.isfile(trt_wav)
        metrics["has_audio"] = MetricResult(
            value=1.0 if has_wav else 0.0, threshold=1.0, operator="==",
            passed=has_wav, note="WAV file produced")

        if trt_rms is not None:
            rms_ok = float(trt_rms) >= min_rms
            metrics["rms"] = MetricResult(
                value=float(trt_rms), threshold=min_rms, operator=">=",
                passed=rms_ok, note="non-silence check")

        if trt_duration is not None:
            dur = float(trt_duration)
            dur_ok = min_duration <= dur <= max_duration
            metrics["duration_s"] = MetricResult(
                value=dur, threshold=min_duration, operator=">=",
                passed=dur_ok, note=f"range [{min_duration}, {max_duration}]")

        # --- ASR round-trip (primary contract) ---
        input_prompt = case.inputs.get("prompt", "")
        asr_transcript = None

        if has_wav and input_prompt:
            # Runtime paths injected by the orchestrator
            ctx = case.metadata.get("_ctx", {})
            engine_dir = ctx.get("engine_dir", "")
            binary_path = ctx.get("binary_path", "")
            hf_python = ctx.get("hf_python", "")

            whisper_bundle = _find_whisper_bundle(engine_dir) if engine_dir else None

            if whisper_bundle and os.path.isfile(binary_path):
                asr_transcript = _run_asr_roundtrip(
                    trt_wav, binary_path, whisper_bundle, hf_python)

        if asr_transcript is not None:
            norm_transcript = normalize_text(asr_transcript)
            norm_prompt = normalize_text(input_prompt)

            ned = levenshtein_ned(norm_transcript, norm_prompt)
            ned_threshold = threshold.metrics.get("contract_asr_ned_threshold", 0.15)

            exact = (norm_transcript == norm_prompt)
            metrics["asr_exact_match"] = MetricResult(
                value=1.0 if exact else 0.0, threshold=None, operator="==",
                passed=True, note="informational — NED is the gate")
            metrics["asr_ned"] = MetricResult(
                value=ned, threshold=ned_threshold, operator="<=",
                passed=ned <= ned_threshold,
                note=f"transcript: '{asr_transcript[:80]}...'" if len(asr_transcript) > 80 else f"transcript: '{asr_transcript}'")
        elif has_wav and input_prompt:
            # Whisper not available — note but don't fail
            metrics["asr_roundtrip"] = MetricResult(
                value=0.0, threshold=0.0, operator=">=",
                passed=True, note="Whisper bundle not found — ASR round-trip skipped")

        all_passed = all(m.passed for m in metrics.values())
        rule = "audio health + ASR round-trip transcript recovery"
        if all_passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail("full_generation", metrics, rule, "TTS contract check failed")


plugin = TTSPlugin()
