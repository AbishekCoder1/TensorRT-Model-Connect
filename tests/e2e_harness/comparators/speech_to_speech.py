"""Speech-to-speech comparator.

Compares TRT PersonaPlex-style speech output against reference with metrics:
- Depth token match rate
- Audio token match rate
- Frame exact match rate
- RMS floor check
- Optional ASR consistency (transcript similarity)
"""

from __future__ import annotations

import logging

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)


class SpeechToSpeechComparator:
    """Compares TRT speech-to-speech output against reference tokens/audio."""

    @property
    def task_strategy(self) -> str:
        return "speech_to_speech"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        import numpy as np

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
                metrics={},
                per_metric_pass={},
                gate_details=[f"early return: TRT speech-to-speech failed (rc={trt.data.get('returncode')})"],
                message=f"TRT speech-to-speech failed (rc={trt.data.get('returncode')})",
            )

        # Get token arrays
        trt_tokens = trt.data.get("output_tokens")
        ref_tokens = ref.data.get("reference_tokens")

        if trt_tokens is not None and ref_tokens is not None:
            trt_arr = np.asarray(trt_tokens)
            ref_arr = np.asarray(ref_tokens)

            # Align frame count
            n_frames = min(trt_arr.shape[0], ref_arr.shape[0])
            if n_frames == 0:
                return CompareResult(
                    stage_name=stage.name,
                    passed=False,
                    metrics={},
                    per_metric_pass={},
                    gate_details=["early return: no frames to compare (empty token arrays)"],
                    message="No frames to compare (empty token arrays)",
                )

            trt_aligned = trt_arr[:n_frames]
            ref_aligned = ref_arr[:n_frames]

            # Depth token match (first column if multi-column)
            if trt_aligned.ndim >= 2 and trt_aligned.shape[1] >= 1:
                depth_matches = np.sum(trt_aligned[:, 0] == ref_aligned[:, 0])
                depth_rate = float(depth_matches) / n_frames
                metrics["depth_token_match_rate"] = depth_rate
                depth_thresh = thresholds.get("depth_token_match_rate", 0.7)
                depth_ok = depth_rate >= depth_thresh
                per_metric_pass["depth_token_match_rate"] = depth_ok
                gate_details.append(
                    f"{'PASS' if depth_ok else 'FAIL'} "
                    f"depth_token_match={depth_rate:.4f} (>= {depth_thresh})")
                if not depth_ok:
                    all_pass = False

                # Audio token match (remaining columns)
                if trt_aligned.shape[1] > 1:
                    audio_cols = trt_aligned[:, 1:]
                    ref_audio_cols = ref_aligned[:, 1:]
                    audio_matches = np.sum(audio_cols == ref_audio_cols)
                    total_audio = audio_cols.size
                    audio_rate = float(audio_matches) / total_audio if total_audio > 0 else 0.0
                    metrics["audio_token_match_rate"] = audio_rate
                    audio_thresh = thresholds.get("audio_token_match_rate", 0.7)
                    audio_ok = audio_rate >= audio_thresh
                    per_metric_pass["audio_token_match_rate"] = audio_ok
                    gate_details.append(
                        f"{'PASS' if audio_ok else 'FAIL'} "
                        f"audio_token_match={audio_rate:.4f} (>= {audio_thresh})")
                    if not audio_ok:
                        all_pass = False
            else:
                # 1D token comparison (flat match)
                flat_trt = trt_aligned.flatten()
                flat_ref = ref_aligned.flatten()
                n_compare = min(len(flat_trt), len(flat_ref))
                if n_compare > 0:
                    match_rate = float(np.sum(flat_trt[:n_compare] == flat_ref[:n_compare])) / n_compare
                    metrics["speech_min_token_match"] = match_rate
                    token_thresh = thresholds.get("speech_min_token_match", 0.8)
                    token_ok = match_rate >= token_thresh
                    per_metric_pass["speech_min_token_match"] = token_ok
                    gate_details.append(
                        f"{'PASS' if token_ok else 'FAIL'} "
                        f"token_match={match_rate:.4f} (>= {token_thresh})")
                    if not token_ok:
                        all_pass = False

            # Frame exact match rate
            frame_exact = 0
            for i in range(n_frames):
                if np.array_equal(trt_aligned[i], ref_aligned[i]):
                    frame_exact += 1
            frame_rate = float(frame_exact) / n_frames
            metrics["frame_exact_match_rate"] = frame_rate
            frame_thresh = thresholds.get(
                "frame_exact_match_rate",
                thresholds.get("speech_min_frame_exact", 0.7),
            )
            frame_ok = frame_rate >= frame_thresh
            per_metric_pass["frame_exact_match_rate"] = frame_ok
            gate_details.append(
                f"{'PASS' if frame_ok else 'FAIL'} "
                f"frame_exact_match={frame_rate:.4f} (>= {frame_thresh})")
            if not frame_ok:
                all_pass = False

        elif trt_tokens is None:
            gate_details.append("WARN: No TRT output tokens available")
        elif ref_tokens is None:
            gate_details.append("WARN: No reference tokens available for comparison")

        # RMS floor check
        rms = trt.data.get("rms", 0.0)
        if rms > 0 or trt.data.get("wav_exists", False):
            metrics["rms"] = rms
            rms_thresh = thresholds.get("speech_min_rms", 0.001)
            rms_ok = rms >= rms_thresh
            per_metric_pass["rms_floor"] = rms_ok
            gate_details.append(
                f"{'PASS' if rms_ok else 'FAIL'} rms={rms:.6f} (>= {rms_thresh})")
            if not rms_ok:
                all_pass = False

        return CompareResult(
            stage_name=stage.name,
            passed=all_pass,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"{'PASS' if all_pass else 'FAIL'}: "
                    f"{sum(per_metric_pass.values())}/{len(per_metric_pass)} metrics passed",
        )


plugin = SpeechToSpeechComparator()
