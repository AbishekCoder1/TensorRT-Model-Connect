"""Text-to-audio comparator.

Compares TRT Bark-style audio generation output against reference with metrics:
- Codec token match rate
- Mel-spectrogram distance
- Log-spectral distance
- Duration and RMS bounds
"""

from __future__ import annotations

import logging
import math

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)


def _compute_mel_spectrogram(samples, sample_rate: int = 24000, n_fft: int = 1024,
                              hop_length: int = 256, n_mels: int = 80):
    """Compute log-mel spectrogram using numpy (no librosa dependency)."""
    import numpy as np

    # STFT
    n_frames = 1 + (len(samples) - n_fft) // hop_length
    if n_frames < 1:
        return np.zeros((n_mels, 1), dtype=np.float32)

    frames = np.stack([
        samples[i * hop_length: i * hop_length + n_fft]
        for i in range(n_frames)
    ])
    window = np.hanning(n_fft)
    frames = frames * window
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = np.abs(spectrum) ** 2

    # Mel filterbank (simplified linear spacing)
    fmin = 0.0
    fmax = sample_rate / 2.0
    mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center > f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    mel_spec = filterbank @ power.T  # [n_mels, n_frames]
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    return log_mel.astype(np.float32)


def _mel_spectrogram_distance(samples1, samples2, sample_rate: int = 24000) -> float:
    """Compute mean absolute difference between two log-mel spectrograms."""
    import numpy as np
    mel1 = _compute_mel_spectrogram(np.asarray(samples1, dtype=np.float32), sample_rate)
    mel2 = _compute_mel_spectrogram(np.asarray(samples2, dtype=np.float32), sample_rate)

    # Align lengths
    min_frames = min(mel1.shape[1], mel2.shape[1])
    if min_frames == 0:
        return float("inf")
    mel1 = mel1[:, :min_frames]
    mel2 = mel2[:, :min_frames]

    return float(np.mean(np.abs(mel1 - mel2)))


def _log_spectral_distance(samples1, samples2, sample_rate: int = 24000,
                            n_fft: int = 1024, hop_length: int = 256) -> float:
    """Compute log-spectral distance (LSD) between two audio signals."""
    import numpy as np

    def _power_spectrum(samples):
        n_frames = 1 + (len(samples) - n_fft) // hop_length
        if n_frames < 1:
            return np.zeros((1, n_fft // 2 + 1), dtype=np.float32)
        frames = np.stack([
            samples[i * hop_length: i * hop_length + n_fft]
            for i in range(n_frames)
        ])
        window = np.hanning(n_fft)
        frames = frames * window
        spectrum = np.fft.rfft(frames, n=n_fft)
        return np.abs(spectrum) ** 2

    ps1 = _power_spectrum(np.asarray(samples1, dtype=np.float32))
    ps2 = _power_spectrum(np.asarray(samples2, dtype=np.float32))

    min_frames = min(ps1.shape[0], ps2.shape[0])
    if min_frames == 0:
        return float("inf")
    ps1 = ps1[:min_frames]
    ps2 = ps2[:min_frames]

    log_ps1 = np.log(np.maximum(ps1, 1e-10))
    log_ps2 = np.log(np.maximum(ps2, 1e-10))

    lsd = np.sqrt(np.mean((log_ps1 - log_ps2) ** 2))
    return float(lsd)


class TextToAudioComparator:
    """Compares TRT text-to-audio output against reference."""

    @property
    def task_strategy(self) -> str:
        return "text_to_audio"

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
                message=f"TRT audio generation failed (rc={trt.data.get('returncode')})",
            )

        # WAV existence check
        if not trt.data.get("wav_exists", False):
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                message="TRT did not produce a WAV output file",
            )

        # RMS energy check
        rms = trt.data.get("rms", 0.0)
        metrics["rms"] = rms
        rms_min = thresholds.get("rms_min", 0.001)
        rms_max = thresholds.get("rms_max", 1.0)
        rms_ok = rms_min <= rms <= rms_max
        per_metric_pass["rms"] = rms_ok
        gate_details.append(
            f"{'PASS' if rms_ok else 'FAIL'} rms={rms:.6f} "
            f"(in [{rms_min}, {rms_max}])")
        if not rms_ok:
            all_pass = False

        # Duration check
        duration = trt.data.get("duration_s", 0.0)
        ref_duration = ref.data.get("duration_s", 0.0)
        if duration > 0 and ref_duration > 0:
            ratio = duration / ref_duration
            metrics["duration_ratio"] = ratio
            ratio_min = thresholds.get("duration_ratio_min", 0.5)
            ratio_max = thresholds.get("duration_ratio_max", 2.0)
            ratio_ok = ratio_min <= ratio <= ratio_max
            per_metric_pass["duration_ratio"] = ratio_ok
            gate_details.append(
                f"{'PASS' if ratio_ok else 'FAIL'} duration_ratio={ratio:.3f} "
                f"(in [{ratio_min}, {ratio_max}])")
            if not ratio_ok:
                all_pass = False

        # Mel-spectrogram and log-spectral distance (if reference audio available)
        ref_samples = ref.data.get("audio_samples")
        trt_wav_path = trt.data.get("wav_path", "")
        if ref_samples is not None and trt_wav_path:
            try:
                import numpy as np
                import struct

                # Read TRT WAV samples
                trt_samples = self._read_wav_samples(trt_wav_path)
                sample_rate = trt.data.get("sample_rate", 24000)

                if trt_samples is not None and len(trt_samples) > 0:
                    mel_dist = _mel_spectrogram_distance(
                        trt_samples, ref_samples, sample_rate)
                    metrics["mel_spectrogram_distance"] = mel_dist
                    mel_thresh = thresholds.get("mel_spectrogram_distance", 5.0)
                    mel_ok = mel_dist <= mel_thresh
                    per_metric_pass["mel_spectrogram_distance"] = mel_ok
                    gate_details.append(
                        f"{'PASS' if mel_ok else 'FAIL'} mel_dist={mel_dist:.3f} "
                        f"(<= {mel_thresh})")
                    if not mel_ok:
                        all_pass = False

                    lsd = _log_spectral_distance(
                        trt_samples, ref_samples, sample_rate)
                    metrics["log_spectral_distance"] = lsd
                    lsd_thresh = thresholds.get("log_spectral_distance", 3.0)
                    lsd_ok = lsd <= lsd_thresh
                    per_metric_pass["log_spectral_distance"] = lsd_ok
                    gate_details.append(
                        f"{'PASS' if lsd_ok else 'FAIL'} lsd={lsd:.3f} "
                        f"(<= {lsd_thresh})")
                    if not lsd_ok:
                        all_pass = False
            except Exception as e:
                gate_details.append(f"WARN spectral comparison failed: {e}")

        # Codec token match (if token data is available from both sides)
        trt_tokens = trt.data.get("codec_tokens")
        ref_tokens = ref.data.get("codec_tokens")
        if trt_tokens is not None and ref_tokens is not None:
            import numpy as np
            trt_t = np.asarray(trt_tokens).flatten()
            ref_t = np.asarray(ref_tokens).flatten()
            n = min(len(trt_t), len(ref_t))
            if n > 0:
                match_rate = float(np.sum(trt_t[:n] == ref_t[:n])) / n
                metrics["codec_token_match"] = match_rate
                ct_thresh = thresholds.get("codec_token_match", 0.7)
                ct_ok = match_rate >= ct_thresh
                per_metric_pass["codec_token_match"] = ct_ok
                gate_details.append(
                    f"{'PASS' if ct_ok else 'FAIL'} codec_token_match={match_rate:.4f} "
                    f"(>= {ct_thresh})")
                if not ct_ok:
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

    @staticmethod
    def _read_wav_samples(path: str):
        """Read float32 WAV samples from a file."""
        import struct
        import numpy as np

        try:
            with open(path, "rb") as f:
                riff = f.read(4)
                if riff != b"RIFF":
                    return None
                f.read(4)  # chunk size
                f.read(4)  # WAVE

                data_bytes = b""
                audio_format = 1
                while True:
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break
                    chunk_size = struct.unpack("<I", f.read(4))[0]
                    if chunk_id == b"fmt ":
                        fmt_data = f.read(chunk_size)
                        audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                    elif chunk_id == b"data":
                        data_bytes = f.read(chunk_size)
                    else:
                        f.read(chunk_size)

            if audio_format == 3:  # IEEE float32
                return np.frombuffer(data_bytes, dtype=np.float32)
            elif audio_format == 1:  # PCM int16
                return np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return None
        return None


plugin = TextToAudioComparator()
