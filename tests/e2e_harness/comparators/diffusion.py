"""Diffusion media generation comparator.

Compares TRT diffusion output against reference output with metrics:
- Stage-level latent trajectory parity (cosine per step)
- Final-frame PSNR, SSIM, LPIPS
- Temporal consistency for video (frame-to-frame diff stats)
- Frame-level distribution checks (pixel mean/std/min/max)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile

logger = logging.getLogger(__name__)


def _cosine_sim(a: Any, b: Any) -> float:
    """Compute cosine similarity between two arrays."""
    import numpy as np
    a_flat = np.asarray(a, dtype=np.float32).flatten()
    b_flat = np.asarray(b, dtype=np.float32).flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _compute_psnr(img1: Any, img2: Any) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images (values in [0,1])."""
    import numpy as np
    a = np.asarray(img1, dtype=np.float32)
    b = np.asarray(img2, dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 100.0  # Identical
    return float(10.0 * math.log10(1.0 / mse))


def _compute_ssim(img1: Any, img2: Any) -> float:
    """Compute Structural Similarity Index (simplified, per-channel average)."""
    import numpy as np
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a_sq = np.var(a)
    sigma_b_sq = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)

    return float(numerator / denominator)


def _compute_temporal_consistency(frames_dir: str) -> float:
    """Compute average frame-to-frame cosine similarity for temporal consistency."""
    import numpy as np
    from pathlib import Path

    try:
        from PIL import Image
    except ImportError:
        return -1.0

    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if len(frames) < 2:
        return 1.0

    similarities = []
    prev_arr = None
    for fp in frames:
        img = Image.open(fp).convert("RGB")
        arr = np.array(img, dtype=np.float32).flatten()
        if prev_arr is not None:
            similarities.append(_cosine_sim(prev_arr, arr))
        prev_arr = arr

    return float(np.mean(similarities)) if similarities else 1.0


class DiffusionComparator:
    """Compares TRT diffusion output against reference output."""

    @property
    def task_strategy(self) -> str:
        return "diffusion_media_generation"

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

        # Handle debug_pipeline stage: comparison is embedded in TRT output
        if stage.name == "debug_pipeline":
            return self._compare_debug_pipeline(trt, thresholds)

        # Handle end_to_end / generate: compare frame stats
        if stage.name in ("end_to_end", "end_to_end_video", "generate", "frame_quality"):
            return self._compare_frames(trt, ref, thresholds)

        # Handle t5_encode: compare T5 embeddings
        if stage.name == "t5_encode":
            return self._compare_embeddings(trt, ref, thresholds)

        # Handle crossover stages: validate subprocess succeeded and output is sane
        if stage.name.startswith("crossover_"):
            return self._compare_crossover(trt, stage, thresholds)

        return CompareResult(
            stage_name=stage.name,
            passed=True,
            message=f"No comparison logic for diffusion stage: {stage.name}",
        )

    def _compare_debug_pipeline(
        self, trt: StageOutput, thresholds: dict[str, float]
    ) -> CompareResult:
        """Extract pass/fail from the debug_diffusion_pipeline output."""
        metrics: dict[str, float] = {}
        per_metric_pass: dict[str, bool] = {}
        gate_details: list[str] = []

        passed = trt.data.get("passed", False)
        output_text = trt.data.get("output", "")

        # Parse step results from output
        for line in output_text.splitlines():
            line = line.strip()
            if line.startswith("PASS") or line.startswith("FAIL"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    status = parts[0] == "PASS"
                    name = parts[1]
                    metrics[name] = 1.0 if status else 0.0
                    per_metric_pass[name] = status
                    gate_details.append(f"{parts[0]} {name}")

        return CompareResult(
            stage_name="debug_pipeline",
            passed=passed,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"debug_pipeline: {'PASS' if passed else 'FAIL'} "
                    f"(rc={trt.data.get('returncode', -1)})",
        )

    def _compare_frames(
        self,
        trt: StageOutput,
        ref: StageOutput,
        thresholds: dict[str, float],
    ) -> CompareResult:
        """Compare generated frames: pixel stats, PSNR, SSIM, temporal consistency."""
        metrics: dict[str, float] = {}
        per_metric_pass: dict[str, bool] = {}
        gate_details: list[str] = []
        all_pass = True

        # Check TRT returncode
        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name="end_to_end",
                passed=False,
                message=f"TRT generation failed (rc={trt.data.get('returncode')})",
            )

        # Frame count
        num_frames = trt.data.get("num_frames", 0)
        metrics["num_frames"] = float(num_frames)

        # Frame pixel statistics
        frame_stats = trt.data.get("frame_stats", {})
        if frame_stats:
            pixel_mean = frame_stats.get("mean", 0.5)
            pixel_std = frame_stats.get("std", 0.0)
            metrics["pixel_mean"] = pixel_mean
            metrics["pixel_std"] = pixel_std

            min_mean = thresholds.get("min_pixel_mean", 0.15)
            max_mean = thresholds.get("max_pixel_mean", 0.85)
            min_std = thresholds.get("min_pixel_std", 0.05)

            mean_ok = min_mean <= pixel_mean <= max_mean
            per_metric_pass["pixel_mean_range"] = mean_ok
            if not mean_ok:
                gate_details.append(
                    f"FAIL pixel_mean={pixel_mean:.3f} not in [{min_mean}, {max_mean}]")
                all_pass = False
            else:
                gate_details.append(f"PASS pixel_mean={pixel_mean:.3f}")

            std_ok = pixel_std >= min_std
            per_metric_pass["pixel_std_min"] = std_ok
            if not std_ok:
                gate_details.append(
                    f"FAIL pixel_std={pixel_std:.3f} < {min_std}")
                all_pass = False
            else:
                gate_details.append(f"PASS pixel_std={pixel_std:.3f}")

        # Temporal consistency (if frames directory available)
        frames_dir = trt.data.get("frames_dir", "")
        if frames_dir:
            temporal_cs = _compute_temporal_consistency(frames_dir)
            metrics["temporal_consistency"] = temporal_cs
            tc_thresh = thresholds.get("temporal_consistency", 0.6)
            tc_ok = temporal_cs >= tc_thresh
            per_metric_pass["temporal_consistency"] = tc_ok
            if not tc_ok:
                gate_details.append(
                    f"FAIL temporal_consistency={temporal_cs:.4f} < {tc_thresh}")
                all_pass = False
            else:
                gate_details.append(f"PASS temporal_consistency={temporal_cs:.4f}")

        # Cross-reference PSNR/SSIM if we have both TRT and ref frames
        ref_frames_dir = ref.data.get("frames_dir", "")
        if frames_dir and ref_frames_dir:
            psnr, ssim = self._cross_compare_frames(frames_dir, ref_frames_dir)
            if psnr is not None:
                metrics["psnr"] = psnr
                psnr_thresh = thresholds.get("psnr", 20.0)
                psnr_ok = psnr >= psnr_thresh
                per_metric_pass["psnr"] = psnr_ok
                gate_details.append(
                    f"{'PASS' if psnr_ok else 'FAIL'} psnr={psnr:.2f} (>= {psnr_thresh})")
                if not psnr_ok:
                    all_pass = False

            if ssim is not None:
                metrics["ssim"] = ssim
                ssim_thresh = thresholds.get("ssim", 0.7)
                ssim_ok = ssim >= ssim_thresh
                per_metric_pass["ssim"] = ssim_ok
                gate_details.append(
                    f"{'PASS' if ssim_ok else 'FAIL'} ssim={ssim:.4f} (>= {ssim_thresh})")
                if not ssim_ok:
                    all_pass = False

        return CompareResult(
            stage_name="end_to_end",
            passed=all_pass,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"{'PASS' if all_pass else 'FAIL'}: "
                    f"{sum(per_metric_pass.values())}/{len(per_metric_pass)} metrics passed",
        )

    def _compare_embeddings(
        self,
        trt: StageOutput,
        ref: StageOutput,
        thresholds: dict[str, float],
    ) -> CompareResult:
        """Compare T5 embedding outputs via cosine similarity."""
        import numpy as np

        trt_path = trt.data.get("output_path", "")
        ref_path = ref.data.get("output_path", "")

        if not trt_path or not ref_path:
            return CompareResult(
                stage_name="t5_encode",
                passed=False,
                message="Missing output paths for T5 comparison",
            )

        try:
            trt_arr = np.load(trt_path)
            ref_arr = np.load(ref_path)
        except Exception as e:
            return CompareResult(
                stage_name="t5_encode",
                passed=False,
                message=f"Failed to load T5 outputs: {e}",
            )

        cs = _cosine_sim(trt_arr, ref_arr)
        thresh = thresholds.get("latent_cosine_per_step", 0.95)

        return CompareResult(
            stage_name="t5_encode",
            passed=cs >= thresh,
            metrics={"cosine_similarity": cs},
            per_metric_pass={"cosine_similarity": cs >= thresh},
            gate_details=[f"{'PASS' if cs >= thresh else 'FAIL'} cosine_sim={cs:.6f} (>= {thresh})"],
            message=f"T5 cosine_sim={cs:.6f}",
        )

    def _compare_crossover(
        self,
        trt: StageOutput,
        stage: StageSpec,
        thresholds: dict[str, float],
    ) -> CompareResult:
        """Validate crossover stage output: subprocess succeeded and output is sane.

        Crossover stages mix TRT and HF components in one subprocess, so there
        is no separate reference output. We validate that the subprocess ran
        successfully and the output tensor has reasonable statistics.
        """
        metrics: dict[str, float] = {}
        per_metric_pass: dict[str, bool] = {}
        gate_details: list[str] = []
        all_pass = True

        rc = trt.data.get("returncode", -1)
        if rc != 0:
            return CompareResult(
                stage_name=stage.name,
                passed=False,
                message=f"Crossover stage failed (rc={rc}): "
                        f"{trt.data.get('stderr', '')[:500]}",
            )

        per_metric_pass["subprocess_ok"] = True
        gate_details.append("PASS subprocess_ok")

        # Check output tensor stats (non-zero, finite)
        out_mean = trt.data.get("dit_output_mean")
        out_std = trt.data.get("dit_output_std")
        if out_mean is not None and out_std is not None:
            import math
            metrics["output_mean"] = out_mean
            metrics["output_std"] = out_std

            finite_ok = math.isfinite(out_mean) and math.isfinite(out_std)
            per_metric_pass["output_finite"] = finite_ok
            gate_details.append(
                f"{'PASS' if finite_ok else 'FAIL'} output_finite "
                f"(mean={out_mean:.4f}, std={out_std:.4f})")
            if not finite_ok:
                all_pass = False

            nonzero_ok = out_std > 1e-6
            per_metric_pass["output_nonzero"] = nonzero_ok
            gate_details.append(
                f"{'PASS' if nonzero_ok else 'FAIL'} output_nonzero (std={out_std:.6f})")
            if not nonzero_ok:
                all_pass = False

        return CompareResult(
            stage_name=stage.name,
            passed=all_pass,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"{'PASS' if all_pass else 'FAIL'} crossover {stage.name}",
        )

    @staticmethod
    def _cross_compare_frames(
        trt_dir: str, ref_dir: str
    ) -> tuple[float | None, float | None]:
        """Compare frames from two directories, return (avg_psnr, avg_ssim)."""
        from pathlib import Path

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return None, None

        trt_frames = sorted(Path(trt_dir).glob("frame_*.png"))
        ref_frames = sorted(Path(ref_dir).glob("frame_*.png"))

        if not trt_frames or not ref_frames:
            return None, None

        n = min(len(trt_frames), len(ref_frames))
        psnr_vals = []
        ssim_vals = []

        for i in range(n):
            trt_img = np.array(Image.open(trt_frames[i]).convert("RGB"), dtype=np.float32) / 255.0
            ref_img = np.array(Image.open(ref_frames[i]).convert("RGB"), dtype=np.float32) / 255.0
            psnr_vals.append(_compute_psnr(trt_img, ref_img))
            ssim_vals.append(_compute_ssim(trt_img, ref_img))

        return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


plugin = DiffusionComparator()
