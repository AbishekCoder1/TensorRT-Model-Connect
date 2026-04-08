"""Torch-TRT diffusion comparator.

Compares torch-trt engine outputs against HuggingFace diffusers reference:

Per-component stages (t5_encode, dit_step, vae_decode):
  - Cosine similarity between TRT and HF output tensors
  - Finite/non-zero sanity checks

End-to-end stage:
  - Pixel mean/std range (not black/white/flat)
  - PSNR and SSIM against HF reference frames
  - Thresholds are intentionally loose because the C++ pipeline uses a
    simplified DDIM scheduler (not identical to diffusers DPM-Solver++),
    so even with identical engines, final images diverge.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from ._helpers import cosine_similarity

logger = logging.getLogger(__name__)


def _compute_psnr(img1: Any, img2: Any) -> float:
    import numpy as np
    a = np.asarray(img1, dtype=np.float32)
    b = np.asarray(img2, dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def _compute_ssim(img1: Any, img2: Any) -> float:
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


class TorchTrtDiffusionComparator:
    """Compares torch-trt diffusion output against HF reference."""

    @property
    def task_strategy(self) -> str:
        return "torchtrt_diffusion"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        thresholds = threshold.metrics

        if stage.name in ("t5_encode", "dit_step", "vae_decode"):
            return self._compare_component(trt, ref, thresholds, stage.name)

        if stage.name == "end_to_end":
            return self._compare_frames(trt, ref, thresholds)

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.SKIPPED.value,
            metrics={},
            message=f"No comparison logic for stage: {stage.name}",
        )

    def _compare_component(
        self,
        trt: StageOutput,
        ref: StageOutput,
        thresholds: dict[str, float],
        stage_name: str,
    ) -> CompareResult:
        """Compare per-component outputs via cosine similarity."""
        import numpy as np

        metrics: dict[str, MetricResult] = {}

        # Check subprocess succeeded
        trt_rc = trt.data.get("returncode", -1)
        ref_rc = ref.data.get("returncode", -1)
        if trt_rc != 0:
            return CompareResult(
                stage_name=stage_name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"TRT subprocess failed (rc={trt_rc}): "
                        f"{trt.data.get('stderr', '')[:500]}",
            )
        if ref_rc != 0:
            return CompareResult(
                stage_name=stage_name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"HF reference subprocess failed (rc={ref_rc}): "
                        f"{ref.data.get('stderr', '')[:500]}",
            )

        trt_path = trt.data.get("output_path", "")
        ref_path = ref.data.get("output_path", "")
        if not trt_path or not ref_path:
            return CompareResult(
                stage_name=stage_name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Missing output paths: trt={trt_path!r} ref={ref_path!r}",
            )

        try:
            trt_arr = np.load(trt_path).astype(np.float64).flatten()
            ref_arr = np.load(ref_path).astype(np.float64).flatten()
        except Exception as e:
            return CompareResult(
                stage_name=stage_name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Failed to load outputs: {e}",
            )

        # Cosine similarity
        cs = cosine_similarity(trt_arr, ref_arr)

        # Stage-specific default thresholds
        default_thresholds = {
            "t5_encode": 0.95,
            "dit_step": 0.90,
            "vae_decode": 0.95,
        }
        thresh_key = f"{stage_name}_cosine"
        thresh = thresholds.get(thresh_key,
                    thresholds.get("component_cosine",
                        default_thresholds.get(stage_name, 0.90)))
        cs_ok = cs >= thresh
        metrics["cosine_similarity"] = MetricResult(
            value=cs, threshold=thresh, operator=">=", passed=cs_ok,
        )

        # Sanity: output is finite and non-zero
        trt_mean = float(np.mean(np.abs(trt_arr)))
        finite_ok = np.all(np.isfinite(trt_arr))
        nonzero_ok = trt_mean > 1e-8
        metrics["output_finite"] = MetricResult(
            value=1.0 if finite_ok else 0.0, threshold=1.0,
            operator=">=", passed=bool(finite_ok),
        )
        metrics["output_nonzero"] = MetricResult(
            value=trt_mean, threshold=1e-8,
            operator=">", passed=nonzero_ok,
        )

        all_pass = cs_ok and finite_ok and nonzero_ok
        return CompareResult(
            stage_name=stage_name,
            status=StageStatus.PASSED.value if all_pass else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="cosine_similarity >= threshold AND output is finite and non-zero",
            message=f"{stage_name}: cosine_sim={cs:.6f} "
                    f"(threshold={thresh})",
        )

    def _compare_frames(
        self,
        trt: StageOutput,
        ref: StageOutput,
        thresholds: dict[str, float],
    ) -> CompareResult:
        """Compare generated frames: pixel stats + PSNR/SSIM against HF."""
        metrics: dict[str, MetricResult] = {}
        all_pass = True

        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"TRT generation failed (rc={trt.data.get('returncode')})",
            )

        num_frames = trt.data.get("num_frames", 0)
        min_frames = 1
        frames_ok = num_frames >= min_frames
        metrics["num_frames"] = MetricResult(
            value=float(num_frames), threshold=float(min_frames),
            operator=">=", passed=frames_ok,
        )
        if not frames_ok:
            all_pass = False

        frame_stats = trt.data.get("frame_stats", {})
        if frame_stats:
            pixel_mean = frame_stats.get("mean", 0.5)
            pixel_std = frame_stats.get("std", 0.0)

            min_mean = thresholds.get("min_pixel_mean", 0.10)
            max_mean = thresholds.get("max_pixel_mean", 0.90)
            mean_ok = min_mean <= pixel_mean <= max_mean
            metrics["pixel_mean_range"] = MetricResult(
                value=pixel_mean, threshold=None, operator="in_range",
                passed=mean_ok, note=f"[{min_mean}, {max_mean}]",
            )
            if not mean_ok:
                all_pass = False

            min_std = thresholds.get("min_pixel_std", 0.05)
            std_ok = pixel_std >= min_std
            metrics["pixel_std_min"] = MetricResult(
                value=pixel_std, threshold=min_std,
                operator=">=", passed=std_ok,
            )
            if not std_ok:
                all_pass = False

        # Cross-compare frames against HF reference
        frames_dir = trt.data.get("frames_dir", "")
        ref_frames_dir = ref.data.get("frames_dir", "")
        if frames_dir and ref_frames_dir:
            psnr, ssim = self._cross_compare_frames(frames_dir, ref_frames_dir)
            if psnr is not None:
                # Low threshold: scheduler differences cause large divergence
                psnr_thresh = thresholds.get("psnr", 2.0)
                if psnr_thresh >= 0:
                    psnr_ok = psnr >= psnr_thresh
                    metrics["psnr"] = MetricResult(
                        value=psnr, threshold=psnr_thresh,
                        operator=">=", passed=psnr_ok,
                    )
                    if not psnr_ok:
                        all_pass = False

            if ssim is not None:
                ssim_thresh = thresholds.get("ssim", -1.0)
                if ssim_thresh >= 0:
                    ssim_ok = ssim >= ssim_thresh
                    metrics["ssim"] = MetricResult(
                        value=ssim, threshold=ssim_thresh,
                        operator=">=", passed=ssim_ok,
                    )
                    if not ssim_ok:
                        all_pass = False

        n_gated = sum(1 for m in metrics.values() if m.threshold is not None)
        n_passed = sum(1 for m in metrics.values()
                       if m.threshold is not None and m.passed)
        return CompareResult(
            stage_name="end_to_end",
            status=StageStatus.PASSED.value if all_pass else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all gated metrics must pass",
            message=f"{'PASS' if all_pass else 'FAIL'}: "
                    f"{n_passed}/{n_gated} metrics passed",
        )

    @staticmethod
    def _cross_compare_frames(
        trt_dir: str, ref_dir: str
    ) -> tuple[float | None, float | None]:
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
            t_img = np.array(Image.open(trt_frames[i]).convert("RGB"),
                             dtype=np.float32) / 255.0
            r_img = np.array(Image.open(ref_frames[i]).convert("RGB"),
                             dtype=np.float32) / 255.0
            psnr_vals.append(_compute_psnr(t_img, r_img))
            ssim_vals.append(_compute_ssim(t_img, r_img))

        return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


plugin = TorchTrtDiffusionComparator()
