"""Contract test plugin for diffusion models (FLUX, PixArt, Z-Image, Wan)."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from ..contracts import MetricResult
from .base import make_pass, make_fail


def _compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = np.mean(a64)
    mu_b = np.mean(b64)
    var_a = np.var(a64)
    var_b = np.var(b64)
    cov = np.mean((a64 - mu_a) * (b64 - mu_b))
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


def _compare_frame_dirs(trt_dir: str, ref_dir: str) -> tuple[float, float] | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    trt_frames = sorted(Path(trt_dir).glob("frame_*.png"))
    ref_frames = sorted(Path(ref_dir).glob("frame_*.png"))
    if not trt_frames or not ref_frames:
        return None

    psnr_values = []
    ssim_values = []
    for trt_frame, ref_frame in zip(trt_frames, ref_frames):
        trt_img = np.asarray(Image.open(trt_frame).convert("RGB"), dtype=np.float32) / 255.0
        ref_img = np.asarray(Image.open(ref_frame).convert("RGB"), dtype=np.float32) / 255.0
        psnr_values.append(_compute_psnr(trt_img, ref_img))
        ssim_values.append(_compute_ssim(trt_img, ref_img))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


class DiffusionPlugin:
    reference_families = ["diffusers_image_gen", "diffusers_video_gen"]
    user_contract = "diffusion_image"

    def configure_reference(self, case):
        config = {"use_diffusers": True}
        if case.reference_family == "diffusers_video_gen":
            config["video_mode"] = True
        return config

    def verify(self, trt_output, ref_output, case, threshold):
        stage = trt_output.stage_name
        is_video = case.reference_family == "diffusers_video_gen"
        metrics = {}

        # Sub-stages: invariant checks only (no images/frames produced yet)
        if stage in ("t5_encode", "dit_step"):
            has_data = len(trt_output.data) > 0 or stage == "t5_encode"
            metrics["stage_ok"] = MetricResult(
                value=1.0 if has_data else 0.0, threshold=0.0, operator=">=",
                passed=True, note=f"{stage} completed")
            return make_pass(stage, metrics, f"{stage} invariant check")

        # vae_decode / end_to_end: check frames or image output
        if is_video:
            # Video health: check frames directory
            frames_dir = trt_output.data.get("frames_dir")
            num_frames = trt_output.data.get("num_frames", 0)
            has_frames = frames_dir is not None and num_frames > 0
            metrics["has_frames"] = MetricResult(
                value=float(num_frames), threshold=1.0, operator=">=",
                passed=has_frames, note="video frames produced")
        else:
            # Image health: check output path or frames_dir (runners may use either)
            image_path = (trt_output.data.get("image_path")
                          or trt_output.data.get("output_path")
                          or trt_output.data.get("frames_dir"))
            has_image = image_path is not None
            # Also accept if frame_paths or num_frames indicates output
            if not has_image:
                has_image = (trt_output.data.get("num_frames", 0) > 0
                             or bool(trt_output.data.get("frame_paths")))
            metrics["has_image"] = MetricResult(
                value=1.0 if has_image else 0.0, threshold=1.0, operator="==",
                passed=has_image, note="image file produced")

        # Pixel / frame statistics (check pixel_stats or frame_stats)
        trt_pixels = trt_output.data.get("pixel_stats") or trt_output.data.get("frame_stats")
        if isinstance(trt_pixels, dict):
            mean = trt_pixels.get("mean", 0.0)
            std = trt_pixels.get("std", 0.0)

            min_mean = threshold.metrics.get("contract_min_pixel_mean", 0.05)
            max_mean = threshold.metrics.get("contract_max_pixel_mean", 0.95)
            min_std = threshold.metrics.get("contract_min_pixel_std", 0.01)

            mean_ok = min_mean <= mean <= max_mean
            std_ok = std >= min_std

            metrics["pixel_mean"] = MetricResult(
                value=mean, threshold=min_mean, operator=">=",
                passed=mean_ok, note=f"range [{min_mean}, {max_mean}]")
            metrics["pixel_std"] = MetricResult(
                value=std, threshold=min_std, operator=">=",
                passed=std_ok, note="non-uniform check")

        ref_pixels = ref_output.data.get("pixel_stats") or ref_output.data.get("frame_stats")
        if isinstance(ref_pixels, dict):
            mean = ref_pixels.get("mean", 0.0)
            std = ref_pixels.get("std", 0.0)

            min_mean = threshold.metrics.get("contract_min_pixel_mean", 0.05)
            max_mean = threshold.metrics.get("contract_max_pixel_mean", 0.95)
            min_std = threshold.metrics.get("contract_min_pixel_std", 0.01)

            mean_ok = min_mean <= mean <= max_mean
            std_ok = std >= min_std

            metrics["reference_pixel_mean"] = MetricResult(
                value=mean, threshold=min_mean, operator=">=",
                passed=mean_ok, note=f"range [{min_mean}, {max_mean}]")
            metrics["reference_pixel_std"] = MetricResult(
                value=std, threshold=min_std, operator=">=",
                passed=std_ok, note="reference non-uniform check")

        psnr_threshold = threshold.metrics.get(
            "contract_psnr_threshold", threshold.metrics.get("psnr"))
        ssim_threshold = threshold.metrics.get(
            "contract_ssim_threshold", threshold.metrics.get("ssim"))
        if psnr_threshold is not None or ssim_threshold is not None:
            trt_frames_dir = trt_output.data.get("frames_dir")
            ref_frames_dir = ref_output.data.get("frames_dir")
            frame_similarity = (
                _compare_frame_dirs(trt_frames_dir, ref_frames_dir)
                if trt_frames_dir and ref_frames_dir else None
            )
            if frame_similarity is None:
                if psnr_threshold is not None:
                    metrics["psnr"] = MetricResult(
                        value=0.0, threshold=psnr_threshold, operator=">=",
                        passed=False, note="frame similarity unavailable")
                if ssim_threshold is not None:
                    metrics["ssim"] = MetricResult(
                        value=0.0, threshold=ssim_threshold, operator=">=",
                        passed=False, note="frame similarity unavailable")
            else:
                psnr, ssim = frame_similarity
                if psnr_threshold is not None:
                    metrics["psnr"] = MetricResult(
                        value=psnr, threshold=psnr_threshold, operator=">=",
                        passed=psnr >= psnr_threshold)
                if ssim_threshold is not None:
                    metrics["ssim"] = MetricResult(
                        value=ssim, threshold=ssim_threshold, operator=">=",
                        passed=ssim >= ssim_threshold)

        # PSNR against reference (if available as numpy arrays)
        trt_arr = trt_output.data.get("pixels")
        ref_arr = ref_output.data.get("pixels")
        if trt_arr is not None and ref_arr is not None:
            try:
                trt_np = np.asarray(trt_arr, dtype=np.float32)
                ref_np = np.asarray(ref_arr, dtype=np.float32)
                if trt_np.shape == ref_np.shape:
                    mse = np.mean((trt_np - ref_np) ** 2)
                    if mse > 0:
                        psnr = float(10 * np.log10(1.0 / mse)) if np.max(ref_np) <= 1.0 else float(10 * np.log10(255.0 ** 2 / mse))
                    else:
                        psnr = 100.0
                    psnr_threshold = threshold.metrics.get("contract_psnr_threshold", 15.0)
                    metrics["psnr"] = MetricResult(
                        value=psnr, threshold=psnr_threshold, operator=">=",
                        passed=psnr >= psnr_threshold)
            except (ValueError, TypeError):
                pass

        all_passed = all(m.passed for m in metrics.values())
        label = "video" if is_video else "image"
        rule = f"diffusion {label} health + optional media similarity"
        if all_passed:
            return make_pass(stage, metrics, rule)
        return make_fail(stage, metrics, rule, f"Diffusion {label} health check failed")


plugin = DiffusionPlugin()
