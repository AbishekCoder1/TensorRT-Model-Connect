"""Contract test plugin for diffusion models (FLUX, PixArt, Z-Image, Wan)."""
from __future__ import annotations
import numpy as np
from ..contracts import (CompareResult, E2ECase, MetricResult, StageOutput, ThresholdProfile)
from .base import make_pass, make_fail, make_error


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
