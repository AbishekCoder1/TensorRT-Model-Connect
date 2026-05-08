"""Contract test plugin for segmentation models (SegFormer, SAM)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ..contracts import MetricResult
from .base import make_pass, make_fail, make_error


def _compute_iou(pred, gt):
    """Compute mean IoU between prediction and ground truth class maps."""
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)
    classes = np.union1d(np.unique(pred), np.unique(gt))
    if len(classes) == 0:
        return 1.0
    ious = []
    for c in classes:
        p = (pred == c)
        g = (gt == c)
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious.append(float(intersection / union))
    return float(np.mean(ious)) if ious else 0.0


def _pixel_accuracy(pred, gt):
    """Fraction of pixels with matching class."""
    pred = np.asarray(pred, dtype=np.int32).flatten()
    gt = np.asarray(gt, dtype=np.int32).flatten()
    if len(pred) != len(gt):
        return 0.0
    return float((pred == gt).mean())


def _resolve_mask_list(data):
    masks = data.get("masks") or []
    if masks:
        return masks

    masks_path = data.get("masks_path")
    if not masks_path:
        return []

    path = Path(masks_path)
    if not path.is_file():
        return []

    loaded = np.load(path, allow_pickle=False)
    if loaded.ndim == 2:
        return [loaded]
    return [loaded[i] for i in range(loaded.shape[0])]


def _compute_binary_iou(pred, gt):
    """Compute IoU between two binary masks."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _verify_prompted_masks(trt_output, ref_output, threshold):
    trt_masks = _resolve_mask_list(trt_output.data)
    ref_masks = _resolve_mask_list(ref_output.data)

    if not trt_masks or not ref_masks:
        return make_error("full_inference", "Missing prompted segmentation masks")

    metrics = {
        "trt_num_masks": MetricResult(
            value=float(len(trt_masks)), threshold=None, operator=">=", passed=True),
        "ref_num_masks": MetricResult(
            value=float(len(ref_masks)), threshold=None, operator=">=", passed=True),
    }

    num_masks_threshold = threshold.metrics.get("num_masks_consistency")
    if num_masks_threshold is not None:
        same_count = len(trt_masks) == len(ref_masks)
        metrics["num_masks_consistency"] = MetricResult(
            value=1.0 if same_count else 0.0,
            threshold=1.0,
            operator="==",
            passed=same_count,
        )

    iou_values = []
    for i in range(min(len(trt_masks), len(ref_masks))):
        trt_mask = np.asarray(trt_masks[i], dtype=bool)
        ref_mask = np.asarray(ref_masks[i], dtype=bool)
        if trt_mask.shape != ref_mask.shape:
            try:
                from PIL import Image
                trt_img = Image.fromarray(trt_mask.astype(np.uint8) * 255)
                trt_img = trt_img.resize((ref_mask.shape[1], ref_mask.shape[0]), Image.NEAREST)
                trt_mask = np.asarray(trt_img, dtype=np.uint8).astype(bool)
            except ImportError:
                return make_error(
                    "full_inference",
                    f"Shape mismatch {trt_mask.shape} vs {ref_mask.shape} and PIL unavailable",
                )
        iou = _compute_binary_iou(trt_mask, ref_mask)
        iou_values.append(iou)
        metrics[f"mask_{i}_iou"] = MetricResult(
            value=iou, threshold=None, operator=">=", passed=True,
            note="per-mask informational",
        )

    if not iou_values:
        return make_error("full_inference", "No prompted segmentation masks were comparable")

    mean_iou = sum(iou_values) / len(iou_values)
    iou_threshold = threshold.metrics.get("iou_per_prompt", 0.5)
    metrics["iou_per_prompt"] = MetricResult(
        value=mean_iou,
        threshold=iou_threshold,
        operator=">=",
        passed=mean_iou >= iou_threshold,
    )

    rule = "mean prompted-mask IoU >= threshold"
    gated = [m for m in metrics.values() if m.threshold is not None]
    passed = all(m.passed for m in gated)
    if passed:
        return make_pass("full_inference", metrics, rule)
    return make_fail(
        "full_inference",
        metrics,
        rule,
        f"Prompted segmentation quality: mean_iou={mean_iou:.3f}",
    )


class SegmentationPlugin:
    reference_families = ["segmentation_segformer", "prompted_segmentation_sam"]
    user_contract = "segmentation_mask"

    def configure_reference(self, case):
        if case.reference_family == "prompted_segmentation_sam":
            return {"sam_mode": True}
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        if case.reference_family == "prompted_segmentation_sam":
            return _verify_prompted_masks(trt_output, ref_output, threshold)

        trt_mask = trt_output.data.get("class_map")
        if trt_mask is None:
            trt_mask = trt_output.data.get("mask")
        ref_mask = ref_output.data.get("class_map")
        if ref_mask is None:
            ref_mask = ref_output.data.get("mask")

        if trt_mask is None or ref_mask is None:
            return make_error("full_inference", "Missing mask/class_map in output data")

        trt_arr = np.asarray(trt_mask, dtype=np.int32)
        ref_arr = np.asarray(ref_mask, dtype=np.int32)

        # Resize if shapes differ
        if trt_arr.shape != ref_arr.shape:
            try:
                from PIL import Image
                ref_img = Image.fromarray(ref_arr.astype(np.uint8))
                ref_img = ref_img.resize((trt_arr.shape[1], trt_arr.shape[0]), Image.NEAREST)
                ref_arr = np.array(ref_img, dtype=np.int32)
            except ImportError:
                return make_error("full_inference", f"Shape mismatch {trt_arr.shape} vs {ref_arr.shape} and PIL unavailable")

        miou = _compute_iou(trt_arr, ref_arr)
        pixel_acc = _pixel_accuracy(trt_arr, ref_arr)

        miou_threshold = threshold.metrics.get("contract_miou_threshold", 0.5)
        pixel_threshold = threshold.metrics.get("contract_pixel_accuracy", 0.85)

        metrics = {
            "mIoU": MetricResult(value=miou, threshold=miou_threshold, operator=">=", passed=miou >= miou_threshold),
            "pixel_accuracy": MetricResult(value=pixel_acc, threshold=pixel_threshold, operator=">=", passed=pixel_acc >= pixel_threshold),
        }

        passed = miou >= miou_threshold and pixel_acc >= pixel_threshold
        rule = "mIoU >= threshold AND pixel_accuracy >= threshold"
        if passed:
            return make_pass("full_inference", metrics, rule)
        return make_fail("full_inference", metrics, rule,
                        f"Segmentation quality: mIoU={miou:.3f} pixel_acc={pixel_acc:.3f}")


plugin = SegmentationPlugin()
