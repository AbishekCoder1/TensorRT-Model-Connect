"""Segmentation, prompted segmentation, and object detection comparators.

SegmentationComparator:
  Metrics: mIoU, pixel_accuracy, boundary_f_score
  Canonicalization: same class ID mapping, deterministic resize

PromptedSegmentationComparator:
  Metrics: iou_per_prompt, mask_rank_consistency, num_masks_consistency

ObjectDetectionComparator:
  Metrics: mAP@IoU thresholds, class_precision, class_recall, box_iou_distribution
  Canonicalization: deterministic NMS, stable sort

All three comparators are auto-discovered: the primary ``plugin`` is
SegmentationComparator. PromptedSegmentationComparator and
ObjectDetectionComparator are registered at import time via direct
registry calls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..contracts import (
    CompareResult,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared numerical helpers
# ---------------------------------------------------------------------------

def _safe_import_numpy():
    import numpy as np
    return np


def _compute_iou(mask_a, mask_b) -> float:
    """Compute IoU between two binary masks (numpy arrays)."""
    np = _safe_import_numpy()
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _compute_pixel_accuracy(pred, gt) -> float:
    """Pixel-wise accuracy between predicted and ground-truth class maps."""
    np = _safe_import_numpy()
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)
    if pred.shape != gt.shape:
        # Resize pred to gt shape via nearest-neighbor
        try:
            from PIL import Image
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_pil.resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.array(pred_resized).astype(np.int32)
        except ImportError:
            logger.warning("PIL not available for resize; shape mismatch")
            return 0.0
    return float((pred == gt).mean())


def _compute_miou(pred, gt, num_classes: int | None = None) -> float:
    """Mean Intersection-over-Union across all present classes."""
    np = _safe_import_numpy()
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)

    if pred.shape != gt.shape:
        try:
            from PIL import Image
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_pil.resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.array(pred_resized).astype(np.int32)
        except ImportError:
            return 0.0

    if num_classes is None:
        all_ids = np.union1d(np.unique(pred), np.unique(gt))
    else:
        all_ids = np.arange(num_classes)

    iou_sum = 0.0
    count = 0
    for cls in all_ids:
        pred_mask = pred == cls
        gt_mask = gt == cls
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            continue
        iou_sum += intersection / union
        count += 1

    return float(iou_sum / count) if count > 0 else 0.0


def _compute_boundary_f_score(pred, gt, tolerance: int = 2) -> float:
    """Boundary F-score: precision/recall of predicted boundary pixels.

    Uses morphological gradient to extract boundaries, then computes
    F1 between predicted and ground-truth boundary pixels within a
    tolerance band.
    """
    np = _safe_import_numpy()
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)

    if pred.shape != gt.shape:
        try:
            from PIL import Image
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_pil.resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.array(pred_resized).astype(np.int32)
        except ImportError:
            return 0.0

    # Extract boundaries via simple gradient (pixel differs from neighbor)
    def _boundary(seg):
        b = np.zeros_like(seg, dtype=bool)
        b[:-1, :] |= seg[:-1, :] != seg[1:, :]
        b[1:, :] |= seg[:-1, :] != seg[1:, :]
        b[:, :-1] |= seg[:, :-1] != seg[:, 1:]
        b[:, 1:] |= seg[:, :-1] != seg[:, 1:]
        return b

    pred_boundary = _boundary(pred)
    gt_boundary = _boundary(gt)

    if not gt_boundary.any() and not pred_boundary.any():
        return 1.0
    if not gt_boundary.any() or not pred_boundary.any():
        return 0.0

    # Dilate ground-truth boundary for tolerance
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_boundary, structure=struct)
    pred_dilated = binary_dilation(pred_boundary, structure=struct)

    precision = float(pred_boundary[gt_dilated].sum() / pred_boundary.sum())
    recall = float(gt_boundary[pred_dilated].sum() / gt_boundary.sum())

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _compute_box_iou(box_a: list, box_b: list) -> float:
    """IoU between two axis-aligned boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# SegmentationComparator
# ---------------------------------------------------------------------------


class SegmentationComparator:
    """Compare TRT vs reference semantic segmentation outputs."""

    @property
    def task_strategy(self) -> str:
        return "segmentation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        np = _safe_import_numpy()

        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: list[str] = []

        trt_map = trt.data.get("class_map")
        ref_map = ref.data.get("class_map")

        if trt_map is None:
            gate_details.append("TRT class_map is None")
            return CompareResult(
                stage_name=stage.name, passed=False,
                metrics=metrics, per_metric_pass=per_metric_pass,
                gate_details=gate_details,
                message="No TRT segmentation output",
            )
        if ref_map is None:
            gate_details.append("Reference class_map is None")
            return CompareResult(
                stage_name=stage.name, passed=False,
                metrics=metrics, per_metric_pass=per_metric_pass,
                gate_details=gate_details,
                message="No reference segmentation output",
            )

        trt_map = np.asarray(trt_map, dtype=np.int32)
        ref_map = np.asarray(ref_map, dtype=np.int32)

        # Pixel accuracy
        pixel_acc = _compute_pixel_accuracy(trt_map, ref_map)
        metrics["pixel_accuracy"] = pixel_acc

        pixel_acc_thresh = threshold.metrics.get("pixel_accuracy", 0.85)
        passed_pa = pixel_acc >= pixel_acc_thresh
        per_metric_pass["pixel_accuracy"] = passed_pa
        gate_details.append(
            f"pixel_accuracy: {pixel_acc:.4f} >= {pixel_acc_thresh} "
            f"-> {'PASS' if passed_pa else 'FAIL'}"
        )

        # mIoU
        miou = _compute_miou(trt_map, ref_map)
        metrics["mIoU"] = miou

        miou_thresh = threshold.metrics.get("mIoU", 0.5)
        passed_miou = miou >= miou_thresh
        per_metric_pass["mIoU"] = passed_miou
        gate_details.append(
            f"mIoU: {miou:.4f} >= {miou_thresh} "
            f"-> {'PASS' if passed_miou else 'FAIL'}"
        )

        # Boundary F-score (optional, graceful fallback if scipy unavailable)
        try:
            bf = _compute_boundary_f_score(trt_map, ref_map)
            metrics["boundary_f_score"] = bf

            bf_thresh = threshold.metrics.get("boundary_f_score")
            if bf_thresh is not None:
                passed_bf = bf >= bf_thresh
                per_metric_pass["boundary_f_score"] = passed_bf
                gate_details.append(
                    f"boundary_f_score: {bf:.4f} >= {bf_thresh} "
                    f"-> {'PASS' if passed_bf else 'FAIL'}"
                )
        except ImportError:
            gate_details.append("scipy not available; skipping boundary_f_score")

        # Class distribution summary
        trt_classes = int(len(np.unique(trt_map)))
        ref_classes = int(len(np.unique(ref_map)))
        metrics["trt_num_classes"] = float(trt_classes)
        metrics["ref_num_classes"] = float(ref_classes)
        gate_details.append(
            f"Unique classes: TRT={trt_classes}, Ref={ref_classes}"
        )

        overall = all(per_metric_pass.values())
        return CompareResult(
            stage_name=stage.name,
            passed=overall,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"Segmentation: {'PASS' if overall else 'FAIL'} "
                    f"(pixel_acc={pixel_acc:.4f}, mIoU={miou:.4f})",
        )


# ---------------------------------------------------------------------------
# PromptedSegmentationComparator
# ---------------------------------------------------------------------------


class PromptedSegmentationComparator:
    """Compare TRT vs reference prompted segmentation (e.g. SAM) outputs."""

    @property
    def task_strategy(self) -> str:
        return "prompted_segmentation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        np = _safe_import_numpy()

        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: list[str] = []

        trt_masks = trt.data.get("masks", [])
        ref_masks = ref.data.get("masks", [])
        trt_scores = trt.data.get("mask_scores", [])
        ref_scores = ref.data.get("mask_scores", [])

        # Number of masks consistency
        metrics["trt_num_masks"] = float(len(trt_masks))
        metrics["ref_num_masks"] = float(len(ref_masks))

        num_expected = trt.data.get("num_expected_masks") or ref.data.get("num_expected_masks")
        if num_expected is not None:
            metrics["expected_num_masks"] = float(num_expected)

        num_masks_thresh = threshold.metrics.get("num_masks_consistency")
        if num_masks_thresh is not None:
            passed_nm = len(trt_masks) == len(ref_masks)
            per_metric_pass["num_masks_consistency"] = passed_nm
            gate_details.append(
                f"num_masks: TRT={len(trt_masks)} vs Ref={len(ref_masks)} "
                f"-> {'MATCH' if passed_nm else 'MISMATCH'}"
            )
        else:
            gate_details.append(
                f"num_masks: TRT={len(trt_masks)}, Ref={len(ref_masks)}"
            )

        # Per-mask IoU (matched by rank/index)
        iou_values: list[float] = []
        n_compare = min(len(trt_masks), len(ref_masks))
        for i in range(n_compare):
            trt_m = np.asarray(trt_masks[i], dtype=bool)
            ref_m = np.asarray(ref_masks[i], dtype=bool)

            # Resize if shapes differ
            if trt_m.shape != ref_m.shape:
                try:
                    from PIL import Image
                    trt_pil = Image.fromarray(trt_m.astype(np.uint8) * 255)
                    trt_resized = trt_pil.resize(
                        (ref_m.shape[1], ref_m.shape[0]), Image.NEAREST)
                    trt_m = np.array(trt_resized).astype(bool)
                except ImportError:
                    gate_details.append(f"Mask {i}: shape mismatch, PIL unavailable")
                    continue

            iou = _compute_iou(trt_m, ref_m)
            iou_values.append(iou)
            metrics[f"mask_{i}_iou"] = iou

        if iou_values:
            mean_iou = sum(iou_values) / len(iou_values)
            # "iou_per_prompt" matches threshold key from prompted_segmentation.json
            metrics["iou_per_prompt"] = mean_iou

            iou_thresh = threshold.metrics.get("iou_per_prompt", 0.5)
            passed_iou = mean_iou >= iou_thresh
            per_metric_pass["iou_per_prompt"] = passed_iou
            gate_details.append(
                f"iou_per_prompt: {mean_iou:.4f} >= {iou_thresh} "
                f"-> {'PASS' if passed_iou else 'FAIL'}"
            )
        elif n_compare == 0:
            gate_details.append("No masks to compare")

        # Mask rank consistency: top-scoring masks should match
        if trt_scores and ref_scores and len(trt_scores) >= 2 and len(ref_scores) >= 2:
            trt_rank = sorted(range(len(trt_scores)),
                              key=lambda i: trt_scores[i], reverse=True)
            ref_rank = sorted(range(len(ref_scores)),
                              key=lambda i: ref_scores[i], reverse=True)
            n_rank = min(len(trt_rank), len(ref_rank))
            rank_matches = sum(
                1 for i in range(n_rank)
                if i < len(trt_rank) and i < len(ref_rank)
                and trt_rank[i] == ref_rank[i]
            )
            rank_consistency = rank_matches / n_rank if n_rank > 0 else 0.0
            metrics["mask_rank_consistency"] = rank_consistency

            rank_thresh = threshold.metrics.get("mask_rank_consistency")
            if rank_thresh is not None:
                passed_rank = rank_consistency >= rank_thresh
                per_metric_pass["mask_rank_consistency"] = passed_rank
                gate_details.append(
                    f"mask_rank_consistency: {rank_consistency:.4f} >= "
                    f"{rank_thresh} -> {'PASS' if passed_rank else 'FAIL'}"
                )

        overall = all(per_metric_pass.values()) if per_metric_pass else (n_compare > 0)
        return CompareResult(
            stage_name=stage.name,
            passed=overall,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"Prompted segmentation: {'PASS' if overall else 'FAIL'}",
        )


# ---------------------------------------------------------------------------
# ObjectDetectionComparator
# ---------------------------------------------------------------------------


class ObjectDetectionComparator:
    """Compare TRT vs reference object detection outputs.

    Canonicalization:
      - Deterministic NMS: suppress overlapping boxes within each class
        using a fixed IoU threshold to ensure consistent detection sets.
      - Stable sort by (class_id asc, score desc, box coords) to ensure
        deterministic ordering before comparison.

    Metric keys aligned with thresholds/defaults/object_detection.json:
      mAP_50, mAP_75, class_precision, class_recall, box_iou_mean
    """

    @property
    def task_strategy(self) -> str:
        return "object_detection"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: Dict[str, float] = {}
        per_metric_pass: Dict[str, bool] = {}
        gate_details: list[str] = []

        # Extract and canonicalize detections
        trt_dets = self._canonicalize(
            trt.data.get("boxes", []),
            trt.data.get("scores", []),
            trt.data.get("class_ids", []),
        )
        ref_dets = self._canonicalize(
            ref.data.get("boxes", []),
            ref.data.get("scores", []),
            ref.data.get("class_ids", []),
        )

        trt_boxes, trt_scores, trt_classes = trt_dets
        ref_boxes, ref_scores, ref_classes = ref_dets

        metrics["trt_num_detections"] = float(len(trt_boxes))
        metrics["ref_num_detections"] = float(len(ref_boxes))

        gate_details.append(
            f"Detections (post-NMS): TRT={len(trt_boxes)}, Ref={len(ref_boxes)}"
        )

        if not ref_boxes:
            gate_details.append("No reference detections to compare against")
            return CompareResult(
                stage_name=stage.name,
                passed=len(trt_boxes) == 0,
                metrics=metrics,
                per_metric_pass=per_metric_pass,
                gate_details=gate_details,
                message="No reference detections",
            )

        # mAP at standard IoU thresholds — keys match threshold file
        for iou_thresh, key in [(0.5, "mAP_50"), (0.75, "mAP_75")]:
            ap = self._compute_ap(
                trt_boxes, trt_scores, trt_classes,
                ref_boxes, ref_classes,
                iou_threshold=iou_thresh,
            )
            metrics[key] = ap

            ap_gate = threshold.metrics.get(key)
            if ap_gate is not None:
                passed_ap = ap >= ap_gate
                per_metric_pass[key] = passed_ap
                gate_details.append(
                    f"{key}: {ap:.4f} >= {ap_gate} "
                    f"-> {'PASS' if passed_ap else 'FAIL'}"
                )

        # Class precision and recall
        if trt_classes and ref_classes:
            trt_cls_set = set(trt_classes)
            ref_cls_set = set(ref_classes)
            cls_precision = (
                len(trt_cls_set & ref_cls_set) / len(trt_cls_set)
                if trt_cls_set else 0.0
            )
            cls_recall = len(trt_cls_set & ref_cls_set) / len(ref_cls_set)
            metrics["class_precision"] = cls_precision
            metrics["class_recall"] = cls_recall

            for mkey in ["class_precision", "class_recall"]:
                mthresh = threshold.metrics.get(mkey)
                if mthresh is not None:
                    passed_m = metrics[mkey] >= mthresh
                    per_metric_pass[mkey] = passed_m
                    gate_details.append(
                        f"{mkey}: {metrics[mkey]:.4f} >= {mthresh} "
                        f"-> {'PASS' if passed_m else 'FAIL'}"
                    )

        # Box IoU distribution (match each TRT box to closest ref box)
        if trt_boxes and ref_boxes:
            box_ious = []
            for trt_box in trt_boxes:
                best_iou = max(
                    _compute_box_iou(trt_box, ref_box) for ref_box in ref_boxes
                )
                box_ious.append(best_iou)
            import statistics
            metrics["box_iou_mean"] = statistics.mean(box_ious)
            metrics["box_iou_median"] = statistics.median(box_ious)
            if len(box_ious) > 1:
                metrics["box_iou_min"] = min(box_ious)

            box_iou_thresh = threshold.metrics.get("box_iou_mean")
            if box_iou_thresh is not None:
                passed_bi = metrics["box_iou_mean"] >= box_iou_thresh
                per_metric_pass["box_iou_mean"] = passed_bi
                gate_details.append(
                    f"box_iou_mean: {metrics['box_iou_mean']:.4f} >= "
                    f"{box_iou_thresh} -> {'PASS' if passed_bi else 'FAIL'}"
                )

        overall = all(per_metric_pass.values()) if per_metric_pass else (len(trt_boxes) > 0)
        return CompareResult(
            stage_name=stage.name,
            passed=overall,
            metrics=metrics,
            per_metric_pass=per_metric_pass,
            gate_details=gate_details,
            message=f"Object detection: {'PASS' if overall else 'FAIL'}",
        )

    @staticmethod
    def _canonicalize(
        boxes: list,
        scores: list[float],
        class_ids: list[int],
        nms_iou_threshold: float = 0.5,
    ) -> tuple[list, list[float], list[int]]:
        """Canonicalize detections: deterministic NMS + stable sort.

        1. Apply per-class NMS with fixed IoU threshold to suppress
           overlapping boxes deterministically.
        2. Stable sort remaining detections by (class_id asc, score desc,
           box coords) to ensure reproducible ordering.
        """
        if not boxes:
            return [], [], []

        n = min(len(boxes), len(scores), len(class_ids))
        boxes = boxes[:n]
        scores = scores[:n]
        class_ids = class_ids[:n]

        # Per-class NMS
        keep_indices: list[int] = []
        unique_classes = sorted(set(class_ids))
        for cls in unique_classes:
            cls_indices = [i for i in range(n) if class_ids[i] == cls]
            # Sort by score descending within class (stable)
            cls_indices.sort(key=lambda i: (-scores[i], tuple(boxes[i])))
            suppressed = set()
            for i, idx_i in enumerate(cls_indices):
                if idx_i in suppressed:
                    continue
                keep_indices.append(idx_i)
                for idx_j in cls_indices[i + 1:]:
                    if idx_j in suppressed:
                        continue
                    iou = _compute_box_iou(boxes[idx_i], boxes[idx_j])
                    if iou >= nms_iou_threshold:
                        suppressed.add(idx_j)

        # Stable sort: (class_id asc, score desc, box coords for tie-breaking)
        keep_indices.sort(
            key=lambda i: (class_ids[i], -scores[i], tuple(boxes[i]))
        )

        out_boxes = [boxes[i] for i in keep_indices]
        out_scores = [scores[i] for i in keep_indices]
        out_classes = [class_ids[i] for i in keep_indices]
        return out_boxes, out_scores, out_classes

    @staticmethod
    def _compute_ap(
        pred_boxes: list,
        pred_scores: list[float],
        pred_classes: list[int],
        gt_boxes: list,
        gt_classes: list[int],
        iou_threshold: float = 0.5,
    ) -> float:
        """Compute Average Precision at a given IoU threshold.

        Uses stable sort by (score desc, box coords) and greedy matching.
        """
        if not pred_boxes or not gt_boxes:
            return 0.0

        # Stable sort predictions by descending score, tie-break on box coords
        indices = sorted(
            range(len(pred_scores)),
            key=lambda i: (-pred_scores[i], tuple(pred_boxes[i])),
        )

        gt_matched = [False] * len(gt_boxes)
        tp_list: list[int] = []
        fp_list: list[int] = []

        for idx in indices:
            pred_box = pred_boxes[idx]
            pred_cls = pred_classes[idx] if idx < len(pred_classes) else -1

            best_iou = 0.0
            best_gt = -1
            for gi, gt_box in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                gt_cls = gt_classes[gi] if gi < len(gt_classes) else -1
                if pred_cls != gt_cls and pred_cls != -1 and gt_cls != -1:
                    continue
                iou = _compute_box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            if best_iou >= iou_threshold and best_gt >= 0:
                gt_matched[best_gt] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

        # Compute precision-recall curve and AP
        tp_cumsum = []
        fp_cumsum = []
        tp_acc = 0
        fp_acc = 0
        for tp, fp in zip(tp_list, fp_list):
            tp_acc += tp
            fp_acc += fp
            tp_cumsum.append(tp_acc)
            fp_cumsum.append(fp_acc)

        n_gt = len(gt_boxes)
        precisions = [
            tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i])
            for i in range(len(tp_cumsum))
        ]
        recalls = [tp_cumsum[i] / n_gt for i in range(len(tp_cumsum))]

        # AP via 11-point interpolation
        ap = 0.0
        for t in [i / 10.0 for i in range(11)]:
            prec_at_recall = [
                precisions[i] for i in range(len(recalls)) if recalls[i] >= t
            ]
            if prec_at_recall:
                ap += max(prec_at_recall) / 11.0

        return ap


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

# Primary plugin for auto-discovery
plugin = SegmentationComparator()

# Register additional comparators at import time
def _register_extras():
    try:
        from ..registry import register_comparator
        register_comparator(PromptedSegmentationComparator())
        register_comparator(ObjectDetectionComparator())
    except Exception:
        try:
            from tests.e2e_harness.registry import register_comparator
            register_comparator(PromptedSegmentationComparator())
            register_comparator(ObjectDetectionComparator())
        except Exception:
            pass

_register_extras()
