from __future__ import annotations

import json

import numpy as np

from tests.e2e_harness.comparators.segmentation import PromptedSegmentationComparator
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest


def test_prompted_segmentation_comparator_loads_reference_masks_from_npy(tmp_path) -> None:
    masks_path = tmp_path / "hf_sam_masks.npy"
    np.save(
        masks_path,
        np.array(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    )

    trt = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [
                np.array([[1, 0], [0, 1]], dtype=np.uint8),
                np.array([[0, 1], [1, 0]], dtype=np.uint8),
                np.array([[1, 1], [0, 0]], dtype=np.uint8),
            ],
            "mask_scores": [0.9, 0.8, 0.7],
        },
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "masks_path": str(masks_path),
            "iou_scores": [0.9, 0.8, 0.7],
        },
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={
            "num_masks_consistency": 1.0,
            "iou_per_prompt": 0.7,
        },
    )

    result = PromptedSegmentationComparator().compare(
        trt, ref, threshold, StageSpec(name="full_inference")
    )

    assert result.status == "passed"
    assert result.metrics["num_masks_consistency"].passed
    assert result.metrics["iou_per_prompt"].passed


def test_manifest_loader_promotes_num_expected_masks_into_inputs(tmp_path) -> None:
    manifest_path = tmp_path / "sam.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "sam-vit-base",
                "hf_id": "facebook/sam-vit-base",
                "bundle": "sam-vit-base.trtfb",
                "family": "sam",
                "runtime_strategy": "prompted_segmentation",
                "test_type": "prompted_segmentation",
                "test_image": "data/test_img.jpeg",
                "point_x": 0.5,
                "point_y": 0.5,
                "num_expected_masks": 3,
            }
        ),
        encoding="utf-8",
    )

    case = load_manifest(manifest_path)

    assert case.inputs["num_expected_masks"] == 3
    assert case.threshold_overrides["num_expected_masks"] == 3
