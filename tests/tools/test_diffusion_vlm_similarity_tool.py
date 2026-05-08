from __future__ import annotations

import json

from tools.evaluate_diffusion_vlm_similarity import (
    _apply_gate,
    _discover_pairs,
    _parse_json,
)


def test_vlm_similarity_gate_fails_low_semantic_similarity():
    result = _apply_gate({
        "semantic_similarity_0_to_5": 0.1,
        "trt_prompt_alignment_0_to_5": 0.0,
        "trt_visual_quality_0_to_5": 0.0,
    })

    assert result["failed"]
    assert any("semantic_similarity" in reason for reason in result["reasons"])


def test_vlm_similarity_gate_allows_minor_quality_delta():
    result = _apply_gate({
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
    })

    assert not result["failed"]


def test_parse_json_normalizes_internvl_quality_key_typo():
    parsed = _parse_json("""
```json
{"hf_visual_quality_5_to_5": 5, "semantic_similarity_0_to_5": 4}
```
""")

    assert parsed["hf_visual_quality_0_to_5"] == 5


def test_parse_json_recovers_scores_from_truncated_vlm_json():
    parsed = _parse_json("""
```json
{
  "semantic_similarity_0_to_5": 4.5,
  "trt_prompt_alignment_0_to_5": 4,
  "trt_visual_quality_0_to_5": 4,
  "is_regression": false,
  "reason": "repeated text
""")

    assert parsed["semantic_similarity_0_to_5"] == 4.5
    assert parsed["trt_prompt_alignment_0_to_5"] == 4
    assert parsed["trt_visual_quality_0_to_5"] == 4
    assert parsed["is_regression"] is False


def test_discover_pairs_accepts_ref_frames_alias(tmp_path):
    model_dir = tmp_path / "artifacts" / "pixart"
    (model_dir / "frames").mkdir(parents=True)
    (model_dir / "ref_frames").mkdir()
    (model_dir / "frames" / "frame_0000.png").write_bytes(b"trt")
    (model_dir / "ref_frames" / "frame_0000.png").write_bytes(b"ref")
    (model_dir / "result.json").write_text(
        json.dumps({
            "case_name": "pixart",
            "case_config": {
                "task_strategy": "diffusion_media_generation",
                "inputs": {"prompt": "a cat"},
            },
        }),
        encoding="utf-8",
    )

    pairs = _discover_pairs(tmp_path / "artifacts")

    assert len(pairs) == 1
    assert pairs[0]["case_name"] == "pixart"
    assert pairs[0]["hf_image"].endswith("ref_frames/frame_0000.png")
