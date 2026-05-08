#!/usr/bin/env python3
"""Compare TRT and HF diffusion outputs with a paired vision-language judge.

This is an optional artifact-review tool. It feeds both images to the same
VLM prompt so the model can compare them directly, instead of captioning each
image independently.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_SCORED_NUMERIC_KEYS = (
    "semantic_similarity_0_to_5",
    "trt_prompt_alignment_0_to_5",
    "hf_prompt_alignment_0_to_5",
    "trt_visual_quality_0_to_5",
    "hf_visual_quality_0_to_5",
    "hf_visual_quality_5_to_5",
)
_SCORED_STRING_KEYS = (
    "trt_description",
    "hf_description",
    "trt_relative_to_hf",
    "reason",
)
_GATE_RULE = (
    "fail if semantic_similarity_0_to_5 < 3.0, "
    "trt_prompt_alignment_0_to_5 < 3.0, or trt_visual_quality_0_to_5 < 2.5"
)


def _load_image(path: Path, max_side: int) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def _frames_in(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    frames = sorted(path.glob("frame_*.png"))
    if frames:
        return frames
    return sorted(path.glob("*.png"))


def _select_frame(frames: list[Path]) -> Path | None:
    if not frames:
        return None
    return frames[(len(frames) - 1) // 2]


def _discover_pairs(artifacts_dir: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for result_path in sorted(artifacts_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        case = result.get("case_config", {})
        if case.get("task_strategy") != "diffusion_media_generation":
            continue

        model_dir = result_path.parent
        trt_frame = _select_frame(_frames_in(model_dir / "frames"))
        hf_frame = _select_frame(_frames_in(model_dir / "hf_frames"))
        if hf_frame is None:
            hf_frame = _select_frame(_frames_in(model_dir / "ref_frames"))
        if trt_frame is None or hf_frame is None:
            continue

        inputs = case.get("inputs", {})
        pairs.append({
            "case_name": result.get("case_name") or case.get("name") or model_dir.name,
            "prompt": inputs.get("prompt", ""),
            "trt_image": str(trt_frame),
            "hf_image": str(hf_frame),
        })
    return pairs


def _parse_json(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    if not match:
        return _parse_scored_fields(text)
    try:
        parsed = json.loads(match.group(0))
        if (
            "hf_visual_quality_0_to_5" not in parsed
            and "hf_visual_quality_5_to_5" in parsed
        ):
            parsed["hf_visual_quality_0_to_5"] = parsed["hf_visual_quality_5_to_5"]
        parsed["raw"] = text
        return parsed
    except json.JSONDecodeError:
        return _parse_scored_fields(text)


def _parse_scored_fields(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"raw": text}
    for key in _SCORED_NUMERIC_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            parsed[key] = float(match.group(1))
    for key in _SCORED_STRING_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)
        if match:
            parsed[key] = match.group(1)
    match = re.search(r'"is_regression"\s*:\s*(true|false)', text, re.IGNORECASE)
    if match:
        parsed["is_regression"] = match.group(1).lower() == "true"
    if (
        "hf_visual_quality_0_to_5" not in parsed
        and "hf_visual_quality_5_to_5" in parsed
    ):
        parsed["hf_visual_quality_0_to_5"] = parsed["hf_visual_quality_5_to_5"]
    return parsed


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_gate(judgment: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    similarity = _as_float(judgment.get("semantic_similarity_0_to_5"))
    alignment = _as_float(judgment.get("trt_prompt_alignment_0_to_5"))
    quality = _as_float(judgment.get("trt_visual_quality_0_to_5"))

    if similarity is None:
        reasons.append("missing semantic_similarity_0_to_5")
    elif similarity < 3.0:
        reasons.append(f"semantic_similarity_0_to_5={similarity:.2f} < 3.0")

    if alignment is None:
        reasons.append("missing trt_prompt_alignment_0_to_5")
    elif alignment < 3.0:
        reasons.append(f"trt_prompt_alignment_0_to_5={alignment:.2f} < 3.0")

    if quality is None:
        reasons.append("missing trt_visual_quality_0_to_5")
    elif quality < 2.5:
        reasons.append(f"trt_visual_quality_0_to_5={quality:.2f} < 2.5")

    return {
        "failed": bool(reasons),
        "rule": _GATE_RULE,
        "reasons": reasons,
    }


def _judge_pair(
    model: Any,
    processor: Any,
    device: str,
    pair: dict[str, Any],
    *,
    max_side: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    trt_image = _load_image(Path(pair["trt_image"]), max_side)
    hf_image = _load_image(Path(pair["hf_image"]), max_side)
    prompt = pair.get("prompt") or ""

    question = f"""You are comparing two diffusion pipeline outputs for the same prompt.
Image 1 is TensorRT/TRT output. Image 2 is Hugging Face Diffusers reference.
Prompt: {prompt}

Compare Image 1 to Image 2. Focus on semantic content, scene layout, prompt alignment,
major missing objects, visible artifacts, and whether Image 1 is a user-visible regression.
Do not require pixel-level identity.
Set "is_regression" to true only for gate-level failures such as garbage output,
severe artifacts, missing or incorrect primary subjects, or poor prompt alignment.
Do not mark lower detail than the HF reference alone as a regression.

Return only JSON with these keys:
{{
  "trt_description": string,
  "hf_description": string,
  "semantic_similarity_0_to_5": number,
  "trt_prompt_alignment_0_to_5": number,
  "hf_prompt_alignment_0_to_5": number,
  "trt_visual_quality_0_to_5": number,
  "hf_visual_quality_0_to_5": number,
  "trt_relative_to_hf": "better" | "similar" | "worse",
  "is_regression": boolean,
  "reason": string
}}
Keep "reason" under 30 words."""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": trt_image},
            {"type": "image", "image": hf_image},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[trt_image, hf_image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    judged = _parse_json(answer.strip())
    judged["vlm_gate"] = _apply_gate(judged)
    return {**pair, "vlm_judgment": judged}


def main() -> int:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--case", action="append", default=[],
                        help="Optional case name filter. May be passed more than once.")
    args = parser.parse_args()

    pairs = _discover_pairs(args.artifacts_dir)
    if args.case:
        wanted = set(args.case)
        pairs = [pair for pair in pairs if pair["case_name"] in wanted]
    if not pairs:
        raise SystemExit(f"No TRT/HF diffusion frame pairs found in {args.artifacts_dir}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        min_pixels=224 * 224,
        max_pixels=args.max_side * args.max_side,
    )

    results = []
    any_gate_failed = False
    for pair in pairs:
        result = _judge_pair(
            model, processor, args.device, pair,
            max_side=args.max_side, max_new_tokens=args.max_new_tokens)
        results.append(result)
        judgment = result["vlm_judgment"]
        any_gate_failed = any_gate_failed or bool(judgment.get("vlm_gate", {}).get("failed"))
        print(
            f"{result['case_name']}: similarity="
            f"{judgment.get('semantic_similarity_0_to_5')} "
            f"trt_quality={judgment.get('trt_visual_quality_0_to_5')} "
            f"hf_quality={judgment.get('hf_visual_quality_0_to_5')} "
            f"regression={judgment.get('is_regression')} "
            f"gate_failed={judgment.get('vlm_gate', {}).get('failed')}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model_id": args.model_id,
        "artifacts_dir": str(args.artifacts_dir),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 1 if any_gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
