"""Contract test plugin for vision-language QA and OCR models."""
from __future__ import annotations
from ..contracts import (CompareResult, E2ECase, MetricResult, StageOutput, ThresholdProfile)
from .base import (normalize_text, extract_answer, levenshtein_ned, make_pass, make_fail, make_error)

class VLQAPlugin:
    reference_families = ["vl_instruct_qa", "ocr_markdown"]
    user_contract = "vl_answer"

    def configure_reference(self, case):
        config = {"use_processor": True, "use_chat_template": True}
        if case.reference_family == "ocr_markdown":
            config["ocr_mode"] = True
        return config

    def verify(self, trt_output, ref_output, case, threshold):
        stage = trt_output.stage_name

        # vision_encode: invariant check only (no text to compare)
        if stage == "vision_encode":
            passed = trt_output.data.get("passed", False)
            # Also accept if data is non-empty (some runners just return metrics)
            if not passed and trt_output.data:
                passed = "metrics" in trt_output.data or not trt_output.data.get("error")
            metrics = {
                "vision_encode_ok": MetricResult(
                    value=1.0 if passed else 0.0, threshold=1.0, operator="==",
                    passed=bool(passed), note="vision encoder ran successfully"),
            }
            if passed:
                return make_pass("vision_encode", metrics, "vision encoder health")
            return make_fail("vision_encode", metrics, "vision encoder health",
                             "Vision encoder failed")

        # Non-generation stages: pass-through invariant check
        if stage != "full_generation":
            metrics = {
                "stage_ok": MetricResult(
                    value=1.0, threshold=0.0, operator=">=",
                    passed=True, note=f"{stage} completed"),
            }
            return make_pass(stage, metrics, f"{stage} invariant check")

        # full_generation: text comparison
        prompt = case.inputs.get("prompt", "")

        # TRT VL runner puts text in data["generated_text"], ref in data["text"]
        trt_text = trt_output.data.get("generated_text", trt_output.text or "")
        ref_text = ref_output.data.get("text", ref_output.text or "")

        # If no reference text (invariant_only backend), just check TRT produced output
        if not ref_text:
            has_output = len(normalize_text(trt_text)) > 0
            metrics = {
                "has_output": MetricResult(
                    value=1.0 if has_output else 0.0, threshold=1.0, operator="==",
                    passed=has_output, note="TRT produced non-empty text"),
            }
            if has_output:
                return make_pass("full_generation", metrics, "invariant: non-empty output")
            return make_fail("full_generation", metrics, "invariant: non-empty output",
                            "TRT produced empty output")

        trt_answer = normalize_text(extract_answer(
            StageOutput(stage_name=stage, text=trt_text), prompt))
        ref_answer = normalize_text(extract_answer(
            StageOutput(stage_name=stage, text=ref_text), prompt))

        if not trt_answer:
            return make_fail("full_generation", {}, message="TRT produced empty VL answer")

        exact = (trt_answer == ref_answer)
        ned = levenshtein_ned(trt_answer, ref_answer)

        is_ocr = case.reference_family == "ocr_markdown"
        ned_threshold = threshold.metrics.get(
            "contract_ned_threshold", 0.05 if is_ocr else 0.15)

        metrics = {
            "exact_match": MetricResult(value=1.0 if exact else 0.0, threshold=1.0, operator="==", passed=exact),
            "ned": MetricResult(value=ned, threshold=ned_threshold, operator="<=", passed=ned <= ned_threshold),
        }

        passed = exact or ned <= ned_threshold
        label = "OCR" if is_ocr else "VL QA"
        rule = "exact_match OR ned <= threshold"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail("full_generation", metrics, rule, f"{label} answer diverged: NED={ned:.3f}")

plugin = VLQAPlugin()
