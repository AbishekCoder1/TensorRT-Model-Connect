"""Contract checks for stochastic text sampling features."""

from __future__ import annotations

from ..contracts import E2ECase, MetricResult, StageOutput, ThresholdProfile
from .base import make_fail, make_pass, normalize_text, strip_prompt_echo


class SamplingPlugin:
    reference_families = ["sampling_top_p"]
    user_contract = "sampling"

    def configure_reference(self, case: E2ECase) -> dict:
        return {}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ):
        stage = trt_output.stage_name
        if stage != "full_generation":
            metrics = {
                "stage_ok": MetricResult(
                    value=1.0,
                    threshold=1.0,
                    operator="==",
                    passed=True,
                    note=f"{stage} completed",
                )
            }
            return make_pass(stage, metrics, f"{stage} invariant check")

        cpp_rc = int((trt_output.data or {}).get("cpp_returncode", -1))
        command = [
            str(x)
            for x in (trt_output.metadata or {}).get("cpp", {}).get("command", [])
        ]
        if not command:
            command = [str(x) for x in (trt_output.data or {}).get("command", [])]

        prompt = str(case.inputs.get("prompt", ""))
        text = normalize_text(strip_prompt_echo(trt_output.text or "", prompt))
        has_text = bool(text)
        rc_ok = cpp_rc == 0

        required_flags = []
        if float(case.inputs.get("top_p", 1.0)) < 1.0 - 1e-6:
            required_flags.append("--top-p")
        if float(case.inputs.get("temperature", 1.0)) != 1.0:
            required_flags.append("--temperature")
        if int(case.inputs.get("top_k", 1)) != 1:
            required_flags.append("--top-k")
        if int(case.inputs.get("seed", -1)) >= 0:
            required_flags.append("--seed")

        missing_flags = [flag for flag in required_flags if flag not in command]
        flags_ok = not missing_flags

        metrics = {
            "cpp_returncode_ok": MetricResult(
                value=1.0 if rc_ok else 0.0,
                threshold=1.0,
                operator="==",
                passed=rc_ok,
                note=f"cpp_returncode={cpp_rc}",
            ),
            "has_output": MetricResult(
                value=1.0 if has_text else 0.0,
                threshold=1.0,
                operator="==",
                passed=has_text,
                note="TRT produced non-empty sampled text",
            ),
            "sampling_flags_forwarded": MetricResult(
                value=1.0 if flags_ok else 0.0,
                threshold=1.0,
                operator="==",
                passed=flags_ok,
                note=(
                    "missing: " + ", ".join(missing_flags)
                    if missing_flags
                    else "all requested flags present"
                ),
            ),
        }

        passed = rc_ok and has_text and flags_ok
        rule = "cpp_returncode_ok AND has_output AND sampling_flags_forwarded"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            "Top-p sampling contract failed",
        )


plugin = SamplingPlugin()
